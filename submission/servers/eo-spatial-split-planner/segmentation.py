"""Small, paired semantic-segmentation experiment for the spatial split demo.

This module is deliberately separate from the MCP server: PyTorch is an optional
demo dependency, and no pretrained weights or network downloads are required.
The comparison measures a change in evaluation after excluding nearby training
patches; it does not assume that a buffer must improve a model's score.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

IGNORE_LABEL = -100
SPLITS = {"train": 1, "validation": 2, "test": 3}


def worldcover_to_classes(worldcover):
    """Map WorldCover 2021 to vegetation, built-up, water; ignore other codes."""
    source = np.asarray(worldcover)
    labels = np.full(source.shape, IGNORE_LABEL, dtype=np.int16)
    labels[np.isin(source, [10, 20, 30, 40, 90, 95, 100])] = 0
    labels[source == 50] = 1
    labels[source == 80] = 2
    return labels


def segmentation_metrics(target, prediction, num_classes: int) -> dict:
    """Pixel confusion matrix and IoU, excluding labels outside [0, K)."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError("target and prediction must have identical shapes")
    valid = (target >= 0) & (target < num_classes)
    if not np.any(valid):
        raise ValueError("No valid reference pixels are available for evaluation")
    if np.any((prediction[valid] < 0) | (prediction[valid] >= num_classes)):
        raise ValueError("prediction contains an invalid class on evaluated pixels")
    matrix = np.bincount(
        num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64),
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)
    reference_count, prediction_count = matrix.sum(axis=1), matrix.sum(axis=0)
    intersection = np.diag(matrix)
    union = reference_count + prediction_count - intersection
    iou = np.divide(intersection, union, out=np.full(num_classes, np.nan), where=union > 0)
    return {
        "pixel_count": int(matrix.sum()),
        "pixel_accuracy": float(intersection.sum() / matrix.sum()),
        "mean_iou": float(np.nanmean(iou)),
        "per_class_iou": [float(value) if np.isfinite(value) else None for value in iou],
        "reference_class_counts": reference_count.tolist(),
        "predicted_class_counts": prediction_count.tolist(),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": "rows = reference, columns = prediction",
    }


def _records(patches, shape):
    records = []
    for index, patch in enumerate(patches):
        split = patch.get("split", {1: "train", 2: "validation", 3: "test"}.get(patch.get("split_code")))
        if split not in SPLITS:
            raise ValueError("Each patch must have split train, validation, or test")
        bounds = tuple(patch[key] for key in ("row_start", "row_stop", "col_start", "col_stop"))
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in bounds):
            raise ValueError("Patch bounds must be integer pixel coordinates")
        r0, r1, c0, c1 = map(int, bounds)
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            raise ValueError("Patch bounds must lie inside the input image")
        records.append({"patch_id": patch.get("patch_id", index), "split": split,
                        "bounds": (r0, r1, c0, c1)})
    return records


def _validate_inputs(values, labels, patches, unbuffered_mask, buffered_mask, class_names):
    values, labels = np.asarray(values), np.asarray(labels)
    if values.ndim != 3 or not 1 <= values.shape[0] <= 16:
        raise ValueError("values must have shape [bands, rows, columns], with 1 to 16 bands")
    if values.shape[1:] != labels.shape or labels.ndim != 2 or labels.dtype.kind not in "ui":
        raise ValueError("labels must be an aligned 2-D integer class raster")
    if labels.size > 4_000_000:
        raise ValueError("Crop the segmentation demo to at most 4,000,000 pixels")
    if not 2 <= len(class_names) <= 32:
        raise ValueError("class_names must define 2 to 32 classes")
    first, second = np.asarray(unbuffered_mask), np.asarray(buffered_mask)
    if first.shape != labels.shape or second.shape != labels.shape:
        raise ValueError("Both paired split masks must match the label raster")
    if not np.all(np.isin(first, [0, 1, 2, 3])) or not np.all(np.isin(second, [0, 1, 2, 3])):
        raise ValueError("Split masks must use 0=excluded, 1=train, 2=validation, 3=test")
    for code in (2, 3):
        if not np.array_equal(first == code, second == code):
            raise ValueError("Validation and test pixel locations must be identical in both conditions")
    if np.any((second == 1) & (first != 1)):
        raise ValueError("Buffered training pixels must be a subset of unbuffered training pixels")
    valid = np.all(np.isfinite(values), axis=0) & (labels >= 0) & (labels < len(class_names))
    records = _records(patches, labels.shape)
    coverage = np.zeros(labels.shape, dtype=np.uint8)
    for patch in records:
        r0, r1, c0, c1 = patch["bounds"]
        region = coverage[r0:r1, c0:c1]
        if np.any(region):
            raise ValueError("Candidate rectangle patches must not overlap")
        region[:] = SPLITS[patch["split"]]
    for mask in (first, second):
        if np.any((mask > 0) & (mask != coverage)):
            raise ValueError("Split masks must agree with the candidate patch geometry")
        for name, code in SPLITS.items():
            if not np.any(valid & (mask == code)):
                raise ValueError(f"Each condition needs valid labelled {name} pixels")
    return values.astype(np.float32), labels, records, first, second, valid


def training_normalization(values, train_mask):
    """Fit channel statistics on training pixels only; no scene-wide fitting."""
    values, train_mask = np.asarray(values), np.asarray(train_mask, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != train_mask.shape:
        raise ValueError("Training mask must match [bands, rows, columns] values")
    effective = train_mask & np.all(np.isfinite(values), axis=0)
    if not np.any(effective):
        raise ValueError("No finite training pixels for normalization")
    mean = values[:, effective].mean(axis=1, dtype=np.float64)
    std = values[:, effective].std(axis=1, dtype=np.float64)
    std = np.where(std > 1e-6, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def _active_patches(records, mask, valid, split):
    result = []
    for patch in records:
        if patch["split"] != split:
            continue
        r0, r1, c0, c1 = patch["bounds"]
        rows, columns = np.nonzero(valid[r0:r1, c0:c1] & (mask[r0:r1, c0:c1] == SPLITS[split]))
        if rows.size:
            result.append({**patch, "valid_rows": rows, "valid_columns": columns})
    return result


def _minimum_distance(first, second, resolution_m):
    distances = []
    for left in first:
        a0, a1, b0, b1 = left["bounds"]
        for right in second:
            c0, c1, d0, d1 = right["bounds"]
            dy = max(c0 - a1, a0 - c1, 0)
            dx = max(d0 - b1, b0 - d1, 0)
            distances.append(math.hypot(dx, dy) * resolution_m)
    return float(min(distances)) if distances else None


def _sample_batch(image, targets, patches, crop_size, batch_size, generator):
    crop = min(crop_size, min(min(patch["bounds"][1] - patch["bounds"][0],
                                    patch["bounds"][3] - patch["bounds"][2]) for patch in patches))
    images, labels = [], []
    for _ in range(batch_size):
        patch = patches[int(generator.integers(len(patches)))]
        r0, r1, c0, c1 = patch["bounds"]
        anchor = int(generator.integers(len(patch["valid_rows"])))
        row, column = int(patch["valid_rows"][anchor]), int(patch["valid_columns"][anchor])
        # Pick a crop containing one valid training pixel. This never crosses
        # rectangle boundaries, including when labels contain nodata holes.
        row = int(generator.integers(max(0, row - crop + 1), min(row, r1 - r0 - crop) + 1))
        column = int(generator.integers(max(0, column - crop + 1), min(column, c1 - c0 - crop) + 1))
        images.append(image[:, r0 + row:r0 + row + crop, c0 + column:c0 + column + crop])
        labels.append(targets[r0 + row:r0 + row + crop, c0 + column:c0 + column + crop])
    return np.stack(images), np.stack(labels)


def _evaluate(model, image, labels, mask, valid, patches, split, torch, num_classes):
    from torch.nn import functional

    predictions = np.full(labels.shape, -1, dtype=np.int16)
    loss_sum, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for patch in patches:
            r0, r1, c0, c1 = patch["bounds"]
            selected = valid[r0:r1, c0:c1] & (mask[r0:r1, c0:c1] == SPLITS[split])
            target = np.where(selected, labels[r0:r1, c0:c1], IGNORE_LABEL).astype(np.int64)
            logits = model(torch.from_numpy(image[:, r0:r1, c0:c1][None].copy()))
            loss_sum += float(functional.cross_entropy(
                logits, torch.from_numpy(target[None]), ignore_index=IGNORE_LABEL, reduction="sum"))
            count += int(selected.sum())
            predicted = logits.argmax(dim=1)[0].cpu().numpy()
            predictions[r0:r1, c0:c1][selected] = predicted[selected]
    target = np.where(valid & (mask == SPLITS[split]), labels, IGNORE_LABEL)
    return {**segmentation_metrics(target, predictions, num_classes), "cross_entropy": loss_sum / count}, predictions


def run_segmentation_benchmark(
    values,
    labels,
    patches,
    unbuffered_mask,
    buffered_mask,
    *,
    resolution_m: float,
    buffer_m: float,
    output_dir,
    class_names=("vegetation", "built-up", "water"),
    steps: int = 150,
    batch_size: int = 8,
    crop_size: int = 32,
    learning_rate: float = 0.003,
    seed: int = 42,
    validation_interval: int = 25,
    cpu_threads: int = 2,
    label_provenance: dict | None = None,
    include_size_matched_control: bool = False,
) -> dict:
    """Train the same tiny CNN twice on paired, random 2-D rectangle splits.

    Each mask uses 0=excluded, 1=train, 2=validation, 3=test. Candidate patches
    provide row_start/stop, col_start/stop, and split. Fixed held-out masks must
    match exactly; buffered training is a subset. Test labels are used once per
    selected model, after validation checkpoint selection has finished.
    """
    try:
        import torch
        from torch import nn
        from torch.nn import functional
    except ImportError as error:
        raise ImportError("Install requirements-demo.txt to run the optional segmentation benchmark") from error

    for name, value, maximum in (("steps", steps, 2_000), ("batch_size", batch_size, 32),
                                 ("crop_size", crop_size, 128), ("validation_interval", validation_interval, 2_000),
                                 ("cpu_threads", cpu_threads, 8)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    if not math.isfinite(resolution_m) or resolution_m <= 0 or not math.isfinite(buffer_m) or buffer_m < 0:
        raise ValueError("resolution_m must be positive and buffer_m nonnegative, in metres")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 1:
        raise ValueError("learning_rate must be finite and between 0 and 1")
    values, labels, records, unbuffered, buffered, valid = _validate_inputs(
        values, labels, patches, unbuffered_mask, buffered_mask, class_names)
    buffered_train = _active_patches(records, buffered, valid, "train")
    buffered_heldout = (_active_patches(records, buffered, valid, "validation")
                        + _active_patches(records, buffered, valid, "test"))
    separation = _minimum_distance(buffered_train, buffered_heldout, resolution_m)
    if separation + 1e-6 < buffer_m:
        raise ValueError("Buffered training rectangle footprints violate the requested held-out guard")
    experiment_masks = [("without_buffer", unbuffered), ("with_buffer", buffered)]
    if include_size_matched_control:
        candidate_train = _active_patches(records, unbuffered, valid, "train")
        selected = np.random.default_rng(seed + 1).choice(len(candidate_train), len(buffered_train), replace=False)
        control_mask = unbuffered.copy()
        control_mask[control_mask == 1] = 0
        for index in selected:
            r0, r1, c0, c1 = candidate_train[int(index)]["bounds"]
            control_mask[r0:r1, c0:c1] = unbuffered[r0:r1, c0:c1]
        experiment_masks.append(("size_matched_without_buffer", control_mask))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    conditions, prediction_arrays = {}, {}
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    try:
        for condition, mask in experiment_masks:
            train = _active_patches(records, mask, valid, "train")
            validation = _active_patches(records, mask, valid, "validation")
            test = _active_patches(records, mask, valid, "test")
            mean, std = training_normalization(values, valid & (mask == 1))
            image = (values - mean[:, None, None]) / std[:, None, None]
            image = np.where(np.isfinite(image), image, 0).astype(np.float32)
            targets = np.where(valid & (mask == 1), labels, IGNORE_LABEL).astype(np.int64)
            generator = np.random.default_rng(seed)
            curve, best_state, best_iou, best_step = [], None, -1.0, None
            # fork_rng prevents this reproducible demo changing its caller's
            # global torch random stream. Both models start at identical weights.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = nn.Sequential(
                    nn.Conv2d(values.shape[0], 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(16, len(class_names), 1),
                )
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
                for step in range(1, steps + 1):
                    model.train()
                    batch_image, batch_labels = _sample_batch(image, targets, train, crop_size, batch_size, generator)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(batch_image))
                    loss = functional.cross_entropy(logits, torch.from_numpy(batch_labels), ignore_index=IGNORE_LABEL)
                    loss.backward()
                    optimizer.step()
                    if step == 1 or step % validation_interval == 0 or step == steps:
                        measured, _ = _evaluate(model, image, labels, mask, valid, validation,
                                                "validation", torch, len(class_names))
                        curve.append({"step": step, "training_cross_entropy": float(loss.detach()),
                                      "validation_cross_entropy": measured["cross_entropy"],
                                      "validation_mean_iou": measured["mean_iou"]})
                        if measured["mean_iou"] > best_iou:
                            best_iou, best_step = measured["mean_iou"], step
                            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                model.load_state_dict(best_state)
                # Held-out test evaluation occurs only after selecting by validation.
                test_metrics, predictions = _evaluate(model, image, labels, mask, valid, test,
                                                       "test", torch, len(class_names))
                torch.save({"state_dict": best_state, "input_channels": values.shape[0],
                            "class_names": list(class_names), "normalization_mean": mean.tolist(),
                            "normalization_std": std.tolist(), "selected_step": best_step},
                           output / f"{condition}_model.pt")
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
            counts = {name: np.bincount(labels[valid & (mask == code)], minlength=len(class_names)).tolist()
                      for name, code in SPLITS.items()}
            conditions[condition] = {
                "selected_step": best_step,
                "selected_validation_mean_iou": best_iou,
                "training_steps": steps,
                "training_patch_count": len(train), "validation_patch_count": len(validation),
                "test_patch_count": len(test),
                "training_patch_ids": [patch["patch_id"] for patch in train],
                "validation_patch_ids": [patch["patch_id"] for patch in validation],
                "test_patch_ids": [patch["patch_id"] for patch in test],
                "labelled_class_counts": counts,
                "training_labelled_pixel_count": int(np.sum(counts["train"])),
                "training_labelled_area_m2": float(np.sum(counts["train"]) * resolution_m**2),
                "minimum_train_validation_patch_edge_distance_m": _minimum_distance(train, validation, resolution_m),
                "minimum_train_test_patch_edge_distance_m": _minimum_distance(train, test, resolution_m),
                "normalization": {"fitted_on": "this condition's finite, labelled training pixels only",
                                  "channel_mean": mean.tolist(), "channel_std": std.tolist()},
                "learning_curve": curve, "test": test_metrics,
                "checkpoint_file": f"{condition}_model.pt",
            }
            prediction_arrays[condition] = predictions
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)

    retained = conditions["with_buffer"]["training_labelled_pixel_count"]
    original = conditions["without_buffer"]["training_labelled_pixel_count"]
    report = {
        "experiment": "paired tiny CNN semantic segmentation on fixed held-out rectangle patches",
        "primary_conditions": ["without_buffer", "with_buffer"],
        "class_names": list(class_names), "label_provenance": label_provenance or {},
        "model": {"architecture": "3 x Conv2d(3x3, 16 channels) + ReLU, then Conv2d(1x1, K classes)",
                  "trainable_parameters": parameter_count, "pretrained_weights": False,
                  "device": "cpu", "torch_version": torch.__version__},
        "controls": {"same_model_initialization_seed": seed, "same_optimizer_steps": steps,
                     "optimizer": "AdamW", "learning_rate": learning_rate, "weight_decay": 1e-4,
                     "batch_size": batch_size, "maximum_crop_size_px": crop_size,
                     "validation_interval": validation_interval, "checkpoint_selection": "maximum validation mean IoU; earliest tie",
                     "test_evaluations_per_condition": 1, "held_out_pixel_locations_identical": True,
                     "cpu_threads": cpu_threads, "training_class_weighting": "none",
                     "size_matched_control": {
                         "enabled": include_size_matched_control, "subsampling_seed": seed + 1,
                         "description": "Randomly retain the same number of training rectangles as the buffered condition, without a distance filter. Labelled pixel counts can still differ."}},
        "buffer_m": float(buffer_m), "resolution_m": float(resolution_m),
        "retained_training_labelled_fraction": retained / original,
        "conditions": conditions,
        "test_mean_iou_difference_without_minus_with_buffer": (
            conditions["without_buffer"]["test"]["mean_iou"] - conditions["with_buffer"]["test"]["mean_iou"]),
        "interpretation": (
            "A buffer increases geometric separation while reducing the training pool. "
            "Either model can score higher. This single-scene, single-seed comparison cannot establish "
            "a causal accuracy benefit or prove independence. Reference-map agreement is not field-validated accuracy."),
        "elapsed_seconds": time.monotonic() - started,
        "artifacts": {"predictions": "segmentation_predictions.npz", "comparison": "segmentation_comparison.png",
                      "test_patch_montage": "test_patch_montage.png"},
    }
    np.savez_compressed(output / report["artifacts"]["predictions"], **prediction_arrays,
                        **{f"{name}_split_mask": mask for name, mask in experiment_masks},
                        labels=labels, valid_mask=valid, unbuffered_mask=unbuffered, buffered_mask=buffered)
    _plot_comparison(values, labels, valid, unbuffered, buffered, prediction_arrays, report,
                     output / report["artifacts"]["comparison"])
    _plot_test_patch_montage(values, labels, valid, records, buffered, prediction_arrays, report,
                            output / report["artifacts"]["test_patch_montage"])
    (output / "segmentation_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _plot_comparison(values, labels, valid, unbuffered, buffered, predictions, report, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    colors = ["#57a345", "#d16b52", "#3888c2"]
    if len(report["class_names"]) > len(colors):
        colors = list(plt.get_cmap("turbo")(np.linspace(0, 1, len(report["class_names"]))))
    palette = ListedColormap(colors[:len(report["class_names"])]).with_extremes(bad="#eeeeee")
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    # Sentinel demo feature order is B05/B03/B04/B08: NIR/red/green false colour.
    channels = [3, 2, 1] if values.shape[0] >= 4 else [0, 0, 0]
    rgb = np.moveaxis(values[channels], 0, -1).copy()
    for channel in range(3):
        finite = rgb[..., channel][np.isfinite(rgb[..., channel])]
        lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
        rgb[..., channel] = np.clip((rgb[..., channel] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    axes[0, 0].imshow(np.nan_to_num(rgb))
    axes[0, 0].set_title("Sentinel-2 features (NIR/red/green false colour)")
    axes[0, 1].imshow(np.ma.masked_where(~valid, labels), cmap=palette, vmin=-0.5, vmax=len(report["class_names"]) - 0.5,
                      interpolation="nearest")
    axes[0, 1].set_title("Aligned reference classes")
    split_palette = ListedColormap(["#eeeeee", "#57a345", "#e6b34c", "#725ba7"])
    axes[0, 2].imshow(buffered, cmap=split_palette, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4), interpolation="nearest")
    # Thin red contours show candidate train rectangles removed by the buffer.
    removed = (unbuffered == 1) & (buffered == 0)
    if np.any(removed) and not np.all(removed):
        axes[0, 2].contour(removed, levels=[0.5], colors="#ce4b42", linewidths=0.5)
    axes[0, 2].set_title(f"Retained rectangle patches; {report['buffer_m']:,.0f} m requested guard")
    for axis, condition in zip(axes[1, :2], ("without_buffer", "with_buffer"), strict=True):
        prediction = predictions[condition]
        axis.imshow(np.ma.masked_where(prediction < 0, prediction), cmap=palette,
                    vmin=-0.5, vmax=len(report["class_names"]) - 0.5, interpolation="nearest")
        metric = report["conditions"][condition]["test"]
        axis.set_title(f"{condition.replace('_', ' ').capitalize()}: test mIoU {metric['mean_iou']:.3f}")
    for condition, color in (("without_buffer", "#ce7540"), ("with_buffer", "#356fa6"),
                             ("size_matched_without_buffer", "#7c579d")):
        if condition not in report["conditions"]:
            continue
        curve = report["conditions"][condition]["learning_curve"]
        axes[1, 2].plot([point["step"] for point in curve],
                        [point["validation_mean_iou"] for point in curve], marker="o", color=color,
                        label=condition.replace("_", " "))
    axes[1, 2].set(xlabel="Optimizer step", ylabel="Validation mean IoU", ylim=(0, 1))
    axes[1, 2].set_title("Select by validation; evaluate test once")
    axes[1, 2].legend()
    for axis in list(axes[0]) + list(axes[1, :2]):
        axis.set_xticks([])
        axis.set_yticks([])
    handles = [Patch(color=colors[index], label=name) for index, name in enumerate(report["class_names"])]
    axes[0, 1].legend(handles=handles, loc="lower right", fontsize=8)
    axes[0, 2].legend(handles=[Patch(color=split_palette.colors[code], label=name)
                               for name, code in SPLITS.items()], loc="lower right", fontsize=8)
    figure.suptitle("Paired segmentation demo: same held-out patches, model seed and training budget")
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_test_patch_montage(values, labels, valid, records, buffered, predictions, report, path):
    """Show up to 16 deterministic, evenly spaced test rectangles, without score selection."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    patches = sorted(_active_patches(records, buffered, valid, "test"), key=lambda patch: patch["patch_id"])
    total_test_patches = len(patches)
    if total_test_patches > 16:
        patches = [patches[int(index)] for index in np.linspace(0, total_test_patches - 1, 16, dtype=int)]
    report["montage_sampling"] = {"total_test_patches": total_test_patches,
                                  "displayed_patch_ids": [patch["patch_id"] for patch in patches],
                                  "selection": "up to 16 evenly spaced indices in sorted test patch IDs; no score selection"}
    conditions = ["without_buffer", "with_buffer"]
    if "size_matched_without_buffer" in predictions:
        conditions.append("size_matched_without_buffer")
    headers = ["Sentinel-2\nNIR / red / green", "WorldCover 2021\nreference",
               "No TRAIN erosion\nprediction", f"{report['buffer_m'] / 1000:.2f} km buffer\nprediction"]
    if len(conditions) == 3:
        headers.append("Size matched, no TRAIN erosion\nprediction")
    colors = ["#57a345", "#d16b52", "#3888c2"]
    if len(report["class_names"]) > len(colors):
        colors = list(plt.get_cmap("turbo")(np.linspace(0, 1, len(report["class_names"]))))
    palette = ListedColormap(colors[:len(report["class_names"])]).with_extremes(bad="#eeeeee")
    channels = [3, 2, 1] if values.shape[0] >= 4 else [0, 0, 0]
    rgb = np.moveaxis(values[channels], 0, -1).copy()
    for channel in range(3):
        finite = rgb[..., channel][np.isfinite(rgb[..., channel])]
        lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0, 1)
        rgb[..., channel] = np.clip((rgb[..., channel] - lo) / max(float(hi - lo), 1e-6), 0, 1)
    figure, axes = plt.subplots(len(patches), len(headers), squeeze=False,
                                figsize=(3.1 * len(headers), 2.3 * len(patches)), layout="constrained")
    for column, title in enumerate(headers):
        axes[0, column].set_title(title, fontsize=11)
    for row, patch in enumerate(patches):
        r0, r1, c0, c1 = patch["bounds"]
        selected = valid[r0:r1, c0:c1] & (buffered[r0:r1, c0:c1] == 3)
        axes[row, 0].imshow(np.nan_to_num(rgb[r0:r1, c0:c1]), interpolation="nearest")
        axes[row, 0].set_ylabel(f"Patch {patch['patch_id']}\n{int(selected.sum()):,} valid pixels", fontsize=10)
        panels = [np.ma.masked_where(~selected, labels[r0:r1, c0:c1])]
        panels.extend(np.ma.masked_where(~selected, predictions[condition][r0:r1, c0:c1])
                      for condition in conditions)
        for column, panel in enumerate(panels, start=1):
            axes[row, column].imshow(panel, cmap=palette, vmin=-0.5,
                                     vmax=len(report["class_names"]) - 0.5, interpolation="nearest")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(f"{len(patches)} of {total_test_patches} fixed test rectangles; deterministic sample, metrics use all", fontsize=15)
    figure.supxlabel("Reference and prediction colors: vegetation = green; built-up = terracotta; water = blue. Excluded pixels = grey.", fontsize=10)
    figure.savefig(path, dpi=140)
    plt.close(figure)
