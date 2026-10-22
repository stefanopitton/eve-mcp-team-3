"""Export paired image/label patches using the checkpoint's dataset conventions.

Pixels remain in their original radiometric units. Training-only percentile
statistics are saved for optional preprocessing; exporting never normalizes data.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

UPSTREAM_COMMIT_URL = "https://github.com/stefanopitton/lulc-classification/commit/ac7dfc6578c2c408c87398e15073354752bbd32b"
UPSTREAM_COMMIT = UPSTREAM_COMMIT_URL.rsplit("/", 1)[-1]
IGNORE_LABEL = -100
SPLIT_CODES = {"train": 1, "val": 2, "test": 3}
CONDITIONS = {"buffered": ("buffered_retained", "split_mask"),
              "unbuffered": ("unbuffered_retained", "unbuffered_split_mask")}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _json_bytes(value):
    return (json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n").encode()


def _npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _write_member(archive, filename, data):
    # Fixed timestamps make unchanged .npy members and repeat exports comparable.
    info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, data, compresslevel=9)


def _coordinate(patch, short, long):
    value = patch.get(short, patch.get(long))
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Patch coordinate {short}/{long} must be an integer")
    return int(value)


def _prepare_records(patches, shape):
    records, identifiers = [], set()
    occupied = np.zeros(shape, dtype=bool)
    for index, patch in enumerate(patches):
        identifier = patch.get("patch_id", index)
        if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)) or identifier < 0:
            raise ValueError("patch_id must be a nonnegative integer")
        identifier = int(identifier)
        if identifier in identifiers:
            raise ValueError("patch_id values must be unique")
        identifiers.add(identifier)
        split = {"validation": "val"}.get(patch.get("split"), patch.get("split"))
        if split not in SPLIT_CODES:
            raise ValueError("Each patch split must be train, val/validation or test")
        r0 = _coordinate(patch, "r0", "row_start")
        r1 = _coordinate(patch, "r1", "row_stop")
        c0 = _coordinate(patch, "c0", "col_start")
        c1 = _coordinate(patch, "c1", "col_stop")
        if not (0 <= r0 < r1 <= shape[0] and 0 <= c0 < c1 <= shape[1]):
            raise ValueError("Patch coordinates must define a nonempty window inside the input raster")
        if np.any(occupied[r0:r1, c0:c1]):
            raise ValueError("Candidate patches must not overlap")
        occupied[r0:r1, c0:c1] = True
        for key, _ in CONDITIONS.values():
            if key not in patch:
                raise ValueError(f"Patch records must include {key}")
        records.append({"patch_id": identifier, "split": split, "r0": r0, "r1": r1,
                        "c0": c0, "c1": c1,
                        "buffered_retained": bool(patch["buffered_retained"]),
                        "unbuffered_retained": bool(patch["unbuffered_retained"])})
    if not records:
        raise ValueError("The plan must contain candidate patches")
    records.sort(key=lambda patch: patch["patch_id"])
    for split in ("val", "test"):
        first = {p["patch_id"] for p in records if p["split"] == split and p["buffered_retained"]}
        second = {p["patch_id"] for p in records if p["split"] == split and p["unbuffered_retained"]}
        if first != second:
            raise ValueError("Paired conditions must retain identical validation/test patch membership")
    return records


def _normalization(training, channel_names):
    """Mirror script 04's seed-3, 25% spatial sampling and training-only stats."""
    rng = np.random.default_rng(3)
    samples, fallback, finite_count, geometric_count = [], None, 0, 0
    for data, finite in training:
        geometric_count += int(finite.size)
        finite_count += int(finite.sum())
        if fallback is None and np.any(finite):
            fallback = data[:, finite][:, :1]
        selected = rng.choice([True, False], size=finite.shape, p=(0.25, 0.75)) & finite
        if np.any(selected):
            samples.append(data[:, selected])
    if finite_count == 0:
        raise ValueError("Each condition needs at least one jointly finite training pixel")
    used_fallback = not samples
    if used_fallback:
        # Tiny fixtures can receive an empty Bernoulli draw. No held-out data is
        # substituted: use the first finite training pixel and report it.
        samples = [fallback]
    sampled = np.concatenate(samples, axis=1).astype("float64")
    return {
        "normalization": {
            "percentiles": [1, 99], "low": np.percentile(sampled, 1, axis=1).tolist(),
            "high": np.percentile(sampled, 99, axis=1).tolist(),
            "median": np.median(sampled, axis=1).tolist(),
            "mean": np.mean(sampled, axis=1).tolist(),
            "std": np.std(sampled, axis=1, ddof=0).tolist(),
        },
        "channel_names": list(channel_names), "seed": 3, "stats_retention_fraction": 0.25,
        "selected_train_pixel_count": geometric_count,
        "finite_train_pixel_count": finite_count, "sampled_pixel_count": int(sampled.shape[1]),
        "sampled_pixel_count_per_channel": [int(sampled.shape[1])] * len(channel_names),
        "sample_method": "Independent Bernoulli 0.25 draw per training patch; jointly finite training pixels only",
        "empty_draw_training_pixel_fallback": used_fallback,
        "fit_scope": "Retained training patches in this condition only; validation/test excluded",
        "applied_to_exported_arrays": False,
        "optional_application": "Clip each channel at low/high, scale to [0,1], and impute missing values with the normalized training median; constant channels map to zero",
        "upstream_commit": UPSTREAM_COMMIT,
    }


def export_patch_datasets(
    values, labels, plan, output_dir,
    channel_names=("B05", "B03", "B04", "B08"),
    class_names=("vegetation", "built-up", "water"), source_metadata=None,
):
    """Write paired ZIP datasets, coordinate metadata and training-only statistics.

    Each ZIP contains ``train/``, ``val/`` and ``test/`` float32 ``*_features.npy``
    and int16 ``*_labels.npy`` members. Invalid targets use -100. Feature arrays
    are slices of the original, unnormalized image, including missing values.
    Validation/test membership and member bytes must match between conditions.
    """
    values, labels = np.asarray(values), np.asarray(labels)
    if values.ndim != 3 or labels.ndim != 2 or values.shape[1:] != labels.shape:
        raise ValueError("values must be C,H,W and labels H,W on the same grid")
    if len(channel_names) != values.shape[0] or not values.shape[0]:
        raise ValueError("channel_names must contain exactly one name per feature channel")
    if any(not isinstance(name, str) or not name for name in channel_names):
        raise ValueError("channel_names must contain nonempty strings")
    if any(name.lower() == "scl" for name in channel_names):
        raise ValueError("SCL is a mask, not a continuous feature channel in this export")
    if not 0 < len(class_names) <= np.iinfo(np.int16).max:
        raise ValueError("class_names must describe at least one supported target class")
    features = values.astype("float32")
    valid_features = np.all(np.isfinite(features), axis=0)
    valid_labels = np.isfinite(labels) & (labels >= 0) & (labels < len(class_names))
    valid_labels &= labels == np.floor(labels)
    records = _prepare_records(plan.get("patches", []), labels.shape)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = {}
    heldout_hashes = {}
    for condition, (retention, mask_key) in CONDITIONS.items():
        mask = plan.get(mask_key)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != labels.shape or not np.all(np.isin(mask, [0, 1, 2, 3])):
                raise ValueError("Split masks must have the input grid shape and codes 0/1/2/3")
        selected = [record for record in records if record[retention]]
        if not any(record["split"] == "train" for record in selected):
            raise ValueError(f"{condition} condition has no training patches")
        entries, exported, training = {}, [], []
        patch_counts = dict.fromkeys(SPLIT_CODES, 0)
        class_counts = {split: np.zeros(len(class_names), dtype=np.int64) for split in SPLIT_CODES}
        for record in selected:
            r0, r1, c0, c1 = (record[key] for key in ("r0", "r1", "c0", "c1"))
            window = np.s_[r0:r1, c0:c1]
            split = record["split"]
            in_split = np.ones((r1-r0, c1-c0), dtype=bool) if mask is None else mask[window] == SPLIT_CODES[split]
            finite = valid_features[window] & in_split
            valid = finite & valid_labels[window]
            patch_features = features[:, r0:r1, c0:c1].copy()
            patch_labels = np.full(valid.shape, IGNORE_LABEL, dtype="int16")
            patch_labels[valid] = labels[window][valid].astype("int16")
            stem = f"{split}/patch_{record['patch_id']:04d}"
            feature_name, label_name = stem + "_features.npy", stem + "_labels.npy"
            entries[feature_name] = _npy_bytes(patch_features)
            entries[label_name] = _npy_bytes(patch_labels)
            hashes = {"features": hashlib.sha256(entries[feature_name]).hexdigest(),
                      "labels": hashlib.sha256(entries[label_name]).hexdigest()}
            if split != "train":
                key = (split, record["patch_id"])
                if key in heldout_hashes and heldout_hashes[key] != hashes:
                    raise ValueError("Paired validation/test arrays differ between conditions")
                heldout_hashes[key] = hashes
            counts = np.bincount(patch_labels[valid], minlength=len(class_names))
            class_counts[split] += counts
            patch_counts[split] += 1
            exported.append({**record, "features": feature_name, "labels": label_name,
                             "shape_pixels": list(valid.shape), "class_counts": counts.tolist(),
                             "finite_feature_pixels": int(finite.sum()), "labelled_pixels": int(valid.sum()),
                             "ignored_pixels": int(valid.size - valid.sum()),
                             "feature_invalid_fraction": float(1 - finite.mean()),
                             "label_ignore_fraction": float(1 - valid.mean()), "sha256": hashes})
            if split == "train":
                training.append((patch_features, finite))
        normalization = _normalization(training, channel_names)
        metadata = {
            "schema_version": "eo-spatial-patch-export-v1", "condition": condition,
            "channel_names": list(channel_names), "num_channels": values.shape[0],
            "class_names": {str(i): name for i, name in enumerate(class_names)},
            "label_mapping": {str(i): i for i in range(len(class_names))},
            "ignore_label": IGNORE_LABEL, "feature_dtype": "float32", "label_dtype": "int16",
            "normalization_applied": False, "patches_per_split": patch_counts,
            "class_counts_per_split": {split: counts.tolist() for split, counts in class_counts.items()},
            "patches": exported, "candidate_patch_count": len(records),
            "geometry": plan.get("report", {}), "source_metadata": source_metadata or {},
            "upstream_reference": {
                "repository": "https://github.com/stefanopitton/lulc-classification",
                "branch": "stepit/eve-hackathon-checkpoint", "commit": UPSTREAM_COMMIT,
                "script": "scripts/04_extract_patches.py",
                "source_url": f"https://github.com/stefanopitton/lulc-classification/blob/{UPSTREAM_COMMIT}/scripts/04_extract_patches.py",
            },
            "explicit_differences": [
                "The caller supplies the spatial plan; its geometry and retention records are authoritative.",
                "This bounded demo uses the supplied calibrated channels instead of requiring four seasonal 13-channel composites.",
                "Three grouped model classes are supplied by the demo; the original WorldCover-to-group mapping is recorded in source_metadata.",
                "Missing labels are int16 -100 instead of upstream uint8 255; missing feature values remain unchanged.",
                "All retained patches are exported, with invalid pixels ignored rather than an additional post-plan patch quality exclusion.",
                "Normalization sampling uses jointly finite training pixels; an empty Bernoulli draw falls back to one training pixel and is reported.",
                "Validation and test membership are held fixed across the paired buffer conditions.",
            ],
        }
        prepared[condition] = (entries, metadata, normalization)

    manifest = {"schema_version": "eo-spatial-patch-export-v1", "conditions": {},
                "paired_heldout_identical": True, "upstream_commit": UPSTREAM_COMMIT}
    for condition, (entries, metadata, normalization) in prepared.items():
        directory = output / condition
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path, normalization_path = directory / "patches_metadata.json", directory / "normalization_params.json"
        metadata_bytes, normalization_bytes = _json_bytes(metadata), _json_bytes(normalization)
        metadata_path.write_bytes(metadata_bytes)
        normalization_path.write_bytes(normalization_bytes)
        archive_path = output / f"{condition}_patches.zip"
        temporary = archive_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, data in sorted(entries.items()):
                _write_member(archive, name, data)
            _write_member(archive, "patches_metadata.json", metadata_bytes)
            _write_member(archive, "normalization_params.json", normalization_bytes)
        temporary.replace(archive_path)
        manifest["conditions"][condition] = {
            "archive_path": str(archive_path), "metadata_path": str(metadata_path),
            "normalization_path": str(normalization_path), "archive_bytes": archive_path.stat().st_size,
            "archive_below_5_mib": archive_path.stat().st_size < 5 * 1024**2,
            "patch_counts": metadata["patches_per_split"],
            "class_counts": metadata["class_counts_per_split"],
            "selected_train_pixel_count": normalization["selected_train_pixel_count"],
            "finite_train_pixel_count": normalization["finite_train_pixel_count"],
            "sampled_train_pixel_count": normalization["sampled_pixel_count"],
        }
    manifest["manifest_path"] = str(output / "patch_export_manifest.json")
    # The live API returns absolute paths. The saved manifest is portable with
    # the exported directory, including when extracted on another machine.
    persisted = _jsonable(manifest)
    for exported in persisted["conditions"].values():
        for key in ("archive_path", "metadata_path", "normalization_path"):
            exported[key] = str(Path(exported[key]).relative_to(output))
    persisted["manifest_path"] = "patch_export_manifest.json"
    persisted["path_scope"] = "Paths relative to the directory containing this manifest"
    Path(manifest["manifest_path"]).write_bytes(_json_bytes(persisted))
    return manifest
