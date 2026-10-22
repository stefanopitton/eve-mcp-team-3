"""Scientific controls for the optional paired semantic-segmentation demo."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "segmentation.py"
SPEC = importlib.util.spec_from_file_location("eo_segmentation", MODULE_PATH)
segmentation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(segmentation)


def paired_data():
    labels = (np.indices((24, 32)).sum(axis=0) % 3).astype(np.int16)
    values = np.stack([labels / 3, np.indices(labels.shape)[0] / 24]).astype(np.float32)
    patches = []
    unbuffered = np.zeros(labels.shape, dtype=np.uint8)
    buffered = np.zeros(labels.shape, dtype=np.uint8)
    # Training occurs in the west; fixed validation/test rectangles in the east.
    for index, (r0, c0, split) in enumerate([(0, 0, "train"), (16, 0, "train"),
                                             (0, 24, "validation"), (16, 24, "test")]):
        patch = {"patch_id": index, "row_start": r0, "row_stop": r0 + 8,
                 "col_start": c0, "col_stop": c0 + 8, "split": split}
        patches.append(patch)
        unbuffered[r0:r0 + 8, c0:c0 + 8] = segmentation.SPLITS[split]
        if index != 1:
            buffered[r0:r0 + 8, c0:c0 + 8] = segmentation.SPLITS[split]
    return values, labels, patches, unbuffered, buffered


def test_confusion_matrix_iou_and_ignore_labels():
    result = segmentation.segmentation_metrics(
        np.array([0, 0, 1, 1, 2, -100, 255]), np.array([0, 1, 1, 1, 0, -1, -1]), 3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert result["pixel_accuracy"] == pytest.approx(3 / 5)
    assert result["per_class_iou"] == pytest.approx([1 / 3, 2 / 3, 0])
    assert result["mean_iou"] == pytest.approx(1 / 3)
    absent = segmentation.segmentation_metrics(np.array([0, 1]), np.array([0, 1]), 3)
    assert absent["per_class_iou"] == [1, 1, None]
    json.dumps(absent, allow_nan=False)


def test_worldcover_mapping_retains_meaningful_classes_and_ignores_others():
    actual = segmentation.worldcover_to_classes(np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]))
    np.testing.assert_array_equal(actual, [-100, 0, 0, 0, 0, 1, -100, -100, 2, 0, 0, 0])


def test_normalization_uses_only_retained_training_pixels():
    values, _, _, unbuffered, buffered = paired_data()
    mean, std = segmentation.training_normalization(values, buffered == 1)
    changed = values.copy()
    changed[:, buffered != 1] += 10_000
    actual_mean, actual_std = segmentation.training_normalization(changed, buffered == 1)
    np.testing.assert_array_equal(actual_mean, mean)
    np.testing.assert_array_equal(actual_std, std)
    assert not np.array_equal(mean, segmentation.training_normalization(changed, unbuffered == 1)[0])


def test_paired_comparison_rejects_changed_heldout_locations():
    values, labels, patches, first, second = paired_data()
    second[0, 24] = 0
    with pytest.raises(ValueError, match="Validation and test pixel locations"):
        segmentation._validate_inputs(values, labels, patches, first, second, ["a", "b", "c"])


def test_patch_sampling_never_includes_a_neighbouring_split():
    values, labels, patches, unbuffered, buffered = paired_data()
    records = segmentation._records(patches, labels.shape)
    valid = np.ones(labels.shape, dtype=bool)
    active = segmentation._active_patches(records, buffered, valid, "train")
    targets = np.where(buffered == 1, labels, -100)
    _, batch = segmentation._sample_batch(values, targets, active, 32, 16, np.random.default_rng(9))
    assert batch.shape == (16, 8, 8)
    assert np.all(batch >= 0)
    assert all(patch["split"] == "train" for patch in active)
    assert np.count_nonzero(unbuffered == 1) > np.count_nonzero(buffered == 1)


def test_tiny_training_runs_same_budget_and_fixed_test_once(tmp_path):
    pytest.importorskip("torch")
    values, labels, patches, first, second = paired_data()
    report = segmentation.run_segmentation_benchmark(
        values, labels, patches, first, second, resolution_m=10, buffer_m=100,
        output_dir=tmp_path, steps=2, batch_size=2, crop_size=8, validation_interval=1,
        include_size_matched_control=True)
    first_run = report["conditions"]["without_buffer"]
    second_run = report["conditions"]["with_buffer"]
    matched_run = report["conditions"]["size_matched_without_buffer"]
    assert first_run["training_steps"] == second_run["training_steps"] == 2
    assert first_run["test_patch_ids"] == second_run["test_patch_ids"]
    assert first_run["validation_patch_ids"] == second_run["validation_patch_ids"]
    assert first_run["test"]["pixel_count"] == second_run["test"]["pixel_count"] == 64
    assert report["controls"]["test_evaluations_per_condition"] == 1
    assert report["retained_training_labelled_fraction"] == 0.5
    assert matched_run["training_patch_count"] == second_run["training_patch_count"] == 1
    assert matched_run["test_patch_ids"] == first_run["test_patch_ids"]
    assert matched_run["test"]["reference_class_counts"] == first_run["test"]["reference_class_counts"]
    assert second_run["minimum_train_test_patch_edge_distance_m"] >= 100
    assert (tmp_path / "segmentation_report.json").is_file()
    assert (tmp_path / "segmentation_predictions.npz").is_file()
    assert (tmp_path / "segmentation_comparison.png").is_file()
    json.dumps(report, allow_nan=False)


def test_rejects_buffer_label_when_patch_footprints_are_too_close(tmp_path):
    pytest.importorskip("torch")
    values, labels, patches, first, second = paired_data()
    with pytest.raises(ValueError, match="footprints violate"):
        segmentation.run_segmentation_benchmark(
            values, labels, patches, first, second, resolution_m=10, buffer_m=1_000,
            output_dir=tmp_path, steps=1)
