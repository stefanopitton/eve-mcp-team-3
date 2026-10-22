"""Verify exported pixels, paired held-out identity and training-only statistics."""

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from patch_export import export_patch_datasets


def _fixture():
    values = np.arange(4*8*12, dtype="float32").reshape(4, 8, 12) / 100
    labels = np.indices((8, 12)).sum(axis=0).astype("int16") % 3
    patches = [
        {"patch_id": 0, "split": "train", "r0": 0, "r1": 4, "c0": 0, "c1": 4,
         "buffered_retained": True, "unbuffered_retained": True},
        {"patch_id": 1, "split": "train", "r0": 0, "r1": 4, "c0": 4, "c1": 8,
         "buffered_retained": False, "unbuffered_retained": True},
        {"patch_id": 2, "split": "validation", "r0": 4, "r1": 8, "c0": 0, "c1": 4,
         "buffered_retained": True, "unbuffered_retained": True},
        {"patch_id": 3, "split": "test", "r0": 4, "r1": 8, "c0": 4, "c1": 8,
         "buffered_retained": True, "unbuffered_retained": True},
    ]
    masks = {}
    for condition, key in (("split_mask", "buffered_retained"),
                           ("unbuffered_split_mask", "unbuffered_retained")):
        mask = np.zeros(labels.shape, dtype="uint8")
        for p in patches:
            if p[key]:
                mask[p["r0"]:p["r1"], p["c0"]:p["c1"]] = {"train": 1, "validation": 2, "test": 3}[p["split"]]
        masks[condition] = mask
    return values, labels, {"patches": patches, **masks, "report": {"method": "fixture rectangles", "seed": 3}}


def _array(archive, member):
    return np.load(io.BytesIO(archive.read(member)), allow_pickle=False)


def test_roundtrip_preserves_features_coordinates_and_ignores_invalid_targets(tmp_path):
    values, labels, plan = _fixture()
    values[1, 0, 1] = np.nan
    labels[1, 0] = -100
    labels[1, 1] = 255
    plan["split_mask"][2, 2] = plan["unbuffered_split_mask"][2, 2] = 0
    result = export_patch_datasets(values, labels, plan, tmp_path,
                                   source_metadata={"item_id": "test", "year": np.int64(2021)})
    persisted = json.loads(Path(result["manifest_path"]).read_text())
    for exported in persisted["conditions"].values():
        for key in ("archive_path", "metadata_path", "normalization_path"):
            assert not Path(exported[key]).is_absolute()
            assert (tmp_path / exported[key]).is_file()
    with zipfile.ZipFile(result["conditions"]["buffered"]["archive_path"]) as archive:
        x = _array(archive, "train/patch_0000_features.npy")
        y = _array(archive, "train/patch_0000_labels.npy")
        np.testing.assert_array_equal(x, values[:, :4, :4])
        assert x.dtype == np.float32 and y.dtype == np.int16
        expected = labels[:4, :4].copy()
        expected[0, 1] = expected[1, 0] = expected[1, 1] = expected[2, 2] = -100
        np.testing.assert_array_equal(y, expected)
        metadata = json.loads(archive.read("patches_metadata.json"))
        assert metadata["patches"][0]["r0"] == 0 and metadata["patches"][0]["c1"] == 4
        assert metadata["patches"][0]["ignored_pixels"] == 4
        assert metadata["source_metadata"]["year"] == 2021
        assert metadata["channel_names"] == ["B05", "B03", "B04", "B08"]
        assert metadata["upstream_reference"]["commit"].startswith("ac7dfc6")
        assert archive.read("normalization_params.json") == Path(result["conditions"]["buffered"]["normalization_path"]).read_bytes()


def test_paired_heldout_members_are_byte_identical(tmp_path):
    values, labels, plan = _fixture()
    result = export_patch_datasets(values, labels, plan, tmp_path)
    assert result["paired_heldout_identical"]
    with zipfile.ZipFile(result["conditions"]["buffered"]["archive_path"]) as first, \
            zipfile.ZipFile(result["conditions"]["unbuffered"]["archive_path"]) as second:
        heldout = [name for name in first.namelist() if name.startswith(("val/", "test/"))]
        assert len(heldout) == 4
        for name in heldout:
            assert first.read(name) == second.read(name)
        assert "train/patch_0001_features.npy" not in first.namelist()
        assert "train/patch_0001_features.npy" in second.namelist()
    assert result["conditions"]["buffered"]["patch_counts"] == {"train": 1, "val": 1, "test": 1}


def test_normalization_is_reproducible_and_ignores_validation_test(tmp_path):
    values, labels, plan = _fixture()
    first = export_patch_datasets(values, labels, plan, tmp_path / "first")
    changed = values.copy()
    changed[:, 4:, :] += 1_000_000
    second = export_patch_datasets(changed, labels, plan, tmp_path / "second")
    for condition in ("buffered", "unbuffered"):
        a = Path(first["conditions"][condition]["normalization_path"]).read_bytes()
        b = Path(second["conditions"][condition]["normalization_path"]).read_bytes()
        assert a == b
        norm = json.loads(a)
        assert norm["sampled_pixel_count"] > 0
        assert len(norm["normalization"]["low"]) == values.shape[0]
        assert not norm["applied_to_exported_arrays"]
    assert first["conditions"]["buffered"]["selected_train_pixel_count"] == 16
    assert first["conditions"]["unbuffered"]["selected_train_pixel_count"] == 32
    # Changing the extra unbuffered training patch affects only that condition.
    changed = values.copy()
    changed[:, :4, 4:8] += 1000
    third = export_patch_datasets(changed, labels, plan, tmp_path / "third")
    for condition, expected_same in (("buffered", True), ("unbuffered", False)):
        a = Path(first["conditions"][condition]["normalization_path"]).read_bytes()
        b = Path(third["conditions"][condition]["normalization_path"]).read_bytes()
        assert (a == b) == expected_same


def test_long_coordinate_names_and_constant_training_channel_statistics(tmp_path):
    values, labels, plan = _fixture()
    constants = np.array([2, 4, 8, 16], dtype="float32")
    values[:, :4, :4] = constants[:, None, None]
    values[0, 0, 0] = np.nan
    for patch in plan["patches"]:
        for short, long in (("r0", "row_start"), ("r1", "row_stop"),
                            ("c0", "col_start"), ("c1", "col_stop")):
            patch[long] = patch.pop(short)
    result = export_patch_datasets(values, labels, plan, tmp_path)
    norm = json.loads(Path(result["conditions"]["buffered"]["normalization_path"]).read_text())
    assert norm["finite_train_pixel_count"] == 15
    for field in ("low", "high", "mean", "median"):
        np.testing.assert_array_equal(norm["normalization"][field], constants)
    np.testing.assert_array_equal(norm["normalization"]["std"], [0, 0, 0, 0])


def test_rejects_changed_heldout_membership_or_masks(tmp_path):
    values, labels, plan = _fixture()
    plan["patches"][2]["buffered_retained"] = False
    with pytest.raises(ValueError, match="identical validation/test"):
        export_patch_datasets(values, labels, plan, tmp_path / "membership")
    values, labels, plan = _fixture()
    plan["split_mask"][4, 0] = 0
    with pytest.raises(ValueError, match="arrays differ"):
        export_patch_datasets(values, labels, plan, tmp_path / "pixels")


@pytest.mark.parametrize("problem", ["no_training", "nan_training", "coordinates", "channels"])
def test_rejects_invalid_or_unusable_inputs(tmp_path, problem):
    values, labels, plan = _fixture()
    options = {}
    if problem == "no_training":
        plan["patches"][0]["buffered_retained"] = False
    elif problem == "nan_training":
        values[:, :4, :4] = np.nan
    elif problem == "coordinates":
        plan["patches"][0]["r1"] = 100
    else:
        options["channel_names"] = ["B05"]
    with pytest.raises(ValueError):
        export_patch_datasets(values, labels, plan, tmp_path, **options)
