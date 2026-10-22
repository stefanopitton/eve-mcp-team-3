"""Reference extraction parity and rectangular, quality-aware adaptations."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))
import block_patches as blocks  # noqa: E402

SPEC = importlib.util.spec_from_file_location("eo_block_split_core", SERVER_DIR / "core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def _plan(valid=None, **overrides):
    arguments = dict(pixel_size_m=10, buffer_m=30, patch_size_pixels=(4, 5),
                     block_size_pixels=(32, 40), seed_candidates=20, edge_policy="crop")
    arguments.update(overrides)
    return blocks.plan_reference_blocks(np.ones((98, 142), bool) if valid is None else valid,
                                       **arguments)


def _literal_reference_buffer(patch_labels, radius):
    """Independent literal upstream shift-and-compare algorithm."""
    padded = np.pad(patch_labels, radius, mode="edge")
    expected = np.ones(patch_labels.shape, bool)
    for row_shift in range(-radius, radius + 1):
        for col_shift in range(-radius, radius + 1):
            start_y, start_x = row_shift + radius, col_shift + radius
            expected &= patch_labels == padded[
                start_y:start_y + patch_labels.shape[0],
                start_x:start_x + patch_labels.shape[1]]
    return expected


def test_block_assignment_matches_pinned_reference_and_new_default_fractions():
    # Frozen from checkpoint assign_blocks(seed=3,grid_size=3) after +1 code mapping.
    expected = [[1, 3, 2], [3, 1, 1], [1, 1, 1]]
    np.testing.assert_array_equal(blocks.assign_blocks(3, (3, 3)), expected)
    labels = blocks.assign_blocks(0, (4, 5))
    assert np.count_nonzero(labels == 2) == 2  # ceil(20 * .1)
    assert np.count_nonzero(labels == 3) == 4  # ceil(20 * .2)
    assert np.count_nonzero(labels == 1) == 14
    for code in (1, 2, 3):
        positions = np.argwhere(labels == code)
        assert len(np.unique(positions[:, 0])) >= 2
        assert len(np.unique(positions[:, 1])) >= 2
    assert not np.array_equal(labels, blocks.assign_blocks(4, (4, 5)))
    assert core.create_spatial_split.__kwdefaults__["split_ratios"] == (0.7, 0.1, 0.2)


@pytest.mark.parametrize("radius", [0, 1, 2, 3])
def test_symmetric_buffer_matches_upstream_shift_comparisons(radius):
    labels = np.repeat(np.repeat(blocks.assign_blocks(7, (3, 3)), 5, axis=0), 5, axis=1)
    np.testing.assert_array_equal(blocks.create_buffer_mask(labels, radius),
                                  _literal_reference_buffer(labels, radius))


def test_symmetric_buffer_removes_both_sides_and_includes_diagonals():
    labels = np.ones((8, 10), dtype=np.uint8)
    labels[:, 5:] = 2
    keep = blocks.create_buffer_mask(labels, (1, 2))
    assert np.all(keep[:, :3]) and np.all(keep[:, 7:])
    assert not np.any(keep[:, 3:7])  # two full patch columns on BOTH sides
    diagonal = np.ones((5, 5), dtype=np.uint8)
    diagonal[0, 0] = 2
    assert not blocks.create_buffer_mask(diagonal, 1)[1, 1]


def test_homogeneous_edges_holes_and_same_split_block_seams_create_no_buffer():
    labels = np.ones((12, 16), dtype=np.uint8)
    active = np.ones(labels.shape, dtype=bool)
    active[3:8, 4:9] = False
    # Missing patches carry another stored split code but create no interface.
    labels[~active] = 2
    keep = blocks.create_buffer_mask(labels, (2, 3), active_mask=active)
    assert np.all(keep)
    assert np.all(keep[0]) and np.all(keep[-1])
    assert np.all(keep[:, 0]) and np.all(keep[:, -1])
    # A seam between two adjacent blocks assigned train is just more train.
    same_split_blocks = np.repeat(np.repeat(np.ones((2, 3), np.uint8), 4, 0), 5, 1)
    assert np.all(blocks.create_buffer_mask(same_split_blocks, 2,
                                           active_mask=np.ones(same_split_blocks.shape, bool)))


def test_only_nearby_active_foreign_splits_trigger_buffer_and_inactive_returns_true():
    labels = np.ones((7, 12), dtype=np.uint8)
    labels[:, 6:] = 2
    active = np.ones(labels.shape, dtype=bool)
    active[:, 5:7] = False
    keep = blocks.create_buffer_mask(labels, 1, active_mask=active)
    assert np.all(keep)  # inactive corridor already separates active patches
    active[:, 5:7] = True
    touching = blocks.create_buffer_mask(labels, 1, active_mask=active)
    assert not np.any(touching[:, 5:7])
    assert np.all(touching[:, :5]) and np.all(touching[:, 7:])
    isolated = np.zeros(labels.shape, dtype=bool)
    isolated[0, 0] = True
    isolated[-1, -1] = True
    assert np.all(blocks.create_buffer_mask(labels, 2, active_mask=isolated))
    with pytest.raises(ValueError, match="active_mask"):
        blocks.create_buffer_mask(labels, 1, active_mask=np.ones((3, 3), bool))


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_active_full_grid_retains_exact_upstream_geometry(radius):
    labels = np.repeat(np.repeat(blocks.assign_blocks(7, (3, 3)), 5, axis=0), 5, axis=1)
    np.testing.assert_array_equal(blocks.create_buffer_mask(labels, radius,
                                                            active_mask=np.ones(labels.shape, bool)),
                                  _literal_reference_buffer(labels, radius))


def test_planner_ignores_quality_excluded_corridor_at_unlike_block_interface():
    valid = np.ones((98, 142), bool)
    # Seed zero assigns top-middle block train and top-right block validation.
    # Two absent patch columns between them already supply separation.
    valid[:32, 75:85] = False
    planned = _plan(valid, seed_candidates=1)
    meta = planned["metadata"]
    assert meta["block_labels"][0][1:3] == [1, 2]
    assert meta["buffer_keep_mask"][2][14] is True
    assert meta["buffer_keep_mask"][2][17] is True
    assert meta["buffer_keep_mask"][2][15:17] == [True, True]
    assert meta["quality_keep_mask"][2][15:17] == [False, False]
    assert meta["buffer_removed_mask"][2][15:17] == [False, False]
    assert meta["buffer_scope"] == "only nearby quality-eligible patches assigned to different splits"
    for patch in planned["patches"]:
        if not patch["quality_retained"]:
            assert patch["buffer_excluded"] is False
            assert patch["drop_reason"] == "quality_threshold"


def test_rectangular_radii_and_primary_demo_grid_are_feasible():
    planned = blocks.plan_reference_blocks(np.ones((850, 699), bool),
                                           pixel_size_m=20, buffer_m=1298,
                                           patch_size_pixels=(32, 40), block_size_pixels=(256, 320),
                                           seed_candidates=20, edge_policy="crop")
    meta = planned["metadata"]
    assert meta["buffer_radius_patches"] == [3, 2]
    assert meta["reference_buffer_radius_patches"] == [3, 2]
    assert meta["buffer_radius_quantized_m"] == [1920, 1600]
    assert meta["block_grid_shape"] == [3, 2]
    assert meta["patch_grid_shape"] == [24, 16]
    assert meta["supported_raster_shape"] == [768, 640]
    assert len(planned["patches"]) == 384


def test_seed_selection_maximizes_smallest_quality_filtered_split_with_stable_ties():
    planned = _plan()
    meta = planned["metadata"]
    candidates = meta["seed_search"]["candidates"]
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    best_score = max(candidate["minimum_retained_patch_count"] for candidate in eligible)
    first_best = next(candidate for candidate in eligible
                      if candidate["minimum_retained_patch_count"] == best_score)
    assert meta["selected_seed"] == first_best["seed"]
    assert meta["seed_search"]["selected_minimum_retained_patch_count"] == best_score
    assert meta["seed_search"]["start_inclusive"] == 0
    assert meta["seed_search"]["stop_exclusive"] == 20
    json.dumps(meta, allow_nan=False)


def test_custom_split_fractions_report_requested_and_rounded_block_fractions():
    planned = _plan(split_ratios=(0.5, 0.25, 0.25))
    meta = planned["metadata"]
    assert meta["requested_block_split_ratios"] == {"train": 0.5, "validation": 0.25, "test": 0.25}
    assert meta["realized_block_split_ratios"] == {"train": 1 / 3, "validation": 1 / 3, "test": 1 / 3}


def test_default_blocks_group_three_patches_in_each_axis():
    planned = blocks.plan_reference_blocks(np.ones((96, 120), bool),
                                           pixel_size_m=10, buffer_m=0,
                                           patch_size_pixels=(4, 5), seed_candidates=3)
    meta = planned["metadata"]
    assert meta["block_size_pixels"] == [12, 15]
    assert meta["patches_per_block"] == [3, 3]
    assert meta["block_grid_shape"] == [8, 8]
    assert meta["edge_policy"] == "extend"


def test_ignored_label_codes_are_excluded_and_coverage_counts_valid_classes():
    valid = np.ones((98, 142), bool)
    labels = (np.indices(valid.shape).sum(0) % 3).astype(np.int16)
    labels[:4, :5] = -100
    planned = _plan(valid, class_labels=labels, require_class_coverage=True)
    meta = planned["metadata"]
    assert meta["classes_present_in_supported_grid"] == [0, 1, 2]
    assert meta["class_coverage_complete"] is True
    assert meta["quality_keep_mask"][0][0] is False
    assert not np.any(planned["valid_mask"][:4, :5])
    assert all(all(count > 0 for count in counts) for counts in meta["class_pixel_counts"].values())


def test_quality_filtering_happens_before_seed_coverage_claims():
    valid = np.ones((98, 142), bool)
    labels = np.zeros(valid.shape, dtype=np.int16)
    # The only occurrence of class 1 is in a patch with 95% missing features.
    # A seed search that ignores quality could incorrectly claim coverage.
    valid[:4, :5] = False
    valid[0, 0] = True
    labels[0, 0] = 1
    with pytest.raises(ValueError, match="every present class"):
        _plan(valid, class_labels=labels, require_class_coverage=True)
    diagnostic = _plan(valid, class_labels=labels, require_class_coverage=False)
    assert diagnostic["metadata"]["class_coverage_complete"] is False
    assert diagnostic["metadata"]["quality_rejected_patch_count"] == 1
    assert diagnostic["metadata"]["quality_rejected_valid_pixel_count"] == 1


def test_core_grid_masks_preserve_whole_patches_and_identical_paired_holdouts():
    shape = (98, 142)
    field = np.random.default_rng(4).normal(size=shape)
    field[:2, :5] = np.nan  # first 4x5 patch is exactly half missing: retain it
    labels = (np.indices(shape).sum(0) % 3).astype(np.int16)
    plan = core.create_spatial_split(
        field, pixel_size_m=10, buffer_m=30, patch_layout="reference_blocks",
        patch_size_pixels=(4, 5), block_size_pixels=(32, 40), seed_candidates=20,
        class_labels=labels, require_class_coverage=True,
        max_pairs_per_lag=100, max_lag_m=200, edge_policy="crop")
    actual, paired, candidate = (plan[key] for key in
                                 ("split_mask", "unbuffered_split_mask", "candidate_split_mask"))
    report = plan["report"]
    assert report["patch_layout"] == "reference_blocks"
    assert report["quality_keep_mask"][0][0] is True
    assert report["edge_discarded_pixel_count"] == 98 * 142 - 96 * 120
    assert report["edge_discarded_valid_pixel_count"] == 98 * 142 - 96 * 120
    for mask in (actual, paired, candidate):
        assert not np.any(mask[96:]) and not np.any(mask[:, 120:])
        assert not np.any(mask[~np.isfinite(field)])
    for code in (2, 3):
        np.testing.assert_array_equal(actual == code, paired == code)
    assert np.count_nonzero(paired == 1) > np.count_nonzero(actual == 1)
    for patch in plan["patches"]:
        window = np.s_[patch["row_start"]:patch["row_stop"], patch["col_start"]:patch["col_stop"]]
        support = np.isfinite(field[window])
        assert np.all(actual[window][support] == (patch["split_code"] if patch["retained"] else 0))
        assert np.all(candidate[window][support] == patch["split_code"])
    assert report["spatial_diagnostics"]["requested_buffer_met"] is True
    assert report["splits"]["buffer_discarded_candidate_pixel_count"] == report["buffer_discarded_quality_candidate_pixel_count"]
    json.dumps(report, allow_nan=False)


def test_default_edge_extension_covers_full_aoi_and_retains_original_block_grid():
    shape = (98, 142)
    field = np.random.default_rng(18).normal(size=shape)
    plan = core.create_spatial_split(
        field, pixel_size_m=10, buffer_m=30, patch_layout="reference_blocks",
        patch_size_pixels=(4, 5), block_size_pixels=(32, 40), seed_candidates=20,
        max_pairs_per_lag=50, max_lag_m=200)
    report, patches = plan["report"], plan["patches"]
    assert report["edge_policy"] == "extend"
    assert report["supported_raster_shape"] == list(shape)
    assert report["complete_block_raster_shape"] == [96, 120]
    assert report["edge_discarded_pixel_count"] == 0
    assert report["edge_discarded_valid_pixel_count"] == 0
    assert report["patch_grid_shape"] == [25, 29]
    assert report["block_grid_shape"] == [3, 3]  # no extra random edge blocks
    assert len(report["block_windows"]) == 9
    assert report["block_windows"][-1]["height"] == 34
    assert report["block_windows"][-1]["width"] == 62
    assert np.all(plan["candidate_split_mask"] > 0)
    assert np.any(plan["split_mask"][-1]) and np.any(plan["split_mask"][:, -1])
    coverage = np.zeros(shape, np.uint8)
    block_labels = np.array(report["block_labels"])
    for patch in patches:
        window = np.s_[patch["row_start"]:patch["row_stop"], patch["col_start"]:patch["col_stop"]]
        coverage[window] += 1
        assert patch["block_row"] == min(patch["patch_row"] // 8, 2)
        assert patch["block_col"] == min(patch["patch_col"] // 8, 2)
        assert patch["split_code"] == block_labels[patch["block_row"], patch["block_col"]]
    assert np.all(coverage == 1)
    assert sum(patch["pixel_area"] for patch in patches) == field.size
    # The extended region inherits the same last block label across both axes.
    assert report["patch_label_grid"][-1][-1] == report["block_labels"][-1][-1]
    assert patches[-1]["row_stop"] == 98 and patches[-1]["col_stop"] == 142
    assert patches[-1]["pixel_area"] == 2 * 2
    assert patches[-1]["nan_fraction"] == 0


def test_clipped_terminal_patch_quality_uses_actual_area_and_never_external_padding():
    valid = np.ones((98, 142), bool)
    valid[-2:, -2:] = False
    valid[-1, -1] = valid[-1, -2] = True  # 2/4 actual pixels, not 2/20 padded pixels
    planned = blocks.plan_reference_blocks(
        valid, pixel_size_m=10, buffer_m=0, patch_size_pixels=(4, 5),
        block_size_pixels=(32, 40), seed_candidates=20, max_nan_fraction=0.5)
    corner = planned["patches"][-1]
    assert corner["edge_clipped"] is True
    assert corner["pixel_area"] == 4
    assert corner["valid_pixel_count"] == 2
    assert corner["nan_fraction"] == 0.5
    assert corner["quality_retained"] is True
    assert corner["retained"] is True
    assert planned["metadata"]["edge_discarded_pixel_count"] == 0


def test_divisible_fully_valid_extension_matches_crop_and_upstream_geometry_exactly():
    arguments = dict(pixel_size_m=10, buffer_m=30, patch_size_pixels=(4, 5),
                     block_size_pixels=(32, 40), seed_candidates=20)
    valid = np.ones((96, 120), bool)
    extended = blocks.plan_reference_blocks(valid, **arguments)
    cropped = blocks.plan_reference_blocks(valid, edge_policy="crop", **arguments)
    for key in ("selected_seed", "block_labels", "patch_label_grid", "keep_mask", "quality_keep_mask"):
        assert extended["metadata"][key] == cropped["metadata"][key]
    assert extended["patches"] == cropped["patches"]
    grid = np.array(extended["metadata"]["patch_label_grid"], dtype=np.uint8)
    np.testing.assert_array_equal(extended["metadata"]["keep_mask"], _literal_reference_buffer(grid, 1))


@pytest.mark.parametrize("overrides,match", [
    ({"patch_size_pixels": (4, 0)}, "patch_size_pixels"),
    ({"block_size_pixels": (31, 40)}, "integer multiple"),
    ({"block_size_pixels": 1000}, "block_size_pixels"),
    ({"seed_candidates": 0}, "seed_candidates"),
    ({"seed_candidates": True}, "seed_candidates"),
    ({"max_nan_fraction": -0.1}, "max_nan_fraction"),
    ({"max_nan_fraction": np.nan}, "max_nan_fraction"),
    ({"require_class_coverage": True}, "class_labels"),
    ({"require_class_coverage": "yes"}, "boolean"),
    ({"split_ratios": (0.5, 0.5, 0.5)}, "split_ratios"),
    ({"split_ratios": (0.8, 0.2, 0)}, "split_ratios"),
    ({"buffer_m": 1e9}, "too small"),
    ({"edge_policy": "pad"}, "edge_policy"),
])
def test_invalid_block_grid_configuration_has_actionable_errors(overrides, match):
    with pytest.raises(ValueError, match=match):
        _plan(**overrides)


def test_caps_and_label_alignment_prevent_unbounded_or_ambiguous_work():
    with pytest.raises(ValueError, match="4096"):
        _plan(np.ones((256, 256), bool), patch_size_pixels=1, block_size_pixels=32)
    with pytest.raises(ValueError, match="class_labels"):
        _plan(class_labels=np.ones((98, 141), dtype=int))
    with pytest.raises(ValueError, match="integer"):
        _plan(class_labels=np.ones((98, 142), dtype=float))
    with pytest.raises(ValueError, match="three nonempty"):
        blocks.assign_blocks(0, (1, 2))
    with pytest.raises(ValueError, match="n_patches must be None"):
        core.create_spatial_split(np.arange(1600).reshape(40, 40), pixel_size_m=10,
                                  buffer_m=0, patch_layout="reference_blocks", n_patches=10,
                                  max_pairs_per_lag=20, max_lag_m=100)


def test_patch_radius_extra_footprints_still_pass_exact_global_geometry_check():
    field = np.random.default_rng(1).normal(size=(98, 142))
    plan = core.create_spatial_split(field, pixel_size_m=10, buffer_m=1,
                                     patch_radius_m=45, patch_layout="reference_blocks",
                                     patch_size_pixels=(4, 5), block_size_pixels=(32, 40),
                                     seed_candidates=20, max_pairs_per_lag=50, max_lag_m=100)
    assert plan["report"]["spatial_diagnostics"]["requested_buffer_met"] is True
    assert plan["report"]["buffer_radius_patches"] == [2, 2]
