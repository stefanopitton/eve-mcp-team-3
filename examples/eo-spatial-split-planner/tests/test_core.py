"""Procedure-level scientific regression checks and exact split geometry."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

CORE_PATH = Path(__file__).resolve().parents[1] / "core.py"
SPEC = importlib.util.spec_from_file_location("eo_split_core", CORE_PATH)
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


def test_seeded_statistics_match_pinned_reference_implementation():
    """Frozen outputs from lulc-classification ac7dfc6578c2c408c87398e15073354752bbd32b.

    This fixture also detects accidental capping after rounded offsets migrate
    between bins: the first bin legitimately has >2048 accepted pairs.
    """
    field = np.arange(24 * 32, dtype=np.float32).reshape(24, 32) % 97
    field[:3, :5] = np.nan
    edges = np.array([10, 15, 22, 33, 50, 75, 112, 170, 250], dtype=float)
    result = core.estimate_variogram(field, 10, edges, n_pairs_per_lag=2048, seed=3)
    np.testing.assert_array_equal(result["counts"], [2399, 813, 2368, 1248, 1720, 1254, 831, 385])
    np.testing.assert_array_equal(result["sq_diff_sum"],
                                  [3697226, 942521, 3776618, 1837921, 2467232, 1876904, 1210705, 563616])
    np.testing.assert_allclose(result["gamma"],
                               [770.5764902042517, 579.6562115621156, 797.4277871621622,
                                736.3465544871794, 717.2186046511628, 748.3668261562998,
                                728.4626955475331, 731.9688311688311], rtol=0, atol=0)
    np.testing.assert_allclose(result["rho"],
                               [-0.0016959642088567595, 0.24648714938544247, -0.036600786946678676,
                                0.04280057688576122, 0.06766558431578107, 0.027175057845515882,
                                0.05304904641815289, 0.04849131341374713], rtol=0, atol=0)
    assert result["variance"] == 769.2718327090955
    assert result["seed"] == 3
    json.dumps(result, allow_nan=False)


def test_default_log_bins_seed_and_raw_float64_variance():
    field = np.random.default_rng(7).normal(size=(64, 64))
    result = core.estimate_variogram(field, 10, n_pairs_per_lag=2000)
    assert len(result["counts"]) == 24
    assert result["lag_edges_m"][0] == 10
    assert result["lag_edges_m"][-1] == 12_800
    np.testing.assert_allclose(np.diff(np.log(result["lag_edges_m"])), np.log(1280) / 24)
    edges = np.array(result["lag_edges_m"])
    np.testing.assert_allclose(result["lag_centers_m"], np.sqrt(edges[:-1] * edges[1:]))
    assert result["variance"] == np.nanvar(field, dtype=np.float64)
    assert result["seed"] == 3
    assert core.estimate_variogram.__defaults__[1] == 200_000


def test_variogram_uses_raw_values_and_rho_is_scale_and_shift_free():
    field = np.random.default_rng(91).normal(size=(70, 90))
    options = dict(resolution_m=10, max_lag_m=300, n_pairs_per_lag=2000)
    raw = core.estimate_variogram(field, **options)
    scaled = core.estimate_variogram(37 * field + 120, **options)
    np.testing.assert_array_equal(raw["counts"], scaled["counts"])
    np.testing.assert_allclose(np.asarray(scaled["sq_diff_sum"]), np.asarray(raw["sq_diff_sum"]) * 37**2)
    assert scaled["variance"] == pytest.approx(raw["variance"] * 37**2)
    np.testing.assert_allclose(np.asarray(scaled["gamma"], dtype=float),
                               np.asarray(raw["gamma"], dtype=float) * 37**2, equal_nan=True)
    np.testing.assert_allclose(np.asarray(scaled["rho"], dtype=float), np.asarray(raw["rho"], dtype=float), equal_nan=True)


def test_rho_is_not_clipped_even_below_minus_one():
    checkerboard = np.indices((3, 3)).sum(axis=0) % 2 * 2.0 - 1
    result = core.estimate_variogram(checkerboard, 1, [0.99, 1.01], n_pairs_per_lag=2000)
    assert result["gamma"] == [2.0]
    assert result["rho"][0] == pytest.approx(-1.025)
    assert result["rho"][0] < -1


def test_offsets_are_rebinned_by_actual_distance_with_half_open_edges(monkeypatch):
    class ControlledGenerator:
        def uniform(self, low, high, size):
            if low == 0 and high == 2 * np.pi:
                return np.array([0.0, np.pi / 4])
            return np.full(size, (low + high) / 2)

        def integers(self, low, high, size):
            return np.full(size, 2, dtype=int)

    monkeypatch.setattr(core.np.random, "default_rng", lambda seed: ControlledGenerator())
    # Both bins' cardinal offsets round to distance 1, exactly the second
    # bin's lower edge. Diagonal offsets reach sqrt(2), its excluded upper edge.
    result = core.estimate_variogram(np.tile(np.arange(5.0), (5, 1)), 1,
                                      [0.9, 1.0, np.sqrt(2)], n_pairs_per_lag=2)
    assert result["counts"] == [0, 2]
    assert result["sq_diff_sum"] == [0, 2]
    assert result["gamma"] == [None, 0.5]
    assert result["rho"][0] is None
    json.dumps(result, allow_nan=False)


def test_zero_rounded_offsets_and_empty_bins_are_excluded():
    result = core.estimate_variogram(np.arange(25.0).reshape(5, 5), 1,
                                      [0.01, 0.1, 0.2], n_pairs_per_lag=100)
    assert result["counts"] == [0, 0]
    assert result["gamma"] == result["rho"] == [None, None]
    assert result["sampled_lag_min_m"] is None
    assert result["sampled_lag_max_m"] is None
    json.dumps(result, allow_nan=False)


def test_nonfinite_values_and_mask_are_excluded_without_imputation():
    field = np.random.default_rng(33).normal(size=(48, 60)).astype(np.float32)
    field[:5] = np.nan
    field[7, 8] = np.inf
    mask = np.ones_like(field, dtype=bool)
    mask[:, :7] = False
    effective_field = field.copy()
    effective_field[~np.isfinite(effective_field) | ~mask] = np.nan
    args = dict(resolution_m=10, max_lag_m=300, n_pairs_per_lag=1000)
    masked = core.estimate_variogram(field, valid_mask=mask, **args)
    explicit_nan = core.estimate_variogram(effective_field, **args)
    assert masked == explicit_nan
    assert masked["valid_pixel_count"] == np.isfinite(effective_field).sum()
    assert masked["nan_fraction"] == pytest.approx(np.isnan(effective_field).mean())


def test_external_generators_are_reproducible_and_sums_remain_mergeable():
    field = np.random.default_rng(8).normal(size=(32, 40))
    args = dict(resolution_m=10, max_lag_m=200, n_pairs_per_lag=1000)
    first = core.estimate_variogram(field, rng=np.random.default_rng(3), **args)
    second = core.estimate_variogram(field, rng=np.random.default_rng(3), seed=999, **args)
    assert first == second
    third = core.estimate_variogram(field, seed=71, **args)
    merged_counts = np.array(first["counts"]) + third["counts"]
    merged_sums = np.array(first["sq_diff_sum"]) + third["sq_diff_sum"]
    populated = merged_counts > 0
    combined_gamma = merged_sums[populated] / (2 * merged_counts[populated])
    counts_first = np.array(first["counts"])[populated]
    counts_third = np.array(third["counts"])[populated]
    gamma_first = np.nan_to_num(np.array(first["gamma"], dtype=float)[populated])
    gamma_third = np.nan_to_num(np.array(third["gamma"], dtype=float)[populated])
    np.testing.assert_allclose(combined_gamma,
                               (gamma_first * counts_first + gamma_third * counts_third) / merged_counts[populated])
    assert first["seed"] is None


def test_white_noise_empirical_semivariance_is_near_variance():
    field = np.random.default_rng(71).normal(size=(160, 160))
    result = core.estimate_variogram(field, 20, max_lag_m=600, n_pairs_per_lag=10_000)
    counts = np.asarray(result["counts"])
    rho = np.asarray(result["rho"], dtype=float)
    assert np.max(np.abs(rho[counts > 500])) < 0.08
    gamma = np.asarray(result["gamma"], dtype=float)
    np.testing.assert_allclose(gamma[counts > 0],
                               np.asarray(result["sq_diff_sum"])[counts > 0] / (2 * counts[counts > 0]))
    assert "directions" not in result
    assert "global_mean" not in result


def test_spherical_fit_recovers_known_nugget_total_sill_and_range():
    lags = np.geomspace(10, 1000, 24)
    gamma = core.spherical_variogram(lags, 0.15, 1.8, 250)
    counts = np.linspace(500, 30_000, lags.size)
    fit = core.fit_spherical_variogram(lags, gamma, counts)
    assert fit["nugget"] == pytest.approx(0.15, rel=1e-5)
    assert fit["partial_sill"] == pytest.approx(1.8, rel=1e-5)
    assert fit["sill"] == pytest.approx(1.95, rel=1e-5)
    assert fit["practical_range_m"] == pytest.approx(250, rel=1e-5)
    assert fit["fit_rss"] < 1e-15
    assert core.spherical_variogram(np.array([250, 500]), 0.15, 1.8, 250).tolist() == [1.95, 1.95]


def test_fit_uses_pair_counts_for_weighting_and_reports_unweighted_rss():
    lags = np.geomspace(10, 1000, 24)
    observed = core.spherical_variogram(lags, 0.15, 1.8, 250)
    observed[-1] += 4
    counts = np.full(lags.size, 10_000)
    counts[-1] = 1
    fit = core.fit_spherical_variogram(lags, observed, counts)
    assert fit["practical_range_m"] == pytest.approx(250, rel=0.01)
    assert fit["sill"] == pytest.approx(1.95, rel=0.001)
    expected_rss = np.sum((observed - core.spherical_variogram(lags, fit["nugget"], fit["partial_sill"], fit["practical_range_m"]))**2)
    assert fit["fit_rss"] == pytest.approx(expected_rss)
    assert fit["fit_rss"] > 15


def test_fit_requires_four_populated_bins():
    with pytest.raises(ValueError, match="at least 4 populated"):
        core.fit_spherical_variogram([10, 20, 30, 40], [1, 2, 3, None], [50, 50, 50, 0])


def test_log_lag_interpolation_drops_missing_bins_without_extrapolation():
    centers, rho = [10, 20, 40, 160], [1.0, None, 0.5, -0.5]
    assert core.interpolate_rho(centers, rho, 20) == pytest.approx(0.75)
    assert core.interpolate_rho(centers, rho, 80) == pytest.approx(0.0)
    assert core.interpolate_rho(centers, rho, 10) == 1
    assert core.interpolate_rho(centers, rho, 160) == -0.5
    for lag in (9.999, 160.001, 0, np.nan, np.inf):
        with pytest.raises(ValueError, match="sampled lag-center range"):
            core.interpolate_rho(centers, rho, lag)
    with pytest.raises(ValueError, match="at least 2 populated"):
        core.interpolate_rho([10, 20], [None, 0.5], 20)


@pytest.mark.parametrize("rho,lag,fit_resolved,accepted", [
    (0.1, 50, True, False), (0.099, 50, True, True),
    (-0.3, 49, True, False), (-0.3, 50, False, False),
])
def test_agent_decision_requires_strict_empirical_threshold_and_fitted_range(rho, lag, fit_resolved, accepted):
    fit = {"fit_resolved": fit_resolved, "practical_range_m": 50 if fit_resolved else None}
    decision = core._decision(rho, lag, fit, 0.1)
    assert decision["accepted"] is accepted


def test_tool_contract_reports_rho_at_requested_lag_and_fit_context():
    field = gaussian_filter(np.random.default_rng(8).normal(size=(192, 192)), sigma=5)
    result = core.evaluate_autocorrelation(field, 20, 600, band="B05", max_lag_m=1000, n_pairs_per_lag=3000)
    assert result["band"] == "B05"
    assert result["lag_m"] == 600
    assert result["rho_at_lag"] == core.interpolate_rho(result["lag_centers_m"], result["rho"], 600)
    assert result["fit_status"] == "resolved"
    assert result["fit_resolved"] is True
    assert result["decision"]["accepted"] is (result["rho_at_lag"] < 0.1 and result["practical_range_m"] <= 600)
    for key in ("practical_range_m", "nugget", "sill", "fit_rss", "counts", "variance", "nan_fraction"):
        assert key in result
    json.dumps(result, allow_nan=False)


def test_auto_buffer_passes_both_empirical_and_fitted_criteria():
    field = gaussian_filter(np.random.default_rng(8).normal(size=(192, 192)), sigma=5)
    result = core.create_spatial_split(field, pixel_size_m=20, max_lag_m=1000,
                                       max_pairs_per_lag=5000, n_patches=12, split_ratios=(0.6, 0.2, 0.2))
    report = result["report"]
    diagnostic = report["autocorrelation"]
    assert report["buffer_source"] == "variogram_range_and_empirical_rho"
    assert report["requested_buffer_m"] >= diagnostic["practical_range_m"]
    assert core.interpolate_rho(diagnostic["lag_centers_m"], diagnostic["rho"], report["requested_buffer_m"]) < 0.1
    assert report["spatial_diagnostics"]["requested_buffer_met"]
    actual = report["buffer_autocorrelation"]
    assert actual["lag_m"] == report["requested_buffer_m"]
    assert actual["within_sampled_support"] is True
    assert actual["decision"]["accepted"] is True


@pytest.mark.parametrize("buffer_m,within_support", [(20, False), (40, True)])
@pytest.mark.parametrize("lag_of_interest", [None, 2560])
def test_actual_buffer_assessment_is_separate_from_a_lag_of_interest(buffer_m, within_support, lag_of_interest):
    field = gaussian_filter(np.random.default_rng(8).normal(size=(192, 192)), sigma=5)
    report = core.create_spatial_split(
        field, pixel_size_m=20, buffer_m=buffer_m, lag_m=lag_of_interest,
        max_lag_m=4000, max_pairs_per_lag=3000,
    )["report"]
    actual = report["buffer_autocorrelation"]
    assert actual["lag_m"] == buffer_m
    assert actual["within_sampled_support"] is within_support
    assert actual["decision"]["accepted"] is False
    assert actual["decision"]["lag_at_least_practical_range"] is False
    if within_support:
        assert actual["rho_at_lag"] > 0.1
    else:
        assert actual["rho_at_lag"] is None
        assert actual["decision"]["rho_below_threshold"] is False
    if lag_of_interest is not None:
        assert report["autocorrelation"]["lag_m"] == lag_of_interest
        assert report["autocorrelation"]["decision"]["accepted"] is True
    else:
        assert "decision" not in report["autocorrelation"]
    assert report["spatial_diagnostics"]["requested_buffer_met"] is True
    json.dumps(report, allow_nan=False)


def test_unobserved_range_never_falls_back_to_largest_tested_lag():
    gradient = np.tile(np.arange(80, dtype=float), (80, 1))
    args = dict(pixel_size_m=10, max_lag_m=200, max_pairs_per_lag=3000)
    analysis = core.analyze_autocorrelation(gradient, **args)
    assert analysis["recommended_buffer_m"] is None
    assert any("sampled" in warning for warning in analysis["warnings"])
    with pytest.raises(ValueError, match="No automatic buffer"):
        core.create_spatial_split(gradient, **args)
    explicit = core.create_spatial_split(gradient, buffer_m=30, split_ratios=(0.6, 0.2, 0.2), **args)
    assert explicit["report"]["buffer_source"] == "user_supplied"
    assert any("not certified" in warning for warning in explicit["report"]["warnings"])


def test_unresolved_fit_reports_no_range_and_cannot_accept_a_lag():
    field = np.random.default_rng(3).normal(size=(48, 48))
    result = core.evaluate_autocorrelation(field, 10, 12, max_lag_m=15, n_pairs_per_lag=1000)
    assert result["fit_status"] == "unresolved"
    assert result["practical_range_m"] is None
    assert result["recommended_buffer_m"] is None
    assert result["decision"]["accepted"] is False
    json.dumps(result, allow_nan=False)


def test_evaluating_lag_outside_populated_centers_raises():
    field = np.random.default_rng(8).normal(size=(48, 48))
    with pytest.raises(ValueError, match="sampled lag-center range"):
        core.evaluate_autocorrelation(field, 10, 500, max_lag_m=200, n_pairs_per_lag=1000)


def test_buffer_has_exact_verified_separation_and_matching_random_baseline():
    values = np.random.default_rng(1).normal(size=(96, 128))
    output = core.create_spatial_split(values, pixel_size_m=10, buffer_m=80,
                                       patch_radius_m=15, seed=7, max_pairs_per_lag=1000, max_lag_m=400)
    labels, report = output["split_mask"], output["report"]
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert report["spatial_diagnostics"]["requested_buffer_met"]
    assert report["spatial_diagnostics"]["minimum_footprint_gap_lower_bound_m"] >= 80
    stats = report["splits"]
    assert stats["discarded_valid_pixel_count"] > 0
    assert stats["counts"] == report["random_baseline"]["counts"]
    np.testing.assert_array_equal(labels > 0, output["random_split_mask"] > 0)
    for name, requested in report["requested_split_ratios"].items():
        # Whole rectangles are dropped instead of silently reallocating patch
        # assignments to recover the original ratio after buffer exclusions.
        initial_fraction = report["patch_counts"][name]["candidate"] / report["n_candidate_patches"]
        assert abs(initial_fraction - requested) <= 1 / report["n_candidate_patches"]
    assert stats["candidate_pixel_count"] + stats["unselected_valid_pixel_count"] == stats["valid_pixel_count"]
    assert stats["buffer_discarded_candidate_pixel_count"] == stats["candidate_pixel_count"] - stats["retained_pixel_count"]
    assert stats["buffer_discarded_training_pixel_count"] == stats["unbuffered_retained_pixel_count"] - stats["retained_pixel_count"]
    assert report["random_baseline"]["separation"]["minimum_center_distance_m"] == 10
    json.dumps(report, allow_nan=False)


def test_rectangle_patches_are_random_in_both_axes_and_never_overlap():
    field = np.random.default_rng(18).normal(size=(192, 256))
    plan = core.create_spatial_split(field, pixel_size_m=10, buffer_m=0,
                                     patch_size_pixels=(12, 20), n_patches=30,
                                     max_pairs_per_lag=100, max_lag_m=300)
    patches, labels, report = plan["patches"], plan["split_mask"], plan["report"]
    coverage = np.zeros(field.shape, dtype=int)
    for patch in patches:
        window = np.s_[patch["row_start"]:patch["row_stop"], patch["col_start"]:patch["col_stop"]]
        assert patch["row_stop"] - patch["row_start"] == 12
        assert patch["col_stop"] - patch["col_start"] == 20
        assert np.all(labels[window] == patch["split_code"])
        coverage[window] += 1
    assert coverage.max() == 1
    assert np.count_nonzero(labels) == 30 * 12 * 20
    for split in ("train", "validation", "test"):
        selected = [patch for patch in patches if patch["split"] == split]
        assert len({patch["row_start"] for patch in selected}) >= 3
        assert len({patch["col_start"] for patch in selected}) >= 3
    assert len({p["row_start"] % 15 for p in patches}) > 1
    assert len({p["col_start"] % 25 for p in patches}) > 1
    assert "stripe_axis" not in report and "stripe_order" not in report
    assert report["splits"]["unselected_valid_pixel_count"] > 0
    np.testing.assert_array_equal(plan["split_mask"], plan["unbuffered_split_mask"])


def test_buffer_drops_whole_rectangles_and_paired_comparison_has_fixed_holdouts():
    field = np.random.default_rng(18).normal(size=(192, 256))
    args = dict(pixel_size_m=10, patch_size_pixels=(12, 20), n_patches=30,
                max_pairs_per_lag=100, max_lag_m=300, seed=3)
    plan = core.create_spatial_split(field, buffer_m=120, patch_radius_m=5, **args)
    zero = core.create_spatial_split(field, buffer_m=0, patch_radius_m=0, **args)
    buffered, unbuffered = plan["split_mask"], plan["unbuffered_split_mask"]
    original_keys = ("patch_id", "row_start", "row_stop", "col_start", "col_stop", "split", "split_code")
    assert [{key: p[key] for key in original_keys} for p in plan["patches"]] == [
        {key: p[key] for key in original_keys} for p in zero["patches"]]
    assert np.count_nonzero(buffered == 1) < np.count_nonzero(unbuffered == 1)
    for code in (2, 3):
        np.testing.assert_array_equal(buffered == code, unbuffered == code)
    np.testing.assert_array_equal(buffered == 3, zero["split_mask"] == 3)
    for patch in plan["patches"]:
        window = np.s_[patch["row_start"]:patch["row_stop"], patch["col_start"]:patch["col_stop"]]
        assert np.all(buffered[window] == (patch["split_code"] if patch["retained"] else 0))
        assert patch["buffered_pixel_count"] == (12 * 20 if patch["retained"] else 0)
        assert patch["keep"] is patch["retained"]
        assert patch["unbuffered_retained"] is bool(patch["retained"] or patch["split_code"] == 1)
    assert plan["report"]["spatial_diagnostics"]["minimum_footprint_gap_lower_bound_m"] >= 120
    assert plan["report"]["unbuffered_baseline"]["separation"]["minimum_footprint_gap_lower_bound_m"] < 120


@pytest.mark.parametrize("patch_size", [0, -1, True, 4.5, (0, 10), (10,), (3, 4, 5), (50, 100), (np.nan, 4)])
def test_invalid_rectangle_dimensions_are_rejected(patch_size):
    with pytest.raises(ValueError, match="patch_size_pixels"):
        core.create_spatial_split(np.arange(1600).reshape(40, 40), pixel_size_m=10,
                                  buffer_m=0, patch_size_pixels=patch_size,
                                  max_pairs_per_lag=20, max_lag_m=100)


@pytest.mark.parametrize("n_patches", [0, 1, 2, True, 3.5, 513])
def test_invalid_rectangle_counts_are_rejected(n_patches):
    with pytest.raises(ValueError, match="n_patches"):
        core.create_spatial_split(np.arange(1600).reshape(40, 40), pixel_size_m=10,
                                  buffer_m=0, n_patches=n_patches,
                                  max_pairs_per_lag=20, max_lag_m=100)


def test_too_many_or_too_large_rectangles_give_actionable_errors():
    field = np.arange(1600).reshape(40, 40)
    args = dict(pixel_size_m=10, buffer_m=0, max_pairs_per_lag=20, max_lag_m=100)
    with pytest.raises(ValueError, match="decrease n_patches or patch_size_pixels"):
        core.create_spatial_split(field, patch_size_pixels=16, n_patches=6, **args)
    with pytest.raises(ValueError, match="choose smaller patch_size_pixels"):
        core.create_spatial_split(field, patch_size_pixels=32, **args)


def test_exact_distance_transform_matches_pairwise_bruteforce():
    labels = np.zeros((11, 13), dtype=np.uint8)
    labels[1:3, 1:4] = 1
    labels[5:7, 4:6] = 2
    labels[8:, 10:] = 3
    report = core.separation_diagnostics(labels, pixel_size_m=10)
    for pair, (first, second) in zip(report["pairwise"], ((1, 2), (1, 3), (2, 3)), strict=True):
        expected = cdist(np.argwhere(labels == first) * 10,
                         np.argwhere(labels == second) * 10).min()
        assert pair["minimum_center_distance_m"] == pytest.approx(expected)


def test_nodata_and_explicit_mask_are_never_used_and_seed_reproduces():
    values = np.random.default_rng(3).normal(size=(64, 80))
    values[:5] = np.nan
    values[8:10] = np.inf
    mask = np.ones_like(values, dtype=bool)
    mask[:, :6] = False
    args = dict(pixel_size_m=20, buffer_m=40, valid_mask=mask, seed=123,
                max_pairs_per_lag=500, max_lag_m=500)
    first = core.create_spatial_split(values, **args)
    second = core.create_spatial_split(values, **args)
    np.testing.assert_array_equal(first["split_mask"], second["split_mask"])
    np.testing.assert_array_equal(first["random_split_mask"], second["random_split_mask"])
    assert first["report"] == second["report"]
    assert np.all(first["split_mask"][~np.isfinite(values) | ~mask] == 0)
    expected_valid = np.count_nonzero(np.isfinite(values) & mask)
    assert first["report"]["autocorrelation"]["valid_pixel_count"] == expected_valid


@pytest.mark.parametrize("buffer_m", [None, 0, 10])
def test_constant_signal_must_raise_even_with_explicit_buffer(buffer_m):
    with pytest.raises(ValueError, match="variance must be finite and > 0"):
        core.create_spatial_split(np.ones((40, 40)), pixel_size_m=10, buffer_m=buffer_m,
                                  max_pairs_per_lag=1000)


def test_infeasible_buffer_gives_actionable_error():
    with pytest.raises(ValueError, match="AOI is too small"):
        core.create_spatial_split(np.arange(400).reshape(20, 20), pixel_size_m=10, buffer_m=500,
                                  max_pairs_per_lag=500, max_lag_m=100)


@pytest.mark.parametrize("ratios", [(0.5, 0.2, 0.2), (0, 0.5, 0.5), (0.5, 0.5), (np.nan, 0.3, 0.7)])
def test_invalid_ratios_rejected(ratios):
    with pytest.raises(ValueError, match="split_ratios"):
        core.create_spatial_split(np.ones((20, 20)), pixel_size_m=10,
                                  buffer_m=0, split_ratios=ratios)


@pytest.mark.parametrize("pixel_size", [0, -1, np.nan, np.inf])
def test_invalid_metric_scale_rejected(pixel_size):
    with pytest.raises(ValueError, match="pixel_size_m"):
        core.analyze_autocorrelation(np.ones((20, 20)), pixel_size_m=pixel_size)


def test_all_nodata_and_wrong_mask_rejected():
    with pytest.raises(ValueError, match="three finite"):
        core.analyze_autocorrelation(np.full((10, 10), np.nan), pixel_size_m=10)
    with pytest.raises(ValueError, match="same shape"):
        core.analyze_autocorrelation(np.ones((10, 10)), np.ones((9, 10)), pixel_size_m=10)


@pytest.mark.parametrize("patch_radius", [-1, np.nan, np.inf])
def test_invalid_patch_radius_rejected(patch_radius):
    with pytest.raises(ValueError, match="patch_radius_m"):
        core.create_spatial_split(np.ones((20, 20)), pixel_size_m=10,
                                  buffer_m=0, patch_radius_m=patch_radius)


@pytest.mark.parametrize("edges", [[0, 10], [10, 10], [20, 10], [10, np.inf], [10, np.nan], [10]])
def test_invalid_lag_edges_rejected(edges):
    with pytest.raises(ValueError, match="lag_edges"):
        core.estimate_variogram(np.arange(100).reshape(10, 10), 10, edges, n_pairs_per_lag=10)


def test_total_attempt_limit_rejects_work_before_pair_allocation():
    with pytest.raises(ValueError, match="attempted pairs"):
        core.estimate_variogram(np.arange(100).reshape(10, 10), 10,
                                  n_bins=128, n_pairs_per_lag=200_000)
