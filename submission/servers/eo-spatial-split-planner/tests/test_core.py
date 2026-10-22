"""Scientific and geometric checks independent of external services."""

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


def test_white_noise_has_observed_short_dependence_and_sample_counts():
    values = np.random.default_rng(71).normal(size=(160, 160))
    result = core.analyze_autocorrelation(values, pixel_size_m=20, max_pairs_per_lag=1000)
    assert result["range_status"] == "observed"
    assert result["recommended_buffer_m"] <= 200
    for lag in result["lags"]:
        for direction in lag["directions"].values():
            assert 0 < direction["pair_count"] <= 1000
            assert direction["semivariance"] >= 0
    json.dumps(result, allow_nan=False)


def test_empirical_variogram_matches_known_gradient_and_range_is_censored():
    values = np.tile(np.arange(64, dtype=float), (64, 1))
    result = core.analyze_autocorrelation(values, pixel_size_m=10, max_lag_m=100,
                                          max_pairs_per_lag=10_000)
    first = result["lags"][0]["directions"]
    assert first["east_west"]["semivariance"] == pytest.approx(0.5)
    assert first["north_south"]["semivariance"] == 0
    assert first["east_west"]["correlation"] == pytest.approx(1)
    assert result["range_status"] == "not_observed"
    assert result["observed_range_m"] is None
    assert result["recommended_buffer_m"] == result["max_tested_lag_m"]
    assert any("not observed" in warning for warning in result["warnings"])


def test_smooth_field_has_longer_range_than_white_noise():
    noise = np.random.default_rng(8).normal(size=(192, 192))
    smooth = gaussian_filter(noise, sigma=5)
    white_result = core.analyze_autocorrelation(noise, pixel_size_m=20, max_pairs_per_lag=2000)
    smooth_result = core.analyze_autocorrelation(smooth, pixel_size_m=20, max_pairs_per_lag=2000)
    assert smooth_result["recommended_buffer_m"] > white_result["recommended_buffer_m"]


def test_buffer_has_exact_verified_separation_and_matching_random_baseline():
    values = np.random.default_rng(1).normal(size=(96, 128))
    output = core.create_spatial_split(values, pixel_size_m=10, buffer_m=80,
                                       patch_radius_m=15, seed=7, max_pairs_per_lag=1000)
    labels, report = output["split_mask"], output["report"]
    assert set(np.unique(labels)) == {0, 1, 2, 3}
    assert report["spatial_diagnostics"]["requested_buffer_met"]
    assert report["spatial_diagnostics"]["minimum_footprint_gap_lower_bound_m"] >= 80
    stats = report["splits"]
    assert stats["discarded_valid_pixel_count"] > 0
    assert stats["counts"] == report["random_baseline"]["counts"]
    np.testing.assert_array_equal(labels > 0, output["random_split_mask"] > 0)
    for name, requested in report["requested_split_ratios"].items():
        assert abs(stats["fractions_of_retained"][name] - requested) < 0.025
    assert report["random_baseline"]["separation"]["minimum_center_distance_m"] == 10
    json.dumps(report, allow_nan=False)


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
                max_pairs_per_lag=500)
    first = core.create_spatial_split(values, **args)
    second = core.create_spatial_split(values, **args)
    np.testing.assert_array_equal(first["split_mask"], second["split_mask"])
    np.testing.assert_array_equal(first["random_split_mask"], second["random_split_mask"])
    assert first["report"] == second["report"]
    assert np.all(first["split_mask"][~np.isfinite(values) | ~mask] == 0)
    expected_valid = np.count_nonzero(np.isfinite(values) & mask)
    assert first["report"]["autocorrelation"]["valid_pixel_count"] == expected_valid


def test_constant_signal_cannot_invent_a_data_derived_buffer():
    values = np.ones((40, 40))
    analysis = core.analyze_autocorrelation(values, pixel_size_m=10)
    assert analysis["range_status"] == "constant"
    assert analysis["recommended_buffer_m"] is None
    with pytest.raises(ValueError, match="explicit buffer"):
        core.create_spatial_split(values, pixel_size_m=10)
    explicit = core.create_spatial_split(values, pixel_size_m=10, buffer_m=0)
    assert explicit["report"]["splits"]["retention_fraction"] == 1


def test_unobserved_range_is_explicitly_provisional():
    gradient = np.tile(np.arange(80, dtype=float), (80, 1))
    output = core.create_spatial_split(gradient, pixel_size_m=10, max_pairs_per_lag=1000)
    assert output["report"]["autocorrelation"]["range_status"] == "not_observed"
    assert output["report"]["buffer_source"] == "largest_tested_lag_provisional"
    assert any("not observed" in item for item in output["report"]["warnings"])


def test_infeasible_buffer_gives_actionable_error():
    with pytest.raises(ValueError, match="AOI is too small"):
        core.create_spatial_split(np.arange(400).reshape(20, 20), pixel_size_m=10, buffer_m=500)


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
