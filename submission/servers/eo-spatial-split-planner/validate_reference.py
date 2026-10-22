"""Compare numerical output with the pinned external checkpoint implementation.

An existing checkout is supplied explicitly; the test does not fetch code or
execute a repository entrypoint. Only its numerical autocorrelation module is
loaded. Results record the exact source commit and hashes, without local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from core import evaluate_autocorrelation
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
REFERENCE_COMMIT_URL = "https://github.com/stefanopitton/lulc-classification/commit/ac7dfc6578c2c408c87398e15073354752bbd32b"
COMMIT = REFERENCE_COMMIT_URL.rsplit("/", 1)[-1]


def compare(reference, field, resolution_m, lag_m, draws, label):
    local = evaluate_autocorrelation(field, resolution_m, lag_m, seed=3,
                                     n_pairs_per_lag=draws, max_lag_m=12800, n_bins=24)
    edges = np.asarray(local["lag_edges_m"])
    centers = np.asarray(local["lag_centers_m"])
    mean, variance = reference.field_moments(field)
    pairs = reference.pair_statistics(np.random.default_rng(3), field, edges, draws, resolution_m, mean)
    curves = reference.reduce_stats(pairs, variance, centers)
    for name, upstream in [("counts", curves.counts), ("sq_diff_sum", pairs.sq_diff_sum),
                           ("gamma", curves.gamma), ("rho", curves.rho_variogram)]:
        np.testing.assert_array_equal(np.asarray(local[name], dtype=float), upstream)
    assert local["variance"] == variance
    expected_rho = reference.value_at_lag(centers, curves.rho_variogram, lag_m)
    assert local["rho_at_lag"] == expected_rho
    fit = reference.fit_variogram(centers, curves.gamma, curves.counts, "spherical")
    np.testing.assert_allclose([local["nugget"], local["partial_sill"], local["practical_range_m"], local["fit_rss"]],
                               [fit.nugget, fit.sill, fit.practical_range, fit.rss], rtol=1e-12, atol=1e-14)
    assert local["sill"] == local["nugget"] + local["partial_sill"]
    return {"field": label, "shape": list(field.shape), "resolution_m": resolution_m,
            "seed": 3, "n_pairs_per_lag": draws, "n_bins": 24, "lag_m": lag_m,
            "exact_matches": ["counts", "sq_diff_sum", "gamma", "rho", "variance", "rho_at_lag"],
            "spherical_fit_matches": True, "sill_mapping": "reference sill parameter is partial_sill; output sill = nugget + partial_sill",
            "rho_at_lag": local["rho_at_lag"], "practical_range_m": local["practical_range_m"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.reference_dir), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise SystemExit(f"Reference checkout must be pinned at {COMMIT}")
    source = args.reference_dir / "src/lulc/validation/autocorrelation.py"
    spec = importlib.util.spec_from_file_location("checkpoint_autocorrelation", source)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    field = gaussian_filter(np.random.default_rng(71).normal(size=(160, 192)), 3).astype("float32")
    field[12:23, 40:55] = np.nan
    reports = [compare(reference, field, 10, 400, 4000, "synthetic float32 continuous field with NaNs")]
    raster = HERE / "artifacts/analysis_band.tif"
    with rasterio.open(raster) as src:
        field = src.read(1, masked=True).astype("float32").filled(np.nan)
        resolution = src.res[0]
    reports.append(compare(reference, field, resolution, 2560, 200000, "actual cached Sentinel-2 B05"))
    result = {"status": "passed", "repository": "https://github.com/stefanopitton/lulc-classification",
              "branch": "stepit/eve-hackathon-checkpoint", "commit": actual,
              "source_file": "src/lulc/validation/autocorrelation.py",
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "comparisons": reports}
    (HERE / "artifacts/reference_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
