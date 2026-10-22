"""Variogram-only lag evaluation and spatial partition planning for continuous EO bands."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from core import create_spatial_split, evaluate_autocorrelation
from imagery import read_local, read_sentinel2
from mcp.server.fastmcp import FastMCP

SCHEMA_VERSION = "0.4.0"
REFERENCE_COMMIT_URL = "https://github.com/stefanopitton/lulc-classification/commit/ac7dfc6578c2c408c87398e15073354752bbd32b"
REFERENCE_IMPLEMENTATION = {
    "repository": "https://github.com/stefanopitton/lulc-classification",
    "branch": "stepit/eve-hackathon-checkpoint",
    "commit": REFERENCE_COMMIT_URL.rsplit("/", 1)[-1],
    "source": "src/lulc/validation/autocorrelation.py",
    "procedure": "PROCEDURE.md (variogram and spherical-model POC subset)",
    "patch_source": "scripts/04_extract_patches.py and src/lulc/data/patches.py",
}
mcp = FastMCP("EO Spatial Split Planner", host="0.0.0.0", port=8000, stateless_http=True)


def _sampling_parameters(seed, n_bins, n_pairs_per_lag, min_lag_m, max_lag_m, correlation_threshold):
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(n_bins, int) or not 4 <= n_bins <= 64:
        raise ValueError("n_bins must be an integer between 4 and 64.")
    if not isinstance(n_pairs_per_lag, int) or n_pairs_per_lag < 1 or n_pairs_per_lag * n_bins > 8_000_000:
        raise ValueError("Use positive n_pairs_per_lag and at most 8 million total offset draws (n_bins * n_pairs_per_lag).")
    if not math.isfinite(max_lag_m) or max_lag_m <= 0:
        raise ValueError("max_lag_m must be finite and positive.")
    if min_lag_m is not None and (not math.isfinite(min_lag_m) or not 0 < min_lag_m < max_lag_m):
        raise ValueError("min_lag_m must be finite, positive and smaller than max_lag_m.")
    if not math.isfinite(correlation_threshold) or not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold must be finite and between zero and one.")
    return {"seed": seed, "n_bins": n_bins, "n_pairs_per_lag": n_pairs_per_lag,
            "min_lag_m": min_lag_m, "max_lag_m": max_lag_m,
            "correlation_threshold": correlation_threshold}


def _publish(values: np.ndarray, profile: dict, provenance: dict, report: dict,
             split_mask: np.ndarray | None = None,
             unbuffered_split_mask: np.ndarray | None = None,
             candidate_split_mask: np.ndarray | None = None) -> dict:
    """Persist the complete analysis; retain required per-bin counts in responses."""
    report.update(schema_version=SCHEMA_VERSION, provenance=provenance,
                  reference_implementation=REFERENCE_IMPLEMENTATION)
    fingerprint = hashlib.sha256(json.dumps(report, sort_keys=True, allow_nan=False).encode())
    fingerprint.update(str(profile["crs"]).encode())
    fingerprint.update(str(profile["transform"]).encode())
    for start in range(0, values.shape[0], 256):
        fingerprint.update(np.asarray(values[start:start + 256], dtype="<f4").tobytes())
    run_id = fingerprint.hexdigest()[:16]
    root = Path(os.getenv("EO_SPLIT_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "eo-spatial-split-planner")))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=root))
    output_profile = profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "photometric", "interleave"):
        output_profile.pop(key, None)
    output_profile.update(driver="GTiff", count=1, compress="deflate", dtype="float32", nodata=np.nan)
    source_path, report_path = directory / "analysis_band.tif", directory / "report.json"
    with rasterio.open(source_path, "w", **output_profile) as dst:
        dst.write(np.where(np.isfinite(values), values, np.nan).astype("float32"), 1)
    artifacts = {"analysis_band": str(source_path), "report_json": str(report_path),
                 "scope": "Paths on the MCP server filesystem"}
    if split_mask is not None:
        mask_path = directory / "split_mask.tif"
        output_profile.update(dtype="uint8", nodata=0)
        with rasterio.open(mask_path, "w", **output_profile) as dst:
            dst.write(split_mask, 1)
            dst.update_tags(labels="0=nodata_or_buffer,1=train,2=validation,3=test", seed=report["seed"])
        artifacts["split_mask"] = str(mask_path)
    if unbuffered_split_mask is not None:
        unbuffered_path = directory / "unbuffered_split_mask.tif"
        output_profile.update(dtype="uint8", nodata=0)
        with rasterio.open(unbuffered_path, "w", **output_profile) as dst:
            dst.write(unbuffered_split_mask, 1)
            dst.update_tags(labels="0=outside_or_nodata,1=train,2=validation,3=test",
                            pairing="Same validation/test rectangles as buffered mask")
        artifacts["unbuffered_split_mask"] = str(unbuffered_path)
    if candidate_split_mask is not None:
        candidate_path = directory / "candidate_split_mask.tif"
        output_profile.update(dtype="uint8", nodata=0)
        with rasterio.open(candidate_path, "w", **output_profile) as dst:
            dst.write(candidate_split_mask, 1)
            dst.update_tags(labels="0=outside_or_invalid,1=train,2=validation,3=test",
                            description="Every quality-eligible candidate patch before cross-split buffering")
        artifacts["candidate_split_mask"] = str(candidate_path)
    if "patches" in report:
        patch_path = directory / "patches.json"
        patch_path.write_text(json.dumps(report["patches"], indent=2))
        artifacts["patch_manifest"] = str(patch_path)
    report.update(artifacts=artifacts, run_fingerprint=run_id)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    compact = {k: v for k, v in report.items() if k not in {"sq_diff_sum", "patches", "patch_label_grid", "keep_mask", "seed_diagnostics", "buffer_keep_mask", "quality_keep_mask", "candidate_keep_mask", "buffer_removed_mask", "block_windows"}}
    if "seed_search" in compact:
        compact["seed_search"] = {key: value for key, value in compact["seed_search"].items() if key != "candidates"}
    if "autocorrelation" in compact:
        compact["autocorrelation"] = {k: v for k, v in compact["autocorrelation"].items() if k != "sq_diff_sum"}
    return {"status": "ok", **compact}


def _evaluate(values, profile, provenance, lag_m, parameters, band=None):
    report = evaluate_autocorrelation(values, resolution_m=abs(profile["transform"].a),
                                     lag_m=lag_m, band=band, **parameters)
    return _publish(values, profile, provenance, report)


def _plan(values, profile, provenance, buffer_m, split_ratios, patch_radius_m, lag_m, parameters,
          patch_size_pixels=None, n_patches=None, patch_layout="reference_blocks",
          block_size_pixels=None, seed_candidates=200, max_nan_fraction=0.5, edge_policy="extend",
          group_assignment="compact_regions", buffer_policy="one_sided"):
    params = parameters.copy()
    params["n_lags"] = params.pop("n_bins")
    params["max_pairs_per_lag"] = params.pop("n_pairs_per_lag")
    result = create_spatial_split(values, pixel_size_m=abs(profile["transform"].a),
                                  split_ratios=[0.7, 0.1, 0.2] if split_ratios is None else split_ratios,
                                  buffer_m=buffer_m, patch_radius_m=patch_radius_m,
                                  lag_m=lag_m, patch_size_pixels=patch_size_pixels,
                                  n_patches=n_patches, patch_layout=patch_layout,
                                  block_size_pixels=block_size_pixels, seed_candidates=seed_candidates,
                                  max_nan_fraction=max_nan_fraction, edge_policy=edge_policy,
                                  group_assignment=group_assignment, buffer_policy=buffer_policy, **params)
    return _publish(values, profile, provenance, result["report"], result["split_mask"],
                    result["unbuffered_split_mask"], result.get("candidate_split_mask"))


def _error(exc: Exception) -> dict:
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return {"status": "error", "error": {"code": "invalid_input_or_unresolved_analysis", "message": str(exc)}}
    # Remote-reader exceptions can contain signed URLs; omit their text.
    return {"status": "error", "error": {"code": "imagery_or_processing_failed",
            "message": f"{type(exc).__name__}: retrieval or processing failed. Check connectivity, native pixel limits and lag support; retry or use a local raster."}}


@mcp.tool()
def evaluate_spatial_autocorrelation(
    raster_path: str, lag_m: float = 2560, band_index: int = 1, band: str | None = None,
    seed: int = 3, n_bins: int = 24, n_pairs_per_lag: int = 200_000,
    min_lag_m: float | None = None, max_lag_m: float = 12_800,
    correlation_threshold: float = 0.1,
) -> dict:
    """Evaluate a continuous raster band's dependence at a specified lag in metres.

    raster_path is a local cloud-masked GeoTIFF on this server (up to 32 million
    square, north-up pixels in a projected metre CRS). band_index selects a
    1-based band; band optionally labels it. Input values are not normalized.
    Monte Carlo sampling uses isotropic offsets, actual rounded-distance bins,
    finite pairs, 24 logarithmic bins and seed 3. min_lag_m defaults to one pixel.
    Returns rho_at_lag = 1 - gamma/variance, practical_range_m from a weighted
    spherical fit, nugget, total sill, fit_rss, per-bin counts, variance and missing
    fraction. The lag passes only if rho_at_lag < correlation_threshold AND
    lag_m >= practical_range_m, with a resolved fit. Increase the lag and repeat
    if it fails. Lags outside populated bin-center support return an error.
    Stationarity/isotropy are assumed; no label or model independence is proved.
    """
    try:
        parameters = _sampling_parameters(seed, n_bins, n_pairs_per_lag, min_lag_m, max_lag_m, correlation_threshold)
        values, profile, provenance = read_local(raster_path, band_index)
        return _evaluate(values, profile, provenance, lag_m, parameters, band or f"band_{band_index}")
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def evaluate_sentinel2_autocorrelation(
    bbox: list[float], lag_m: float = 2560, band: str = "B05",
    start_date: str = "2024-06-01", end_date: str = "2024-08-31",
    item_id: str | None = None, max_cloud_cover: float = 10, seed: int = 3,
    n_bins: int = 24, n_pairs_per_lag: int = 200_000,
    min_lag_m: float | None = None, max_lag_m: float = 12_800,
    correlation_threshold: float = 0.1,
) -> dict:
    """Measure a Sentinel-2 band's variogram-derived rho at lag_m and fit its range.

    bbox is WGS84 [west,south,east,north], at most one degree and 1 million native
    pixels. Dates are ISO dates up to 366 days apart. B05 is native 20 m red edge;
    B08 is native 10 m NIR. Public Planetary Computer STAC provides numeric L2A
    pixels with SCL masking; item_id pins a scene. Uses the same variogram-only
    procedure and acceptance rule as evaluate_spatial_autocorrelation. Returns
    rho_at_lag, practical_range_m, nugget, total sill, fit_rss, counts, variance,
    missing fraction, decision, provenance, and full-report/numeric-band paths.
    """
    try:
        parameters = _sampling_parameters(seed, n_bins, n_pairs_per_lag, min_lag_m, max_lag_m, correlation_threshold)
        values, profile, provenance = read_sentinel2(bbox, start_date, end_date, band, item_id, max_cloud_cover)
        return _evaluate(values, profile, provenance, lag_m, parameters, provenance["band"])
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def plan_sentinel2_split(
    bbox: list[float], start_date: str = "2024-06-01", end_date: str = "2024-08-31",
    band: str = "B05", item_id: str | None = None, max_cloud_cover: float = 10,
    buffer_m: float | None = None, seed: int = 3, split_ratios: list[float] | None = None,
    patch_radius_m: float = 0, lag_m: float | None = None,
    n_bins: int = 24, n_pairs_per_lag: int = 200_000,
    min_lag_m: float | None = None, max_lag_m: float = 12_800,
    correlation_threshold: float = 0.1,
    patch_size_pixels: int | list[int] | None = None, n_patches: int | None = None,
    patch_layout: str = "reference_blocks", block_size_pixels: int | list[int] | None = None,
    seed_candidates: int = 200, max_nan_fraction: float = 0.5, edge_policy: str = "extend",
    group_assignment: str = "compact_regions", buffer_policy: str = "one_sided",
) -> dict:
    """Plan two-dimensional train/validation/test patches using the checkpoint workflow.

    Imagery inputs match evaluate_sentinel2_autocorrelation. split_ratios is
    [train,validation,test], defaults to [0.7,0.1,0.2], must be positive and sum
    to one. Requested fractions allocate whole blocks; report realized counts
    after rounding, quality filtering and buffers. patch_layout='reference_blocks'
    adapts scripts/04_extract_patches.py: tile all blocks into small rectangular
    patches. group_assignment='compact_regions' clusters the randomized groups
    into contiguous split regions to reduce wasted guard interfaces.
    buffer_policy='one_sided' retains test, then validation, then training;
    lower-priority whole patches are excluded only if too close to another split.
    Candidate seeds maximize retained valid pixels subject to nonempty splits.
    Explicit random_groups plus symmetric policy reproduces the older reference
    assignment/buffering; it can discard much more data.
    Only a nearby quality-eligible patch from a DIFFERENT split triggers removal;
    same-split seams, AOI edges, nodata and quality-rejected regions do not.
    block_size_pixels must be a multiple of patch size. edge_policy='extend'
    keeps the full AOI by extending the last blocks and clipping terminal patches.
    There is no exterior buffer or cropped bottom/right strip. 'crop' is an explicit
    reference-compatibility option only. seed_candidates ranks seeds 0..N-1 by
    total retained valid pixels with all splits nonempty, without accuracy-based selection. Class
    coverage is unverified because this tool receives a continuous raster.
    max_nan_fraction filters patch validity. n_patches must be omitted for this
    dense mode. patch_layout='random_rectangles' retains the sparse jittered
    alternative with n_patches and priority-based whole-patch removal.
    buffer_m=None measures a guard using the variogram; an explicit guard has
    its own dependence decision. Whole-patch exclusion can exceed the requested
    gap. Paired masks restore training patches only and fix buffered heldouts.
    candidate_split_mask
    shows all quality-filtered patches without any buffer, with other heldouts.
    Returns requested/realized fractions, masks, records and measured distances.
    """
    try:
        parameters = _sampling_parameters(seed, n_bins, n_pairs_per_lag, min_lag_m, max_lag_m, correlation_threshold)
        values, profile, provenance = read_sentinel2(bbox, start_date, end_date, band, item_id, max_cloud_cover)
        return _plan(values, profile, provenance, buffer_m, split_ratios, patch_radius_m, lag_m, parameters,
                     patch_size_pixels, n_patches, patch_layout, block_size_pixels,
                     seed_candidates, max_nan_fraction, edge_policy, group_assignment, buffer_policy)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def plan_spatial_split(
    raster_path: str, band_index: int = 1, buffer_m: float | None = None,
    seed: int = 3, split_ratios: list[float] | None = None, patch_radius_m: float = 0,
    lag_m: float | None = None, n_bins: int = 24, n_pairs_per_lag: int = 200_000,
    min_lag_m: float | None = None, max_lag_m: float = 12_800,
    correlation_threshold: float = 0.1,
    patch_size_pixels: int | list[int] | None = None, n_patches: int | None = None,
    patch_layout: str = "reference_blocks", block_size_pixels: int | list[int] | None = None,
    seed_candidates: int = 200, max_nan_fraction: float = 0.5, edge_policy: str = "extend",
    group_assignment: str = "compact_regions", buffer_policy: str = "one_sided",
) -> dict:
    """Plan local EO patches with configurable fractions, default [0.7,0.1,0.2].

    Raster requirements match evaluate_spatial_autocorrelation. patch_layout
    defaults to 'reference_blocks': fixed rectangular patch tiles in randomized
    compact split regions in both axes. One-sided exact-footprint guards retain
    test, then validation, then train; only OTHER active splits trigger exclusion.
    Explicit random_groups/symmetric reproduces checkpoint script04 buffering.
    Same-split seams, AOI edges,
    nodata and quality-rejected regions do not trigger buffering. block_size_pixels must be divisible by patch_size_pixels.
    edge_policy='extend' keeps full AOI coverage, extends terminal blocks and
    clips terminal patches; it creates no outside buffer. 'crop' is opt-in. seed_candidates ranks
    seeds0..N-1 by total retained valid pixels with nonempty splits, without class labels; report
    class coverage remains unverified. max_nan_fraction excludes poor patches.
    See plan_sentinel2_split for full semantics, including rounded block counts,
    requested versus realized fractions, paired fixed heldouts and the raw
    candidate mask. 'random_rectangles' enables sparse jittered rectangles and
    n_patches. Dataset nodata is honored; callers provide cloud-masked values.
    """
    try:
        parameters = _sampling_parameters(seed, n_bins, n_pairs_per_lag, min_lag_m, max_lag_m, correlation_threshold)
        values, profile, provenance = read_local(raster_path, band_index)
        return _plan(values, profile, provenance, buffer_m, split_ratios, patch_radius_m, lag_m, parameters,
                     patch_size_pixels, n_patches, patch_layout, block_size_pixels,
                     seed_candidates, max_nan_fraction, edge_policy, group_assignment, buffer_policy)
    except Exception as exc:
        return _error(exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    args = parser.parse_args()
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")
