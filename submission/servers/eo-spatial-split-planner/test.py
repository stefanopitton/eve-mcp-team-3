"""MCP protocol checks and a real Sentinel-2 variogram demonstration.

python test.py                              # synthetic raster, real MCP stdio
python test.py --real                       # acquire native Sentinel-2 B05
python test.py --real --raster-cache artifacts/analysis_band.tif

The cache option reuses the numeric satellite pixels already acquired; it never
substitutes synthetic pixels. All scientific outputs use PROCEDURE.md defaults.
The 2024 scene is a secondary variogram/MCP demonstration; a separate labeled
segmentation benchmark compares models with and without training buffers.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
SEED = 3
LAG_M = 2560.0
PATCH_SIZE_PIXELS = [48, 64]
N_PATCHES = 12
EXPECTED_TOOLS = {
    "evaluate_spatial_autocorrelation", "evaluate_sentinel2_autocorrelation",
    "plan_spatial_split", "plan_sentinel2_split",
}


async def call_mcp(name: str, arguments: dict) -> dict:
    params = StdioServerParameters(command=sys.executable,
                                  args=[str(HERE / "server.py"), "--transport", "stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert {t.name for t in listing.tools} == EXPECTED_TOOLS
            result = await session.call_tool(name, arguments)
            assert not result.isError, result
            return json.loads(next(c.text for c in result.content if c.type == "text"))


def full_report(result: dict) -> dict:
    assert result["status"] == "ok", result
    return json.loads(Path(result["artifacts"]["report_json"]).read_text())


def verify_variogram(analysis: dict) -> dict:
    """Independently verify the reported sums, variance conversion and decision."""
    edges = np.asarray(analysis["lag_edges_m"], dtype=float)
    centers = np.asarray(analysis["lag_centers_m"], dtype=float)
    counts = np.asarray(analysis["counts"], dtype=np.int64)
    sums = np.asarray(analysis["sq_diff_sum"], dtype=float)
    gamma = np.asarray(analysis["gamma"], dtype=float)
    rho = np.asarray(analysis["rho"], dtype=float)
    populated = counts > 0
    assert analysis["estimator"] == "isotropic Monte Carlo empirical variogram"
    assert analysis["variance"] > 0 and 0 <= analysis["nan_fraction"] <= 1
    assert centers.size == 24 and edges.size == 25
    assert analysis["seed"] == SEED
    assert analysis["n_pairs_per_lag"] == 200_000
    assert np.all(np.diff(edges) > 0) and np.all(counts >= 0)
    np.testing.assert_allclose(centers, np.sqrt(edges[:-1] * edges[1:]), rtol=1e-14)
    np.testing.assert_allclose(np.diff(np.log(edges)), np.diff(np.log(edges))[0], rtol=1e-13)
    np.testing.assert_allclose(gamma[populated], sums[populated] / (2 * counts[populated]), rtol=1e-13)
    np.testing.assert_allclose(rho[populated], 1 - gamma[populated] / analysis["variance"], rtol=1e-13)
    assert np.all(np.isnan(gamma[~populated])) and np.all(np.isnan(rho[~populated]))
    expected_rho = np.interp(np.log(analysis["lag_m"]), np.log(centers[populated]), rho[populated])
    np.testing.assert_allclose(analysis["rho_at_lag"], expected_rho, rtol=1e-13)
    decision = analysis["decision"]
    below = analysis["rho_at_lag"] < decision["correlation_threshold"]
    assert decision["rho_below_threshold"] == below
    if analysis["fit_status"] == "resolved":
        beyond = analysis["lag_m"] >= analysis["practical_range_m"]
        assert decision["lag_at_least_practical_range"] == beyond
        assert decision["accepted"] == (below and beyond)
        np.testing.assert_allclose(analysis["sill"], analysis["nugget"] + analysis["partial_sill"])
        ratio = np.clip(centers[populated] / analysis["practical_range_m"], 0, 1)
        fitted = analysis["nugget"] + analysis["partial_sill"] * (1.5 * ratio - 0.5 * ratio**3)
        np.testing.assert_allclose(analysis["fit_rss"], np.sum((gamma[populated] - fitted)**2), rtol=1e-9, atol=1e-20)
    else:
        assert decision["accepted"] is False
    return {"logarithmic_bins": "passed", "semivariance_from_pair_sums": "passed",
            "rho_from_variance": "passed", "log_lag_interpolation": "passed",
            "two_condition_decision": "passed", "spherical_fit_rss": analysis["fit_status"]}


def verify_rectangles(report: dict, artifacts: dict | None = None) -> dict:
    """Reconstruct complete patches and verify paired support and pixel budgets."""
    paths = report["artifacts"] if artifacts is None else artifacts
    with rasterio.open(paths["analysis_band"]) as src:
        valid = np.isfinite(src.read(1, masked=True).filled(np.nan))
        source_grid = (src.crs, src.transform, src.shape)
    with rasterio.open(paths["split_mask"]) as src:
        buffered = src.read(1)
        assert (src.crs, src.transform, src.shape) == source_grid
    with rasterio.open(paths["unbuffered_split_mask"]) as src:
        unbuffered = src.read(1)
        assert (src.crs, src.transform, src.shape) == source_grid
    patches = json.loads(Path(paths["patch_manifest"]).read_text())
    assert patches == report["patches"]
    assert report["schema_version"] == "0.3.0"
    assert report["patch_size_pixels"] == PATCH_SIZE_PIXELS
    assert len(patches) == report["n_candidate_patches"] == N_PATCHES
    assert len({p["row_start"] for p in patches}) > 1
    assert len({p["col_start"] for p in patches}) > 1
    reconstructed = {"buffered_retained": np.zeros(valid.shape, dtype="uint8"),
                     "unbuffered_retained": np.zeros(valid.shape, dtype="uint8")}
    occupied = np.zeros(valid.shape, dtype=bool)
    for patch in patches:
        r0, r1 = patch["row_start"], patch["row_stop"]
        c0, c1 = patch["col_start"], patch["col_stop"]
        assert [r1 - r0, c1 - c0] == PATCH_SIZE_PIXELS
        assert 0 <= r0 < r1 <= valid.shape[0] and 0 <= c0 < c1 <= valid.shape[1]
        assert not occupied[r0:r1, c0:c1].any(), "Candidate rectangles overlap"
        occupied[r0:r1, c0:c1] = True
        assert patch["valid_pixel_count"] == int(valid[r0:r1, c0:c1].sum())
        for key, expected in reconstructed.items():
            if patch[key]:
                expected[r0:r1, c0:c1] = np.where(valid[r0:r1, c0:c1], patch["split_code"], 0)
    np.testing.assert_array_equal(buffered, reconstructed["buffered_retained"])
    np.testing.assert_array_equal(unbuffered, reconstructed["unbuffered_retained"])
    for code in (2, 3):
        np.testing.assert_array_equal(buffered == code, unbuffered == code)
    assert np.all(unbuffered[buffered == 1] == 1)
    stats = report["splits"]
    assert stats["candidate_pixel_count"] == int((occupied & valid).sum())
    assert stats["unselected_valid_pixel_count"] == int((~occupied & valid).sum())
    assert stats["buffer_discarded_training_pixel_count"] == int((unbuffered == 1).sum() - (buffered == 1).sum())
    assert stats["retained_pixel_count"] + stats["buffer_discarded_candidate_pixel_count"] + stats["unselected_valid_pixel_count"] == stats["valid_pixel_count"]
    assert report["spatial_diagnostics"]["requested_buffer_met"]
    return {"two_axis_rectangle_placements": "passed", "whole_patch_exclusions": "passed",
            "identical_paired_validation_test_pixels": "passed",
            "candidate_coverage_and_buffer_loss_accounting": "passed", "patch_manifest_reconstructs_masks": "passed"}


def smoke() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.tif"
        rng = np.random.default_rng(SEED)
        values = gaussian_filter(rng.normal(size=(240, 320)), sigma=3).astype("float32")
        values[:12, :20] = np.nan
        with rasterio.open(path, "w", driver="GTiff", count=1, height=240, width=320,
                           crs="EPSG:32633", transform=from_origin(400000, 5800000, 20, 20),
                           nodata=np.nan, dtype="float32") as dst:
            dst.write(values, 1)
        evaluated = asyncio.run(call_mcp("evaluate_spatial_autocorrelation", {
            "raster_path": str(path), "lag_m": LAG_M,
        }))
        analysis = full_report(evaluated)
        checks = verify_variogram(analysis)
        np.testing.assert_allclose(analysis["variance"], np.nanvar(values, dtype=np.float64), rtol=1e-13)
        np.testing.assert_allclose(analysis["nan_fraction"], np.isnan(values).mean())
        result = asyncio.run(call_mcp("plan_spatial_split", {
            "raster_path": str(path), "buffer_m": 100, "seed": SEED, "lag_m": LAG_M,
            "patch_size_pixels": PATCH_SIZE_PIXELS, "n_patches": N_PATCHES,
        }))
        assert result["status"] == "ok", result
        with rasterio.open(result["artifacts"]["split_mask"]) as src:
            assert set(np.unique(src.read(1))) == {0, 1, 2, 3}
            assert src.crs.to_epsg() == 32633
            assert src.transform == from_origin(400000, 5800000, 20, 20)
        split = full_report(result)
        checks.update(verify_rectangles(split))
        assert split["spatial_diagnostics"]["requested_buffer_met"]
        assert split["random_baseline"]["counts"] == split["splits"]["counts"]
        assert split["random_baseline"]["retained_pixel_count"] == split["splits"]["retained_pixel_count"]
        np.testing.assert_array_equal(analysis["counts"], split["autocorrelation"]["counts"])
        np.testing.assert_array_equal(np.asarray(analysis["gamma"], dtype=float),
                                      np.asarray(split["autocorrelation"]["gamma"], dtype=float))
        failure = asyncio.run(call_mcp("evaluate_spatial_autocorrelation", {"raster_path": "/missing/science.tif"}))
        assert failure["status"] == "error"
        outside = asyncio.run(call_mcp("evaluate_spatial_autocorrelation", {"raster_path": str(path), "lag_m": 1}))
        assert outside["status"] == "error", outside
        return {"stdio_tools_list": "passed", "variogram_tool_call": "passed",
                "split_tool_call": "passed", "georeferenced_split_artifact": "passed",
                "exact_split_and_random_counts": "passed", "seeded_estimator_replay": "passed",
                "nan_exclusion_and_float64_variance": "passed", "actionable_invalid_path_error": "passed",
                "out_of_sampled_range_lag_error": "passed", **checks}


def _copy(source: str | Path, destination: Path) -> None:
    if Path(source).resolve() != destination.resolve():
        shutil.copy2(source, destination)


def real(raster_cache: Path | None = None) -> dict:
    ARTIFACTS.mkdir(exist_ok=True)
    source_provenance = None
    if raster_cache is not None:
        raster_cache = raster_cache.resolve(strict=True)
        prior_path = raster_cache.parent / "real_demo_report.json"
        if prior_path.is_file():
            prior = json.loads(prior_path.read_text())
            source_provenance = prior.get("source_provenance", prior.get("provenance"))
        if not source_provenance or source_provenance.get("collection") != "sentinel-2-l2a":
            raise ValueError("The Sentinel-2 cache requires adjacent real_demo_report.json scene provenance; use --real to acquire imagery")
        with rasterio.open(raster_cache) as src:
            assert list(src.shape) == source_provenance["shape"], "Cached raster shape differs from its scene provenance"
            assert str(src.crs) == source_provenance["crs"], "Cached raster CRS differs from its scene provenance"
            assert int(src.read_masks(1).astype(bool).sum()) == source_provenance["valid_pixels"]
        evaluated = asyncio.run(call_mcp("evaluate_spatial_autocorrelation", {
            "raster_path": str(raster_cache), "lag_m": LAG_M,
        }))
        acquisition = "Cached numeric Sentinel-2 raster; no new remote acquisition in this replay"
    else:
        evaluated = asyncio.run(call_mcp("evaluate_sentinel2_autocorrelation", {
            "bbox": [13.30, 52.35, 13.50, 52.50], "start_date": "2024-06-01",
            "end_date": "2024-08-31", "band": "B05", "lag_m": LAG_M,
        }))
        acquisition = "Live Sentinel-2 acquisition followed by local MCP raster handoff"
    variogram = full_report(evaluated)
    source_provenance = source_provenance or variogram["provenance"]
    verification = verify_variogram(variogram)
    band_path = ARTIFACTS / "analysis_band.tif"
    _copy(evaluated["artifacts"]["analysis_band"], band_path)
    input_sha256 = hashlib.sha256(band_path.read_bytes()).hexdigest()
    variogram["source_provenance"] = source_provenance
    variogram["integration_test"] = {
        "transport": "MCP stdio initialize + tools/list + tools/call",
        "real_numeric_pixels": True, "synthetic_fallback": False,
        "acquisition": acquisition, "input_geotiff_sha256": input_sha256,
        "verified_identities": verification,
    }
    variogram["artifacts"] = {"analysis_band": "analysis_band.tif", "report_json": "variogram_report.json",
                               "scope": "Files beside this report in artifacts/"}
    (ARTIFACTS / "variogram_report.json").write_text(json.dumps(variogram, indent=2, allow_nan=False) + "\n")
    # A rejected lag stays rejected: creating a mask cannot override that result.
    result = asyncio.run(call_mcp("plan_spatial_split", {
        "raster_path": str(band_path.resolve()), "buffer_m": LAG_M, "seed": SEED, "lag_m": LAG_M,
        "patch_size_pixels": PATCH_SIZE_PIXELS, "n_patches": N_PATCHES,
    }))
    report = full_report(result)
    rectangle_checks = verify_rectangles(report)
    for key, filename in (("split_mask", "split_mask.tif"),
                          ("unbuffered_split_mask", "unbuffered_split_mask.tif"),
                          ("patch_manifest", "patches.json")):
        _copy(result["artifacts"][key], ARTIFACTS / filename)
    report["source_provenance"] = source_provenance
    report["artifacts"] = {"split_mask": "split_mask.tif", "analysis_band": "analysis_band.tif",
                           "unbuffered_split_mask": "unbuffered_split_mask.tif", "patch_manifest": "patches.json",
                           "report_json": "real_demo_report.json", "variogram_report": "variogram_report.json",
                           "scope": "Files beside this report in artifacts/"}
    report["integration_test"] = {**variogram["integration_test"], "rectangle_checks": rectangle_checks,
                                  "role": "Secondary 2024 variogram and MCP demonstration, separate from the labeled segmentation benchmark"}
    np.testing.assert_array_equal(variogram["counts"], report["autocorrelation"]["counts"])
    np.testing.assert_array_equal(np.asarray(variogram["gamma"], dtype=float),
                                  np.asarray(report["autocorrelation"]["gamma"], dtype=float))
    with rasterio.open(band_path) as src:
        values = src.read(1, masked=True).filled(np.nan).astype("float64")
        pixel_size = src.res[0]
    from core import create_spatial_split

    sensitivity = []
    for buffer in (0.0, LAG_M / 2, LAG_M, LAG_M * 2):
        try:
            check = create_spatial_split(values, pixel_size_m=pixel_size, seed=SEED,
                                         buffer_m=buffer, lag_m=LAG_M,
                                         patch_size_pixels=PATCH_SIZE_PIXELS, n_patches=N_PATCHES)
            check_report = check["report"]
            assert check_report["spatial_diagnostics"]["requested_buffer_met"]
            assert check_report["splits"]["counts"] == check_report["random_baseline"]["counts"]
            sensitivity.append({"buffer_m": buffer, "status": "ok", "splits": check_report["splits"],
                                "patch_counts": check_report["patch_counts"],
                                "spatial_diagnostics": check_report["spatial_diagnostics"],
                                "candidate_coverage_fraction": check_report["splits"]["candidate_pixel_count"] / check_report["splits"]["valid_pixel_count"],
                                "retention_fraction_of_candidates": check_report["splits"]["retention_fraction_of_candidates"],
                                "buffer_discarded_training_pixel_count": check_report["splits"]["buffer_discarded_training_pixel_count"]})
        except ValueError as exc:
            sensitivity.append({"buffer_m": buffer, "status": "infeasible", "message": str(exc)})
    (ARTIFACTS / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2, allow_nan=False) + "\n")
    report["sensitivity"] = sensitivity
    (ARTIFACTS / "real_demo_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def plot_report(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    with rasterio.open(ARTIFACTS / "analysis_band.tif") as src:
        values = src.read(1, masked=True)
        pixel_size = src.res[0]
    with rasterio.open(ARTIFACTS / "split_mask.tif") as src:
        mask = src.read(1)
    with rasterio.open(ARTIFACTS / "unbuffered_split_mask.tif") as src:
        baseline = src.read(1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    lo, hi = np.nanpercentile(values.filled(np.nan), [2, 98])
    axes[0, 0].imshow(values, cmap="gray", vmin=lo, vmax=hi)
    source = report["source_provenance"]
    axes[0, 0].set_title(f"2024 Sentinel-2 {source.get('band', 'band')}; native {pixel_size:g} m")
    colors = ListedColormap(["#111827", "#ef6c45", "#4fbb9b", "#5b83da"])
    axes[0, 1].imshow(mask, cmap=colors, vmin=0, vmax=3)
    axes[0, 1].set_title(f"Random rectangles with {report['requested_buffer_m']:.0f} m guard")
    axes[0, 1].legend(handles=[Patch(color=color, label=label) for color, label in
                              (("#ef6c45", "Train"), ("#4fbb9b", "Validation"), ("#5b83da", "Test"))],
                       loc="upper right", fontsize=8)
    axes[0, 2].imshow(baseline, cmap=colors, vmin=0, vmax=3)
    axes[0, 2].set_title("Paired without guard: identical validation / test")
    for ax in axes[0]:
        ax.set_xlabel("Column (native pixel)")
        ax.set_ylabel("Row (native pixel)")
    ac = report["autocorrelation"]
    lag = np.asarray(ac["lag_centers_m"])
    gamma = np.asarray(ac["gamma"], dtype=float)
    rho = np.asarray(ac["rho"], dtype=float)
    counts = np.asarray(ac["counts"])
    valid = counts > 0
    axes[1, 0].semilogx(lag[valid], gamma[valid], "o", markersize=4, label="Isotropic empirical gamma")
    if ac["fit_status"] == "resolved":
        curve_x = np.geomspace(lag[valid].min(), lag[valid].max(), 500)
        ratio = np.minimum(curve_x / ac["practical_range_m"], 1)
        fitted = ac["nugget"] + ac["partial_sill"] * (1.5 * ratio - 0.5 * ratio**3)
        axes[1, 0].plot(curve_x, fitted, color="#24866d", label="Weighted spherical fit")
        for ax in axes[1, :2]:
            ax.axvline(ac["practical_range_m"], color="#24866d", linestyle=":", label="Fitted practical range")
    axes[1, 0].axhline(ac["variance"], color="#888888", linestyle="--", label="Field variance")
    axes[1, 0].axvline(LAG_M, color="#bd4c31", linestyle="--", label="Requested lag 2560 m")
    axes[1, 0].set(xlabel="Lag (m; log scale)", ylabel="Semivariance (reflectance squared)",
                   title="Variogram only: isotropic Monte Carlo sampling")
    axes[1, 0].legend(fontsize=7)
    axes[1, 1].semilogx(lag[valid], rho[valid], "o-", markersize=4, label="rho = 1 - gamma / variance")
    axes[1, 1].axhline(0.1, color="#888888", linestyle="--", label="Decision threshold 0.1")
    axes[1, 1].axvline(LAG_M, color="#bd4c31", linestyle="--", label="Requested lag 2560 m")
    axes[1, 1].plot([LAG_M], [ac["rho_at_lag"]], "*", color="#bd4c31", markersize=12)
    axes[1, 1].set(xlabel="Lag (m; log scale)", ylabel="Variogram-derived rho",
                   title=f"rho(2560 m)={ac['rho_at_lag']:.3f}; accepted={ac['decision']['accepted']}")
    axes[1, 1].legend(fontsize=7, loc="lower left")
    count_axis = axes[1, 1].twinx()
    count_axis.semilogx(lag, counts / 1000, color="#bbbbbb", alpha=0.6, linewidth=1)
    count_axis.set_ylabel("Valid pair count (thousands)", color="#888888")
    sensitivity = [x for x in report["sensitivity"] if x["status"] == "ok"]
    axes[1, 2].plot([x["buffer_m"] for x in sensitivity],
                    [100 * x["retention_fraction_of_candidates"] for x in sensitivity], "o-", color="#24866d",
                    label="Candidate pixels retained after guard")
    axes[1, 2].axhline(100 * sensitivity[0]["candidate_coverage_fraction"], color="#888888", linestyle="--",
                       label="Candidate coverage of valid AOI (%)")
    axes[1, 2].set(xlabel="Requested guard distance (m)", ylabel="Percent (denominator in legend)",
                   title="Sampled coverage and buffer exclusion are distinct", ylim=(0, 105))
    axes[1, 2].legend(fontsize=7)
    fig.suptitle("Secondary 2024 variogram/MCP demo — 12 random 48 × 64 pixel rectangles; seed 3")
    fig.savefig(ARTIFACTS / "diagnostic.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--raster-cache", type=Path,
                        help="Replay an already acquired numeric Sentinel-2 GeoTIFF through MCP")
    args = parser.parse_args()
    if args.raster_cache and not args.real:
        parser.error("--raster-cache requires --real")
    checks = smoke()
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "mcp_smoke_test.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(checks))
    if args.real:
        report = real(args.raster_cache)
        plot_report(report)
        ac = report["autocorrelation"]
        print(json.dumps({"item_id": report["source_provenance"].get("item_id"),
                          "lag_m": ac["lag_m"], "rho_at_lag": ac["rho_at_lag"],
                          "practical_range_m": ac["practical_range_m"], "decision": ac["decision"],
                          "buffer_m": report["requested_buffer_m"], "splits": report["splits"]}, indent=2))
