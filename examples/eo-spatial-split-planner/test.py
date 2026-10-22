"""MCP stdio smoke test and opt-in real Sentinel-2 experiment.

python test.py         # synthetic raster + real MCP protocol, offline
python test.py --real  # native Sentinel-2 B05 pixels + saved evidence
"""

from __future__ import annotations

import argparse
import asyncio
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


async def call_mcp(name: str, arguments: dict) -> dict:
    params = StdioServerParameters(command=sys.executable,
                                  args=[str(HERE / "server.py"), "--transport", "stdio"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listing = await session.list_tools()
            assert {t.name for t in listing.tools} == {"plan_spatial_split", "plan_sentinel2_split"}
            result = await session.call_tool(name, arguments)
            assert not result.isError, result
            return json.loads(next(c.text for c in result.content if c.type == "text"))


def smoke() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.tif"
        rng = np.random.default_rng(42)
        values = gaussian_filter(rng.normal(size=(120, 180)), sigma=2).astype("float32")
        with rasterio.open(path, "w", driver="GTiff", count=1, height=120, width=180,
                           crs="EPSG:32633", transform=from_origin(400000, 5800000, 20, 20),
                           dtype="float32") as dst:
            dst.write(values, 1)
        result = asyncio.run(call_mcp("plan_spatial_split", {"raster_path": str(path), "buffer_m": 100}))
        assert result["status"] == "ok", result
        with rasterio.open(result["artifacts"]["split_mask"]) as src:
            assert set(np.unique(src.read(1))) == {0, 1, 2, 3}
            assert src.crs.to_epsg() == 32633
        failure = asyncio.run(call_mcp("plan_spatial_split", {"raster_path": "/missing/science.tif"}))
        assert failure["status"] == "error"
        return {"stdio_tools_list": "passed", "raster_tool_call": "passed",
                "georeferenced_split_artifact": "passed", "actionable_invalid_path_error": "passed"}


def real() -> dict:
    result = asyncio.run(call_mcp("plan_sentinel2_split", {
        "bbox": [13.30, 52.35, 13.50, 52.50], "start_date": "2024-06-01",
        "end_date": "2024-08-31", "band": "B05", "seed": 42,
    }))
    assert result["status"] == "ok", result
    ARTIFACTS.mkdir(exist_ok=True)
    report = json.loads(Path(result["artifacts"]["report_json"]).read_text())
    for key, filename in (("split_mask", "split_mask.tif"), ("analysis_band", "analysis_band.tif")):
        shutil.copyfile(result["artifacts"][key], ARTIFACTS / filename)
        report["artifacts"][key] = filename
    report["artifacts"]["report_json"] = "real_demo_report.json"
    report["artifacts"]["scope"] = "Files beside this report in artifacts/"
    report["integration_test"] = {"transport": "MCP stdio initialize + tools/list + tools/call",
                                  "real_numeric_pixels": True, "synthetic_fallback": False}
    with rasterio.open(ARTIFACTS / "analysis_band.tif") as src:
        values = src.read(1, masked=True).filled(np.nan).astype("float64")
        pixel_size = src.res[0]
    from core import create_spatial_split

    reference_buffer = report["requested_buffer_m"]
    sensitivity = []
    for buffer in sorted(set([0, reference_buffer / 2, reference_buffer, reference_buffer * 2])):
        try:
            check = create_spatial_split(values, pixel_size_m=pixel_size, seed=42, buffer_m=buffer)
            check_report = check["report"]
            sensitivity.append({"buffer_m": buffer, "status": "ok", "splits": check_report["splits"],
                                "spatial_diagnostics": check_report["spatial_diagnostics"],
                                "retention_fraction": float((check["split_mask"] > 0).sum() / np.isfinite(values).sum())})
        except ValueError as exc:
            sensitivity.append({"buffer_m": buffer, "status": "infeasible", "message": str(exc)})
    (ARTIFACTS / "sensitivity.json").write_text(json.dumps(sensitivity, indent=2, allow_nan=False))
    report["sensitivity"] = sensitivity
    (ARTIFACTS / "real_demo_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def plot_report(report: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    with rasterio.open(ARTIFACTS / "analysis_band.tif") as src:
        values = src.read(1, masked=True)
    with rasterio.open(ARTIFACTS / "split_mask.tif") as src:
        mask = src.read(1)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    lo, hi = np.nanpercentile(values.filled(np.nan), [2, 98])
    axes[0, 0].imshow(values, cmap="gray", vmin=lo, vmax=hi)
    axes[0, 0].set_title("Actual Sentinel-2 B05, SCL masked, native 20 m")
    axes[0, 1].imshow(mask, cmap=ListedColormap(["#111827", "#ef6c45", "#4fbb9b", "#5b83da"]), vmin=0, vmax=3)
    axes[0, 1].set_title(f"Train / validation / test; {report['requested_buffer_m']:.0f} m guard")
    with rasterio.open(ARTIFACTS / "analysis_band.tif") as src:
        from core import create_spatial_split

        baseline = create_spatial_split(values.filled(np.nan), pixel_size_m=src.res[0], seed=42,
                                        buffer_m=report["requested_buffer_m"])["random_split_mask"]
    axes[0, 2].imshow(baseline, cmap=ListedColormap(["#111827", "#ef6c45", "#4fbb9b", "#5b83da"]), vmin=0, vmax=3)
    axes[0, 2].set_title("Random baseline, same pixels and split counts")
    for ax in axes[0]:
        ax.set_xlabel("Column (native pixel)")
        ax.set_ylabel("Row (native pixel)")
    ac = report["autocorrelation"]
    rows = ac["lags"]
    for direction in rows[0]["directions"]:
        points = [row["directions"][direction] for row in rows]
        xs = [point["distance_m"] for point in points]
        axes[1, 0].plot(xs, [point["correlation"] for point in points], "o-", markersize=3,
                        label=direction.replace("_", " "))
        axes[1, 1].plot(xs, [point["semivariance"] for point in points], "o-", markersize=3)
    axes[1, 0].axhline(0.1, color="#888888", linestyle="--")
    axes[1, 0].set(xlabel="Lag (m)", ylabel="Correlation", title="Empirical spatial dependence")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].axhline(-0.1, color="#888888", linestyle="--")
    axes[1, 1].axhline(ac["global_variance"], color="#888888", linestyle="--")
    axes[1, 1].set(xlabel="Lag (m)", ylabel="Semivariance (reflectance squared)", title="Empirical directional semivariogram")
    sensitivity = [x for x in report["sensitivity"] if x["status"] == "ok"]
    axes[1, 2].plot([x["buffer_m"] for x in sensitivity],
                    [100 * x["retention_fraction"] for x in sensitivity], "o-", color="#24866d")
    axes[1, 2].set(xlabel="Requested guard distance (m)", ylabel="Valid pixels retained (%)",
                   title="Retention versus spatial separation", ylim=(0, 105))
    fig.suptitle("EO Spatial Split Planner — measured image dependence, not a proof of independence")
    fig.savefig(ARTIFACTS / "diagnostic.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    checks = smoke()
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "mcp_smoke_test.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks))
    if args.real:
        report = real()
        plot_report(report)
        print(json.dumps({"item_id": report["provenance"]["item_id"], "buffer_m": report["requested_buffer_m"],
                          "range_status": report["autocorrelation"]["range_status"], "splits": report["splits"]}, indent=2))
