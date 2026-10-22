"""EO Spatial Split Planner: native EO imagery to measured, reproducible spatial partitions."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from core import create_spatial_split
from imagery import read_local, read_sentinel2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("EO Spatial Split Planner", host="0.0.0.0", port=8000, stateless_http=True)


def _export(values: np.ndarray, profile: dict, provenance: dict,
            buffer_m: float | None, seed: int, split_ratios: list[float] | None,
            patch_radius_m: float = 0) -> dict:
    result = create_spatial_split(values, pixel_size_m=abs(profile["transform"].a),
                                  split_ratios=[0.6, 0.2, 0.2] if split_ratios is None else split_ratios,
                                  buffer_m=buffer_m, seed=seed, patch_radius_m=patch_radius_m)
    report = result["report"]
    report.update(schema_version="0.1.0", provenance=provenance)
    fingerprint = hashlib.sha256()
    fingerprint.update(np.nan_to_num(values, nan=-9999).astype("<f8").tobytes())
    fingerprint.update(json.dumps({"provenance": provenance, "buffer_m": buffer_m,
                                   "seed": seed, "ratios": split_ratios,
                                   "patch_radius_m": patch_radius_m}, sort_keys=True).encode())
    run_id = fingerprint.hexdigest()[:16]
    root = Path(os.getenv("EO_SPLIT_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "eo-spatial-split-planner")))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=root))
    mask_path, report_path, source_path = directory / "split_mask.tif", directory / "report.json", directory / "analysis_band.tif"
    output_profile = profile.copy()
    for key in ("blockxsize", "blockysize", "tiled", "photometric", "interleave"):
        output_profile.pop(key, None)
    output_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(mask_path, "w", **output_profile) as dst:
        dst.write(result["split_mask"], 1)
        dst.update_tags(labels="0=nodata_or_buffer,1=train,2=validation,3=test", seed=seed)
    output_profile.update(dtype="float32", nodata=-9999)
    with rasterio.open(source_path, "w", **output_profile) as dst:
        dst.write(np.nan_to_num(values, nan=-9999).astype("float32"), 1)
    report["artifacts"] = {"split_mask": str(mask_path), "analysis_band": str(source_path),
                           "report_json": str(report_path), "scope": "Paths on the MCP server filesystem"}
    report["run_fingerprint"] = run_id
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    # Keep LLM responses compact. Full pair-level lag tables remain in report.json.
    compact = {k: v for k, v in report.items() if k != "autocorrelation"}
    compact["autocorrelation"] = {k: v for k, v in report["autocorrelation"].items()
                                  if k not in {"correlogram", "lags", "directional", "directions", "empirical"}}
    # Later core versions may name the lag table differently; omit nested large lists generically.
    compact["autocorrelation"] = {k: v for k, v in compact["autocorrelation"].items()
                                  if not isinstance(v, list) or len(v) <= 8}
    return {"status": "ok", **compact}


def _error(exc: Exception) -> dict:
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return {"status": "error", "error": {"code": "invalid_input_or_infeasible_split", "message": str(exc)}}
    # GDAL/http exception messages can include transient signed asset URLs.
    return {"status": "error", "error": {"code": "imagery_or_processing_failed",
            "message": f"{type(exc).__name__}: imagery retrieval or processing failed. Check connectivity, scene coverage and the pixel limit; retry or use a local raster."}}


@mcp.tool()
def plan_sentinel2_split(
    bbox: list[float], start_date: str = "2024-06-01", end_date: str = "2024-08-31",
    band: str = "B05", item_id: str | None = None, max_cloud_cover: float = 10,
    buffer_m: float | None = None, seed: int = 42,
    split_ratios: list[float] | None = None, patch_radius_m: float = 0,
) -> dict:
    """Plan train/validation/test spatial splits from real Sentinel-2 L2A imagery.

    bbox is WGS84 [west,south,east,north], at most 1 degree and 1 million native
    pixels. Use a small AOI, e.g. [13.30,52.35,13.50,52.50]. Dates are ISO dates,
    at most 366 days apart. B05 is the 20 m red-edge band. Finds one covering
    scene via public Planetary Computer STAC, cloud masks with SCL and reads
    native COG pixels. item_id can pin a STAC scene returned by another tool.
    buffer_m=None measures a heuristic decorrelation range; unresolved estimates
    are flagged. split_ratios defaults to [0.6,0.2,0.2]. patch_radius_m reserves
    extra space around ML patch centers. seed fixes sampling and spatial order.
    Returns compact diagnostics, exact separation checks, random baseline,
    retention, provenance and GeoTIFF/JSON paths on the server filesystem.
    Spectral decorrelation does not prove label or temporal independence.
    """
    try:
        values, profile, provenance = read_sentinel2(bbox, start_date, end_date, band, item_id, max_cloud_cover)
        return _export(values, profile, provenance, buffer_m, seed, split_ratios, patch_radius_m)
    except Exception as exc:
        return _error(exc)


@mcp.tool()
def plan_spatial_split(
    raster_path: str, band_index: int = 1, buffer_m: float | None = None,
    seed: int = 42, split_ratios: list[float] | None = None, patch_radius_m: float = 0,
) -> dict:
    """Plan reproducible spatial splits for a local cloud-masked EO GeoTIFF.

    raster_path must exist on this MCP server, allowing handoff from another
    imagery MCP tool sharing its filesystem. Requires a north-up, square,
    projected metre grid with at most 1 million pixels. Dataset masks and nodata
    are honored; band_index is 1-based. Buffer, ratios and patch radius have the
    same meaning as plan_sentinel2_split. Returns JSON diagnostics and artifact
    paths, not raster arrays. An image preview/PNG is not a scientific raster.
    """
    try:
        values, profile, provenance = read_local(raster_path, band_index)
        return _export(values, profile, provenance, buffer_m, seed, split_ratios, patch_radius_m)
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
