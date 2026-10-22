"""Bounded, native-resolution Sentinel-2 reads and projected raster validation."""

from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, from_bounds

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_PIXELS = 1_000_000
MAX_LOCAL_PIXELS = 32_000_000
BANDS = {"B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"}
BAND_ALIASES = {
    "blue": "B02", "green": "B03", "red": "B04", "red_edge1": "B05",
    "red_edge2": "B06", "red_edge3": "B07", "nir": "B08",
    "narrow_nir": "B8A", "swir1": "B11", "swir2": "B12",
}


def resolve_band(band: str) -> str:
    """Accept Sentinel-2 asset codes and canonical checkpoint-repository names."""
    native = BAND_ALIASES.get(band.lower(), band.upper())
    if native not in BANDS:
        raise ValueError(f"band must be an asset code {sorted(BANDS)} or a name {sorted(BAND_ALIASES)}; derived indices require a local raster.")
    return native


def validate_grid(src) -> float:
    """Require a square, north-up projected grid in metres; never use degrees as metres."""
    if src.crs is None or not src.crs.is_projected:
        raise ValueError("Raster must have a projected CRS in metres. Reproject geographic grids first.")
    if src.crs.linear_units not in {"metre", "meter"}:
        raise ValueError("Raster CRS units must be metres; reproject feet or other units first.")
    t = src.transform
    if abs(t.b) > 1e-8 or abs(t.d) > 1e-8 or t.a <= 0 or t.e >= 0:
        raise ValueError("Raster must be north-up without rotation or shear.")
    if not math.isclose(t.a, -t.e, rel_tol=1e-6):
        raise ValueError("Raster pixels must be square. Resample to a square metre grid first.")
    return float(t.a)


def read_local(path: str, band_index: int = 1) -> tuple[np.ndarray, dict, dict]:
    """Read one bounded local raster band, including dataset masks and nodata."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError("raster_path must identify a local GeoTIFF visible to this server.")
    with rasterio.open(p) as src:
        pixel_size = validate_grid(src)
        if src.width * src.height > MAX_LOCAL_PIXELS:
            raise ValueError(f"Raster exceeds {MAX_LOCAL_PIXELS:,} pixels; crop it first at native resolution.")
        if not 1 <= band_index <= src.count:
            raise ValueError("band_index is outside the raster band count.")
        arr = src.read(band_index, masked=True).astype("float32").filled(np.nan)
        arr = arr * src.scales[band_index - 1] + src.offsets[band_index - 1]
        profile = src.profile.copy()
    return arr, profile, {"source": "local_raster", "filename": p.name,
                          "band_index": band_index, "pixel_size_m": pixel_size,
                          "cloud_mask": "Only the supplied dataset mask/nodata; supply a cloud-masked raster."}


def validate_bbox(bbox: list[float]) -> list[float]:
    if len(bbox) != 4 or not all(math.isfinite(x) for x in bbox):
        raise ValueError("bbox must be four finite WGS84 values: west,south,east,north.")
    w, s, e, n = map(float, bbox)
    if not (-180 <= w < e <= 180 and -80 <= s < n <= 84):
        raise ValueError("bbox must be within WGS84 Sentinel-2 coverage, without crossing the antimeridian.")
    if e - w > 1 or n - s > 1:
        raise ValueError("Use an AOI at most one degree wide/high; the native read also has a pixel cap.")
    return [w, s, e, n]


def _get_json(url: str, params: dict | None = None) -> dict:
    with httpx.Client(timeout=60, follow_redirects=False) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def find_scene(bbox: list[float], start_date: str, end_date: str,
               item_id: str | None = None, max_cloud_cover: float = 10) -> dict:
    bbox = validate_bbox(bbox)
    start, end = dt.date.fromisoformat(start_date), dt.date.fromisoformat(end_date)
    if start > end or (end - start).days > 366:
        raise ValueError("Use ordered ISO dates spanning at most 366 days.")
    if not math.isfinite(max_cloud_cover) or not 0 <= max_cloud_cover <= 100:
        raise ValueError("max_cloud_cover must be between 0 and 100 percent.")
    if item_id:
        if not re.fullmatch(r"[A-Za-z0-9_\-]{1,200}", item_id):
            raise ValueError("item_id must be a Sentinel-2 STAC identifier, not a URL.")
        scenes = [_get_json(f"{STAC}/collections/{COLLECTION}/items/{item_id}")]
    else:
        result = _get_json(f"{STAC}/search", {
            "collections": COLLECTION, "bbox": ",".join(map(str, bbox)),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": json.dumps({"eo:cloud_cover": {"lte": max_cloud_cover}}),
        })
        scenes = result.get("features", [])
    w, s, e, n = bbox
    candidates = []
    for scene in scenes:
        sw, ss, se, sn = scene["bbox"][:4]
        when = scene["properties"]["datetime"][:10]
        cloud = float(scene["properties"].get("eo:cloud_cover", 100))
        if sw <= w and ss <= s and se >= e and sn >= n and str(start) <= when <= str(end) and cloud <= max_cloud_cover:
            candidates.append(scene)
    if not candidates:
        raise ValueError("No single scene covers the AOI within the dates/cloud limit. Try another AOI or dates.")
    # Reproducible within this bounded first catalog page; users can pin an item ID.
    return min(candidates, key=lambda x: (x["properties"].get("eo:cloud_cover", 100), x["id"]))


def _signed(href: str, token: str) -> str:
    host = urlsplit(href).hostname or ""
    if not host.endswith(".blob.core.windows.net") or urlsplit(href).scheme != "https":
        raise ValueError("Catalog asset does not point to the supported Azure HTTPS archive.")
    return href + "?" + token


def read_sentinel2(bbox: list[float], start_date: str, end_date: str,
                   band: str = "B05", item_id: str | None = None,
                   max_cloud_cover: float = 10) -> tuple[np.ndarray, dict, dict]:
    """Read an AOI in one native-resolution scene; SCL uses nearest neighbour."""
    bbox = validate_bbox(bbox)
    band = resolve_band(band)
    scene = find_scene(bbox, start_date, end_date, item_id, max_cloud_cover)
    asset = scene["assets"][band]
    sas = _get_json(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLLECTION}")["token"]
    # Restrict GDAL's remote reads; never persist signed URLs or include them in errors.
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
           "GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_MAX_RETRY": "2"}
    with rasterio.Env(**env), rasterio.open(_signed(asset["href"], sas)) as src:
        pixel_size = validate_grid(src)
        bounds = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        fractional = from_bounds(*bounds, transform=src.transform)
        col, row = math.floor(fractional.col_off), math.floor(fractional.row_off)
        width = math.ceil(fractional.col_off + fractional.width) - col
        height = math.ceil(fractional.row_off + fractional.height) - row
        if width * height > MAX_PIXELS:
            raise ValueError(f"AOI would read {width * height:,} native pixels (limit {MAX_PIXELS:,}); shrink bbox.")
        window = Window(col, row, width, height).intersection(Window(0, 0, src.width, src.height))
        raw = src.read(1, window=window, masked=True)
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update(width=raw.shape[1], height=raw.shape[0], transform=transform)
        with rasterio.open(_signed(scene["assets"]["SCL"]["href"], sas)) as scl_src:
            with WarpedVRT(scl_src, crs=src.crs, transform=transform, width=raw.shape[1],
                           height=raw.shape[0], resampling=Resampling.nearest) as vrt:
                scl = vrt.read(1)
        w, s, e, n = bbox
        polygon = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        inside = geometry_mask([transform_geom("EPSG:4326", src.crs, polygon)],
                               raw.shape, transform, invert=True)
        valid = ~np.ma.getmaskarray(raw) & (raw.data > 0) & np.isin(scl, [4, 5, 6]) & inside
    rb = asset.get("raster:bands", [{}])[0]
    baseline = float(scene["properties"].get("s2:processing_baseline", "0"))
    scale = float(rb.get("scale", 0.0001))
    offset = float(rb.get("offset", -0.1 if baseline >= 4 else 0))
    values = np.where(valid, raw.data.astype("float64") * scale + offset, np.nan).astype("float32")
    provenance = {
        "source": "Microsoft Planetary Computer STAC + native Sentinel-2 L2A COG windows",
        "catalog_url": STAC, "collection": COLLECTION, "item_id": scene["id"],
        "item_url": f"{STAC}/collections/{COLLECTION}/items/{scene['id']}",
        "datetime": scene["properties"]["datetime"], "band": band, "bbox_wgs84": bbox,
        "scene_cloud_cover_percent": scene["properties"].get("eo:cloud_cover"),
        "processing_baseline": baseline, "reflectance_scale": scale, "reflectance_offset": offset,
        "calibration_source": "STAC raster:bands" if "scale" in rb else "Sentinel-2 L2A processing baseline convention",
        "pixel_size_m": pixel_size, "crs": str(profile["crs"]),
        "shape": list(values.shape), "valid_pixels": int(valid.sum()),
        "cloud_mask": "SCL classes 4 vegetation, 5 nonvegetated, 6 water; all other classes and nodata excluded",
        "acquisition_path": "Public STAC directly; no Sentinel-2 downloader exposed by this EVE account at test time",
        "selection": "Least scene cloud cover among first 100 covering catalog items, then item ID; pin item_id for replay",
    }
    return values, profile, provenance
