"""Prepare bounded Sentinel-2 reflectance and weak WorldCover reference labels.

The annual WorldCover product is an illustrative, machine-generated reference,
not independent field ground truth: its production also uses Sentinel imagery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from imagery import COLLECTION, MAX_PIXELS, STAC, _get_json, _signed, find_scene, read_sentinel2, validate_bbox
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

DEFAULT_BBOX = (13.30, 52.35, 13.50, 52.50)
BAND_NAMES = ("B05", "B03", "B04", "B08")
LABEL_COLLECTION = "esa-worldcover"
WORLD_COVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}
LABEL_CAVEAT = (
    "ESA WorldCover 2021 is a weak, machine-generated annual reference rather than "
    "independent field ground truth. Its classifier uses Sentinel-1 and Sentinel-2, "
    "so this demonstration measures agreement with WorldCover; it cannot establish "
    "true segmentation accuracy or prove that a buffer improves accuracy. Annual "
    "labels may differ from land cover visible on the single acquisition date."
)


def _covering_worldcover(bbox: list[float]) -> dict:
    items = _get_json(f"{STAC}/search", {
        "collections": LABEL_COLLECTION, "bbox": ",".join(map(str, bbox)),
        "datetime": "2021-01-01T00:00:00Z/2021-12-31T23:59:59Z", "limit": 100,
    }).get("features", [])
    w, s, e, n = bbox
    candidates = [item for item in items
                  if item["properties"].get("esa_worldcover:product_version") == "2.0.0"
                  and item["properties"].get("start_datetime", "").startswith("2021")
                  and item["bbox"][0] <= w and item["bbox"][1] <= s
                  and item["bbox"][2] >= e and item["bbox"][3] >= n]
    if not candidates:
        raise ValueError("No single ESA WorldCover 2021 v200 tile covers this AOI.")
    return sorted(candidates, key=lambda item: item["id"])[0]


def fetch_segmentation_data(
    output_dir: str | Path,
    bbox: list[float] | tuple[float, ...] = DEFAULT_BBOX,
    start_date: str = "2021-06-01",
    end_date: str = "2021-08-31",
    item_id: str | None = None,
    max_cloud_cover: float = 10,
    force: bool = False,
) -> dict:
    """Fetch four calibrated bands and 2021 labels on a native 20 m B05 grid.

    Each output pixel has a valid SCL class (4, 5 or 6), four finite reflectance
    values, and a valid WorldCover class. Labels use nearest-neighbour resampling;
    the 10 m reflectance bands are averaged to the native B05 20 m grid. Output
    paths in the returned manifest are absolute. The manifest never stores SAS
    tokens. Reads are capped at one million 20 m pixels.
    """
    bbox = validate_bbox(list(bbox))
    if not start_date.startswith("2021-") or not end_date.startswith("2021-"):
        raise ValueError("Use a 2021 scene with the 2021 annual WorldCover labels.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {"features_path": output / "sentinel2_reflectance_20m.tif",
             "labels_path": output / "worldcover_2021_20m.tif",
             "manifest_path": output / "manifest.json"}
    request = {"bbox_wgs84": bbox, "start_date": start_date, "end_date": end_date,
               "requested_item_id": item_id, "max_cloud_cover": max_cloud_cover,
               "band_names": list(BAND_NAMES), "schema_version": 1}
    if not force and all(path.is_file() for path in paths.values()):
        cached = json.loads(paths["manifest_path"].read_text())
        if cached.get("request") == request:
            cached.update({key: str(path) for key, path in paths.items()})
            return cached

    # The existing reader applies baseline-aware radiometry, the AOI mask and
    # an SCL cloud/shadow/snow mask, preserving the native projected B05 grid.
    scene = find_scene(bbox, start_date, end_date, item_id, max_cloud_cover)
    b05, profile, b05_provenance = read_sentinel2(
        bbox, start_date, end_date, "B05", scene["id"], max_cloud_cover)
    if b05.size > MAX_PIXELS:
        raise ValueError("Segmentation AOI exceeds the one million 20 m pixel limit.")
    valid = np.isfinite(b05)
    if not np.any(valid):
        raise ValueError("No valid cloud-masked Sentinel-2 pixels in this AOI.")
    worldcover = _covering_worldcover(bbox)
    s2_token = _get_json(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{COLLECTION}")["token"]
    label_token = _get_json(
        f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{LABEL_COLLECTION}")["token"]
    vrt_grid = {"crs": profile["crs"], "transform": profile["transform"],
                "width": profile["width"], "height": profile["height"]}
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
           "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif", "GDAL_HTTP_TIMEOUT": "60",
           "GDAL_HTTP_MAX_RETRY": "2"}
    bands, calibrations = [], {}
    baseline = float(scene["properties"].get("s2:processing_baseline", "0"))
    with rasterio.Env(**env):
        for band in BAND_NAMES:
            asset = scene["assets"][band]
            metadata = asset.get("raster:bands", [{}])[0]
            scale = float(metadata.get("scale", 0.0001))
            offset = float(metadata.get("offset", -0.1 if baseline >= 4 else 0))
            if band == "B05":
                calibrated = b05
            else:
                with rasterio.open(_signed(asset["href"], s2_token)) as src:
                    with WarpedVRT(src, **vrt_grid, resampling=Resampling.average,
                                   src_nodata=0, nodata=np.nan, dtype="float32") as vrt:
                        raw = vrt.read(1, masked=True).filled(np.nan)
                calibrated = (raw * scale + offset).astype("float32")
            valid &= np.isfinite(calibrated)
            bands.append(calibrated)
            calibrations[band] = {"scale": scale, "offset": offset,
                                  "calibration_source": "STAC raster:bands" if "scale" in metadata
                                  else "Sentinel-2 L2A processing baseline convention",
                                  "resampling": "none (native 20 m)" if band == "B05"
                                  else "area average from native 10 m to B05 20 m grid"}
        with rasterio.open(_signed(worldcover["assets"]["map"]["href"], label_token)) as src:
            with WarpedVRT(src, **vrt_grid, resampling=Resampling.nearest,
                           src_nodata=0, nodata=0) as vrt:
                labels = vrt.read(1).astype("uint8")
    valid &= np.isin(labels, list(WORLD_COVER_CLASSES))
    if not np.any(valid):
        raise ValueError("No jointly valid Sentinel-2 and WorldCover pixels in AOI.")
    features = np.stack(bands)
    features[:, ~valid] = np.nan
    labels[~valid] = 0
    write_profile = profile.copy()
    # ZSTD is lossless and keeps this four-band demo below the registry's
    # individual-file limit; floating-point prediction enlarged this dataset.
    write_profile.update(driver="GTiff", dtype="float32", count=len(BAND_NAMES),
                         nodata=np.nan, compress="zstd", predictor=1, zstd_level=19, tiled=True)
    with rasterio.open(paths["features_path"], "w", **write_profile) as dst:
        dst.write(features)
        dst.write_mask(valid.astype("uint8") * 255)
        for band_index, band in enumerate(BAND_NAMES, 1):
            dst.set_band_description(band_index, band)
        dst.update_tags(source_item_id=scene["id"], units="surface reflectance",
                        processing_baseline=str(baseline), radiometry="calibrated; do not scale again")
    write_profile.update(dtype="uint8", count=1, nodata=0, compress="deflate", predictor=2, zlevel=9)
    with rasterio.open(paths["labels_path"], "w", **write_profile) as dst:
        dst.write(labels, 1)
        dst.set_band_description(1, "ESA WorldCover 2021 v200 land cover class")
        dst.update_tags(source_item_id=worldcover["id"], label_type="weak reference",
                        attribution="ESA WorldCover 2021 / Contains modified Copernicus Sentinel data (2021)",
                        license="CC-BY-4.0", caveat=LABEL_CAVEAT)
    codes, counts = np.unique(labels[valid], return_counts=True)
    manifest = {
        **{key: str(path) for key, path in paths.items()}, "request": request,
        "shape": list(b05.shape), "pixel_size_m": 20.0, "crs": str(profile["crs"]),
        "transform": list(profile["transform"])[:6], "band_names": list(BAND_NAMES),
        "valid_pixels": int(valid.sum()), "total_pixels": int(valid.size),
        "sentinel2": {**b05_provenance, "calibrations": calibrations,
                      "acquisition_path": "Public Planetary Computer STAC directly; not through EVE"},
        "worldcover": {"collection": LABEL_COLLECTION, "item_id": worldcover["id"],
                       "item_url": f"{STAC}/collections/{LABEL_COLLECTION}/items/{worldcover['id']}",
                       "year": 2021, "product_version": "2.0.0", "native_resolution_m": 10,
                       "resampling": "nearest neighbour onto native B05 20 m grid", "nodata": 0,
                       "class_names": WORLD_COVER_CLASSES,
                       "class_pixels": {str(int(code)): int(count) for code, count in zip(codes, counts, strict=True)},
                       "license": "CC-BY-4.0", "attribution": "ESA WorldCover 2021",
                       "doi": "https://doi.org/10.5281/zenodo.7254221"},
        "label_caveat": LABEL_CAVEAT,
    }
    paths["manifest_path"].write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "artifacts" / "segmentation_data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = fetch_segmentation_data(args.output_dir, force=args.force)
    print(json.dumps({key: result[key] for key in ["features_path", "labels_path", "manifest_path",
                                                "shape", "valid_pixels", "pixel_size_m"]}, indent=2))
