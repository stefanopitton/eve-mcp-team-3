"""Geospatial adapter contracts that prevent meaningless metre diagnostics."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imagery import read_local, resolve_band, validate_bbox  # noqa: E402


def write_raster(path, crs="EPSG:32633", transform=None):
    data = np.arange(100, dtype="float32").reshape(10, 10)
    data[0, 0] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="float32", crs=crs, transform=transform or from_origin(400000, 5800000, 20, 20),
                       nodata=-9999) as dst:
        dst.write(data, 1)


def test_local_mask_and_projection(tmp_path):
    path = tmp_path / "science.tif"
    write_raster(path)
    values, profile, provenance = read_local(str(path))
    assert np.isnan(values[0, 0])
    assert np.isfinite(values).sum() == 99
    assert profile["crs"].to_epsg() == 32633
    assert provenance["pixel_size_m"] == 20


@pytest.mark.parametrize("crs,transform", [
    ("EPSG:4326", from_origin(13, 53, 0.001, 0.001)),
    ("EPSG:2263", from_origin(900000, 200000, 20, 20)),
    ("EPSG:32633", from_origin(400000, 5800000, 20, 10)),
])
def test_rejects_angular_feet_and_nonsquare_grids(tmp_path, crs, transform):
    path = tmp_path / "bad_grid.tif"
    write_raster(path, crs, transform)
    with pytest.raises(ValueError):
        read_local(str(path))


@pytest.mark.parametrize("bbox", [[170, -10, -170, 10], [13, 52, 15, 53], [13, 53, 14, 52], [13, 52, float("nan"), 53]])
def test_rejects_invalid_aoi(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


@pytest.mark.parametrize("name,expected", [("nir", "B08"), ("red_edge1", "B05"), ("B05", "B05"), ("b8a", "B8A")])
def test_checkpoint_band_names(name, expected):
    assert resolve_band(name) == expected


def test_remote_index_requires_a_derived_local_field():
    with pytest.raises(ValueError, match="local raster"):
        resolve_band("ndvi")
