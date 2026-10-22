"""Replay the paired CPU segmentation demo from the cached real EO GeoTIFFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from core import create_spatial_split
from segmentation import run_segmentation_benchmark, worldcover_to_classes

HERE = Path(__file__).resolve().parent


def load_demo_data(data_dir):
    """Load aligned, cloud-masked Sentinel-2 features and external reference labels."""
    directory = Path(data_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    with rasterio.open(directory / "sentinel2_reflectance_20m.tif") as source:
        values = source.read(masked=True).astype("float32").filled(np.nan)
        profile = source.profile.copy()
        resolution = float(source.transform.a)
    with rasterio.open(directory / "worldcover_2021_20m.tif") as source:
        if (source.transform != profile["transform"] or source.crs != profile["crs"]
                or (source.height, source.width) != values.shape[1:]):
            raise ValueError("Sentinel-2 and WorldCover must share exactly the same CRS and pixel grid")
        raw_labels = source.read(1)
    return values, worldcover_to_classes(raw_labels), profile, resolution, manifest


def run_demo(data_dir=HERE / "artifacts" / "segmentation_data",
             output_dir=HERE / "artifacts" / "segmentation_benchmark", *,
             buffer_m=None, split_seed=3, model_seed=42, steps=150,
             max_pairs_per_lag=200_000, include_size_matched_control=True):
    values, labels, profile, resolution, manifest = load_demo_data(data_dir)
    # Patch geometry depends on imagery validity and the split seed, never labels.
    valid_imagery = np.all(np.isfinite(values), axis=0)
    plan = create_spatial_split(
        values[0], valid_imagery, pixel_size_m=resolution, seed=split_seed,
        buffer_m=buffer_m, patch_size_pixels=(48, 64), n_patches=40,
        max_lag_m=12_800, max_pairs_per_lag=max_pairs_per_lag)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, mask in (("unbuffered_patch_split.tif", plan["unbuffered_split_mask"]),
                           ("buffered_patch_split.tif", plan["split_mask"])):
        output_profile = {**profile, "driver": "GTiff", "count": 1, "dtype": "uint8", "nodata": 0,
                          "compress": "deflate"}
        with rasterio.open(output / filename, "w", **output_profile) as destination:
            destination.write(mask, 1)
    (output / "benchmark_plan.json").write_text(json.dumps(plan["report"], indent=2, allow_nan=False) + "\n")
    return run_segmentation_benchmark(
        values, labels, plan["patches"], plan["unbuffered_split_mask"], plan["split_mask"],
        resolution_m=resolution, buffer_m=plan["report"]["requested_buffer_m"],
        output_dir=output, seed=model_seed, steps=steps,
        include_size_matched_control=include_size_matched_control,
        label_provenance={
            "source": "ESA WorldCover 2021 v200, nearest-neighbour aligned to Sentinel-2 B05 native 20 m grid",
            "source_item": manifest["worldcover"]["item_id"],
            "sentinel2_item": manifest["sentinel2"]["item_id"],
            "source_year": 2021, "source_date": manifest["sentinel2"]["datetime"],
            "class_mapping": {"0_vegetation": [10, 20, 30, 40, 90, 95, 100],
                              "1_built_up": [50], "2_water": [80],
                              "ignored": "nodata 0, bare/sparse 60, snow/ice 70, unknown codes"},
            "license": manifest["worldcover"]["license"],
            "attribution": manifest["worldcover"]["attribution"],
            "caveat": manifest["label_caveat"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "artifacts" / "segmentation_data")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "segmentation_benchmark")
    parser.add_argument("--buffer-m", type=float, default=None, help="Omit to estimate from the B05 empirical variogram")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=3)
    parser.add_argument("--max-pairs-per-lag", type=int, default=200_000)
    parser.add_argument("--skip-size-matched-control", action="store_true")
    arguments = parser.parse_args()
    report = run_demo(arguments.data_dir, arguments.output_dir, buffer_m=arguments.buffer_m,
                      split_seed=arguments.split_seed, model_seed=arguments.model_seed,
                      steps=arguments.steps, max_pairs_per_lag=arguments.max_pairs_per_lag,
                      include_size_matched_control=not arguments.skip_size_matched_control)
    print(json.dumps({"output_dir": str(arguments.output_dir),
                      "retained_training_fraction": report["retained_training_labelled_fraction"],
                      "conditions": {name: {"train_patches": value["training_patch_count"],
                                             "test": value["test"]}
                                     for name, value in report["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
