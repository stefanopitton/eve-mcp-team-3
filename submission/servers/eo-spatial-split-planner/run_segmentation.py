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
             max_pairs_per_lag=200_000, include_size_matched_control=True,
             patch_layout="reference_blocks", patch_size_pixels=(32, 40),
             block_size_pixels=(96, 120), split_ratios=(0.7, 0.1, 0.2),
             seed_candidates=200, max_nan_fraction=0.5, require_class_coverage=True,
             n_patches=None, edge_policy="extend", crop_size=16):
    values, labels, profile, resolution, manifest = load_demo_data(data_dir)
    # Reference mode can use class coverage to choose the split seed, as script04
    # does. This is disclosed dataset design; model scores never select geometry.
    valid_imagery = np.all(np.isfinite(values), axis=0)
    use_class_coverage = require_class_coverage and patch_layout == "reference_blocks"
    plan = create_spatial_split(
        values[0], valid_imagery, pixel_size_m=resolution, seed=split_seed,
        buffer_m=buffer_m, patch_layout=patch_layout, patch_size_pixels=patch_size_pixels,
        block_size_pixels=block_size_pixels, split_ratios=split_ratios,
        seed_candidates=seed_candidates, max_nan_fraction=max_nan_fraction,
        class_labels=labels if use_class_coverage else None,
        require_class_coverage=use_class_coverage, n_patches=n_patches, edge_policy=edge_policy,
        max_lag_m=12_800, max_pairs_per_lag=max_pairs_per_lag)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for filename, mask in (("unbuffered_patch_split.tif", plan["unbuffered_split_mask"]),
                           ("buffered_patch_split.tif", plan["split_mask"]),
                           ("candidate_patch_split.tif", plan.get("candidate_split_mask", plan["unbuffered_split_mask"]))):
        output_profile = {**profile, "driver": "GTiff", "count": 1, "dtype": "uint8", "nodata": 0,
                          "compress": "deflate"}
        with rasterio.open(output / filename, "w", **output_profile) as destination:
            destination.write(mask, 1)
    (output / "benchmark_plan.json").write_text(json.dumps(plan["report"], indent=2, allow_nan=False) + "\n")
    from patch_export import export_patch_datasets

    patch_export = export_patch_datasets(
        values, labels, plan, output / "patch_exports", channel_names=manifest["band_names"],
        source_metadata={"sentinel2": manifest["sentinel2"], "worldcover": manifest["worldcover"],
                         "label_caveat": manifest["label_caveat"],
                         "class_mapping": {"0_vegetation": [10, 20, 30, 40, 90, 95, 100],
                                           "1_built_up": [50], "2_water": [80],
                                           "ignored": "0, 60, 70 and unknown codes"}})
    result = run_segmentation_benchmark(
        values, labels, plan["patches"], plan["unbuffered_split_mask"], plan["split_mask"],
        resolution_m=resolution, buffer_m=plan["report"]["requested_buffer_m"],
        output_dir=output, seed=model_seed, steps=steps, crop_size=crop_size,
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
    # Resolve exports beside this report, independently of the original checkout.
    for exported in patch_export["conditions"].values():
        for key in ("archive_path", "metadata_path", "normalization_path"):
            exported[key] = str(Path(exported[key]).relative_to(output.resolve()))
    patch_export["manifest_path"] = str(Path(patch_export["manifest_path"]).relative_to(output.resolve()))
    patch_export["path_scope"] = "Paths relative to the segmentation report directory"
    result["patch_export"] = patch_export
    result["patch_planning"] = {key: plan["report"][key] for key in (
        "method", "patch_layout", "seed", "selected_seed", "requested_split_ratios",
        "patch_size_pixels", "block_size_pixels", "edge_policy", "seed_search", "buffer_radius_patches",
        "paired_comparison", "buffer_scope", "buffer_not_applied_to", "reference_implementation") if key in plan["report"]}
    if "seed_search" in result["patch_planning"]:
        result["patch_planning"]["seed_search"] = {
            key: value for key, value in result["patch_planning"]["seed_search"].items() if key != "candidates"}
    result["interpretation"] += (
        " Reference-block mode uses symmetric erosion of heldout patches. The paired baseline restores "
        "training only and can already have a one-sided heldout guard; this is not a leaky-versus-independent "
        "test. Optional class-coverage seed search uses reference labels for dataset design, not model scores.")
    (output / "segmentation_report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


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
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.7, 0.1, 0.2),
                        metavar=("TRAIN", "VALIDATION", "TEST"), help="Positive fractions summing to one")
    parser.add_argument("--patch-layout", choices=("reference_blocks", "random_rectangles"), default="reference_blocks")
    parser.add_argument("--patch-size-pixels", nargs=2, type=int, default=(32, 40), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--block-size-pixels", nargs=2, type=int, default=(96, 120), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--seed-candidates", type=int, default=200)
    parser.add_argument("--max-nan-fraction", type=float, default=0.5)
    parser.add_argument("--no-class-coverage", action="store_true")
    parser.add_argument("--crop-size", type=int, default=16)
    parser.add_argument("--edge-policy", choices=("extend", "crop"), default="extend")
    parser.add_argument("--n-patches", type=int, default=None, help="Only for random_rectangles")
    arguments = parser.parse_args()
    report = run_demo(arguments.data_dir, arguments.output_dir, buffer_m=arguments.buffer_m,
                      split_seed=arguments.split_seed, model_seed=arguments.model_seed,
                      steps=arguments.steps, max_pairs_per_lag=arguments.max_pairs_per_lag,
                      include_size_matched_control=not arguments.skip_size_matched_control,
                      patch_layout=arguments.patch_layout, patch_size_pixels=arguments.patch_size_pixels,
                      block_size_pixels=arguments.block_size_pixels, split_ratios=arguments.split_ratios,
                      seed_candidates=arguments.seed_candidates, max_nan_fraction=arguments.max_nan_fraction,
                      require_class_coverage=not arguments.no_class_coverage, n_patches=arguments.n_patches, edge_policy=arguments.edge_policy, crop_size=arguments.crop_size)
    print(json.dumps({"output_dir": str(arguments.output_dir),
                      "retained_training_fraction": report["retained_training_labelled_fraction"],
                      "conditions": {name: {"train_patches": value["training_patch_count"],
                                             "test": value["test"]}
                                     for name, value in report["conditions"].items()}}, indent=2))


if __name__ == "__main__":
    main()
