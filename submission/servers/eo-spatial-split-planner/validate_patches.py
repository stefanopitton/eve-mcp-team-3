"""Compare patch geometry against functions from the pinned upstream source.

Only five numerical functions are AST-extracted. No upstream package, config,
downloads or script entrypoint is imported or executed. Supply an existing
checkpoint checkout with --reference-root; this script never fetches code.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path

import block_patches
import numpy as np

HERE = Path(__file__).resolve().parent
REFERENCE_URL = "https://github.com/stefanopitton/lulc-classification/commit/ac7dfc6578c2c408c87398e15073354752bbd32b"
REFERENCE_COMMIT = REFERENCE_URL.rsplit("/", 1)[-1]
SOURCE_FILE = "src/lulc/data/patches.py"
FUNCTIONS = {"assign_blocks", "create_labelled_patches", "create_buffer_mask",
             "counts_classes_per_patch", "select_seed"}


def _load_reference(reference_root):
    root = Path(reference_root).expanduser().resolve(strict=True)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != REFERENCE_COMMIT:
        raise ValueError(f"Reference checkout must be pinned at {REFERENCE_COMMIT}; got {actual}")
    source = (root / SOURCE_FILE).read_bytes()
    committed = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{SOURCE_FILE}"])
    if source != committed:
        raise ValueError("Reference patch source differs from the committed checkpoint")
    tree = ast.parse(source, filename=SOURCE_FILE)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if {node.name for node in functions} != FUNCTIONS:
        raise ValueError("The pinned patch source does not contain all expected numerical functions")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"np": np, "ceil": math.ceil,
                 "LABELS_TO_SET": {"train": 0, "val": 1, "test": 2}}
    exec(compile(module, SOURCE_FILE, "exec"), namespace)
    return namespace, source, actual


def validate_patch_reference(reference_root):
    """Assert exact block assignments, expanded grids, buffers and seed choice."""
    reference, source, actual = _load_reference(reference_root)
    rows, columns = np.indices((240, 240))
    labels = ((rows + columns) % 3).astype("int16")
    fractions = {"train": 0.7, "val": 0.1, "test": 0.2}
    counts = reference["counts_classes_per_patch"](labels, 10, 3)
    expected_seed, expected_grid, expected_keep, expected_blocks = reference["select_seed"](
        counts, 4, 1, 6, 200, fractions)
    actual_plan = block_patches.plan_reference_blocks(
        np.ones(labels.shape, dtype=bool), pixel_size_m=10, buffer_m=1,
        split_ratios=(0.7, 0.1, 0.2), patch_size_pixels=10,
        block_size_pixels=60, seed_candidates=200, max_nan_fraction=0.5,
        class_labels=labels, require_class_coverage=True,
        group_assignment="random_groups", buffer_policy="symmetric")
    metadata = actual_plan["metadata"]
    assert metadata["selected_seed"] == expected_seed
    np.testing.assert_array_equal(np.asarray(metadata["block_labels"]), expected_blocks + 1)
    np.testing.assert_array_equal(np.asarray(metadata["patch_label_grid"]), expected_grid + 1)
    np.testing.assert_array_equal(np.asarray(metadata["keep_mask"]), expected_keep)
    np.testing.assert_array_equal(np.asarray(metadata["buffer_keep_mask"]), expected_keep)
    reference_class_counts = {}
    for name, code in (("train", 0), ("validation", 1), ("test", 2)):
        expected_counts = counts[expected_keep & (expected_grid == code)].sum(axis=0)
        reference_class_counts[name] = expected_counts.tolist()
        np.testing.assert_array_equal(metadata["class_pixel_counts"][name], expected_counts)
    assert metadata["class_coverage_complete"] is True
    retained = {name: int(np.count_nonzero(expected_keep & (expected_grid == code)))
                for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    comparisons = []
    for seed in (0, 3, 17):
        blocks = reference["assign_blocks"](seed, 4, fractions)
        expanded = reference["create_labelled_patches"](blocks, 6)
        local_blocks = block_patches.assign_blocks(seed, (4, 4), (0.7, 0.1, 0.2))
        local_grid = np.repeat(np.repeat(local_blocks, 6, axis=0), 6, axis=1)
        np.testing.assert_array_equal(local_blocks, blocks + 1)
        np.testing.assert_array_equal(local_grid, expanded + 1)
        for radius in (0, 1, 2):
            expected = reference["create_buffer_mask"](expanded, radius)
            observed = block_patches.create_buffer_mask(local_grid, radius)
            np.testing.assert_array_equal(observed, expected)
            comparisons.append({"seed": seed, "radius_patches": radius,
                                "exact_block_assignment": True, "exact_patch_grid": True,
                                "exact_symmetric_buffer_mask": True,
                                "retained_patch_count": int(expected.sum())})
    local_source = Path(block_patches.__file__).read_bytes()
    return {
        "status": "passed", "exact_parity": True,
        "repository": "https://github.com/stefanopitton/lulc-classification",
        "branch": "stepit/eve-hackathon-checkpoint", "commit": actual,
        "source_file": SOURCE_FILE, "source_sha256": hashlib.sha256(source).hexdigest(),
        "local_source_file": "block_patches.py", "local_source_sha256": hashlib.sha256(local_source).hexdigest(),
        "extraction": "AST extraction of five numerical functions only; upstream package and entrypoints not imported",
        "functions_compared": sorted(FUNCTIONS),
        "primary_fixture": {
            "raster_shape": [240, 240], "patch_size_pixels": [10, 10],
            "block_size_pixels": [60, 60], "block_grid_shape": [4, 4],
            "pixel_size_m": 10, "buffer_m": 1, "buffer_radius_patches": 1,
            "class_labels": "(row + column) % 3; all pixels valid; classes 0,1,2",
            "requested_fractions": {"train": 0.7, "validation": 0.1, "test": 0.2},
            "seed_candidates": 200, "selected_seed": int(expected_seed),
            "retained_patch_counts": retained, "retained_class_pixel_counts": reference_class_counts,
            "exact_matches": ["selected_seed", "block_labels (upstream + 1)",
                              "patch_label_grid (upstream + 1)", "symmetric_keep_mask",
                              "retained_class_pixel_counts"],
        },
        "additional_geometry_comparisons": comparisons,
        "explicit_adaptations": [
            "MCP codes 1/2/3 replace upstream train/val/test codes 0/1/2; zero remains excluded.",
            "Rectangular rasters and patches are supported; terminal blocks extend to the full AOI by default, terminal patches clip to actual bounds, and cropping is opt-in.",
            "Input quality and ignored-label filtering occur before seed ranking and class-coverage checks.",
            "Quality-inactive patches do not trigger buffers; the reference applies its label-grid buffer before quality filtering.",
            "Buffers exclude nearby unlike active splits only; same-split seams, AOI edges and missing-data boundaries are retained.",
            "Class coverage is optional and remains unverified when only a continuous raster is supplied.",
            "The paired comparison restores training patches and retains the buffered validation/test support.",
        ],
        "scope": "Exact parity on square all-valid fixtures; rectangular and quality-filter adaptations have separate local tests. No segmentation accuracy or satellite-data claim is made by this validator.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="Existing checkout pinned to the requested checkpoint commit")
    parser.add_argument("--output", type=Path, default=HERE / "artifacts" / "patch_reference_validation.json")
    arguments = parser.parse_args()
    result = validate_patch_reference(arguments.reference_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
