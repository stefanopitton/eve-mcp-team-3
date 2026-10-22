"""Block-assigned rectangular patch grids adapted from the pinned LULC workflow.

The reference shuffles blocks, expands them into a dense patch grid, removes
patches symmetrically along unlike split boundaries, then ranks candidate
seeds by their smallest retained split. This module preserves those rules
while accepting rectangular rasters/patches, extending terminal blocks to the
AOI edges, and recording quality, ignored labels, and seed diagnostics.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

SPLITS = {1: "train", 2: "validation", 3: "test"}
MAX_PATCHES = 4096
PUBLIC_REFERENCE_URL = "https://github.com/stefanopitton/lulc-classification/tree/ac7dfc6578c2c408c87398e15073354752bbd32b"


def _integer(value, name, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not 1 <= value <= maximum):
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _pair(value, name, maximum):
    dimensions = (value, value) if isinstance(value, (int, np.integer)) else value
    if not isinstance(dimensions, (list, tuple)) or len(dimensions) != 2:
        raise ValueError(f"{name} must be a positive integer or a (height, width) pair")
    return tuple(_integer(item, name, bound)
                 for item, bound in zip(dimensions, maximum, strict=True))


def assign_blocks(seed, grid_shape, split_ratios=(0.7, 0.1, 0.2)):
    """Use upstream ceil(test), ceil(validation), remainder(train), then shuffle.

    Output codes adapt upstream's 0/1/2 to the MCP's 1/2/3 so that zero remains
    available for excluded pixels. RNG order is identical to the reference.
    """
    rows, columns = _pair(grid_shape, "grid_shape", (MAX_PATCHES, MAX_PATCHES))
    ratios = np.asarray(split_ratios, dtype=float)
    if (ratios.shape != (3,) or not np.all(np.isfinite(ratios)) or
            np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-8, rtol=0)):
        raise ValueError("split_ratios must contain three positive finite fractions summing to 1")
    count = rows * columns
    if count > MAX_PATCHES:
        raise ValueError(f"The reference block grid may contain at most {MAX_PATCHES} blocks")
    validation, test = math.ceil(count * ratios[1]), math.ceil(count * ratios[2])
    if validation + test >= count:
        raise ValueError("Too few blocks for three nonempty splits after ceil(validation/test); enlarge the AOI or use smaller block_size_pixels")
    labels = np.ones(count, dtype=np.uint8)
    labels[:test] = 3
    labels[test:test + validation] = 2
    np.random.default_rng(seed).shuffle(labels)
    return labels.reshape(rows, columns)


def create_buffer_mask(patch_labels, radius_patches, active_mask=None):
    """Buffer only interfaces between differently assigned active patches.

    The rectangular extension uses a separate radius for each patch dimension.
    With every patch active, this exactly matches upstream's symmetric square
    neighbourhood rule and own-split edge padding. Quality-ineligible patches
    and space outside the grid do not create interfaces. Inactive positions
    return True because quality exclusions are reported separately. Active
    patches are compared with the original active grid, without a cascade.
    """
    labels = np.asarray(patch_labels)
    if labels.ndim != 2 or 0 in labels.shape or labels.dtype.kind not in "ui":
        raise ValueError("patch_labels must be a nonempty 2-D integer grid")
    radii = ((radius_patches, radius_patches)
             if isinstance(radius_patches, (int, np.integer)) else radius_patches)
    if not isinstance(radii, (tuple, list)) or len(radii) != 2:
        raise ValueError("radius_patches must be a nonnegative integer or pair")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
           for value in radii):
        raise ValueError("radius_patches must be nonnegative integers")
    if any(value > bound for value, bound in zip(radii, labels.shape, strict=True)):
        raise ValueError("AOI is too small for this buffer radius; enlarge the AOI or reduce buffer_m")
    size = tuple(2 * int(value) + 1 for value in radii)
    if active_mask is None:
        return minimum_filter(labels, size=size, mode="nearest") == maximum_filter(labels, size=size, mode="nearest")
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != labels.shape:
        raise ValueError("active_mask must have the same shape as patch_labels")
    keep = np.ones(labels.shape, dtype=bool)
    for code in np.unique(labels[active]):
        foreign = active & (labels != code)
        if np.any(foreign):
            unlike_nearby = maximum_filter(foreign, size=size, mode="constant", cval=False)
            keep[active & (labels == code)] &= ~unlike_nearby[active & (labels == code)]
    return keep


def _patch_totals(array, patch_shape, grid_shape):
    rows, columns = grid_shape
    height, width = patch_shape
    values = np.asarray(array)
    padding = ((0, rows * height - values.shape[0]),
               (0, columns * width - values.shape[1]))
    if any(amount for axis in padding for amount in axis):
        values = np.pad(values, padding, mode="constant", constant_values=0)
    return values.reshape(rows, height, columns, width).sum(axis=(1, 3), dtype=np.int64)


def assign_compact_blocks(seed, grid_shape, split_ratios=(0.7, 0.1, 0.2)):
    """Grow two compact corner holdouts; keep the remaining groups for train."""
    counts = np.bincount(assign_blocks(seed, grid_shape, split_ratios).ravel(), minlength=4)
    rows, columns = grid_shape
    rng = np.random.default_rng(seed)
    corner = int(rng.integers(4))
    corners = ((0, 0), (0, columns - 1), (rows - 1, 0), (rows - 1, columns - 1))
    iy, ix = np.indices(grid_shape)
    labels = np.ones(grid_shape, dtype=np.uint8)
    for code, location in ((3, corners[corner]), (2, corners[3 - corner])):
        aspect = float(rng.uniform(0.8, 1.25))
        score = np.abs(iy - location[0]) * aspect + np.abs(ix - location[1]) / aspect
        score = score + rng.uniform(0, 0.25, size=grid_shape)
        score[labels != 1] = np.inf
        selected = np.argsort(score, axis=None, kind="stable")[:counts[code]]
        labels.ravel()[selected] = code
    return labels


def _nearby_patch_footprints(grid_shape, patch_shape, supported_shape, pixel_size_m, required_center):
    iy, ix = np.indices(grid_shape)
    y0, x0 = (iy * patch_shape[0]).ravel(), (ix * patch_shape[1]).ravel()
    y1 = np.minimum(y0 + patch_shape[0], supported_shape[0]) - 1
    x1 = np.minimum(x0 + patch_shape[1], supported_shape[1]) - 1
    dy = np.maximum(np.maximum(y0[:, None] - y1[None, :], y0[None, :] - y1[:, None]), 0)
    dx = np.maximum(np.maximum(x0[:, None] - x1[None, :], x0[None, :] - x1[:, None]), 0)
    return np.hypot(dy, dx) * pixel_size_m + 1e-7 < required_center


def _priority_keep(patch_grid, quality, nearby):
    labels, active = patch_grid.ravel(), quality.ravel()
    keep = np.ones(labels.shape, bool)
    higher = np.zeros(labels.shape, bool)
    for code in (3, 2, 1):
        current = np.flatnonzero(active & (labels == code))
        opponents = np.flatnonzero(higher)
        if current.size and opponents.size:
            keep[current] = ~np.any(nearby[np.ix_(current, opponents)], axis=1)
        higher |= active & keep & (labels == code)
    return keep.reshape(patch_grid.shape)


def plan_reference_blocks(
    valid_mask,
    *,
    pixel_size_m,
    buffer_m,
    split_ratios=(0.7, 0.1, 0.2),
    patch_size_pixels=32,
    block_size_pixels=None,
    seed_candidates=200,
    max_nan_fraction=0.5,
    class_labels=None,
    require_class_coverage=False,
    patch_radius_m=0.0,
    edge_policy="extend",
    group_assignment="compact_regions",
    buffer_policy="one_sided",
):
    """Return complete patch records plus transparent block/seed metadata.

    By default the last existing block in each axis extends to the full AOI;
    terminal patch rectangles are clipped to its true edge. No new independently
    randomized edge blocks are created. edge_policy='crop' reproduces the prior
    complete-block support. Class codes are categorical integers (zero is a
    valid class); negative codes are ignored. Quality and class counting use
    effective valid pixels and each patch's actual area before seed ranking.
    """
    valid = np.asarray(valid_mask, dtype=bool)
    if valid.ndim != 2 or not all(valid.shape):
        raise ValueError("valid_mask must be a nonempty 2-D raster")
    if not math.isfinite(float(pixel_size_m)) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be finite and positive")
    if not math.isfinite(float(buffer_m)) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative")
    if not math.isfinite(float(patch_radius_m)) or patch_radius_m < 0:
        raise ValueError("patch_radius_m must be finite and nonnegative")
    if not math.isfinite(float(max_nan_fraction)) or not 0 <= max_nan_fraction <= 1:
        raise ValueError("max_nan_fraction must be between 0 and 1")
    if not isinstance(require_class_coverage, (bool, np.bool_)):
        raise ValueError("require_class_coverage must be a boolean")
    if edge_policy not in ("extend", "crop"):
        raise ValueError("edge_policy must be extend or crop")
    if group_assignment not in ("compact_regions", "random_groups"):
        raise ValueError("group_assignment must be compact_regions or random_groups")
    if buffer_policy not in ("one_sided", "symmetric"):
        raise ValueError("buffer_policy must be one_sided or symmetric")
    seed_candidates = _integer(seed_candidates, "seed_candidates", 2000)
    patch_shape = _pair(patch_size_pixels, "patch_size_pixels", valid.shape)
    if block_size_pixels is None:
        block_size_pixels = tuple(3 * side for side in patch_shape)
    block_shape = _pair(block_size_pixels, "block_size_pixels", valid.shape)
    if any(block % patch for block, patch in zip(block_shape, patch_shape, strict=True)):
        raise ValueError("block_size_pixels must be an integer multiple of patch_size_pixels in each axis")
    block_grid = tuple(length // block for length, block in zip(valid.shape, block_shape, strict=True))
    complete_block_shape = tuple(count * side for count, side in zip(block_grid, block_shape, strict=True))
    supported_shape = valid.shape if edge_policy == "extend" else complete_block_shape
    grid_shape = tuple(math.ceil(length / patch) for length, patch in zip(supported_shape, patch_shape, strict=True))
    if math.prod(grid_shape) > MAX_PATCHES:
        raise ValueError(f"The reference patch grid exceeds {MAX_PATCHES} patches; increase patch_size_pixels or crop the AOI")
    per_block = tuple(block // patch for block, patch in zip(block_shape, patch_shape, strict=True))
    # This also checks block counts and fractions before seed search begins.
    assign_blocks(0, block_grid, split_ratios)
    reference_radii = tuple(math.ceil(buffer_m / (side * pixel_size_m)) for side in patch_shape)
    # Upstream's per-side dilation already gives a conservative total gap.
    # Extra patch-radius footprints sometimes require one additional row/col.
    required_center = buffer_m + math.sqrt(2) * pixel_size_m + 2 * patch_radius_m
    radii = tuple(max(reference, math.ceil(required_center / (2 * side * pixel_size_m)))
                  if buffer_m > 0 or patch_radius_m > 0 else 0
                  for reference, side in zip(reference_radii, patch_shape, strict=True))
    # Validate excessively large radii without allocating a huge filter kernel.
    create_buffer_mask(np.ones(grid_shape, dtype=np.uint8), radii)
    effective_valid = valid.copy()
    categorical = None
    if class_labels is not None:
        categorical = np.asarray(class_labels)
        if categorical.shape != valid.shape or categorical.dtype.kind not in "ui":
            raise ValueError("class_labels must be an aligned 2-D integer raster with negative ignore codes")
        effective_valid &= categorical >= 0
    elif require_class_coverage:
        raise ValueError("require_class_coverage needs an aligned class_labels raster")
    height, width = supported_shape
    supported_valid = effective_valid[:height, :width]
    valid_counts = _patch_totals(supported_valid, patch_shape, grid_shape)
    patch_areas = _patch_totals(np.ones(supported_shape, dtype=bool), patch_shape, grid_shape)
    quality = (valid_counts > 0) & ((patch_areas - valid_counts) / patch_areas <= max_nan_fraction)
    if not np.all(quality) and (buffer_m > 0 or patch_radius_m > 0):
        # An inactive run may already supply the entire inter-split gap, so
        # unlike active patches need not be eroded on both sides. In that case
        # preserve the core's conservative all-pixel footprint distance bound,
        # including the pixel diagonal and any additional patch radius.
        radii = tuple(max(radius, math.ceil((required_center - pixel_size_m) / (side * pixel_size_m)))
                      for radius, side in zip(radii, patch_shape, strict=True))
        create_buffer_mask(np.ones(grid_shape, dtype=np.uint8), radii)
    classes = ([] if categorical is None else
               np.unique(categorical[:height, :width][supported_valid]).astype(np.int64).tolist())
    if len(classes) > 32:
        raise ValueError("At most 32 nonnegative categorical classes are supported")
    if require_class_coverage and not classes:
        raise ValueError("No valid nonignored classes exist inside the supported AOI window")
    class_counts = (None if categorical is None else np.stack([
        _patch_totals((categorical[:height, :width] == value) & supported_valid, patch_shape, grid_shape)
        for value in classes], axis=-1)) if classes else None
    diagnostics, plans = [], []
    row_blocks = np.minimum(np.arange(grid_shape[0]) // per_block[0], block_grid[0] - 1)
    column_blocks = np.minimum(np.arange(grid_shape[1]) // per_block[1], block_grid[1] - 1)
    nearby = (_nearby_patch_footprints(grid_shape, patch_shape, supported_shape, pixel_size_m,
                                     required_center if buffer_m > 0 or patch_radius_m > 0 else 0)
              if buffer_policy == "one_sided" else None)
    for candidate_seed in range(seed_candidates):
        blocks = (assign_compact_blocks(candidate_seed, block_grid, split_ratios)
                  if group_assignment == "compact_regions" else assign_blocks(candidate_seed, block_grid, split_ratios))
        patch_grid = blocks[row_blocks[:, None], column_blocks[None, :]]
        buffer_keep = (_priority_keep(patch_grid, quality, nearby) if buffer_policy == "one_sided"
                       else create_buffer_mask(patch_grid, radii, active_mask=quality))
        keep = buffer_keep & quality
        counts = {name: int(np.count_nonzero(keep & (patch_grid == code)))
                  for code, name in SPLITS.items()}
        split_classes, missing = {}, {}
        for code, name in SPLITS.items():
            observed = (class_counts[keep & (patch_grid == code)].sum(axis=0)
                        if class_counts is not None else np.zeros(0, dtype=np.int64))
            split_classes[name] = observed.tolist()
            missing[name] = [value for value, count in zip(classes, observed, strict=True) if count == 0]
        coverage_complete = not any(missing.values()) if classes else None
        minimum = min(counts.values())
        eligible = minimum > 0 and (not require_class_coverage or coverage_complete is True)
        diagnostics.append({"seed": candidate_seed, "minimum_retained_patch_count": minimum,
                            "retained_quality_pixel_count": int(valid_counts[keep].sum()),
                            "retained_patch_counts": counts, "class_pixel_counts": split_classes,
                            "missing_classes": missing, "class_coverage_complete": coverage_complete,
                            "eligible": bool(eligible)})
        plans.append((blocks, patch_grid, buffer_keep, keep))
    # Python's stable sort preserves ascending seed order on equal scores.
    score_key = ("retained_quality_pixel_count" if group_assignment == "compact_regions"
                 else "minimum_retained_patch_count")
    ranked = sorted(range(seed_candidates), key=lambda index: diagnostics[index][score_key], reverse=True)
    selected = next((index for index in ranked if diagnostics[index]["eligible"]), None)
    if selected is None:
        reason = " with every present class in every split" if require_class_coverage else ""
        raise ValueError(f"No candidate seed in [0, {seed_candidates}) yields three nonempty quality-filtered reference block splits{reason}; enlarge the AOI, change block/patch sizes, or reduce an explicit buffer")
    blocks, patch_grid, buffer_keep, keep = plans[selected]
    paired_keep = quality & (keep | (patch_grid == 1))
    patches = []
    for row in range(grid_shape[0]):
        for column in range(grid_shape[1]):
            code, count = int(patch_grid[row, column]), int(valid_counts[row, column])
            retained, unbuffered = bool(keep[row, column]), bool(paired_keep[row, column])
            good = bool(quality[row, column])
            row_stop, col_stop = min((row + 1) * patch_shape[0], height), min((column + 1) * patch_shape[1], width)
            area = int(patch_areas[row, column])
            patches.append({"patch_id": row * grid_shape[1] + column,
                            "patch_row": row, "patch_col": column,
                            "block_row": int(row_blocks[row]), "block_col": int(column_blocks[column]),
                            "row_start": row * patch_shape[0], "row_stop": row_stop,
                            "col_start": column * patch_shape[1], "col_stop": col_stop,
                            "split": SPLITS[code], "split_code": code,
                            "pixel_area": area, "edge_clipped": area != math.prod(patch_shape),
                            "valid_pixel_count": count, "nan_fraction": 1 - count / area,
                            "quality_retained": good, "candidate_retained": good,
                            "buffer_excluded": bool(good and not buffer_keep[row, column]),
                            "retained": retained, "keep": retained,
                            "buffered_retained": retained, "unbuffered_retained": unbuffered,
                            "drop_reason": (None if retained else "quality_threshold" if not good
                                            else "one_sided_buffer_at_unlike_split_boundary" if buffer_policy == "one_sided"
                                            else "symmetric_buffer_at_unlike_split_boundary"),
                            "buffered_pixel_count": count if retained else 0,
                            "unbuffered_pixel_count": count if unbuffered else 0})
    block_count = math.prod(block_grid)
    requested = dict(zip(SPLITS.values(), map(float, split_ratios), strict=True))
    block_windows = []
    for row in range(block_grid[0]):
        for column in range(block_grid[1]):
            row_start, col_start = row * block_shape[0], column * block_shape[1]
            row_stop = height if row == block_grid[0] - 1 else (row + 1) * block_shape[0]
            col_stop = width if column == block_grid[1] - 1 else (column + 1) * block_shape[1]
            block_windows.append({"block_row": row, "block_col": column,
                                  "row_start": row_start, "row_stop": row_stop,
                                  "col_start": col_start, "col_stop": col_stop,
                                  "height": row_stop - row_start, "width": col_stop - col_start,
                                  "patch_rows": int(np.count_nonzero(row_blocks == row)),
                                  "patch_columns": int(np.count_nonzero(column_blocks == column)),
                                  "split_code": int(blocks[row, column]), "split": SPLITS[int(blocks[row, column])]})
    metadata = {
        "patch_layout": "reference_blocks", "selected_seed": int(selected),
        "edge_policy": edge_policy,
        "group_assignment": group_assignment, "buffer_policy": buffer_policy,
        "block_size_pixels": list(block_shape), "block_grid_shape": list(block_grid),
        "block_windows": block_windows,
        "patch_grid_shape": list(grid_shape), "patches_per_block": list(per_block),
        "supported_raster_shape": list(supported_shape),
        "complete_block_raster_shape": list(complete_block_shape),
        "clipped_edge_patch_count": sum(patch["edge_clipped"] for patch in patches),
        "terminal_patch_size_pixels": [height - (grid_shape[0] - 1) * patch_shape[0],
                                       width - (grid_shape[1] - 1) * patch_shape[1]],
        "patch_quality_area": "actual clipped patch footprint; array padding is used only for counting",
        "supported_window": {"row_start": 0, "row_stop": height, "col_start": 0, "col_stop": width},
        "edge_discarded_pixel_count": int(valid.size - height * width),
        "edge_discarded_valid_pixel_count": int(effective_valid.sum() - supported_valid.sum()),
        "quality_rejected_patch_count": int(np.count_nonzero(~quality)),
        "quality_rejected_valid_pixel_count": int(valid_counts[~quality].sum()),
        "quality_eligible_candidate_pixel_count": int(valid_counts[quality].sum()),
        "quality_eligible_candidate_patch_count": int(np.count_nonzero(quality)),
        "buffer_discarded_quality_candidate_pixel_count": int(valid_counts[quality & ~buffer_keep].sum()),
        "buffer_discarded_quality_candidate_patch_count": int(np.count_nonzero(quality & ~buffer_keep)),
        "max_nan_fraction": float(max_nan_fraction),
        "class_labels_provided": class_labels is not None,
        "classes_present_in_supported_grid": classes,
        "class_coverage_required": bool(require_class_coverage),
        "class_coverage_complete": diagnostics[selected]["class_coverage_complete"],
        "class_pixel_counts": diagnostics[selected]["class_pixel_counts"],
        "reference_buffer_radius_patches": list(reference_radii),
        "buffer_radius_patches": list(radii),
        "buffer_dilation_side": ("one-sided exact rectangle exclusion: test > validation > train" if buffer_policy == "one_sided"
                                 else "symmetric: only interfaces between distinct active train/validation/test splits discard patches from both sides"),
        "buffer_priority": ["test", "validation", "train"] if buffer_policy == "one_sided" else [],
        "buffer_geometry_method": ("exact complete-rectangle nearest pixel-center distance, including pixel-diagonal and extra patch-radius footprint allowance"
                                   if buffer_policy == "one_sided" else "symmetric Chebyshev patch-grid neighbourhood"),
        "stencil_radius_used": buffer_policy == "symmetric",
        "retention_budget": {"quality_candidate_pixel_count": int(valid_counts[quality].sum()),
                             "retained_quality_pixel_count": int(valid_counts[keep].sum()),
                             "buffer_removed_quality_pixel_count": int(valid_counts[quality & ~keep].sum()),
                             "retained_fraction_of_quality_candidates": float(valid_counts[keep].sum() / valid_counts[quality].sum())},
        "buffer_interface_policy": "Ignore AOI edges, quality-ineligible/nodata patches, and seams between blocks assigned to the same split",
        "buffer_scope": "only nearby quality-eligible patches assigned to different splits",
        "buffer_not_applied_to": ["same-split patch/block seams", "AOI exterior", "nodata or quality-rejected patches"],
        "buffer_radius_quantized_m": [radius * side * pixel_size_m for radius, side in zip(radii, patch_shape, strict=True)],
        "block_labels": blocks.tolist(), "patch_label_grid": patch_grid.tolist(),
        "buffer_keep_mask": buffer_keep.tolist(), "quality_keep_mask": quality.tolist(),
        "buffer_removed_mask": (quality & ~buffer_keep).tolist(),
        "keep_mask": keep.tolist(), "candidate_keep_mask": quality.tolist(),
        "requested_block_split_ratios": requested,
        "realized_block_split_ratios": {name: float(np.count_nonzero(blocks == code) / block_count)
                                       for code, name in SPLITS.items()},
        "realized_candidate_patch_split_ratios": {name: float(np.count_nonzero(quality & (patch_grid == code)) / max(1, np.count_nonzero(quality)))
                                                  for code, name in SPLITS.items()},
        "seed_search": {"start_inclusive": 0, "stop_exclusive": seed_candidates,
                        "selection": ("maximize total retained quality pixels, subject to nonempty splits and requested class coverage; stable ascending-seed ties"
                                      if group_assignment == "compact_regions" else
                                      "maximize smallest quality-filtered buffered split, stable ascending-seed ties, then require requested class coverage"),
                        "selected_seed": int(selected),
                        "score": score_key,
                        "eligible_candidate_count": sum(item["eligible"] for item in diagnostics),
                        "selected_retained_quality_pixel_count": diagnostics[selected]["retained_quality_pixel_count"],
                        "selected_minimum_retained_patch_count": diagnostics[selected]["minimum_retained_patch_count"],
                        "candidates": diagnostics},
        "reference_patch_procedure": {"source": "scripts/04_extract_patches.py and src/lulc/data/patches.py",
                                      "commit": PUBLIC_REFERENCE_URL.rsplit("/", 1)[-1],
                                      "adaptations": ["rectangular grids, patches and blocks",
                                                      ("extend last existing blocks to the AOI and clip terminal patches" if edge_policy == "extend"
                                                       else "explicit complete-block edge cropping"),
                                                      "quality and ignored-label filtering before seed selection",
                                                      "buffer interfaces only between differently assigned quality-eligible patches",
                                                      f"group_assignment={group_assignment}, buffer_policy={buffer_policy}",
                                                      "MCP split codes 1/2/3"]},
    }
    return {"patches": patches, "patch_size_pixels": patch_shape,
            "valid_mask": effective_valid, "metadata": metadata}
