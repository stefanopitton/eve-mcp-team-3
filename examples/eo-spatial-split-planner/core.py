"""Bounded, reproducible exploratory spatial splitting for a projected raster.

Coordinates and all distances must use metres. The I/O adapter must verify the
CRS and reject angular units, rotation, and nonsquare pixels before calling this
module. A measured spectral dependence range is not proof of ML independence.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import OptimizeWarning, curve_fit

SPLIT_NAMES = {1: "train", 2: "validation", 3: "test"}
MAX_PIXELS = 64_000_000


def _grid(values, valid_mask, pixel_size_m):
    data = np.asarray(values)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("values must be a 2-D raster with at least 3 rows and columns")
    if data.dtype.kind not in "buif":
        raise ValueError("values must contain real, numeric, continuous field values")
    if data.size > MAX_PIXELS:
        raise ValueError(f"Raster exceeds the bounded-compute limit of {MAX_PIXELS:,} pixels")
    if not math.isfinite(float(pixel_size_m)) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be a finite, positive distance in metres")
    valid = np.isfinite(data)
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError("valid_mask must have the same shape as values")
        valid &= mask
    if np.count_nonzero(valid) < 3:
        raise ValueError("At least three finite, valid pixels are required")
    return data, valid


def _ratios(split_ratios):
    ratios = np.asarray(split_ratios, dtype=float)
    if ratios.shape != (3,) or not np.all(np.isfinite(ratios)):
        raise ValueError("split_ratios must contain three finite values")
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("split_ratios must be positive and sum to 1")
    return ratios / ratios.sum()


def _positive_integer(value, name, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def _finite_or_none(values):
    """Keep missing bins JSON-safe while preserving their positions."""
    return [float(value) if np.isfinite(value) else None for value in values]


def estimate_variogram(
    field,
    resolution_m: float,
    lag_edges=None,
    n_pairs_per_lag: int = 200_000,
    seed: int = 3,
    *,
    min_lag_m: float | None = None,
    max_lag_m: float = 12_800.0,
    n_bins: int = 24,
    valid_mask=None,
    rng: np.random.Generator | None = None,
):
    """Estimate ONLY the isotropic Monte Carlo empirical variogram.

    Radii are uniform within each requested bin, angles are uniform on the
    circle, and anchors are uniform over the whole raster. Rounded offsets are
    reassigned to their actual-distance bins before accumulation. Missing and
    out-of-bounds pairs are excluded, never replaced or imputed. Sums and counts
    can be added across independent batches; divide only after merging.

    Random draws follow the referenced lulc pair_statistics implementation:
    sample all offsets, rebin/filter, then draw all row and column anchors.
    Population variance and squared differences use float64 on the raw field.
    Missing-bin gamma/rho are None for strict JSON serialization (numerical NaN).
    """
    data, valid = _grid(field, valid_mask, resolution_m)
    n_pairs_per_lag = _positive_integer(n_pairs_per_lag, "n_pairs_per_lag", 2_000_000)
    if rng is not None and not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    if lag_edges is None:
        n_bins = _positive_integer(n_bins, "n_bins", 128)
        minimum = float(resolution_m if min_lag_m is None else min_lag_m)
        maximum = float(max_lag_m)
        if not math.isfinite(minimum) or minimum <= 0:
            raise ValueError("min_lag_m must be finite and positive")
        if not math.isfinite(maximum) or maximum <= minimum:
            raise ValueError("max_lag_m must be finite and greater than min_lag_m")
        edges = np.logspace(np.log10(minimum), np.log10(maximum), n_bins + 1)
        edges[0], edges[-1] = minimum, maximum
    else:
        edges = np.asarray(lag_edges, dtype=np.float64)
        if (edges.ndim != 1 or not 2 <= edges.size <= 129 or
                not np.all(np.isfinite(edges)) or np.any(edges <= 0) or
                np.any(np.diff(edges) <= 0)):
            raise ValueError("lag_edges must contain 2 to 129 finite, positive, strictly increasing edges")
        n_bins = edges.size - 1
    if n_bins * n_pairs_per_lag > 8_000_000:
        raise ValueError("n_bins * n_pairs_per_lag must not exceed 8000000 attempted pairs")
    variance_field = (data if valid_mask is None and not np.any(np.isinf(data))
                      else np.where(valid, data, np.nan))
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.nanvar(variance_field, dtype=np.float64))
    if not math.isfinite(variance) or variance <= 0:
        raise ValueError("Field variance must be finite and > 0; constant or extreme fields cannot define rho")
    generator = np.random.default_rng(seed) if rng is None else rng
    rows, columns = data.shape
    dys, dxs = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        theta = generator.uniform(0.0, 2.0 * np.pi, size=n_pairs_per_lag)
        radius = generator.uniform(lo, hi, size=n_pairs_per_lag)
        dy = np.rint(radius * np.sin(theta) / resolution_m).astype(np.int64)
        dx = np.rint(radius * np.cos(theta) / resolution_m).astype(np.int64)
        nonzero = (dy != 0) | (dx != 0)
        dys.append(dy[nonzero])
        dxs.append(dx[nonzero])
    dy, dx = np.concatenate(dys), np.concatenate(dxs)
    del dys, dxs
    actual_lag = np.hypot(dy, dx) * resolution_m
    bins = np.digitize(actual_lag, edges) - 1
    keep = (bins >= 0) & (bins < n_bins)
    dy, dx, bins = dy[keep], dx[keep], bins[keep]
    del actual_lag, keep
    # Draw anchors over all pixels, not just finite/valid pixels. This accepts
    # the edge effect and missing-data exclusion described in PROCEDURE.md.
    iy = generator.integers(0, rows, size=dy.size)
    ix = generator.integers(0, columns, size=dy.size)
    jy, jx = iy + dy, ix + dx
    del dy, dx
    inside = (jy >= 0) & (jy < rows) & (jx >= 0) & (jx < columns)
    iy, ix, jy, jx, bins = (array[inside] for array in (iy, ix, jy, jx, bins))
    usable = valid[iy, ix] & valid[jy, jx]
    bins = bins[usable]
    first = data[iy[usable], ix[usable]].astype(np.float64)
    second = data[jy[usable], jx[usable]].astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_differences = np.square(first - second)
    if not np.all(np.isfinite(squared_differences)):
        raise ValueError("Squared pair differences overflow float64; rescale extreme input values")
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    sq_diff_sum = np.bincount(bins, weights=squared_differences, minlength=n_bins)
    if not np.all(np.isfinite(sq_diff_sum)):
        raise ValueError("Accumulated squared differences overflow float64; rescale extreme input values")
    centers = np.sqrt(edges[:-1] * edges[1:])
    gamma = np.full(n_bins, np.nan, dtype=np.float64)
    np.divide(sq_diff_sum, 2.0 * counts, out=gamma, where=counts > 0)
    rho = 1.0 - gamma / variance
    populated = counts > 0
    return {
        "estimator": "isotropic Monte Carlo empirical variogram",
        "lag_edges_m": edges.tolist(), "lag_centers_m": centers.tolist(),
        "counts": counts.tolist(), "sq_diff_sum": sq_diff_sum.tolist(),
        "gamma": _finite_or_none(gamma), "rho": _finite_or_none(rho),
        "variance": variance,
        "nan_fraction": float(1.0 - np.count_nonzero(valid) / data.size),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "n_pairs_per_lag": n_pairs_per_lag,
        "seed": int(seed) if rng is None else None,
        "sampled_lag_min_m": float(centers[populated][0]) if np.any(populated) else None,
        "sampled_lag_max_m": float(centers[populated][-1]) if np.any(populated) else None,
    }


def spherical_variogram(lag_m, nugget, partial_sill, practical_range_m):
    """Bounded spherical semivariogram; its exact sill is reached at a."""
    h = np.asarray(lag_m, dtype=np.float64)
    fraction = np.minimum(h / max(float(practical_range_m), np.finfo(float).tiny), 1.0)
    return nugget + partial_sill * (1.5 * fraction - 0.5 * fraction**3)


def fit_spherical_variogram(lag_centers_m, gamma, counts):
    """Fit nonnegative (c0, c1, a), weighted by sigma=1/sqrt(counts)."""
    centers = np.asarray(lag_centers_m, dtype=np.float64)
    semivariance = np.asarray(gamma, dtype=np.float64)
    weights = np.asarray(counts, dtype=np.float64)
    if centers.ndim != 1 or centers.shape != semivariance.shape or centers.shape != weights.shape:
        raise ValueError("lag centers, gamma, and counts must be matching 1-D arrays")
    populated = (np.isfinite(centers) & (centers > 0) & np.isfinite(semivariance) &
                 (semivariance >= 0) & np.isfinite(weights) & (weights > 0))
    if np.count_nonzero(populated) < 4:
        raise ValueError("A spherical fit requires at least 4 populated lag bins")
    h, observed, counts_valid = centers[populated], semivariance[populated], weights[populated]
    initial = (0.0, float(np.max(observed)), float(np.max(h) / 3.0))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", OptimizeWarning)
        try:
            parameters, covariance = curve_fit(
                spherical_variogram, h, observed,
                p0=initial, bounds=(0.0, np.inf), sigma=1.0 / np.sqrt(counts_valid),
                maxfev=10_000,
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            raise ValueError(f"Spherical fit is unresolved: {error}") from error
    if (any(issubclass(item.category, OptimizeWarning) for item in caught) or
            not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(covariance)) or
            parameters[2] <= 0):
        raise ValueError("Spherical fit is unresolved: finite identifiable parameters were not obtained")
    nugget, partial_sill, practical_range = (float(parameter) for parameter in parameters)
    rss = float(np.sum((observed - spherical_variogram(h, *parameters))**2, dtype=np.float64))
    if not math.isfinite(rss):
        raise ValueError("Spherical fit is unresolved: residuals are not finite")
    return {"nugget": nugget, "partial_sill": partial_sill,
            "sill": nugget + partial_sill, "practical_range_m": practical_range,
            "fit_rss": rss}


def interpolate_rho(lag_centers_m, rho, lag_m: float):
    """Read empirical rho by linear interpolation in log(h), without extrapolation."""
    centers = np.asarray(lag_centers_m, dtype=np.float64)
    correlations = np.asarray(rho, dtype=np.float64)
    if centers.ndim != 1 or centers.shape != correlations.shape:
        raise ValueError("lag centers and rho must be matching 1-D arrays")
    populated = np.isfinite(centers) & (centers > 0) & np.isfinite(correlations)
    centers, correlations = centers[populated], correlations[populated]
    if centers.size < 2:
        raise ValueError("Reading rho requires at least 2 populated lag bins")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("Populated lag centers must be strictly increasing")
    if not math.isfinite(float(lag_m)) or lag_m < centers[0] or lag_m > centers[-1]:
        raise ValueError(f"lag_m must lie within the sampled lag-center range [{centers[0]:g}, {centers[-1]:g}] m")
    return float(np.interp(np.log(lag_m), np.log(centers), correlations))


def _decision(rho_at_lag, lag_m, fit, threshold):
    rho_passes = bool(rho_at_lag is not None and rho_at_lag < threshold)
    range_passes = bool(fit["fit_resolved"] and lag_m >= fit["practical_range_m"])
    return {"accepted": rho_passes and range_passes,
            "rho_below_threshold": rho_passes,
            "lag_at_least_practical_range": range_passes,
            "correlation_threshold": float(threshold)}


def _summarize_variogram(estimate, *, correlation_threshold=0.1, lag_m=None, band=None):
    if not math.isfinite(float(correlation_threshold)) or not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold must be between 0 and 1")
    result = dict(estimate)
    result.update({"band": band, "fit_status": "unresolved", "fit_resolved": False,
                   "nugget": None, "partial_sill": None, "sill": None,
                   "practical_range_m": None, "fit_rss": None,
                   "recommended_buffer_m": None, "correlation_threshold": float(correlation_threshold),
                   "warnings": []})
    try:
        result.update(fit_spherical_variogram(result["lag_centers_m"], result["gamma"], result["counts"]))
        result.update({"fit_status": "resolved", "fit_resolved": True})
    except ValueError as error:
        result["warnings"].append(str(error))
    centers = np.asarray(result["lag_centers_m"], dtype=float)
    populated = np.asarray(result["counts"]) > 0
    if result["fit_resolved"] and np.count_nonzero(populated) >= 2:
        supported = centers[populated]
        practical_range = result["practical_range_m"]
        if practical_range > supported[-1]:
            result["warnings"].append("The fitted practical range exceeds sampled lag support; it is extrapolated and cannot certify a gap.")
        else:
            candidates = supported[supported >= practical_range].tolist()
            if supported[0] <= practical_range <= supported[-1]:
                candidates.append(practical_range)
            for candidate in sorted(candidates):
                rho = interpolate_rho(centers, result["rho"], candidate)
                if rho < correlation_threshold:
                    result["recommended_buffer_m"] = float(candidate)
                    break
    if result["recommended_buffer_m"] is None:
        result["warnings"].append("No lag within sampled support satisfies both rho < threshold and lag >= a resolved fitted practical range; increase the lag search or AOI.")
    if lag_m is not None:
        rho = interpolate_rho(centers, result["rho"], lag_m)
        result.update({"lag_m": float(lag_m), "rho_at_lag": rho,
                       "decision": _decision(rho, lag_m, result, correlation_threshold)})
    result["warnings"].append("Stationarity and isotropy are assumed; Monte Carlo point estimates do not establish model or label independence.")
    return result


def evaluate_autocorrelation(
    field,
    resolution_m: float,
    lag_m: float,
    *,
    band=None,
    lag_edges=None,
    n_pairs_per_lag: int = 200_000,
    seed: int = 3,
    min_lag_m: float | None = None,
    max_lag_m: float = 12_800.0,
    n_bins: int = 24,
    valid_mask=None,
    correlation_threshold: float = 0.1,
    rng: np.random.Generator | None = None,
):
    """Evaluate the POC tool contract for one continuous band and lag."""
    estimate = estimate_variogram(
        field, resolution_m, lag_edges, n_pairs_per_lag, seed,
        min_lag_m=min_lag_m, max_lag_m=max_lag_m, n_bins=n_bins,
        valid_mask=valid_mask, rng=rng,
    )
    return _summarize_variogram(estimate, correlation_threshold=correlation_threshold,
                               lag_m=lag_m, band=band)


def analyze_autocorrelation(
    values,
    valid_mask=None,
    *,
    pixel_size_m: float,
    seed: int = 3,
    max_lag_m: float | None = None,
    n_lags: int = 24,
    max_pairs_per_lag: int = 200_000,
    correlation_threshold: float = 0.1,
    lag_m: float | None = None,
    min_lag_m: float | None = None,
):
    """Compatibility adapter using ONLY PROCEDURE.md's variogram estimator.

    Historical argument names map to bin count and attempted pairs per bin.
    No directional Pearson statistic or provisional fallback range is used.
    """
    estimate = estimate_variogram(
        values, pixel_size_m, n_pairs_per_lag=max_pairs_per_lag, seed=seed,
        max_lag_m=12_800.0 if max_lag_m is None else max_lag_m,
        n_bins=n_lags, valid_mask=valid_mask, min_lag_m=min_lag_m,
    )
    return _summarize_variogram(estimate, correlation_threshold=correlation_threshold, lag_m=lag_m)


def _patch_dimensions(patch_size_pixels, shape):
    if patch_size_pixels is None:
        side = max(1, min(64, min(shape) // 12))
        return side, side
    dimensions = ((patch_size_pixels, patch_size_pixels)
                  if isinstance(patch_size_pixels, (int, np.integer))
                  else patch_size_pixels)
    if not isinstance(dimensions, (tuple, list)) or len(dimensions) != 2:
        raise ValueError("patch_size_pixels must be a positive integer or a (height, width) pair")
    return tuple(_positive_integer(value, "patch_size_pixels", bound)
                 for value, bound in zip(dimensions, shape, strict=True))


def _rectangular_patch_candidates(valid, ratios, patch_size_pixels, n_patches, rng):
    """Sample nonoverlapping rectangles in both axes without using class labels.

    Disjoint placement cells make overlap impossible. Each rectangle is jittered
    independently within its cell in row and column, then cells are sampled in
    seeded random order. Unselected cells remain outside the patch experiment.
    """
    height, width = _patch_dimensions(patch_size_pixels, valid.shape)
    stride_y, stride_x = height + height // 4, width + width // 4
    candidates = []
    for top in range(0, valid.shape[0] - height + 1, stride_y):
        for left in range(0, valid.shape[1] - width + 1, stride_x):
            row_slack = min(stride_y - height, valid.shape[0] - top - height)
            col_slack = min(stride_x - width, valid.shape[1] - left - width)
            row = top + int(rng.integers(row_slack + 1))
            col = left + int(rng.integers(col_slack + 1))
            count = int(np.count_nonzero(valid[row:row + height, col:col + width]))
            if count:
                candidates.append({"row_start": row, "row_stop": row + height,
                                   "col_start": col, "col_stop": col + width,
                                   "valid_pixel_count": count})
    if len(candidates) < 3:
        raise ValueError("AOI is too small or too masked for three rectangle patches; enlarge the AOI or choose smaller patch_size_pixels")
    requested = (min(40, max(3, len(candidates) // 4)) if n_patches is None
                 else _positive_integer(n_patches, "n_patches", 512))
    if requested < 3:
        raise ValueError("n_patches must be at least 3 to allocate train, validation, and test")
    if requested > len(candidates):
        raise ValueError(f"Only {len(candidates)} nonempty rectangle placements fit the AOI; decrease n_patches or patch_size_pixels")
    selected = [candidates[int(index)] for index in rng.permutation(len(candidates))[:requested]]
    # Largest-remainder allocation with at least one rectangle per split.
    allocation = np.ones(3, dtype=int)
    for _ in range(requested - 3):
        allocation[int(np.argmax(requested * ratios - allocation))] += 1
    split_codes = np.repeat(np.arange(1, 4), allocation)
    rng.shuffle(split_codes)
    for number, (patch, code) in enumerate(zip(selected, split_codes, strict=True)):
        patch.update({"patch_id": number, "split_code": int(code),
                      "split": SPLIT_NAMES[int(code)]})
    return selected, (height, width), len(candidates)


def _rectangle_center_distance(first, second, pixel_size_m):
    """Exact nearest pixel-center distance of complete axis-aligned rectangles."""
    dy = max(first["row_start"] - (second["row_stop"] - 1),
             second["row_start"] - (first["row_stop"] - 1), 0)
    dx = max(first["col_start"] - (second["col_stop"] - 1),
             second["col_start"] - (first["col_stop"] - 1), 0)
    return math.hypot(dy, dx) * pixel_size_m


def _patch_mask(shape, valid, patches, retention_key):
    labels = np.zeros(shape, dtype=np.uint8)
    for patch in patches:
        if patch[retention_key]:
            window = np.s_[patch["row_start"]:patch["row_stop"],
                           patch["col_start"]:patch["col_stop"]]
            labels[window] = np.where(valid[window], patch["split_code"], 0)
    return labels


def _split_stats(labels, valid_count):
    retained = int(np.count_nonzero(labels))
    return {
        "valid_pixel_count": int(valid_count),
        "retained_pixel_count": retained,
        "discarded_valid_pixel_count": int(valid_count - retained),
        "retention_fraction": retained / int(valid_count),
        "counts": {name: int(np.count_nonzero(labels == number)) for number, name in SPLIT_NAMES.items()},
        "fractions_of_retained": {name: int(np.count_nonzero(labels == number)) / retained if retained else 0.0
                                  for number, name in SPLIT_NAMES.items()},
    }


def separation_diagnostics(labels, *, pixel_size_m: float, patch_radius_m: float = 0.0):
    """Check exact all-pixel nearest grid-center distances using distance transforms.

    Subtracting the whole pixel diagonal and two patch radii yields a
    conservative lower bound on the gap between pixel/patch footprints.
    """
    pairs = []
    for first, second in ((1, 2), (1, 3), (2, 3)):
        first_mask, second_mask = labels == first, labels == second
        center = None
        footprint = None
        if np.any(first_mask) and np.any(second_mask):
            distance = distance_transform_edt(~first_mask, sampling=pixel_size_m)
            center = float(np.min(distance[second_mask]))
            footprint = max(0.0, center - math.sqrt(2) * pixel_size_m - 2 * patch_radius_m)
        pairs.append({
            "first_split": SPLIT_NAMES[first], "second_split": SPLIT_NAMES[second],
            "minimum_center_distance_m": center,
            "minimum_footprint_gap_lower_bound_m": footprint,
        })
    return {"method": "exact Euclidean distance transform over every retained pixel",
            "footprint_gap_method": "max(0, center distance - pixel diagonal - 2 * patch radius)",
            "pairwise": pairs,
            "minimum_center_distance_m": min((p["minimum_center_distance_m"] for p in pairs if p["minimum_center_distance_m"] is not None), default=None),
            "minimum_footprint_gap_lower_bound_m": min((p["minimum_footprint_gap_lower_bound_m"] for p in pairs if p["minimum_footprint_gap_lower_bound_m"] is not None), default=None)}


def create_spatial_split(
    values,
    valid_mask=None,
    *,
    pixel_size_m: float,
    split_ratios: Sequence[float] = (0.7, 0.1, 0.2),
    buffer_m: float | None = None,
    patch_radius_m: float = 0.0,
    seed: int = 3,
    max_lag_m: float | None = None,
    n_lags: int = 24,
    max_pairs_per_lag: int = 200_000,
    min_lag_m: float | None = None,
    lag_m: float | None = None,
    correlation_threshold: float = 0.1,
    patch_size_pixels: int | tuple[int, int] | None = None,
    n_patches: int | None = None,
    patch_layout: str = "random_rectangles",
    block_size_pixels: int | tuple[int, int] | None = None,
    seed_candidates: int = 200,
    max_nan_fraction: float = 0.5,
    class_labels=None,
    require_class_coverage: bool = False,
    edge_policy: str = "extend",
    group_assignment: str = "compact_regions",
    buffer_policy: str = "one_sided",
):
    """Sample random 2-D rectangle patches, buffer them, and verify separation.

    Mask codes: 0 = invalid/discarded, 1 = train, 2 = validation, 3 = test.
    The default random_rectangles layout samples sparse jittered rectangles
    with test > validation > train priority. The reference_blocks layout
    tiles dense patches inside compact seeded groups, retaining whole patches
    with an exact one-sided buffer. The upstream random_groups/symmetric
    settings remain available for reference parity.
    Terminal grid patches clip to the AOI edge when edge_policy='extend'; the
    last existing blocks absorb those edges. Missing pixels remain excluded.

    The paired unbuffered mask restores dropped TRAINING rectangles only;
    validation and test rectangles are identical in both masks, enabling a
    fair model comparison on exactly the same heldout pixels. Sparse rectangle
    geometry and initial assignments depend only on the supplied seed. Reference
    block mode selects a seed using geometric retention and optional categorical
    class coverage; no model score participates in selection. Its paired
    baseline shares the same heldout rectangles under the selected policy.
    The legacy random_split_mask is an additional pixel-shuffled diagnostic.
    """
    data, valid = _grid(values, valid_mask, pixel_size_m)
    ratios = _ratios(split_ratios)
    if patch_layout not in ("random_rectangles", "reference_blocks"):
        raise ValueError("patch_layout must be random_rectangles or reference_blocks")
    if patch_layout == "random_rectangles" and (class_labels is not None or require_class_coverage):
        raise ValueError("class_labels and require_class_coverage are supported by patch_layout='reference_blocks'")
    if not math.isfinite(float(patch_radius_m)) or patch_radius_m < 0:
        raise ValueError("patch_radius_m must be finite and nonnegative")
    autocorrelation = analyze_autocorrelation(data, valid, pixel_size_m=pixel_size_m, seed=seed,
                                             max_lag_m=max_lag_m, n_lags=n_lags,
                                             max_pairs_per_lag=max_pairs_per_lag,
                                             min_lag_m=min_lag_m, lag_m=lag_m,
                                             correlation_threshold=correlation_threshold)
    if buffer_m is None:
        buffer_m = autocorrelation["recommended_buffer_m"]
        if buffer_m is None:
            raise ValueError("No automatic buffer satisfies both empirical rho < threshold and lag >= a resolved spherical practical range within sampled support; increase the AOI/lag search or supply an explicit buffer_m")
        buffer_source = "variogram_range_and_empirical_rho"
    else:
        buffer_source = "user_supplied"
    if not math.isfinite(float(buffer_m)) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative")
    required_center = (float(buffer_m) + math.sqrt(2) * pixel_size_m + 2 * patch_radius_m
                       if buffer_m > 0 or patch_radius_m > 0 else 0.0)
    rng = np.random.default_rng(seed)
    layout_metadata = {"patch_layout": patch_layout}
    if patch_layout == "reference_blocks":
        if n_patches is not None:
            raise ValueError("n_patches must be None for reference_blocks; the complete block/patch grid defines the candidates")
        # Local import keeps the core's pinned variogram API independent of the
        # optional patch-grid adapter and avoids circular imports.
        from block_patches import plan_reference_blocks
        planned = plan_reference_blocks(
            valid, pixel_size_m=pixel_size_m, buffer_m=buffer_m,
            split_ratios=ratios, patch_size_pixels=32 if patch_size_pixels is None else patch_size_pixels,
            block_size_pixels=block_size_pixels, seed_candidates=seed_candidates,
            max_nan_fraction=max_nan_fraction, class_labels=class_labels,
            require_class_coverage=require_class_coverage, patch_radius_m=patch_radius_m,
            edge_policy=edge_policy,
            group_assignment=group_assignment, buffer_policy=buffer_policy,
        )
        patches, dimensions = planned["patches"], planned["patch_size_pixels"]
        valid, layout_metadata = planned["valid_mask"], planned["metadata"]
        placement_count = len(patches)
        rng = np.random.default_rng(layout_metadata["selected_seed"])
    else:
        patches, dimensions, placement_count = _rectangular_patch_candidates(
            valid, ratios, patch_size_pixels, n_patches, rng)
        retained_higher = []
        for code in (3, 2, 1):
            same_split_retained = []
            for patch in patches:
                if patch["split_code"] != code:
                    continue
                keep = all(_rectangle_center_distance(patch, other, pixel_size_m) + 1e-7 >= required_center
                           for other in retained_higher)
                patch.update({"retained": keep, "keep": keep,
                              "candidate_retained": True, "quality_retained": True,
                              "buffered_retained": keep,
                              "unbuffered_retained": bool(keep or code == 1),
                              "drop_reason": None if keep else "within_buffer_of_higher_priority_split",
                              "buffered_pixel_count": patch["valid_pixel_count"] if keep else 0,
                              "unbuffered_pixel_count": patch["valid_pixel_count"] if keep or code == 1 else 0})
                if keep:
                    same_split_retained.append(patch)
            retained_higher.extend(same_split_retained)
    if any(not any(p["retained"] and p["split_code"] == code for p in patches) for code in (1, 2, 3)):
        raise ValueError("AOI is too small or too masked for three nonempty rectangle splits at this buffer; enlarge the AOI, sample fewer heldout patches, choose smaller patch_size_pixels, or reduce an explicit buffer_m")
    labels = _patch_mask(data.shape, valid, patches, "buffered_retained")
    unbuffered_labels = _patch_mask(data.shape, valid, patches, "unbuffered_retained")
    candidate_labels = _patch_mask(data.shape, valid, patches, "candidate_retained")
    stats = _split_stats(labels, int(valid.sum()))
    candidate_count = sum(patch["valid_pixel_count"] for patch in patches)
    paired_count = int(np.count_nonzero(unbuffered_labels))
    stats.update({"candidate_pixel_count": candidate_count,
                  "unbuffered_retained_pixel_count": paired_count,
                  "buffer_discarded_candidate_pixel_count": candidate_count - stats["retained_pixel_count"],
                  "buffer_discarded_training_pixel_count": paired_count - stats["retained_pixel_count"],
                  "unselected_valid_pixel_count": int(valid.sum()) - candidate_count,
                  "retention_fraction_of_candidates": stats["retained_pixel_count"] / candidate_count})
    if patch_layout == "reference_blocks":
        stats.update({key: layout_metadata[key] for key in (
            "edge_discarded_pixel_count", "edge_discarded_valid_pixel_count",
            "quality_rejected_valid_pixel_count", "quality_eligible_candidate_pixel_count")})
        stats["buffer_discarded_candidate_pixel_count"] = layout_metadata["buffer_discarded_quality_candidate_pixel_count"]
        stats["retention_fraction_of_quality_candidates"] = stats["retained_pixel_count"] / layout_metadata["quality_eligible_candidate_pixel_count"]
    diagnostics = separation_diagnostics(labels, pixel_size_m=pixel_size_m, patch_radius_m=patch_radius_m)
    diagnostics["requested_buffer_met"] = all(p["minimum_footprint_gap_lower_bound_m"] is not None and
                                               p["minimum_footprint_gap_lower_bound_m"] + 1e-7 >= buffer_m
                                               for p in diagnostics["pairwise"])
    if not diagnostics["requested_buffer_met"]:
        raise RuntimeError("Internal separation validation failed")
    random_labels = np.zeros_like(labels)
    retained = labels > 0
    random_labels[retained] = rng.permutation(labels[retained])
    baseline = {"description": "seeded random shuffle of the same retained pixels and exact split counts",
                **_split_stats(random_labels, int(valid.sum())),
                "separation": separation_diagnostics(random_labels, pixel_size_m=pixel_size_m,
                                                      patch_radius_m=patch_radius_m)}
    warnings = list(autocorrelation["warnings"])
    try:
        buffer_rho = interpolate_rho(autocorrelation["lag_centers_m"], autocorrelation["rho"], buffer_m)
        buffer_within_support = True
    except ValueError:
        buffer_rho = None
        buffer_within_support = False
    buffer_autocorrelation = {
        "lag_m": float(buffer_m), "rho_at_lag": buffer_rho,
        "within_sampled_support": buffer_within_support,
        "decision": _decision(buffer_rho, buffer_m, autocorrelation, correlation_threshold),
    }
    if buffer_source == "user_supplied" and not buffer_autocorrelation["decision"]["accepted"]:
        warnings.append("The user-supplied buffer is not certified by both variogram criteria within sampled support; geometric separation alone does not imply decorrelation.")
    ratio_error = max(abs(stats["fractions_of_retained"][name] - requested)
                      for name, requested in zip(SPLIT_NAMES.values(), ratios, strict=True))
    if ratio_error > 0.03:
        warnings.append("Requested ratios define block assignment fractions; ceil rounding, extended edge blocks, quality filtering, and buffers change patch/pixel fractions."
                        if patch_layout == "reference_blocks" else
                        "Whole-rectangle buffer exclusions and nodata change the retained split ratios; requested ratios apply to the initial candidate patch assignment.")
    if stats.get("retention_fraction_of_quality_candidates", stats["retention_fraction_of_candidates"]) < 0.5:
        warnings.append("More than half the quality-eligible candidate pixels are excluded by the requested separation; unselected imagery and quality exclusions are reported separately.")
    if buffer_m < (autocorrelation["recommended_buffer_m"] or 0):
        warnings.append("The supplied buffer is smaller than the exploratory data-derived recommendation.")
    if patch_layout == "reference_blocks":
        if buffer_policy == "one_sided":
            warnings.append("Whole patches are removed only when their complete footprint is too close to a retained higher-priority split (test, then validation, then train). Compact group selection maximizes retained quality pixels subject to split/class coverage; no model score is used.")
        else:
            warnings.append("Whole patches are removed symmetrically only near a different active split; quantization and both-sided removal can produce a gap larger than requested.")
            warnings.append("The paired restored-training baseline still inherits the heldout side of the symmetric buffer; inspect its actual separation rather than assuming it fails the requested guard.")
        warnings.append("AOI edges, nodata/quality exclusions, and seams within the same split do not create buffers.")
        if class_labels is not None:
            warnings.append("Optional class labels affect patch quality and seed coverage only; the continuous-field variogram is computed independently before label filtering.")
    else:
        warnings.append("Rectangles are sampled randomly in both spatial directions without class stratification. Repeat across seeds, AOIs, and seasons before model claims; buffer exclusions reduce training support.")
    report = {
        "method": "seeded random two-dimensional rectangle patches with whole-patch buffer exclusion",
        "seed": int(seed), "pixel_size_m": float(pixel_size_m),
        "requested_split_ratios": dict(zip(SPLIT_NAMES.values(), ratios.tolist(), strict=True)),
        "requested_buffer_m": float(buffer_m),
        "recommended_buffer_m": autocorrelation["recommended_buffer_m"],
        "buffer_source": buffer_source, "patch_radius_m": float(patch_radius_m),
        "patch_size_pixels": list(dimensions), "n_candidate_patches": len(patches),
        "n_available_patch_placements": placement_count,
        "patch_counts": {name: {"candidate": sum(p["split_code"] == code for p in patches),
                                  "quality_candidate": sum(p["split_code"] == code and p["candidate_retained"] for p in patches),
                                  "buffered": sum(p["split_code"] == code and p["retained"] for p in patches),
                                  "unbuffered": sum(p["split_code"] == code and p["unbuffered_retained"] for p in patches)}
                         for code, name in SPLIT_NAMES.items()},
        "buffer_priority": ["test", "validation", "train"],
        "paired_comparison": "Unbuffered restores candidate training rectangles; both masks share identical retained validation and test rectangles.",
        "patches": patches,
        "mask_codes": {"0": "invalid_or_discarded", "1": "train", "2": "validation", "3": "test"},
        "autocorrelation": autocorrelation, "buffer_autocorrelation": buffer_autocorrelation,
        "splits": stats,
        "spatial_diagnostics": diagnostics, "random_baseline": baseline,
        "unbuffered_baseline": {**_split_stats(unbuffered_labels, int(valid.sum())),
                                "separation": separation_diagnostics(unbuffered_labels, pixel_size_m=pixel_size_m,
                                                                      patch_radius_m=patch_radius_m)},
        "warnings": warnings,
    }
    report.update(layout_metadata)
    report["realized_buffered_patch_split_ratios"] = {
        name: report["patch_counts"][name]["buffered"] / sum(p["retained"] for p in patches)
        for name in SPLIT_NAMES.values()}
    report["realized_paired_unbuffered_patch_split_ratios"] = {
        name: report["patch_counts"][name]["unbuffered"] / sum(p["unbuffered_retained"] for p in patches)
        for name in SPLIT_NAMES.values()}
    if patch_layout == "reference_blocks":
        report["method"] = ("compact block-assigned rectangular patch grid with one-sided exact-footprint buffers"
                            if buffer_policy == "one_sided" else
                            "reference block-assigned rectangular patch grid with symmetric whole-patch buffers")
        report["seed"] = layout_metadata["selected_seed"]
        report["autocorrelation_seed"] = int(seed)
        report["paired_comparison"] = ("Restore quality-eligible training patches only; share identical retained validation/test patches. Compare the measured training-to-heldout gap under both conditions."
                                       if buffer_policy == "one_sided" else
                                       "Restore quality-eligible training patches only; share identical symmetrically buffered validation/test patches. The baseline retains the heldout side of the buffer and may already meet the requested gap.")
        report["candidate_grid"] = _split_stats(candidate_labels, int(valid.sum()))
    return {"split_mask": labels, "unbuffered_split_mask": unbuffered_labels,
            "candidate_split_mask": candidate_labels,
            "random_split_mask": random_labels, "patches": patches, "report": report}
