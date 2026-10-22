"""Bounded, reproducible exploratory spatial splitting for a projected raster.

Coordinates and all distances must use metres. The I/O adapter must verify the
CRS and reject angular units, rotation, and nonsquare pixels before calling this
module. A measured spectral dependence range is not proof of ML independence.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

SPLIT_NAMES = {1: "train", 2: "validation", 3: "test"}
MAX_PIXELS = 16_000_000


def _grid(values, valid_mask, pixel_size_m):
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 3:
        raise ValueError("values must be a 2-D raster with at least 3 rows and columns")
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
    if np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1.0, atol=1e-8):
        raise ValueError("split_ratios must be positive and sum to 1")
    return ratios / ratios.sum()


def analyze_autocorrelation(
    values,
    valid_mask=None,
    *,
    pixel_size_m: float,
    seed: int = 42,
    max_lag_m: float | None = None,
    n_lags: int = 20,
    max_pairs_per_lag: int = 20_000,
    correlation_threshold: float = 0.1,
):
    """Return sampled directional Pearson correlograms and semivariograms.

    Each lag/direction samples at most ``max_pairs_per_lag`` unique source
    pixels, without replacement. Pair counts are sampled counts, not counts of
    independent observations. A range is observed only after three consecutive
    lag bins have |correlation| <= threshold in all four directions. The third
    bin's largest actual lag is the conservative proposed buffer. A range not
    observed is censored: the largest tested distance is a provisional buffer,
    never a measured decorrelation distance.
    """
    data, valid = _grid(values, valid_mask, pixel_size_m)
    if not isinstance(n_lags, (int, np.integer)) or not 3 <= n_lags <= 64:
        raise ValueError("n_lags must be an integer between 3 and 64")
    if not isinstance(max_pairs_per_lag, (int, np.integer)) or not 30 <= max_pairs_per_lag <= 100_000:
        raise ValueError("max_pairs_per_lag must be an integer between 30 and 100000")
    if not 0 < correlation_threshold < 1:
        raise ValueError("correlation_threshold must be between 0 and 1")
    if max_lag_m is None:
        max_lag_m = max(1, min(data.shape) // 4) * float(pixel_size_m)
    if not math.isfinite(float(max_lag_m)) or max_lag_m < pixel_size_m:
        raise ValueError("max_lag_m must be finite and at least one pixel")
    max_steps = min(int(max_lag_m / pixel_size_m), min(data.shape) - 1)
    steps = np.unique(np.rint(np.linspace(1, max_steps, min(n_lags, max_steps))).astype(int))
    flat_valid = np.flatnonzero(valid)
    signal = data.flat[flat_valid]
    variance = float(np.var(signal))
    if not math.isfinite(variance):
        raise ValueError("Signal variance is not finite; rescale extreme input values")
    result = {
        "estimator": "directional paired Pearson correlation and empirical semivariance",
        "range_rule": "three consecutive lag bins with absolute Pearson correlation <= threshold in all four directions",
        "correlation_threshold": float(correlation_threshold),
        "valid_pixel_count": int(flat_valid.size),
        "global_mean": float(np.mean(signal)),
        "global_variance": variance,
        "max_pairs_per_direction_per_lag": int(max_pairs_per_lag),
        "seed": int(seed),
        "lags": [],
        "range_status": "not_observed",
        "observed_range_m": None,
        "recommended_buffer_m": None,
        "warnings": [],
    }
    if variance <= np.finfo(float).eps * max(float(np.mean(signal * signal)), np.finfo(float).tiny):
        result["range_status"] = "constant"
        result["max_tested_lag_m"] = 0.0
        result["warnings"].append("Signal is constant or numerically constant; correlation and a data-derived buffer are undefined.")
        return result
    rng = np.random.default_rng(seed)
    directions = (("east_west", 0.0, 1.0), ("north_south", 1.0, 0.0),
                  ("northwest_southeast", 2**-0.5, 2**-0.5),
                  ("northeast_southwest", 2**-0.5, -(2**-0.5)))
    rows, cols = data.shape
    max_tested = 0.0
    passing = []
    for step in steps:
        lag = {"nominal_lag_m": float(step * pixel_size_m), "directions": {}}
        passes = True
        for name, dy, dx in directions:
            row_offset = int(round(step * dy))
            col_offset = int(round(step * dx))
            distance = float(math.hypot(row_offset, col_offset) * pixel_size_m)
            sample_size = min(flat_valid.size, max_pairs_per_lag * 3)
            sources = flat_valid[rng.choice(flat_valid.size, size=sample_size, replace=False)]
            rr, cc = np.divmod(sources, cols)
            target_rows, target_cols = rr + row_offset, cc + col_offset
            inside = (target_rows >= 0) & (target_rows < rows) & (target_cols >= 0) & (target_cols < cols)
            sources = sources[inside]
            targets = (target_rows[inside] * cols + target_cols[inside]).astype(np.int64)
            usable = valid.flat[targets]
            sources, targets = sources[usable][:max_pairs_per_lag], targets[usable][:max_pairs_per_lag]
            count = int(sources.size)
            correlation = None
            semivariance = None
            if count:
                first, second = data.flat[sources], data.flat[targets]
                semivariance = float(0.5 * np.mean((first - second) ** 2))
                first_zero, second_zero = first - first.mean(), second - second.mean()
                denominator = float(np.linalg.norm(first_zero) * np.linalg.norm(second_zero))
                if denominator > 0:
                    correlation = float(np.clip(np.dot(first_zero, second_zero) / denominator, -1, 1))
                max_tested = max(max_tested, distance)
            lag["directions"][name] = {
                "distance_m": distance,
                "offset_pixels": [row_offset, col_offset],
                "pair_count": count,
                "correlation": correlation,
                "semivariance": semivariance,
            }
            passes &= count >= 30 and correlation is not None and abs(correlation) <= correlation_threshold
        passing.append(passes)
        result["lags"].append(lag)
    result["max_tested_lag_m"] = max_tested
    for index in range(2, len(passing)):
        if all(passing[index - 2:index + 1]):
            proposed = max(d["distance_m"] for d in result["lags"][index]["directions"].values())
            result["range_status"] = "observed"
            result["observed_range_m"] = proposed
            result["recommended_buffer_m"] = proposed
            break
    if result["range_status"] == "not_observed":
        result["recommended_buffer_m"] = max_tested if max_tested > 0 else None
        result["warnings"].append("Decorrelation was not observed within tested lags. The proposed buffer is only the largest tested lag; expand the AOI/lag search and do not interpret it as sufficient separation.")
    result["warnings"].append("Spectral dependence in this scene is exploratory; labels, model features, season, and landscape can have longer ranges. No statistical independence guarantee is made.")
    return result


def _stripe_candidate(valid, ratios, order, axis, gap_pixels):
    """Allocate three stripes with complete discarded gaps and approximate ratios."""
    histogram = valid.sum(axis=1 - axis, dtype=np.int64)
    prefix = np.r_[0, np.cumsum(histogram)]
    length = histogram.size
    if length < 2 * gap_pixels + 3:
        return None
    ordered_ratios = ratios[np.asarray(order) - 1]
    lower, upper = 0.0, float(prefix[-1])
    candidates = []
    for _ in range(48):
        retained_target = (lower + upper) / 2
        end_first = int(np.searchsorted(prefix, retained_target * ordered_ratios[0], side="left"))
        start_second = min(length, end_first + gap_pixels)
        end_second = int(np.searchsorted(prefix, prefix[start_second] + retained_target * ordered_ratios[1], side="left"))
        start_third = min(length, end_second + gap_pixels)
        remaining = int(prefix[-1] - prefix[start_third])
        if remaining > retained_target * ordered_ratios[2]:
            lower = retained_target
        else:
            upper = retained_target
        if end_second > length:
            continue
        counts = np.array([prefix[min(end_first, length)],
                           prefix[min(end_second, length)] - prefix[start_second], remaining])
        if np.any(counts <= 0):
            continue
        fractions = counts / counts.sum()
        error = float(np.max(np.abs(fractions - ordered_ratios)))
        candidates.append((error, -int(counts.sum()), (end_first, start_second, end_second, start_third)))
    if not candidates:
        return None
    error, negative_retained, bounds = min(candidates)
    return {"axis": axis, "order": order, "bounds": bounds, "ratio_error": error,
            "retained": -negative_retained}


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
    split_ratios: Sequence[float] = (0.6, 0.2, 0.2),
    buffer_m: float | None = None,
    patch_radius_m: float = 0.0,
    seed: int = 42,
    max_lag_m: float | None = None,
    n_lags: int = 20,
    max_pairs_per_lag: int = 20_000,
):
    """Measure dependence, allocate contiguous stripes, and verify separation.

    Mask codes: 0 = invalid/discarded, 1 = train, 2 = validation, 3 = test.
    The random baseline shuffles exactly the retained labels over the same
    retained pixels, making its counts, imagery support, and retention equal.
    """
    data, valid = _grid(values, valid_mask, pixel_size_m)
    ratios = _ratios(split_ratios)
    if not math.isfinite(float(patch_radius_m)) or patch_radius_m < 0:
        raise ValueError("patch_radius_m must be finite and nonnegative")
    autocorrelation = analyze_autocorrelation(data, valid, pixel_size_m=pixel_size_m, seed=seed,
                                             max_lag_m=max_lag_m, n_lags=n_lags,
                                             max_pairs_per_lag=max_pairs_per_lag)
    if buffer_m is None:
        buffer_m = autocorrelation["recommended_buffer_m"]
        if buffer_m is None:
            raise ValueError("A data-derived buffer is unavailable for this signal; supply an explicit buffer_m")
        buffer_source = "measured_dependence" if autocorrelation["range_status"] == "observed" else "largest_tested_lag_provisional"
    else:
        buffer_source = "user_supplied"
    if not math.isfinite(float(buffer_m)) or buffer_m < 0:
        raise ValueError("buffer_m must be finite and nonnegative")
    required_center = (float(buffer_m) + math.sqrt(2) * pixel_size_m + 2 * patch_radius_m
                       if buffer_m > 0 or patch_radius_m > 0 else 0.0)
    gap_pixels = max(0, int(math.ceil(required_center / pixel_size_m - 1e-12)) - 1)
    rng = np.random.default_rng(seed)
    order = tuple(int(v) for v in rng.permutation((1, 2, 3)))
    candidates = [candidate for axis in (0, 1)
                  if (candidate := _stripe_candidate(valid, ratios, order, axis, gap_pixels)) is not None]
    if not candidates:
        raise ValueError("AOI is too small or too masked for three nonempty spatial splits at this buffer; enlarge the AOI or choose a smaller explicit buffer")
    # Prefer approximate requested ratios, then retain as much imagery as possible.
    acceptable = [c for c in candidates if c["ratio_error"] <= 0.03]
    chosen = max(acceptable, key=lambda c: c["retained"]) if acceptable else min(candidates, key=lambda c: c["ratio_error"])
    labels = np.zeros(data.shape, dtype=np.uint8)
    end_first, start_second, end_second, start_third = chosen["bounds"]
    coordinate = np.arange(data.shape[chosen["axis"]])
    stripe = np.zeros(coordinate.size, dtype=np.uint8)
    stripe[:end_first] = order[0]
    stripe[start_second:end_second] = order[1]
    stripe[start_third:] = order[2]
    labels[:] = stripe[:, None] if chosen["axis"] == 0 else stripe[None, :]
    labels[~valid] = 0
    stats = _split_stats(labels, int(valid.sum()))
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
    if chosen["ratio_error"] > 0.03:
        warnings.append("Coarse grid or nodata prevents split ratios within 3 percentage points; inspect reported counts.")
    if stats["retention_fraction"] < 0.5:
        warnings.append("More than half the valid imagery is discarded by the requested separation.")
    if buffer_m < (autocorrelation["recommended_buffer_m"] or 0):
        warnings.append("The supplied buffer is smaller than the exploratory data-derived recommendation.")
    warnings.append("Partitions are three contiguous geographic stripes, without class stratification. Spatial holdout can expose geographic distribution shift; repeat across AOIs/seasons before model claims.")
    report = {
        "method": "three contiguous geographic stripes with discarded separation gaps",
        "seed": int(seed), "pixel_size_m": float(pixel_size_m),
        "requested_split_ratios": dict(zip(SPLIT_NAMES.values(), ratios.tolist(), strict=True)),
        "requested_buffer_m": float(buffer_m),
        "recommended_buffer_m": autocorrelation["recommended_buffer_m"],
        "buffer_source": buffer_source, "patch_radius_m": float(patch_radius_m),
        "discarded_gap_width_pixels": gap_pixels,
        "stripe_axis": "rows" if chosen["axis"] == 0 else "columns",
        "stripe_order": [SPLIT_NAMES[number] for number in order],
        "mask_codes": {"0": "invalid_or_discarded", "1": "train", "2": "validation", "3": "test"},
        "autocorrelation": autocorrelation, "splits": stats,
        "spatial_diagnostics": diagnostics, "random_baseline": baseline, "warnings": warnings,
    }
    return {"split_mask": labels, "random_split_mask": random_labels, "report": report}
