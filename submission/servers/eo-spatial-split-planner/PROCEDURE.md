# Spatial autocorrelation of a continuous variable — variogram procedure (POC)

How this project measures spatial autocorrelation for a continuous field, such
as the reflectance of a single optical band. This version is deliberately
minimal: **only the variogram is used.** Implementation:
`src/lulc/validation/autocorrelation.py`; configuration: `[autocorrelation]` in
`config/analysis.toml`; worked example: `notebooks/autocorrelation.py`.

## 1. Idea

The AOI is a **single realization** of a random field, so ensemble statistics
cannot be computed directly. We assume the field is approximately stationary and
estimate the **semivariogram** by Monte Carlo sampling of pixel pairs: for each
separation (lag) bin, sample many random pairs whose distance falls in that bin
and average half their squared difference.

The tool is evaluated at one lag of interest and answers: _are two pixels
separated by this distance still correlated?_

## 2. Why only the variogram

- Everything else is derived from it: the correlation-like number is
  `rho = 1 - gamma / variance`.
- It does not depend on the global mean, so it stays valid when the field has a
  large-scale trend across the AOI. Covariance-based estimates do not.
- Its fitted **practical range** gives the agent a natural stopping rule.
- Less code and less to explain. Moran's I, Geary's C, the covariance curve and
  the exponential model are left out of the POC.

## 3. Input

- `field`: 2-D `float32` array `[H, W]`, one band of one composite (here
  5120 × 5120 at 10 m). Used **as is, without normalization**: `rho` is
  scale-free, and normalizing would only add a dependence on the training split.
- `NaN` is allowed (no cloud-free observation). NaN pixels are never imputed;
  pairs touching a NaN are discarded.
- `resolution_m`: pixel size in metres (10).
- `lag_edges`: bin edges in metres (see §4).
- `n_pairs_per_lag`: pairs sampled per bin (200 000).
- A seeded `np.random.Generator` for reproducibility (`seed = 3`).

## 4. Lag bins

Bins are **logarithmically spaced** between `min_lag_m` (10 m, one pixel) and
`max_lag_m` (12 800 m) with `n_bins = 24`:

```python
lag_edges = np.logspace(log10(min_lag_m), log10(max_lag_m), n_bins + 1)
lag_centers = np.sqrt(lag_edges[:-1] * lag_edges[1:])  # geometric mean
```

Why log: correlation decays fast, so linear bins would spend their resolution on
the flat tail. Why cap at 12.8 km: beyond that the number of admissible pairs
collapses and estimates become unstable.

## 5. Procedure

### Step 1 — Field variance

Computed once over the whole field, ignoring NaN, in float64:

```
variance = nanvar(field)        # population variance
```

It must be finite and `> 0`, otherwise raise. It is only used to turn `gamma`
into `rho`; the variogram itself needs no mean.

### Step 2 — Sample pair offsets

For each bin `[lag_lo, lag_hi)`, draw `n_pairs_per_lag` offsets in polar
coordinates:

```
theta  ~ U(0, 2π)
radius ~ U(lag_lo, lag_hi)               # metres
dy = round(radius · sin θ / resolution_m)
dx = round(radius · cos θ / resolution_m)
```

The uniform angle makes the estimate **isotropic** (averaged over directions).
Offsets with `dy = dx = 0` are dropped.

Rounding to integer pixels can push the true distance
`hypot(dy, dx) · resolution_m` into a neighbouring bin, so the bin of each pair
is **recomputed from the rounded offset** (`np.digitize(lag, lag_edges) - 1`,
bins are `[lo, hi)`). Pairs falling outside `[0, n_bins)` are discarded, so a
pair is always counted in the bin matching its actual distance.

### Step 3 — Place the pairs

- Anchor: `(iy, ix)` uniform over the raster.
- Partner: `(iy + dy, ix + dx)`.
- Discard pairs whose partner falls outside the raster.
- Discard pairs where either value is non-finite.

Long-lag pairs near the borders are more likely to be discarded; this edge
effect is accepted.

### Step 4 — Accumulate per-bin sums

With `z_i`, `z_j` the anchor and partner values (float64), for each bin `k`:

```
counts[k]      = number of valid pairs
sq_diff_sum[k] = Σ (z_i - z_j)²
```

Keeping sums rather than averages lets batches be **added** and merged without
re-reading data.

### Step 5 — Semivariance and correlation

Bins with `counts == 0` stay `NaN`.

```
gamma(h) = sq_diff_sum / (2 · counts)
rho(h)   = 1 - gamma(h) / variance
```

Interpretation of `rho`:

- `≈ 1`: neighbours are nearly identical.
- `≈ 0`: uncorrelated; one pixel tells nothing about the other.
- `< 0`: pairs differ more than random (noise or periodic structure).

### Step 6 — Fit a spherical model

Fit to the valid `(lag_center, gamma)` points (at least 4 populated bins):

```
γ(h) = c0 + c1 · (1.5 · min(h/a, 1) − 0.5 · min(h/a, 1)³)
```

- `c0` — **nugget**: variance at zero separation (sensor noise, sub-pixel
  heterogeneity, geolocation error). A large nugget means the field is noisy at
  short range.
- `c1` — partial sill; total sill is `c0 + c1`.
- `a` — **practical range**: distance beyond which pairs are effectively
  uncorrelated.

Details: `scipy.optimize.curve_fit`, parameters bounded to `≥ 0`, initial guess
`(0, max(gamma), max(lag)/3)`, weights `sigma = 1/√counts` so well-sampled bins
pull harder. Also report the residual sum of squares as a fit-quality check.

### Step 7 — Read `rho` at the lag of interest

The lag of interest rarely coincides with a bin centre, so `rho` is **linearly
interpolated in `log(h)`** (`np.interp(log(lag), log(centers), rho)`) after
dropping NaN bins. Raise if the lag is outside the sampled range or fewer than 2
bins are populated.

In this project the lag of interest is the inter-split gap,
`patches.size · resolution` = 256 × 10 = 2560 m.

## 6. Tool contract

**Input:** `band`, `lag_m`.

**Output:**

| Field                | Use                                               |
| -------------------- | ------------------------------------------------- |
| `rho_at_lag`         | Primary decision value                            |
| `practical_range_m`  | Distance at which correlation vanishes (fit)      |
| `nugget`, `sill`     | Noise level and total variance explained by space |
| `fit_rss`            | Fit quality                                       |
| `counts` per bin     | Reliability of each point on the curve            |
| `variance`, NaN frac | Context for the field                             |

**Decision rule for the agent:** accept a lag when `rho_at_lag` is below a chosen
threshold (e.g. 0.1) **and** `lag_m ≥ practical_range_m`. Otherwise increase the
lag and repeat. The two conditions guard each other: `rho` is interpolated
directly from noisy bins, while the range comes from a smooth fit.

## 7. Caveats

- **Monte Carlo noise.** Results depend on the seed. With 200 000 pairs per bin
  the noise is small except where `counts` is low (long lags, heavy NaN). Check
  `counts` before trusting a bin.
- **Stationarity and isotropy are assumed.** Directional structure (field rows,
  rivers) is averaged away. A strong trend makes `variance` too large, which
  biases `rho` upwards; the variogram itself is less affected.
- **NaN handling is exclusion.** Heavily clouded seasons sample a biased subset
  of the AOI.
- **Point estimates only.** No significance test is attached to `rho`.
- **Continuous variables only.** For categorical labels the variance of class
  codes is meaningless; use per-class indicator fields instead.
- Estimates are per band and per season; spectral indices (NDVI, NDBI, NDWI) are
  treated like any other continuous band.

## 8. Left out of the POC

Covariance-based correlation, Geary's C as a separate output, Moran's I with
permutation p-values, the exponential model, and a cross-check against
`scikit-gstat`. All are natural extensions once the variogram loop works.
