# EO Spatial Split Planner — draft patch demo

An MCP server for EVE that measures spatial autocorrelation in continuous Earth Observation data and samples **random rectangular train/validation/test patches in both spatial axes**. It removes whole patches to create a measured guard, exports paired buffered/unbuffered masks, and reports retained support and actual separation. The notebook now trains a small semantic-segmentation CNN on real imagery to compare prediction behavior with and without excluding nearby training patches.

The implementation is aligned with the variogram subset of the reference repository's frozen checkpoint [`ac7dfc6`](https://github.com/stefanopitton/lulc-classification/tree/ac7dfc6578c2c408c87398e15073354752bbd32b): [scientific functions](https://github.com/stefanopitton/lulc-classification/blob/ac7dfc6578c2c408c87398e15073354752bbd32b/src/lulc/validation/autocorrelation.py), [configuration](https://github.com/stefanopitton/lulc-classification/blob/ac7dfc6578c2c408c87398e15073354752bbd32b/config/analysis.toml), and [worked example](https://github.com/stefanopitton/lulc-classification/blob/ac7dfc6578c2c408c87398e15073354752bbd32b/notebooks/autocorrelation.py). That checkpoint also contains additional estimators; `PROCEDURE.md` selects only its variogram and spherical model for this POC. Here `partial_sill` is the spherical model's second coefficient, and output `sill` means the total `nugget + partial_sill`.

## Random rectangles and paired masks

Seeded selection and row/column jitter place nonoverlapping rectangles in two dimensions. The primary demo requests 40 rectangles of 48 × 64 pixels (960 × 1,280 m at 20 m resolution), with seed 3 and initial split ratios 0.6/0.2/0.2. Patch placement and split assignment use the supplied finite-pixel mask without class stratification. The cache already applies imagery/reference-availability exclusions; vegetation/built-up/water identities and model scores do not select patch geometry.

The buffer applies to whole rectangle footprints. Test has priority, then validation, then training; a conflicting lower-priority rectangle is removed rather than cropped. The paired condition without a training buffer restores candidate training rectangles while retaining **identical validation and test patches**. In the recorded experiment, 24 training candidates become 8 buffered training patches, with 6 validation and 8 test patches fixed in both conditions. Two candidate validation patches are excluded to keep the held-out geometry compatible.

`patches.json` records coordinates, assigned split and keep flags. The masks use `0=excluded`, `1=train`, `2=validation`, `3=test`. Unselected valid AOI pixels, invalid pixels and buffer-discarded candidate patches are reported separately: the whole-scene exclusion count must not be described as pixels lost to the guard. Requested ratios describe the initial candidate assignment; retained ratios change after exclusions.

`buffer_autocorrelation` evaluates the actual guard separately from any query `lag_m`. A passing result at a larger query lag cannot endorse a smaller guard. Too few retained patches produces an actionable error. Exact retained-pixel center distances and conservative footprint-gap bounds are reported alongside rectangle coordinates.

## MCP tools

| Tool | Input | Result |
| --- | --- | --- |
| `evaluate_sentinel2_autocorrelation` | WGS84 bbox, continuous band, lag in metres; optional dates/scene | Variogram estimate and accept/increase-lag decision on real Sentinel-2 |
| `evaluate_spatial_autocorrelation` | Local numeric GeoTIFF, lag, 1-based band index | The same evaluation for an imagery-tool handoff |
| `plan_sentinel2_split` | Sentinel-2 search, `patch_size_pixels`, `n_patches`, optional guard | Random rectangular patches, paired masks and separation evidence |
| `plan_spatial_split` | Local raster, rectangle settings, optional guard | The same patch planning for an existing raster |

Planning accepts `patch_size_pixels=[48,64]` (or an integer for a square) and `n_patches=40`. Band aliases such as `red_edge1`/`nir` resolve to `B05`/`B08`. The evaluator reports `rho_at_lag`, `practical_range_m`, nugget, partial/total sill, fit RSS, variance, missing fraction and per-bin accepted counts. Complete reports, numeric imagery, buffered and paired unbuffered masks, and patch records are files on the MCP server. Another machine needs shared storage or artifact transfer to read them.

The optional segmentation experiment is a separate CPU runner; PyTorch is not required by the four MCP tools.

## Real 2021 segmentation comparison

The primary notebook uses Berlin `[13.30,52.35,13.50,52.50]`, with Sentinel-2 scene `S2B_MSIL2A_20210617T100559_R022_T33UUU_20210624T052802` acquired on 2021-06-17. Features are B05/B03/B04/B08 surface reflectance on B05's native 20 m grid (850 × 699 pixels, EPSG:32633). The 10 m bands are area-averaged to that grid. Per-pixel SCL classes 4/5/6 are retained, and processing-baseline calibration is recorded. Acquisition is through public Planetary Computer STAC.

ESA WorldCover 2021 v200 is nearest-neighbor aligned to the same grid, mapped to vegetation, built-up and water, with other labels ignored. Matching imagery and label years avoids using a 2021 reference for a 2024 scene. WorldCover is a **weak annual, machine-generated reference using Sentinel-1/Sentinel-2**, so the scores measure reference-map agreement rather than independent field accuracy. Annual labels can differ from land cover visible on the single acquisition date. Labels are distributed under CC BY 4.0; see the [WorldCover 2021 record](https://doi.org/10.5281/zenodo.7254221) and saved attribution.

The B05 variogram selects a **1,298.24 m** guard. The experiment trains a 5,283-parameter CNN from scratch: three 3 × 3 convolutional layers with 16 channels and ReLU, followed by a 1 × 1 class head. All conditions start from seed-42 weights and use AdamW (learning rate 0.003, weight decay 0.0001), 150 optimizer steps, batch size 8 and 32 × 32 training crops on CPU. Normalization is fitted on each condition's labelled training pixels only. Validation every 25 steps selects the highest-mIoU checkpoint; the common test support is scored once per selected model.

| Condition | Train patches | Labelled train pixels | Test mIoU | Test pixel accuracy | Minimum train–test rectangle-edge gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| Without training buffer | 24 | 70,237 | 0.5307 | 0.9016 | 100 m |
| With measured buffer | 8 | 23,845 | 0.4790 | 0.8519 | 1,700 m |
| Size-matched, without distance filter | 8 | 23,326 | 0.4785 | 0.8515 | 220 m |

All models use the same 6 validation/8 test patches and 23,240 labelled test pixels. The buffered condition retains 33.95% of the unbuffered labelled training pixels. The size-matched control samples 8 training rectangles using seed 43; it matches patch count, while pixel counts and class composition can differ. Water has only 229 test pixels, and every model has water IoU 0 in this run.

The buffer increases separation and reduces the training pool. The larger unbuffered training set scores higher; the size-matched control is close to the buffered result. **This does not isolate leakage as the cause of the score difference or establish that a buffer improves accuracy.** A lower buffered score is a valid observed result. Repeat across seeds, independent labels, regions and seasons before making model claims.

See the executed [notebook](../../EO_SPATIAL_SPLIT_PLANNER.ipynb), [data manifest](artifacts/segmentation_data/manifest.json), [patch plan](artifacts/segmentation_benchmark/benchmark_plan.json), and [trained results](artifacts/segmentation_benchmark/segmentation_report.json). Saved masks, predictions, checkpoints and figures make the comparison inspectable.

![Real imagery, weak labels, random rectangular patches, model predictions and validation curves](artifacts/segmentation_benchmark/segmentation_comparison.png)

## Variogram procedure

1. Read the selected band without normalizing it. Calculate population variance over finite pixels in float64; reject zero or non-finite variance.
2. Create 24 logarithmic bins between the native pixel size and 12,800 m, with geometric bin centers. For each source bin, draw 200,000 radii uniformly in that bin and angles uniformly over `[0, 2π)`, using `np.random.default_rng(3)`.
3. Round offsets to integer pixels, remove zero offsets, and recompute each pair's destination bin from its actual rounded distance. Draw anchors uniformly over the raster. Discard out-of-bounds partners and pairs touching non-finite values. Sampling is isotropic, and edge rejection is accepted.
4. Accumulate squared differences and counts by destination bin. Empty bins remain missing. Calculate `gamma = sum_squared_differences / (2 * counts)` and `rho = 1 - gamma / variance`.
5. Fit `gamma(h) = c0 + c1 * (1.5*t - 0.5*t**3)`, where `t = min(h/a, 1)`, using at least four populated bins. The nonnegative parameters are nugget `c0`, partial sill `c1`, and practical range `a`; total sill is `c0 + c1`. `scipy.optimize.curve_fit` uses `sigma = 1/sqrt(counts)` and the procedure's initial guess. Report fit RSS so the fit can be inspected.
6. Interpolate empirical `rho` linearly in `log(lag)` after dropping missing bins. Reject requests outside the populated bin-center range or with fewer than two populated bins.
7. Accept a requested separation only when **`rho_at_lag < 0.1` and `lag_m >= practical_range_m`**. Otherwise increase the lag and evaluate again. A failed or unresolved fit cannot establish acceptance.

Counts are accepted pairs after all exclusions, and can differ from 200,000 because rounded offsets move across source-bin boundaries. This POC uses only the variogram; it does not calculate directional Pearson curves, Moran's I, Geary's C, or an exponential fit.


## Assumptions

The AOI is one realization of a random field. Approximate stationarity and isotropy are assumed; uniform angles average directional structure away. Differencing removes dependence on an arbitrary constant mean, but **does not remove spatial trend**. Trend affects gamma and global variance, and can affect rho. Inspect the fit, accepted counts, cloud exclusions and directional structure before relying on the range. Negative rho values are retained.

The range and decision are continuous-band-specific point estimates without a significance test. Arbitrary categorical class codes have no meaningful numeric variance; use per-class indicator fields for a separate analysis. A B05-derived buffer does not prove independence in labels, other bands or repeated acquisitions. The model comparison also changes training volume, spatial support, normalization and class mix. The size-matched control reduces one confound but cannot establish a causal benefit.

## Can EVE use GIS MCP imagery?

**Yes, with a prepared numeric-raster handoff.** GIS MCP documents [`download_satellite_imagery`](https://gis-mcp.com/data-gathering/satellite_imagery/), returning a GeoTIFF `file_path` and scene metadata. A deployed GIS MCP server must be registered with EVE first; the documentation URL is not an MCP endpoint. The inspected account has no GIS MCP server enabled. **GIS acquisition was not invoked or integration-tested here**; actual pixels in the demo use this project's Planetary Computer reader. The source-audited [contract](artifacts/gis_mcp_contract.json) records this limitation and the requirements below.

Request B05 and SCL together on their common 20 m grid. Verify the returned scene ID, acquisition date, assets and AOI crop; search fallbacks can relax date/cloud constraints. Apply reflectance scale/offset from source metadata or processing baseline, plus the SCL per-pixel cloud/shadow/nodata mask. The audited GIS writer does not calibrate values or preserve scale/offset metadata. Retain a projected metre grid. Its stack writer cannot mix 10 m B03/B04/B08 with 20 m B05/SCL directly; obtain and align additional bands separately for the four-band model.

`file_path` is local to the GIS server. Share storage or transfer it using that server's documented `/storage/download` route before passing the prepared path to `evaluate_spatial_autocorrelation` and `plan_spatial_split`. Scene cloud percentage alone is not a pixel mask, and rendered previews are unsuitable scientific input.

## Reproduce

From the `agentic-eo` directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r examples/eo-spatial-split-planner/requirements.txt pytest
.venv/bin/python -m pip install -r examples/eo-spatial-split-planner/requirements-demo.txt
.venv/bin/python examples/eo-spatial-split-planner/segmentation_data.py
.venv/bin/python examples/eo-spatial-split-planner/run_segmentation.py --steps 150
.venv/bin/python -m pytest examples/eo-spatial-split-planner/tests -q
.venv/bin/python examples/eo-spatial-split-planner/test.py
```

Data acquisition reuses matching cached inputs unless `--force` is passed. The model runner retrains from those real rasters; it uses the full procedure defaults of 24 bins, 200,000 attempted pairs per source bin and split seed 3. The notebook also retrains by default with `TRAIN_MODELS=True`; `RECOMPUTE_IMAGES=True` refreshes imagery. The saved [reference-validation artifact](artifacts/reference_validation.json) records parity with the frozen variogram checkpoint.

Start the MCP server with:

```bash
.venv/bin/python examples/eo-spatial-split-planner/server.py --transport stdio
.venv/bin/python examples/eo-spatial-split-planner/server.py --transport http
```

HTTP uses port 8000 and `/mcp`. `EO_SPLIT_OUTPUT_DIR` selects persistent artifact storage; otherwise a server temporary directory is used. Local rasters must be square, north-up and projected in metres, with at most 32 million pixels. Remote Sentinel-2 acquisition is capped at one million pixels and a one-degree AOI span per axis.

Example `plan_spatial_split` arguments:

```json
{
  "raster_path": "/shared/sentinel2_reflectance_20m.tif",
  "band_index": 1,
  "seed": 3,
  "patch_size_pixels": [48, 64],
  "n_patches": 40
}
```

Omitting `buffer_m` measures a sufficient guard from the variogram. An explicit guard is reported with its own decision and can be rejected scientifically even if a geometric split is feasible. An imagery-discovery scene ID can be passed as `item_id` to a Sentinel-2 tool.

## EVE and draft submission

Gateway discovery and live EVE traces are saved under `artifacts/`. The recorded account did not expose a dedicated Sentinel-2 downloader; public STAC acquisition is explicit in the tool provenance. Inspect trace verification, schema/geometry version and `review_notes` before treating model prose as evidence. Older successful traces do not validate revised rectangle behavior. The live MCP imagery replay can use the separate 2024 B05 scene recorded in its provenance; the primary matched-year 2021 segmentation experiment is a distinct local CPU run.

Optional EVE testing uses repository helpers and credentials from `.env` without saving secrets:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python examples/eo-spatial-split-planner/test_eve.py --discover-only
.venv/bin/python examples/eo-spatial-split-planner/test_eve.py
```

The registry draft is staged under [`submission/servers/eo-spatial-split-planner/`](../../submission/servers/eo-spatial-split-planner/), with proposed wording in [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md). Refresh the staged copy and run `.venv/bin/python submission/validate_submission.py` after changes. The optional PyTorch demo is separate from the base server; Docker build and publishing status are separate from numerical, geospatial and MCP validation. This remains a draft demo.
