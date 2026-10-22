## Draft demo: EO Spatial Split Planner with random patches

Nearby EO training and evaluation patches can share spatial information. This draft MCP server measures a continuous band's variogram and samples random rectangular patches in both spatial axes, then removes whole rectangles to satisfy a measured guard. It exports paired buffered/unbuffered masks, patch coordinates and keep flags, retention and cross-split distances. Validation and test locations remain fixed when nearby training patches are restored for the unbuffered condition.

Four MCP tools evaluate a local or Sentinel-2 raster at a requested lag and plan its patch split. Planning exposes `patch_size_pixels` and `n_patches`. The variogram follows `PROCEDURE.md` and frozen checkpoint `ac7dfc6`: 24 logarithmic bins, 200,000 offset draws per source bin, rounded-distance reassignment, finite-pair exclusion, seed 3, count-weighted spherical fit and log-lag interpolation. Acceptance requires `rho_at_lag < 0.1` and `lag_m >= practical_range_m`; the actual guard has a separate `buffer_autocorrelation` decision.

The executed notebook trains a 5,283-parameter CPU CNN on real 2021 Sentinel-2 B05/B03/B04/B08 reflectance with aligned ESA WorldCover 2021 weak labels. Forty 48 × 64-pixel rectangle candidates and a measured 1,298.24 m guard yield 24 training patches without the training filter versus 8 buffered, with identical 6 validation/8 test patches. All models start from the same weights and use 150 optimizer steps, training-only normalization and validation-selected checkpoints. Test evaluation occurs once per selected model.

| Condition | Train patches | Test mIoU | Minimum train–test patch-edge gap |
| --- | ---: | ---: | ---: |
| Without training buffer | 24 | 0.5307 | 100 m |
| With measured buffer | 8 | 0.4790 | 1,700 m |
| Size-matched, without distance filter | 8 | 0.4785 | 220 m |

The buffered condition retains 33.95% of labelled training pixels. It scores below the full unbuffered training pool, while the size-matched control is close. This demonstrates the separation/data-volume tradeoff; it does not attribute the score gap to leakage or show an accuracy benefit from buffering. Water is rare and all conditions score water IoU 0. WorldCover itself uses Sentinel imagery, so these scores measure agreement with a weak annual reference, not independent field accuracy.

The notebook includes before/after rectangular patch maps, imagery/reference/prediction views, per-class metrics, class counts, confusion matrices and validation curves. Optional PyTorch training remains separate from the four-tool MCP server. Reports, masks, prediction arrays, model checkpoints and attribution are included in the demo artifacts.

GIS MCP can provide a conditional numeric-raster handoff through its documented `download_satellite_imagery` tool, after EVE registers a deployed server and calibration, SCL masking, scene verification, grid alignment and shared storage/transfer are handled. The GIS source contract was reviewed; no GIS server was invoked or tested through this account. Actual demo imagery is acquired from public Planetary Computer STAC.

Reproduce from this server directory:

```bash
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python segmentation_data.py
.venv/bin/python run_segmentation.py --steps 150
.venv/bin/python -m pytest tests -q
.venv/bin/python test.py
```

Stationarity and isotropy remain assumptions; pair differencing is not trend removal. One AOI/acquisition/seed and weak labels do not establish true accuracy, independence in all bands, or temporal transferability. This is a draft demo for review; Docker verification and publishing are separate steps.
