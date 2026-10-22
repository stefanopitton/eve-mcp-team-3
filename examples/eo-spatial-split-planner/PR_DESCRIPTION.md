## Draft demo: EO Spatial Split Planner — full-AOI submission

This submission modifies `START_HERE.ipynb` into a runnable EO Spatial Split Planner demo, with the exact EVE system/user/follow-up prompts, raw agent responses, verification and corrections shown inline. A judging-evidence table maps usefulness (40%), agent readiness (30%), quality (20%) and demo (10%) to actual artifacts. Network replay uses the repository `.env` only when explicitly enabled; saved real-data/model evidence is the default submission path.

Four MCP tools evaluate spatial autocorrelation in local/Sentinel-2 imagery and plan patch datasets. The variogram follows `PROCEDURE.md`: 24 logarithmic bins, 200,000 draws per source bin, seed 3, rounded-distance binning, finite pairs, weighted spherical fit and log-lag interpolation. Lag acceptance requires rho below 0.1 and a lag covering the fitted practical range.

The patch workflow adapts checkpoint `scripts/04_extract_patches.py` with configurable **70%/10%/20%** split targets, 32 × 40-pixel model patches and nominal 96 × 120-pixel random assignment groups. **`edge_policy="extend"` covers the full AOI:** final groups extend to the image edge and terminal patches are clipped, eliminating bottom/right crop bands. The 2021 demo uses 40 groups and a 27 × 18 patch grid over all 850 × 699 pixels; the requested group allocation is 28 train/4 validation/8 test. Actual retained proportions and class-aware offline seed selection are reported.

Only different active train/validation/test splits trigger guards. Same-split seams, AOI edges, nodata and quality-rejected cells do not create buffers. The notebook distinguishes orange guard exclusions from grey nodata/quality exclusions, with patch/group lines. Extracted `.npy` arrays, metadata, train-only statistics and portable ZIP archives are inspected directly.

A small CPU CNN is trained in three conditions on actual matched-2021 Sentinel-2/WorldCover data: eligible training support, symmetric training exclusions and a size-matched control. Validation/test support is identical. Restoring training retains the held-out side of the symmetric guard, so the comparison is not presented as leaky versus independent. Test mIoU is 0.3527 for 306 training patches without training erosion, 0.3055 for 124 buffered patches and 0.3259 for the 124-patch size-matched control. All share 14 validation/10 test patches and 12,601 labelled test pixels, with only 11 water pixels. Refreshed reports show class counts and measured distances; no accuracy improvement is assumed. WorldCover is a weak annual classifier-derived reference.

The current live EVE replay validates hosted geocoding, evaluation of the cached actual 2021 Sentinel-2 raster and local patch planning; it does not perform a new imagery download or train the CNN. GIS MCP is a reviewed conditional raster handoff, not an integration invoked through this account. Actual imagery uses public Planetary Computer STAC. Pinned-source parity, tests, protocol checks and complete reviewed traces are included.

```bash
.venv/bin/python -m pip install -r examples/eo-spatial-split-planner/requirements-demo.txt
.venv/bin/python examples/eo-spatial-split-planner/run_segmentation.py --steps 150 --split-ratios 0.7 0.1 0.2 --edge-policy extend
.venv/bin/python -m pytest examples/eo-spatial-split-planner/tests -q
.venv/bin/python examples/eo-spatial-split-planner/test.py
.venv/bin/python submission/validate_submission.py
```

This is a draft demo for review. Approximate stationarity/isotropy, a selected split, weak reference labels and one acquisition limit the conclusions. Docker verification and registry publishing are separate steps.
