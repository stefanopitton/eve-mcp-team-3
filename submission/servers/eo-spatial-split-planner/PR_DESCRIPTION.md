## Draft demo: preserve START_HERE and retain scene pixels

This draft preserves every original `START_HERE.ipynb` cell outside “Your turn” and its notebook metadata. Only the three placeholder cells in that section are replaced with a concise MCP/EVE demo summary; the full workflow stays in `EO_SPATIAL_SPLIT_PLANNER.ipynb`. CI verifies the protected template cells against the original first commit.

The same real Sentinel-2 scene now retains **414,474 / 530,246 usable candidate pixels (78.2%, previously 31.0%)** with the unchanged **1,298.2 m** guard. Compact seeded regions and one-sided cross-split guards keep **486 small patches across 40 assignment groups** over the full scene, with zero edge trimming. Retained patches: **200 train, 35 validation, 127 test**. No buffers arise at the scene exterior or same-split seams.

Four MCP tools evaluate spatial autocorrelation in local/Sentinel-2 imagery and plan patch datasets. The variogram follows `PROCEDURE.md`: 24 logarithmic bins, 200,000 draws per source bin, seed 3, rounded-distance binning, finite pairs, weighted spherical fit and log-lag interpolation. Lag acceptance requires rho below 0.1 and a lag covering the fitted practical range.

The patch workflow adapts checkpoint `scripts/04_extract_patches.py` with configurable **70%/10%/20%** split targets, 32 × 40-pixel model patches and nominal 96 × 120-pixel random assignment groups. **`edge_policy="extend"` covers the full AOI:** final groups extend to the image edge and terminal patches are clipped, eliminating bottom/right crop bands. The 2021 demo uses 40 groups and a 27 × 18 patch grid over all 850 × 699 pixels; the requested group allocation is 28 train/4 validation/8 test. Actual retained proportions and class-aware offline seed selection are reported.

Only different active train/validation/test splits trigger guards. Same-split seams, AOI edges, nodata and quality-rejected cells do not create buffers. The notebook distinguishes orange guard exclusions from grey nodata/quality exclusions, with patch/group lines. Extracted `.npy` arrays, metadata, train-only statistics and portable ZIP archives are inspected directly.

The final retention settings cluster groups into seeded **compact split regions** and use a **one-sided** test > validation > train guard. Only conflicting lower-priority whole patches are removed, reducing interface length and exclusions while retaining the full small-patch AOI layout. This is an explicit extension of the checkpoint's independently shuffled groups and symmetric erosion; reference parity is not presented as validation of the new policy. Same-split seams, nodata/quality holes and outer edges never trigger a guard.

A small CPU CNN compares eligible training support, guarded training support and a size-matched control on fixed validation/test locations. Final reports record actual retention, mIoU, class counts and measured distances; no accuracy improvement or independence is assumed. WorldCover remains a weak classifier-derived reference. The detailed notebook and the short “Your turn” summary replay cached results and invoke the current local MCP planner. The three test mIoUs are 0.5252 (without buffer), 0.5203 (with buffer) and 0.5229 (size-matched control); this demonstrates separation rather than an assumed accuracy gain. The water class has only 4 validation and 42 test pixels.

A fresh live EVE run using the repository `.env` successfully called hosted geocoding, our variogram evaluator and the current compact/one-sided planner. Both notebooks display the actual agent/tool timeline via `show_trace`, including tool arguments, outputs and the final answer. START preserves the original “Running the agent” tutorial and adds the concise EO run inside “Your turn,” with optional `create_agent` replay code. Exact prompts, raw replies and reviewed corrections are included. The trace uses cached actual 2021 imagery; acquisition and CNN training are separate steps. GIS MCP remains a reviewed conditional handoff, not a service invoked in this run.

```bash
.venv/bin/python -m pip install -r requirements-demo.txt
.venv/bin/python run_segmentation.py --steps 150 --split-ratios 0.7 0.1 0.2 --edge-policy extend --group-assignment compact_regions --buffer-policy one_sided
.venv/bin/python -m pytest tests -q
.venv/bin/python test.py
.venv/bin/python submission/validate_submission.py
```

This is a draft demo for review. Approximate stationarity/isotropy, a selected split, weak reference labels and one acquisition limit the conclusions. Docker verification and registry publishing are separate steps.
