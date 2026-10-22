# EO Spatial Split Planner

A draft MCP server for EVE that turns a numeric Earth Observation raster into a reproducible train/validation/test partition informed by measured spatial dependence. It implements the [EO Spatial Split Planner concept note](https://docs.google.com/document/d/12IIlu0eEh_bVcY3pnii98iE5CrnEY80DatgcOOmFLEM/edit): expose the tradeoff between spatial separation and retained training data, with evidence an analyst can inspect.

The server reads actual native-resolution Sentinel-2 L2A pixels or a local GeoTIFF. It returns empirical directional dependence, a buffered stripe partition, a matched random split comparison, retention and separation checks, provenance, and artifact paths. The prototype uses contiguous stripes, rather than a checkerboard or a fitted variogram model.

## Tools

| Tool | Input | Result |
| --- | --- | --- |
| `plan_sentinel2_split` | WGS84 bbox, dates, band; optional scene ID, buffer, seed, ratios and patch radius | A real scene's measured dependence, spatial partition and evidence |
| `plan_spatial_split` | Path to a numeric, cloud-masked raster on the server; 1-based band index | The same analysis for imagery provided by another workflow or MCP tool |

Common options are `buffer_m=None` for a measured heuristic, `seed=42`, `split_ratios=[0.6, 0.2, 0.2]`, and `patch_radius_m=0`. A patch radius reserves additional space around sample centers. The buffer is a distance in metres; inspect the returned separation checks when using nonzero patch footprints.

The compact JSON response includes server-local paths for `split_mask.tif`, `analysis_band.tif`, and `report.json`. Mask labels are `0=nodata_or_buffer`, `1=train`, `2=validation`, and `3=test`. A client on another machine needs a separate artifact transfer mechanism to retrieve these files. Every successful run has a deterministic content/parameter fingerprint; paths are unique per invocation.

## Method and limits

1. Read one spectral band and its valid-pixel mask on a projected grid in metres.
2. Measure paired Pearson correlation and empirical semivariance in east/west, north/south, and both diagonal directions. Derive a heuristic range only after three consecutive lag bins have absolute correlation at most 0.1 in all four directions. Report unresolved estimates and sampled pair counts; paired pixels are not independent observations.
3. Construct reproducible contiguous train/validation/test stripes, excluding guard regions between partitions and allowing for sample-patch footprints. Preserve nodata exclusions.
4. Check the actual minimum cross-partition distances, retained pixel counts and achieved proportions. Compare with a seeded random shuffle on exactly the same retained pixels and split counts.
5. Export the masks and full diagnostics so that the planning decision is reviewable.

The requested split ratios are targets; buffer exclusions, nodata and stripe geometry affect achieved proportions. Increasing a buffer can leave too little area for all three splits. Such cases return an actionable error rather than a usable-looking empty split.

The default analysis samples up to 20,000 pairs per direction and lag over up to 20 lag bins, extending to approximately one quarter of the shorter image dimension. If no range is observed, the largest tested distance becomes an explicitly provisional buffer; it is not evidence of sufficient decorrelation. Constant imagery requires an explicit buffer.

The measured range is empirical and band-specific. It is not a guarantee of statistical independence. A single AOI can be nonstationary or anisotropic, and a range can remain unresolved over the available distances. Validate the target labels, other predictive bands, land-cover balance, overlapping scenes, repeated sites and acquisition times for the intended ML task. No prediction model is fitted, and the separation comparison does not measure downstream model accuracy. A temporal or region holdout may still be necessary.

For local input, the grid must be north-up, square, projected in metres, and contain at most 1 million pixels. Dataset masks and nodata are honored; the server does not invent a cloud mask for arbitrary local files. Raster previews and rendered PNGs are unsuitable input.

## Real Sentinel-2 acquisition

The default demo analyzes `[13.30,52.35,13.50,52.50]` around southern Berlin over `2024-06-01` to `2024-08-31`, using B05 at its native 20 m resolution. Scene search uses the public Microsoft Planetary Computer STAC collection `sentinel-2-l2a`. It selects the least cloudy covering scene among the first 100 returned items, breaking ties by item ID. Supply `item_id` to pin a scene for replay.

The server reads windowed numeric cloud-optimized GeoTIFF pixels and uses nearest-neighbor SCL alignment. It retains SCL classes 4 (vegetation), 5 (nonvegetated) and 6 (water), excludes all other classes and nodata, and applies the asset's reflectance scale/offset or the recorded Sentinel-2 processing-baseline convention. The report records the scene ID, acquisition date, CRS, native pixel spacing, mask, calibration, AOI and source URL. The AOI must be within Sentinel-2 coverage, span at most one degree in either direction, and read no more than 1 million native pixels.

A catalogue cloud percentage is a scene-level filter; it does not establish that every AOI pixel is clear. The retained SCL classes are an explicit masking choice, with their usual classification uncertainty. Network access is necessary for this path. No Planetary Computer account or API key is required; transient signed archive URLs are not persisted.

## Recorded real-data result

The saved experiment used scene `S2A_MSIL2A_20240828T102021_R065_T32UQD_20240828T150452`, acquired on 2024-08-28. The native B05 window has 876 × 731 pixels in EPSG:32632, with 562,947 valid pixels after masking.

| Measurement | Result |
| --- | ---: |
| Heuristic directional range | 1,923.33 m, observed by the stated rule |
| Retained valid pixels | 431,857 / 562,947 (76.71%) |
| Train / validation / test counts | 258,623 / 86,580 / 86,654 |
| Minimum spatial center separation | 1,960 m |
| Conservative minimum footprint gap | 1,931.72 m |
| Matched random minimum footprint gap | 0 m |

The 131,090 discarded valid pixels are guard-gap exclusions. Cloud/nodata exclusions are already absent from the 562,947 valid-pixel denominator. These results quantify one scene and partition; they do not measure model accuracy. See the [full report](artifacts/real_demo_report.json), [buffer sensitivity](artifacts/sensitivity.json), and diagnostic below.

![Actual B05 imagery, spatial and random partitions, directional dependence, semivariance, and buffer sensitivity](artifacts/diagnostic.png)

## Run and test

Run these commands from the `agentic-eo` directory. A Python virtual environment with the server dependencies is sufficient for the planner:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r examples/eo-spatial-split-planner/requirements.txt pytest
.venv/bin/python examples/eo-spatial-split-planner/test.py
.venv/bin/python -m pytest examples/eo-spatial-split-planner/tests -q
.venv/bin/python examples/eo-spatial-split-planner/test.py --real
```

Validation completed with 29 unit tests passing, the offline stdio MCP checks passing, and the real Sentinel-2 experiment passing. The offline checks exercise synthetic cases and geometry; `--real` acquires actual satellite pixels and writes `artifacts/real_demo_report.json`, `diagnostic.png`, `analysis_band.tif`, `split_mask.tif`, and `sensitivity.json`. The diagnostic contains measured dependence and the resulting partition. The sensitivity output exposes retention and separation for alternative buffer choices.

[`EO_SPATIAL_SPLIT_PLANNER.ipynb`](../../EO_SPATIAL_SPLIT_PLANNER.ipynb) displays the saved real-data report, figure, MCP schema and EVE evidence. Its default uses cached results; `RECOMPUTE=True` requests another real acquisition. [`START_HERE_no_instructions.ipynb`](../../START_HERE_no_instructions.ipynb) preserves all seven original code cells, removes the original notebook's Markdown cells, and clears saved outputs and execution counts. The original notebook is preserved.

Start the MCP server over stdio or streamable HTTP:

```bash
.venv/bin/python examples/eo-spatial-split-planner/server.py --transport stdio
.venv/bin/python examples/eo-spatial-split-planner/server.py --transport http
```

HTTP uses port 8000 and `/mcp`; artifact storage defaults to a server temporary directory. Set `EO_SPLIT_OUTPUT_DIR` to choose a persistent output directory.

Example tool arguments:

```json
{
  "bbox": [13.30, 52.35, 13.50, 52.50],
  "start_date": "2024-06-01",
  "end_date": "2024-08-31",
  "band": "B05",
  "max_cloud_cover": 10,
  "seed": 42
}
```

The local-raster handoff accepts, for example, `{"raster_path":"/shared/cloud_masked_b05.tif","band_index":1,"seed":42}`. That path must be visible to the planner's server process. A scene ID from an imagery discovery tool can instead be supplied as `item_id` to `plan_sentinel2_split`.

## EVE integration

The live account's gateway discovery is saved in `artifacts/eve_gateway_discovery.json`. At testing, the account exposed no dedicated Sentinel-2/CDSE imagery server, and both Sentinel-2 searches through the enabled `eodashboard.search_collections` tool returned no matching collections. This draft therefore acquires Sentinel-2 through public STAC within the planner and demonstrates the separate EVE model plus hosted geocoder plus local MCP planner chain. The acquisition route is explicit in the output provenance.

EVE testing uses the existing `agentic_eo` helpers and reads credentials from the repository `.env` without putting them in the notebook or evidence files. Install the repository dependencies for that optional path:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python examples/eo-spatial-split-planner/test_eve.py --discover-only
.venv/bin/python examples/eo-spatial-split-planner/test_eve.py
```

The live EVE run completed both the hosted geocoder call and local Sentinel-2 planner call successfully; the saved verification flags and full trace record this. The raw model answer is preserved in the trace. Its prose needs four corrections: discarded valid pixels are guard-gap exclusions; the 1,931.72 m footprint gap is a conservative lower bound while the 1,960 m center distance is exact; the very low cloud percentage is scene-level catalogue metadata; and verified geometry alone does not establish label independence or prevent every form of leakage.

The second command records the currently accessible catalogue. The final command asks the actual EVE model to call the hosted `reverse_geocode_place` tool and the local `plan_sentinel2_split` tool, and saves a sanitized answer/trace in `artifacts/eve_demo.json`. Use `--skip-discovery` to reuse catalogue evidence, or `--raster /path/to/analysis_band.tif` to exercise the local-raster tool through the same agent. A returned tool message may contain an error; inspect the saved trace and verification fields before claiming success.

## Registry submission

The self-contained draft entry is staged at [`submission/servers/eo-spatial-split-planner/`](../../submission/servers/eo-spatial-split-planner/) for the [EVE MCP tool registry](https://github.com/eve-esa/mcp-tool-registry). It includes the server, dependencies, tests and recorded evidence. The optional EVE script stays in this development repository because it imports the hackathon helpers. The proposed PR description is in [`PR_DESCRIPTION.md`](PR_DESCRIPTION.md).

From `agentic-eo`, validate the staged entry with:

```bash
.venv/bin/python submission/validate_submission.py
```

The validator writes [`submission/validation.json`](../../submission/validation.json). To prepare the registry PR, place only `submission/servers/eo-spatial-split-planner/` at `servers/eo-spatial-split-planner/` in a registry checkout. `submission/shared/Dockerfile` supplies the shared base for the staged build layout and is not an additional proposed registry change. The container build was not executed because a Docker daemon was unavailable. The draft has not been published or submitted.
