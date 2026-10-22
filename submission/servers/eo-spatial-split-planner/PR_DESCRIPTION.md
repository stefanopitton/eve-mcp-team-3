## EO Spatial Split Planner

Nearby Earth Observation pixels can leak spatial information across random train/validation/test partitions. This MCP server measures directional dependence in a native-resolution Sentinel-2 band or a local numeric GeoTIFF, constructs buffered contiguous spatial partitions, and returns retention, distance checks, a matched random split comparison, provenance, and GeoTIFF/JSON artifacts.

The two tools are `plan_sentinel2_split` and `plan_spatial_split`. The Sentinel-2 path reads real Planetary Computer L2A COG windows, applies the recorded reflectance calibration and SCL mask, and records the exact scene and processing choices. The local-raster path supports handoff from an imagery tool sharing the server filesystem. The prototype uses stripes and empirical range estimates; it does not fit a prediction model or promise statistical independence of labels, other bands, or repeated acquisitions.

Validation completed: 29 unit tests passed, stdio and streamable HTTP MCP initialization/tool discovery/tool calls passed, the real Sentinel-2 experiment passed, and the EVE agent completed both hosted geocoding and the local planner call. The executed notebook and official registry structural validator also passed. Docker build remains unverified because the local daemon was unavailable.

The 2024-08-28 B05 demo retained 431,857 of 562,947 valid pixels (76.71%), with train/validation/test counts 258,623/86,580/86,654. Its observed heuristic range was 1,923.33 m; the conservative minimum footprint-gap lower bound was 1,931.72 m, compared with 0 m for a random shuffle of the same retained pixels and split counts. Guard exclusions account for the 131,090 discarded valid pixels.

Reproduction commands, run from the development repository's `agentic-eo` directory:

```bash
.venv/bin/python examples/eo-spatial-split-planner/test.py
.venv/bin/python -m pytest examples/eo-spatial-split-planner/tests -q
.venv/bin/python examples/eo-spatial-split-planner/test.py --real
.venv/bin/python examples/eo-spatial-split-planner/test_eve.py
```

The real-data evidence, diagnostic figure, buffer sensitivity results and sanitized EVE trace are included under `artifacts/`. The tested EVE account did not expose a Sentinel-2 imagery downloader, so imagery acquisition uses public STAC inside the planner; the live integration combines the actual EVE model, hosted geocoding and the local planner. Artifact paths returned by the tool refer to the MCP server filesystem.

### Checklist

- [x] Proposed changes are confined to one `servers/eo-spatial-split-planner/` folder.
- [x] `server.py` follows the FastMCP container and streamable HTTP entrypoint contract.
- [x] Every dependency has a version constraint.
- [x] No credentials, `.env` files, or signed archive tokens are included; no server secrets are required.
- [x] No file exceeds 5 MB.
- [x] Local numerical, geospatial, protocol and actual-imagery tests pass.
- [x] Official registry structural validation passes.
- [ ] Docker build succeeds (not run: local Docker daemon unavailable).
