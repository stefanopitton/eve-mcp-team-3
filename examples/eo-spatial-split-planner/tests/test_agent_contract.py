"""Offline contracts for tool discovery, actionable failures and compact results."""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def test_agent_discovers_four_tools_and_full_aoi_planner_defaults():
    tools = {tool.name: tool for tool in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {"evaluate_spatial_autocorrelation", "evaluate_sentinel2_autocorrelation",
                          "plan_spatial_split", "plan_sentinel2_split"}
    for name, required in (("plan_spatial_split", "raster_path"), ("plan_sentinel2_split", "bbox")):
        tool = tools[name]
        assert required in tool.inputSchema["required"]
        properties = tool.inputSchema["properties"]
        assert properties["patch_layout"]["default"] == "reference_blocks"
        assert properties["edge_policy"]["default"] == "extend"
        assert properties["seed_candidates"]["default"] == 200
        assert "0.7" in tool.description and "0.1" in tool.description and "0.2" in tool.description
    for name in ("evaluate_spatial_autocorrelation", "evaluate_sentinel2_autocorrelation"):
        properties = tools[name].inputSchema["properties"]
        assert properties["seed"]["default"] == 3
        assert properties["n_bins"]["default"] == 24
        assert properties["n_pairs_per_lag"]["default"] == 200_000
        assert properties["lag_m"]["default"] == 2560
        assert tools[name].description


def test_invalid_sampling_returns_an_actionable_error_before_opening_a_raster(monkeypatch):
    def must_not_read(*_args, **_kwargs):
        raise AssertionError("invalid arguments must be rejected before raster I/O")

    monkeypatch.setattr(server, "read_local", must_not_read)
    result = server.evaluate_spatial_autocorrelation("missing.tif", seed=-1)
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_input_or_unresolved_analysis"
    assert "seed" in result["error"]["message"]
    assert "nonnegative" in result["error"]["message"]


def test_remote_failure_omits_signed_asset_url_details(monkeypatch):
    def retrieval_failure(*_args, **_kwargs):
        raise RuntimeError("transport failed: https://example.test/band.tif?sig=temporary-signed-query")

    monkeypatch.setattr(server, "read_sentinel2", retrieval_failure)
    result = server.evaluate_sentinel2_autocorrelation([13.3, 52.35, 13.5, 52.5])
    assert result["status"] == "error"
    assert result["error"]["code"] == "imagery_or_processing_failed"
    assert "temporary-signed-query" not in json.dumps(result)
    assert "https://" not in json.dumps(result)
    assert "local raster" in result["error"]["message"]


def test_agent_response_stays_compact_while_full_artifacts_preserve_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("EO_SPLIT_OUTPUT_DIR", str(tmp_path))
    values = np.arange(32 * 48, dtype="float32").reshape(32, 48)
    values[0, 0], values[0, 1] = -9999, np.nan
    transform = from_origin(400000, 5800000, 20, 20)
    profile = {"driver": "GTiff", "height": 32, "width": 48, "count": 1,
               "dtype": "float32", "crs": "EPSG:32633", "transform": transform}
    report = {"seed": 3, "patches": [{"patch_id": i} for i in range(512)],
              "patch_label_grid": [[1] * 16] * 24,
              "buffer_keep_mask": [[True] * 16] * 24,
              "seed_search": {"selected_seed": 0, "candidates": [{"seed": i} for i in range(200)]},
              "autocorrelation": {"counts": [100] * 24, "sq_diff_sum": [20.0] * 24}}
    mask = np.zeros(values.shape, dtype="uint8")
    mask[2:12, 2:14] = 1
    result = server._publish(values, profile, {"source": "offline_contract_fixture"}, report,
                             split_mask=mask, unbuffered_split_mask=mask, candidate_split_mask=mask)
    assert result["status"] == "ok"
    assert len(json.dumps(result)) < 16_000
    assert "patches" not in result and "patch_label_grid" not in result and "buffer_keep_mask" not in result
    assert "candidates" not in result["seed_search"]
    assert result["autocorrelation"]["counts"] == [100] * 24
    assert "sq_diff_sum" not in result["autocorrelation"]
    artifacts = result["artifacts"]
    full = json.loads(Path(artifacts["report_json"]).read_text())
    assert len(full["patches"]) == 512 and len(full["seed_search"]["candidates"]) == 200
    assert full["autocorrelation"]["sq_diff_sum"] == [20.0] * 24
    assert json.loads(Path(artifacts["patch_manifest"]).read_text()) == full["patches"]
    for key in ("analysis_band", "split_mask", "unbuffered_split_mask", "candidate_split_mask"):
        with rasterio.open(artifacts[key]) as source:
            assert source.crs.to_epsg() == 32633 and source.transform == transform
            if key == "analysis_band":
                np.testing.assert_array_equal(source.read(1, masked=True).filled(np.nan), values)
            else:
                np.testing.assert_array_equal(source.read(1), mask)
