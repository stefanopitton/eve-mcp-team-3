"""Optional live EVE integration: catalogue discovery, hosted geocoding, local MCP.

Run from this directory with the hackathon environment, for example:
    ../../.venv/bin/python test_eve.py --discover-only
    ../../.venv/bin/python test_eve.py

Reads the existing repository .env. Never persists the API key or signed URL
query strings. This is a network integration demonstration, not an offline test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "lib"))

# The user requires live tests to use this project's .env, even if the shell
# contains a different EVE configuration. Never print or persist these values.
_eve_env = dotenv_values(REPO_ROOT / ".env")
for _name in ("EVE_API_BASE_URL", "EVE_API_KEY"):
    if not _eve_env.get(_name):
        raise RuntimeError(f"The repository .env must contain {_name} for live EVE tests")
for _name in ("EVE_API_BASE_URL", "EVE_API_KEY", "EVE_LLM_MODEL"):
    if _eve_env.get(_name):
        os.environ[_name] = _eve_env[_name]

from agentic_eo import create_agent  # noqa: E402
from agentic_eo.env import settings  # noqa: E402
from agentic_eo.eve import list_eve_mcp_servers  # noqa: E402
from agentic_eo.mcp_tools import eve_mcp_client  # noqa: E402


def sanitize(value):
    """Remove credentials and URL query strings from saved or printed evidence."""
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if not isinstance(value, str):
        return value
    if settings.eve_api_key:
        value = value.replace(settings.eve_api_key, "<redacted>")

    def strip_query(match):
        parts = urlsplit(match.group(0))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))

    return re.sub(r"https?://[^\s\"'<>]+", strip_query, value)


def write_evidence(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), indent=2) + "\n")


def decoded_tool_output(content):
    """Handle text JSON and the MCP adapter's text-block representation."""
    if isinstance(content, str):
        try:
            return decoded_tool_output(json.loads(content))
        except (json.JSONDecodeError, TypeError):
            return content
    if isinstance(content, list):
        texts = [block.get("text", "") for block in content if isinstance(block, dict)]
        return decoded_tool_output("".join(texts)) if texts else content
    if isinstance(content, dict) and "status" not in content:
        if "structuredContent" in content:
            return decoded_tool_output(content["structuredContent"])
        if "content" in content:
            return decoded_tool_output(content["content"])
        if set(content) == {"result"}:
            return decoded_tool_output(content["result"])
    return content


def preserve_planner_artifacts(planner: dict, output_dir: Path,
                               report_filename: str = "eve_planner_report.json") -> dict:
    """Copy files from the local stdio server's scratch directory into the project."""
    original = planner.get("artifacts", {})
    report_path = Path(original.get("report_json", ""))
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text())
    directory = output_dir / "eve_runs" / report["run_fingerprint"]
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {"scope": "Preserved workspace copies from the local MCP server"}
    for key in ("analysis_band", "split_mask", "unbuffered_split_mask", "candidate_split_mask", "patch_manifest"):
        if key not in original:
            continue
        source = Path(original[key])
        destination = directory / source.name
        shutil.copy2(source, destination)
        artifacts[key] = str(destination.resolve())
    artifacts["report_json"] = str((directory / "report.json").resolve())
    report["source_server_artifacts"] = original
    report["artifacts"] = artifacts
    write_evidence(Path(artifacts["report_json"]), report)
    write_evidence(output_dir / report_filename, report)
    return artifacts


async def discover(output_dir: Path) -> dict:
    """Inspect the user's currently exposed EVE catalogue without assumptions."""
    servers = await asyncio.to_thread(list_eve_mcp_servers, 100)
    enabled = {s["name"] for s in servers if s.get("enabled", True)}
    evidence = {
        "tested_at_utc": datetime.now(UTC).isoformat(),
        "authentication": "EVE_API_KEY loaded from local .env; value omitted",
        "configuration_source": "repository .env (overrides inherited EVE settings)",
        "servers": servers,
        "inspected_tools": {},
        "sentinel2_discovery_calls": [],
    }
    for name in ("geocode", "eodashboard", "eotdl"):
        if name not in enabled:
            continue
        try:
            async with eve_mcp_client(name) as client:
                tools = await client.list_tools()
                evidence["inspected_tools"][name] = [
                    {"name": t.name, "description": t.description,
                     "inputSchema": t.inputSchema}
                    for t in tools
                ]
                if name == "eodashboard":
                    for arguments in ({"keyword": "Sentinel-2"},
                                      {"code": "sentinel-2-l2a"}):
                        result = await client.call_tool("search_collections", arguments)
                        evidence["sentinel2_discovery_calls"].append({
                            "server": name, "tool": "search_collections",
                            "arguments": arguments, "is_error": result.is_error,
                            "content": [getattr(block, "text", "")
                                        for block in result.content],
                        })
        except Exception as exc:
            evidence["inspected_tools"][name] = {"error": str(exc)}
    evidence["imagery_access_note"] = (
        "At this test time the enabled catalogue has no dedicated Sentinel-2 "
        "or CDSE imagery server. The eodashboard discovery results above record "
        "whether its catalogue exposed Sentinel-2. The planner acquires numeric "
        "Sentinel-2 pixels from public STAC itself; hosted EVE geocoding is "
        "tested separately in the agent chain."
    )
    write_evidence(output_dir / "eve_gateway_discovery.json", evidence)
    return evidence


async def run_agent(output_dir: Path, raster_path: str | None = None) -> dict:
    """Require the real EVE model to call the hosted and local MCP tools."""
    if raster_path:
        evaluation_instruction = (
            "Call evaluate_spatial_autocorrelation with raster_path="
            f"{str(Path(raster_path).resolve())!r}, band_index=1, lag_m=2560, "
            "seed=3, n_pairs_per_lag=200000, n_bins=24, max_lag_m=12800. "
        )
        expected_evaluator = "evaluate_spatial_autocorrelation"
    else:
        evaluation_instruction = (
            "Call evaluate_sentinel2_autocorrelation with bbox=[13.30,52.35,13.50,52.50], "
            "start_date='2024-06-01', end_date='2024-08-31', band='B05', "
            "max_cloud_cover=10, lag_m=2560, seed=3, n_pairs_per_lag=200000, "
            "n_bins=24, max_lag_m=12800. Keep this exact small bbox. "
        )
        expected_evaluator = "evaluate_sentinel2_autocorrelation"
    expected_planner = "plan_spatial_split"
    planner_instruction = (
        "Then call plan_spatial_split with raster_path equal to the evaluator's "
        "artifacts.analysis_band path, band_index=1, buffer_m=null (or omit buffer_m for automatic guard), lag_m=2560, "
        "seed=3, patch_layout='reference_blocks', patch_size_pixels=[32,40], "
        "block_size_pixels=[96,120], edge_policy='extend', seed_candidates=200, "
        "group_assignment='compact_regions', buffer_policy='one_sided', and "
        "split_ratios=[0.7,0.1,0.2]. Omit n_patches. Use seeded compact split "
        "regions in both spatial directions and extract fixed rectangular patches, "
        "adapting the checkpoint scripts/04_extract_patches.py workflow. "
        "Report requested versus realized fractions and the ranked seed selection. "
        "Class coverage remains unverified because no categorical labels are "
        "provided to this MCP call; never claim all classes occur in every split. "
        "Report the paired "
        "unbuffered mask with exactly the same validation/test rectangles and "
        "restored training rectangles. Guards retain test, then validation, "
        "then training, and exclude only conflicting lower-priority patches. "
        "The candidate mask has every quality-eligible patch before buffering "
        "and different heldout support. This is a geometric demonstration of the requested gap, "
        "including when the scientific lag decision rejects it. Report an "
        "infeasible split accurately if this AOI cannot support that guard. "
    )
    question = (
        "Create a defensible spatial train/validation/test plan for the small "
        "study area [13.30,52.35,13.50,52.50] west,south,east,north. "
        "First identify its setting by calling reverse_geocode_place with "
        "latitude=52.425, longitude=13.40, zoom=10. "
        + evaluation_instruction + planner_instruction
        + "Report the actual scene/band or raster used, variogram-derived "
        "rho_at_lag, practical_range_m, nugget, sill, fit_rss, per-bin count "
        "reliability, and the two-condition decision. Accept the lag only when "
        "rho_at_lag<0.1 AND lag_m>=practical_range_m with a resolved fit. "
        "Report patch counts before/after the buffer, candidate coverage, "
        "unselected valid pixels, excluded training pixels, and geometric "
        "separation. Keep unselected coverage separate from pixels removed by "
        "the buffer. This live integration test does not train a segmentation "
        "model; do not invent model accuracy. Distinguish observed results from "
        "recommendations, identify tool failures, and cite each tool. "
        "Explain that this unsupervised image analysis cannot establish a "
        "downstream model's prediction accuracy. Keep the answer concise."
    )
    prompt = (
        "You are an Earth Observation research assistant. Use the connected "
        "tools for every numerical result. Complete all three requested tool calls. "
        "Never invent pixels, statistics, an acquisition route, or test results. "
        "Use the exact AOI and dates requested, rather than a geocoder's larger "
        "bounding box. This chain uses geocoding, not a GIS viewer. Geometric "
        "separation and low single-band rho do not prove no data leakage. "
        "If a tool fails, report its error accurately. Keep the final answer under 200 words."
    )
    agent = await create_agent([HERE / "server.py", "geocode"], prompt=prompt)
    try:
        trace, agent_turns = [], []
        next_query = question
        async with asyncio.timeout(300):
            for _ in range(3):
                result = await agent.chat(next_query, echo=False)
                trace.extend(result.trace)
                agent_turns.append({"query": next_query, "answer": result.answer})
                outputs = {step["name"]: decoded_tool_output(step.get("content", ""))
                           for step in trace if step.get("node") == "tools"}
                pending = [name for name in ("reverse_geocode_place", expected_evaluator, expected_planner)
                           if name not in outputs or not isinstance(outputs[name], dict)
                           or (name != "reverse_geocode_place" and outputs[name].get("status") != "ok")]
                if not pending:
                    break
                next_query = (
                    "Continue the existing task by executing the missing calls now: "
                    + ", ".join(pending) + ". Do not stop after announcing a plan. "
                    + evaluation_instruction + planner_instruction
                    + "Reuse successful tool results already in this conversation. Report actual errors; "
                    "do not invent results or change requested scientific parameters."
                )
        evidence = {
            "tested_at_utc": datetime.now(UTC).isoformat(),
            "model": settings.eve_llm_model,
            "configuration_source": "repository .env (overrides inherited EVE settings)",
            "servers": ["local eo-spatial-split-planner", "EVE hosted geocode"],
            "available_tool_names": agent.tool_names,
            "system_prompt": prompt,
            "query": question,
            "answer": result.answer,
            "trace": trace,
            "agent_turns": agent_turns,
            "maximum_agent_turns": 3,
        }
    finally:
        await agent.close()
    completed = {s.get("name") for s in evidence["trace"]
                 if s.get("node") == "tools"}
    tool_results = {
        s["name"]: decoded_tool_output(s.get("content", ""))
        for s in evidence["trace"] if s.get("node") == "tools"
    }
    planner = tool_results.get(expected_planner)
    planner_ok = isinstance(planner, dict) and planner.get("status") == "ok"
    evaluation = tool_results.get(expected_evaluator)
    evaluation_ok = isinstance(evaluation, dict) and evaluation.get("status") == "ok"
    geocode = tool_results.get("reverse_geocode_place")
    geocode_ok = isinstance(geocode, dict) and not geocode.get("error")
    evidence["verification"] = {
        "expected_evaluation_tool": expected_evaluator,
        "evaluation_tool_returned": expected_evaluator in completed,
        "evaluation_status_ok": evaluation_ok,
        "expected_planner_tool": expected_planner,
        "planner_tool_returned": expected_planner in completed,
        "planner_status_ok": planner_ok,
        "hosted_geocode_tool_returned": "reverse_geocode_place" in completed,
        "hosted_geocode_success": geocode_ok,
    }
    if evaluation_ok:
        evidence["workspace_evaluation_artifacts"] = preserve_planner_artifacts(
            evaluation, output_dir, "eve_variogram_report.json")
    if planner_ok:
        evidence["workspace_artifacts"] = preserve_planner_artifacts(planner, output_dir)
    write_evidence(output_dir / "eve_demo.json", evidence)
    if not evaluation_ok or not planner_ok or not geocode_ok:
        raise AssertionError("EVE did not complete all three requested MCP calls successfully; inspect eve_demo.json")
    from test import verify_rectangles, verify_variogram

    evaluation_report = json.loads((output_dir / "eve_variogram_report.json").read_text())
    planner_report = json.loads((output_dir / "eve_planner_report.json").read_text())
    evidence["scientific_verification"] = verify_variogram(evaluation_report)
    evidence["rectangle_verification"] = verify_rectangles(planner_report)
    assert planner_report["schema_version"] == "0.4.0"
    assert planner_report["patch_layout"] == "reference_blocks"
    assert planner_report["patch_size_pixels"] == [32, 40]
    assert planner_report["block_size_pixels"] == [96, 120]
    assert planner_report["requested_split_ratios"] == {"train": 0.7, "validation": 0.1, "test": 0.2}
    assert planner_report["seed_search"]["stop_exclusive"] == 200
    assert planner_report["requested_buffer_m"] == planner_report["recommended_buffer_m"]
    assert planner_report["edge_policy"] == "extend"
    assert planner_report["group_assignment"] == "compact_regions"
    assert planner_report["buffer_policy"] == "one_sided"
    assert planner_report["edge_discarded_pixel_count"] == 0
    assert planner_report["class_labels_provided"] is False
    assert planner_report["class_coverage_complete"] is None
    evidence["verification"].update({
        "planner_schema_0_4_0": True, "reference_block_geometry": True,
        "requested_70_10_20_fractions": True, "full_aoi_no_edge_cropping": True,
        "compact_region_assignment": True, "one_sided_cross_split_buffer": True,
        "class_coverage_unverified_without_categorical_labels": True,
    })
    write_evidence(output_dir / "eve_demo.json", evidence)
    print(sanitize(evidence["answer"]))
    return evidence


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--raster", help="Use an existing numeric raster instead of live imagery acquisition")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts")
    args = parser.parse_args()
    if not settings.eve_configured:
        raise SystemExit("Set EVE_API_KEY and EVE_API_BASE_URL in the repository .env")
    if not args.skip_discovery:
        evidence = await discover(args.output_dir)
        print(f"Recorded {len(evidence['servers'])} EVE servers in {args.output_dir}")
    if not args.discover_only:
        await run_agent(args.output_dir, args.raster)


if __name__ == "__main__":
    asyncio.run(main())
