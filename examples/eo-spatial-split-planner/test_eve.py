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
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "lib"))

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


def preserve_planner_artifacts(planner: dict, output_dir: Path) -> dict:
    """Copy files from the local stdio server's scratch directory into the project."""
    original = planner.get("artifacts", {})
    report_path = Path(original.get("report_json", ""))
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text())
    directory = output_dir / "eve_runs" / report["run_fingerprint"]
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {"scope": "Preserved workspace copies from the local MCP server"}
    for key in ("analysis_band", "split_mask"):
        source = Path(original[key])
        destination = directory / source.name
        shutil.copy2(source, destination)
        artifacts[key] = str(destination.resolve())
    artifacts["report_json"] = str((directory / "report.json").resolve())
    report["source_server_artifacts"] = original
    report["artifacts"] = artifacts
    write_evidence(Path(artifacts["report_json"]), report)
    write_evidence(output_dir / "eve_planner_report.json", report)
    return artifacts


async def discover(output_dir: Path) -> dict:
    """Inspect the user's currently exposed EVE catalogue without assumptions."""
    servers = await asyncio.to_thread(list_eve_mcp_servers, 100)
    enabled = {s["name"] for s in servers if s.get("enabled", True)}
    evidence = {
        "tested_at_utc": datetime.now(UTC).isoformat(),
        "authentication": "EVE_API_KEY loaded from local .env; value omitted",
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
        planner_instruction = (
            "Call plan_spatial_split with raster_path="
            f"{str(Path(raster_path).resolve())!r}, band_index=1, and seed=42."
        )
        expected_planner = "plan_spatial_split"
    else:
        planner_instruction = (
            "Call plan_sentinel2_split with bbox=[13.30,52.35,13.50,52.50], "
            "start_date='2024-06-01', end_date='2024-08-31', band='B05', "
            "max_cloud_cover=10, and seed=42. Keep this exact small bbox."
        )
        expected_planner = "plan_sentinel2_split"
    question = (
        "Create a defensible spatial train/validation/test plan for the small "
        "study area [13.30,52.35,13.50,52.50] west,south,east,north. "
        "First identify its setting by calling reverse_geocode_place with "
        "latitude=52.425, longitude=13.40, zoom=10. "
        + planner_instruction + " Report the actual scene/band or raster used, "
        "spatial correlation/range evidence, split counts, guard buffer, and "
        "leakage comparison when returned. Distinguish observed results from "
        "recommendations, identify tool failures, and cite each tool. "
        "Explain that this unsupervised image analysis cannot establish a "
        "downstream model's prediction accuracy. Keep the answer concise."
    )
    prompt = (
        "You are an Earth Observation research assistant. Use the connected "
        "tools for every numerical result. Complete both requested tool calls. "
        "Never invent pixels, statistics, an acquisition route, or test results. "
        "Use the exact AOI and dates requested, rather than a geocoder's larger "
        "bounding box. If a tool fails, report its error accurately."
    )
    agent = await create_agent([HERE / "server.py", "geocode"], prompt=prompt)
    try:
        result = await asyncio.wait_for(agent.chat(question, echo=False), timeout=300)
        evidence = {
            "tested_at_utc": datetime.now(UTC).isoformat(),
            "model": settings.eve_llm_model,
            "servers": ["local eo-spatial-split-planner", "EVE hosted geocode"],
            "available_tool_names": agent.tool_names,
            "query": question,
            "answer": result.answer,
            "trace": result.trace,
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
    geocode = tool_results.get("reverse_geocode_place")
    geocode_ok = isinstance(geocode, dict) and not geocode.get("error")
    evidence["verification"] = {
        "expected_planner_tool": expected_planner,
        "planner_tool_returned": expected_planner in completed,
        "planner_status_ok": planner_ok,
        "hosted_geocode_tool_returned": "reverse_geocode_place" in completed,
        "hosted_geocode_success": geocode_ok,
    }
    if planner_ok:
        evidence["workspace_artifacts"] = preserve_planner_artifacts(planner, output_dir)
    write_evidence(output_dir / "eve_demo.json", evidence)
    if not planner_ok or not geocode_ok:
        raise AssertionError("EVE did not complete both requested MCP calls successfully; inspect eve_demo.json")
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
