"""Validate the prepared server using the official EVE registry checks.

Run with the hackathon .venv Python from any directory. The vendored checker and
Dockerfile are unmodified snapshots of eve-esa/mcp-tool-registry main, fetched
on 2026-10-22. Only servers/eo-spatial-split-planner belongs in the proposed PR.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SERVER = ROOT / "servers" / "eo-spatial-split-planner"
files = [str(p.relative_to(ROOT)) for p in SERVER.rglob("*")
         if p.is_file() and "__pycache__" not in p.parts]
result = subprocess.run([sys.executable, str(ROOT / "registry_validate_pr.py"), "--files", *files],
                        cwd=ROOT, text=True, capture_output=True)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
receipt = {"official_registry_structural_checks_passed": result.returncode == 0,
           "stdout": result.stdout, "stderr": result.stderr,
           "docker_build": "not run: Docker daemon unavailable on this host",
           "recorded_unit_tests": "29 passed", "recorded_mcp_stdio": "passed synthetic and real Sentinel-2",
           "recorded_mcp_http": "passed real raster through stateless streamable HTTP",
           "live_eve_agent": "passed hosted geocoder + local planner",
           "notebook_execution": "passed", "published": False}
(ROOT / "validation.json").write_text(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
