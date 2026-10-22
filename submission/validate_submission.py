"""Run the vendored official registry structural checker on the prepared server.

Only servers/eo-spatial-split-planner belongs in a registry PR. Other recorded
validation is read from the versioned evidence, never inferred from old results.
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
recorded = SERVER / "artifacts" / "validation_summary.json"
receipt = {"schema_version": "0.3.0",
           "official_registry_structural_checks_passed": result.returncode == 0,
           "stdout": result.stdout, "stderr": result.stderr,
           "recorded_validation": json.loads(recorded.read_text()) if recorded.is_file() else None,
           "official_registry_submission_created": False}
(ROOT / "validation.json").write_text(json.dumps(receipt, indent=2))
raise SystemExit(result.returncode)
