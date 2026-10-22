"""Validate the standalone START_HERE notebook and its submission copy.

By default, check structure, Python compilation and embedded input integrity.
With --execute, run only the notebook in an otherwise empty temporary directory,
without EVE credentials or project imports. Runtime validation lives in the
notebook itself; this script is an optional development check.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("START_HERE.ipynb", "submission/START_HERE.ipynb")
PROJECT_MODULES = {"agentic_eo", "core", "imagery", "block_patches", "patch_export",
                   "segmentation", "segmentation_data", "run_segmentation", "server", "test_eve"}
REQUIRED_CELLS = {"standalone-setup", "embedded-inputs", "inline-core", "inline-block_patches",
                  "inline-patch_export", "inline-segmentation", "inline-server",
                  "run-fresh-experiment", "invoke-notebook-mcp", "verify-fresh-results"}


def validate_standalone(notebook, name):
    """Enforce the single-notebook contract, including real input bytes."""
    missing = REQUIRED_CELLS - {cell.id for cell in notebook.cells}
    if missing:
        raise ValueError(f"{name} is missing standalone workflow cells: {sorted(missing)}")
    if notebook.metadata.get("standalone_workflow", {}).get("external_project_files_required") is not False:
        raise ValueError(f"{name} does not declare a standalone workflow")
    code_cells = 0
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        compile(cell.source, f"{name}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        code_cells += 1
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(f"{name}:cell-{index} imports a relative project module")
                modules = [(node.module or "").split(".")[0]]
            if PROJECT_MODULES.intersection(modules):
                raise ValueError(f"{name}:cell-{index} imports an external project module")
            if isinstance(node, ast.Name) and node.id == "__file__":
                raise ValueError(f"{name}:cell-{index} depends on an external script location")
            if isinstance(node, ast.Attribute) and ast.unparse(node) == "sys.path":
                raise ValueError(f"{name}:cell-{index} changes the project import path")
        if any(output.output_type == "error" for output in cell.get("outputs", [])):
            raise ValueError(f"{name}:cell-{index} contains a failed execution")
    embedded_cell = next(cell for cell in notebook.cells if cell.id == "embedded-inputs")
    embedded = ast.literal_eval(ast.parse(embedded_cell.source).body[0].value)
    for relative, item in embedded.items():
        raw = base64.b64decode(item["base64"], validate=True)
        if len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"{name} has a corrupt embedded input: {relative}")
    expected = {"data/sentinel2_reflectance_20m.tif", "data/worldcover_2021_20m.tif", "data/manifest.json"}
    if not expected.issubset(embedded):
        raise ValueError(f"{name} is missing embedded satellite inputs")
    return {"notebook": name, "code_cells_compiled": code_cells,
            "embedded_files_verified": len(embedded),
            "embedded_bytes": sum(item["bytes"] for item in embedded.values())}


def execute_standalone():
    from nbclient import NotebookClient

    with tempfile.TemporaryDirectory(prefix="start-here-validation-") as directory:
        work = Path(directory)
        shutil.copyfile(ROOT / "START_HERE.ipynb", work / "START_HERE.ipynb")
        notebook = nbformat.read(work / "START_HERE.ipynb", as_version=4)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("EVE_") and key not in {"PYTHONPATH", "EO_SPLIT_OUTPUT_DIR"}}
        client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                                resources={"metadata": {"path": str(work)}})
        client.execute(env=environment)
        if list(work.rglob("*.py")):
            raise ValueError("Standalone execution unexpectedly generated Python scripts")
        receipt = json.loads((work / "start_here_outputs" / "notebook_validation.json").read_text())
        if receipt["status"] != "passed":
            raise ValueError("Notebook runtime validation did not pass")
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the notebook alone in a temporary directory")
    arguments = parser.parse_args()
    reports = []
    for name in NOTEBOOKS:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        reports.append(validate_standalone(notebook, name))
    if (ROOT / NOTEBOOKS[0]).read_bytes() != (ROOT / NOTEBOOKS[1]).read_bytes():
        raise ValueError("The main notebook and submission notebook differ")
    execution = execute_standalone() if arguments.execute else "none"
    print(json.dumps({"status": "passed", "execution": execution, "notebooks": reports}))


if __name__ == "__main__":
    main()
