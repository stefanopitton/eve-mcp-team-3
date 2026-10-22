"""Validate notebook structure and compile code cells without executing them.

This check needs nbformat, but no EVE credentials, satellite acquisition,
notebook kernel, PyTorch model execution, or optional agent dependencies.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("START_HERE.ipynb", "EO_SPATIAL_SPLIT_PLANNER.ipynb")


def _cell_sha256(cell):
    canonical = json.dumps(cell, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_start_here_template(notebook):
    """Protect every original cell outside Your turn, including its outputs."""
    manifest = json.loads((ROOT / "scripts" / "start_here_template.json").read_text(encoding="utf-8"))
    prefix, suffix = manifest["protected_prefix"], manifest["protected_suffix"]
    cells = notebook["cells"]
    if len(cells) < len(prefix) + len(suffix):
        raise ValueError("START_HERE.ipynb is missing protected original template cells")
    protected = [*zip(cells[:len(prefix)], prefix, strict=True),
                 *zip(cells[len(cells) - len(suffix):], suffix, strict=True)]
    for cell, expected in protected:
        if _cell_sha256(cell) != expected["sha256"]:
            raise ValueError(
                f"START_HERE.ipynb changed protected original cell {expected['original_cell_index']}; "
                "edit only the Your turn section and preserve the original cell dictionary, including outputs"
            )
    for key in ("metadata", "nbformat", "nbformat_minor"):
        if notebook.get(key) != manifest[key]:
            raise ValueError(f"START_HERE.ipynb changed original notebook {key}; preserve template metadata/version")
    return {"protected_original_cells": len(protected), "source_commit": manifest["source_commit"]}


def main():
    reports = []
    for name in NOTEBOOKS:
        preservation = None
        if name == "START_HERE.ipynb":
            # Read the original JSON representation before nbformat normalizes
            # source lists into strings or repairs any missing cell IDs.
            raw = json.loads((ROOT / name).read_text(encoding="utf-8"))
            preservation = validate_start_here_template(raw)
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        count = 0
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            compile(cell.source, f"{name}:cell-{index}", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            count += 1
        if count == 0:
            raise ValueError(f"{name} has no executable demonstration cells")
        report = {"notebook": name, "code_cells_compiled": count}
        if preservation is not None:
            report["template_preservation"] = preservation
        reports.append(report)
    print(json.dumps({"status": "passed", "execution": "none", "notebooks": reports}))


if __name__ == "__main__":
    main()
