"""Validate notebook structure and compile code cells without executing them.

This check needs nbformat, but no EVE credentials, satellite acquisition,
notebook kernel, PyTorch model execution, or optional agent dependencies.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("START_HERE.ipynb", "EO_SPATIAL_SPLIT_PLANNER.ipynb")


def main():
    reports = []
    for name in NOTEBOOKS:
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
        reports.append({"notebook": name, "code_cells_compiled": count})
    print(json.dumps({"status": "passed", "execution": "none", "notebooks": reports}))


if __name__ == "__main__":
    main()
