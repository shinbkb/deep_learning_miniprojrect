"""Validate notebook JSON and Python syntax without running training."""
import ast
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = json.loads(path.read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path.name}:cell{index}")
    print(f"OK: {path.name} ({len(notebook['cells'])} cells)")
