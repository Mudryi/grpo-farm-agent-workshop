"""Create dist/student/: the project without checkpoints, outputs or notebook outputs.

    uv run python scripts/make_student_copy.py
"""

import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "dist" / "student"

if DST.exists():
    shutil.rmtree(DST)
shutil.copytree(ROOT, DST, ignore=shutil.ignore_patterns(
    ".venv", "dist", "outputs", "checkpoints", "chat_history", "__pycache__", "*.log", "*_executed.ipynb",
    ".pytest_cache", ".ipynb_checkpoints", "MEASURED.md"))
(DST / "outputs").mkdir()

for name in ["workshop.ipynb", "workshop_uk.ipynb"]:
    nb = nbformat.read(DST / name, as_version=4)
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs, c.execution_count = [], None
    nbformat.write(nb, DST / name)
print("student copy ->", DST)
