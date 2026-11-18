"""Create dist/student/: the project without solutions, checkpoints or notebook outputs.

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
    ".venv", "dist", "outputs", "solutions", "__pycache__", "*.log", "workshop_executed.ipynb",
    ".pytest_cache", ".ipynb_checkpoints", "MEASURED.md"))
(DST / "outputs").mkdir()

nb = nbformat.read(DST / "workshop.ipynb", as_version=4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs, c.execution_count = [], None
nbformat.write(nb, DST / "workshop.ipynb")

# Students have no solutions/, so the tests that import it are dropped
(DST / "tests" / "test_rewards.py").unlink()
print("student copy ->", DST)
