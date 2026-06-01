"""Lightweight project checks that do not require ML dependencies."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / "src" / "ppo_cartpole.py",
        ROOT / "notebooks" / "ppo_cartpole_colab.ipynb",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")

    py_compile.compile(str(ROOT / "src" / "ppo_cartpole.py"), doraise=True)

    notebook = json.loads((ROOT / "notebooks" / "ppo_cartpole_colab.ipynb").read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise SystemExit("Notebook must use nbformat 4")
    if len(notebook.get("cells", [])) < 4:
        raise SystemExit("Notebook looks too small")

    print("Project check passed.")


if __name__ == "__main__":
    main()
