"""Smoke tests to prevent syntax/indentation regressions.

This specifically guards against errors like:
`IndentationError` and `unterminated triple-quoted string literal`.
"""

from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _discover_python_files() -> list[Path]:
    py_files = [
        p
        for p in ROOT.rglob("*.py")
        if ".git" not in p.parts
        and ".venv" not in p.parts
        and "__pycache__" not in p.parts
    ]
    return sorted(py_files)


def test_python_modules_compile() -> None:
    py_files = _discover_python_files()
    assert py_files, "No Python files discovered for syntax smoke test"

    for path in py_files:
        py_compile.compile(str(path), doraise=True)
