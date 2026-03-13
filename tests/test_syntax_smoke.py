"""Smoke tests to prevent syntax/indentation regressions.

This specifically guards against errors like:
`IndentationError` and `unterminated triple-quoted string literal`.
`IndentationError: expected an indented block ...`
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
PY_FILES = [
    "app.py",
    "crime_hotspot_model.py",
    "config.py",
    "dataset_generator.py",
    "models/crime_prediction.py",
    "models/hotspot_clustering.py",
    "analytics/crime_statistics.py",
    "analytics/temporal_analysis.py",
    "utils/preprocessing.py",
    "utils/feature_engineering.py",
    "visualization/map_visualization.py",
    "visualization/heatmap_layer.py",
]


def test_python_modules_compile() -> None:
    for rel in PY_FILES:
        path = Path(rel)
        assert path.exists(), f"Missing expected file: {rel}"
        py_compile.compile(str(path), doraise=True)
