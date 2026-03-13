"""Smoke tests to prevent syntax/indentation regressions.

This specifically guards against errors like:
`IndentationError: expected an indented block ...`
"""

from __future__ import annotations

import py_compile
from pathlib import Path


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
