"""Runtime import smoke tests.

Guards against Streamlit startup crashes such as IndentationError during module import.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_import_core_modules() -> None:
    for module in [
        "crime_hotspot_model",
        "config",
        "models.crime_prediction",
        "models.hotspot_clustering",
        "analytics.crime_statistics",
        "analytics.temporal_analysis",
        "utils.preprocessing",
        "utils.feature_engineering",
        "visualization.map_visualization",
        "visualization.heatmap_layer",
    ]:
        importlib.import_module(module)


def test_load_and_validate_data_works() -> None:
    csm = importlib.import_module("crime_hotspot_model")
    sample = ROOT / "data" / "sample_crime_data.csv"
    df = csm.load_and_validate_data(str(sample))
    assert not df.empty
    assert {"crime_type", "timestamp", "latitude", "longitude", "district"}.issubset(df.columns)
