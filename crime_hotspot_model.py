# Backward-compatible facade for crime intelligence modeling.
#
# Legacy entrypoints are preserved while delegating to the modular architecture.

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from config import MODEL_PATH
from models.crime_prediction import (
    PredictionArtifacts,
    predict_crime_probability,
    save_prediction_model,
    train_crime_model,
)
from models.hotspot_clustering import detect_hotspot_clusters
from utils.preprocessing import clean_crime_data


@dataclass
class ModelArtifacts:
    model: object
    feature_columns: list[str]
    auc: float


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    return clean_crime_data(pd.read_csv(file_path))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    return clean_crime_data(df)


def train_model(training_df: pd.DataFrame) -> ModelArtifacts:
    artifacts: PredictionArtifacts = train_crime_model(training_df)
    return ModelArtifacts(model=artifacts.model, feature_columns=[], auc=artifacts.accuracy)


def score_hotspots(
    _artifacts: ModelArtifacts,
    feature_df: pd.DataFrame,
    prediction_hour: int | None = None,
) -> pd.DataFrame:
    out = feature_df.copy()
    if prediction_hour is not None:
        ts = pd.to_datetime(out["timestamp"])
        out = out[ts.dt.hour == prediction_hour]
    return detect_hotspot_clusters(out)


def predict_crime_risk(latitude: float, longitude: float, hour: int, day: int) -> dict[str, float | str]:
    # Generic context defaults for the advanced model inputs.
    return predict_crime_probability(
        latitude=latitude,
        longitude=longitude,
        hour=hour,
        day_of_week=day,
        month=6,
        crime_frequency=10.0,
        model_path=MODEL_PATH,
    )


def run_pipeline(input_path: str) -> tuple[float, int, str]:
    data = load_and_validate_data(input_path)
    artifacts = train_crime_model(data)
    saved_path = save_prediction_model(artifacts, MODEL_PATH)
    hotspots = detect_hotspot_clusters(data)
    return artifacts.accuracy, len(hotspots), str(saved_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crime hotspot model backend")
    parser.add_argument("--input", required=True, help="Path to historical crime CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric, hotspot_count, saved_path = run_pipeline(args.input)
    print(f"Model accuracy: {metric:.4f}")
    print(f"Hotspot clusters: {hotspot_count}")
    print(f"Saved model: {saved_path}")


if __name__ == "__main__":
    main()
