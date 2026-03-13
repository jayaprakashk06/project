"""Core ML logic for crime hotspot prediction.

This module is intentionally UI-agnostic so it can be imported by Streamlit
(`app.py`) or used from the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}
MODEL_FEATURES = ["hour", "day_of_week", "latitude", "longitude"]


@dataclass
class ModelArtifacts:
    model: RandomForestClassifier
    feature_columns: list[str]
    auc: float


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_columns(df)

    out = df.copy()
    out = out.dropna(subset=["crime_type", "timestamp", "latitude", "longitude"])

    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["timestamp", "latitude", "longitude"])

    out = out[(out["latitude"] >= -90) & (out["latitude"] <= 90)]
    out = out[(out["longitude"] >= -180) & (out["longitude"] <= 180)]

    if out.empty:
        raise ValueError("No valid records remain after cleaning input data.")

    return out.sort_values("timestamp").reset_index(drop=True)


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    return clean_crime_data(pd.read_csv(file_path))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = clean_crime_data(df)
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    return out


def build_training_frame(feature_df: pd.DataFrame, cell_size: float = 0.01) -> pd.DataFrame:
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")

    out = feature_df.copy()
    out["cell_lat"] = (out["latitude"] / cell_size).round().astype(int) * cell_size
    out["cell_lon"] = (out["longitude"] / cell_size).round().astype(int) * cell_size

    grouped = (
        out.groupby(["cell_lat", "cell_lon", "hour", "day_of_week"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    threshold = grouped["incident_count"].median()
    grouped["target"] = (grouped["incident_count"] > threshold).astype(int)

    grouped = grouped.rename(columns={"cell_lat": "latitude", "cell_lon": "longitude"})
    if grouped["target"].nunique() < 2:
        grouped.loc[grouped.index[: max(1, len(grouped) // 3)], "target"] = 1
        grouped.loc[grouped.index[max(1, len(grouped) // 3) :], "target"] = 0

    return grouped


def train_model(training_df: pd.DataFrame) -> ModelArtifacts:
    X = training_df[MODEL_FEATURES]
    y = training_df["target"]

    if y.nunique() < 2:
        raise ValueError("Training target has only one class; cannot train classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        max_depth=12,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return ModelArtifacts(model=model, feature_columns=MODEL_FEATURES, auc=auc)


def predict_risk_score(
    artifacts: ModelArtifacts,
    latitude: float,
    longitude: float,
    hour: int,
    day_of_week: int,
) -> float:
    sample = pd.DataFrame(
        [{
            "hour": hour,
            "day_of_week": day_of_week,
            "latitude": latitude,
            "longitude": longitude,
        }]
    )
    return float(artifacts.model.predict_proba(sample[artifacts.feature_columns])[:, 1][0])


def risk_label(score: float) -> str:
    if score < 0.33:
        return "LOW"
    if score < 0.66:
        return "MEDIUM"
    return "HIGH"


def score_hotspots(
    artifacts: ModelArtifacts,
    feature_df: pd.DataFrame,
    prediction_hour: int | None = None,
) -> pd.DataFrame:
    candidates = build_training_frame(feature_df)
    if prediction_hour is not None:
        if prediction_hour < 0 or prediction_hour > 23:
            raise ValueError("prediction_hour must be in [0, 23]")
        candidates = candidates[candidates["hour"] == prediction_hour].copy()

    if candidates.empty:
        raise ValueError("No hotspot candidates available for scoring")

    X = candidates[MODEL_FEATURES]
    candidates["risk_score"] = artifacts.model.predict_proba(X)[:, 1]

    hotspot_df = candidates.groupby(["latitude", "longitude"], as_index=False)["risk_score"].mean()
    return hotspot_df.sort_values("risk_score", ascending=False)


def run_pipeline(input_path: str) -> tuple[float, int]:
    raw = load_and_validate_data(input_path)
    features = engineer_features(raw)
    training = build_training_frame(features)
    artifacts = train_model(training)
    hotspots = score_hotspots(artifacts, features)
    return artifacts.auc, len(hotspots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crime hotspot model backend")
    parser.add_argument("--input", required=True, help="Path to historical crime CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auc, hotspot_count = run_pipeline(args.input)
    print(f"Model AUC: {auc:.4f}")
    print(f"Generated hotspots: {hotspot_count}")


if __name__ == "__main__":
    main()
