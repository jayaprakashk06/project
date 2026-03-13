"""Core ML logic for crime hotspot prediction.

This module is intentionally UI-agnostic so it can be imported by Streamlit
(`app.py`) or used from the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}
MODEL_FEATURES = ["hour", "day_of_week", "latitude", "longitude"]
RISK_ORDER = ["low", "medium", "high"]
RISK_WEIGHTS = {"low": 0.1, "medium": 0.5, "high": 0.9}


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


def _label_risk_levels(incident_counts: pd.Series) -> pd.Series:
    quantiles = incident_counts.quantile([0.33, 0.66]).values
    q1, q2 = float(quantiles[0]), float(quantiles[1])

    def classify(value: float) -> str:
        if value <= q1:
            return "low"
        if value <= q2:
            return "medium"
        return "high"

    labels = incident_counts.apply(classify)
    if labels.nunique() < 2:
        labels = pd.Series(["low" if i % 2 == 0 else "high" for i in range(len(labels))])
    return labels


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

    grouped["crime_risk"] = _label_risk_levels(grouped["incident_count"])
    grouped = grouped.rename(columns={"cell_lat": "latitude", "cell_lon": "longitude"})
    return grouped


def train_model(training_df: pd.DataFrame) -> ModelArtifacts:
    X = training_df[MODEL_FEATURES]
    y = training_df["crime_risk"]

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

    accuracy = float(model.score(X_test, y_test))
    return ModelArtifacts(model=model, feature_columns=MODEL_FEATURES, auc=accuracy)


def save_model(artifacts: ModelArtifacts, model_path: str = "models/crime_risk_model.joblib") -> str:
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifacts.model,
        "feature_columns": artifacts.feature_columns,
        "metric": artifacts.auc,
    }
    joblib.dump(payload, path)
    return str(path)


def load_model(model_path: str = "models/crime_risk_model.joblib") -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")
    return joblib.load(path)


def predict_crime_risk(
    latitude: float,
    longitude: float,
    hour: int,
    day: int,
    model_path: str = "models/crime_risk_model.joblib",
) -> dict[str, float | str]:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Coordinates are out of valid range.")
    if not (0 <= hour <= 23):
        raise ValueError("hour must be in [0, 23]")
    if not (0 <= day <= 6):
        raise ValueError("day must be in [0, 6]")

    payload = load_model(model_path)
    model: RandomForestClassifier = payload["model"]
    feature_columns: list[str] = payload["feature_columns"]

    sample = pd.DataFrame(
        [{
            "hour": hour,
            "day_of_week": day,
            "latitude": latitude,
            "longitude": longitude,
        }]
    )
    probs = model.predict_proba(sample[feature_columns])[0]
    labels = list(model.classes_)
    proba_map = {label: float(prob) for label, prob in zip(labels, probs)}

    predicted_label = max(proba_map, key=proba_map.get)
    return {
        "crime_risk": predicted_label,
        "prediction_probability": float(proba_map[predicted_label]),
        "low_probability": float(proba_map.get("low", 0.0)),
        "medium_probability": float(proba_map.get("medium", 0.0)),
        "high_probability": float(proba_map.get("high", 0.0)),
    }


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
    probs = artifacts.model.predict_proba(sample[artifacts.feature_columns])[0]
    labels = list(artifacts.model.classes_)
    weighted = 0.0
    for label, prob in zip(labels, probs):
        weighted += RISK_WEIGHTS.get(str(label), 0.5) * float(prob)
    return weighted


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
    proba_matrix = artifacts.model.predict_proba(X)
    class_labels = list(artifacts.model.classes_)

    weighted_scores: list[float] = []
    for row_probs in proba_matrix:
        score = 0.0
        for label, prob in zip(class_labels, row_probs):
            score += RISK_WEIGHTS.get(str(label), 0.5) * float(prob)
        weighted_scores.append(score)

    candidates["risk_score"] = weighted_scores
    hotspot_df = candidates.groupby(["latitude", "longitude"], as_index=False)["risk_score"].mean()
    return hotspot_df.sort_values("risk_score", ascending=False)


def train_and_save_model(
    input_path: str,
    model_path: str = "models/crime_risk_model.joblib",
) -> tuple[ModelArtifacts, str]:
    raw = load_and_validate_data(input_path)
    features = engineer_features(raw)
    training = build_training_frame(features)
    artifacts = train_model(training)
    saved_path = save_model(artifacts, model_path=model_path)
    return artifacts, saved_path


def run_pipeline(input_path: str, model_path: str = "models/crime_risk_model.joblib") -> tuple[float, int, str]:
    artifacts, saved_path = train_and_save_model(input_path, model_path=model_path)
    raw = load_and_validate_data(input_path)
    features = engineer_features(raw)
    hotspots = score_hotspots(artifacts, features)
    return artifacts.auc, len(hotspots), saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crime hotspot model backend")
    parser.add_argument("--input", required=True, help="Path to historical crime CSV")
    parser.add_argument(
        "--model-path",
        default="models/crime_risk_model.joblib",
        help="Path to save trained joblib model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric, hotspot_count, saved_path = run_pipeline(args.input, model_path=args.model_path)
    print(f"Model accuracy: {metric:.4f}")
    print(f"Generated hotspots: {hotspot_count}")
    print(f"Saved model: {saved_path}")


if __name__ == "__main__":
    main()
