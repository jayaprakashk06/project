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
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}
MODEL_FEATURES = ["hour", "day_of_week", "latitude", "longitude"]
RISK_ORDER = ["low", "medium", "high"]
RISK_WEIGHTS = {"low": 0.1, "medium": 0.5, "high": 0.9}
import folium
import pandas as pd
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}


@dataclass
class ModelArtifacts:
    model: object
    model: RandomForestClassifier
    feature_columns: list[str]
    auc: float


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    validate_columns(df)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


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
    return clean_crime_data(df)


def train_model(training_df: pd.DataFrame) -> ModelArtifacts:
    artifacts: PredictionArtifacts = train_crime_model(training_df)
    return ModelArtifacts(model=artifacts.model, feature_columns=[], auc=artifacts.accuracy)


def score_hotspots(
    _artifacts: ModelArtifacts,
    feature_df: pd.DataFrame,
    prediction_hour: int | None = None,
) -> pd.DataFrame:
def score_hotspots(_artifacts: ModelArtifacts, feature_df: pd.DataFrame, prediction_hour: int | None = None) -> pd.DataFrame:
    out = feature_df.copy()
    if prediction_hour is not None:
        ts = pd.to_datetime(out["timestamp"])
        out = out[ts.dt.hour == prediction_hour]
    return detect_hotspot_clusters(out)


def predict_crime_risk(latitude: float, longitude: float, hour: int, day: int) -> dict[str, float | str]:
    # Generic context defaults for the advanced model inputs.
    # Use generic defaults for context features expected by advanced model
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
    df = df.dropna(subset=["timestamp", "latitude", "longitude", "crime_type"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


def engineer_features(df: pd.DataFrame, grid_size: float = 0.01) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month

    out["lat_bin"] = (out["latitude"] / grid_size).round().astype(int)
    out["lon_bin"] = (out["longitude"] / grid_size).round().astype(int)

    out = out.sort_values("timestamp")

    out["target"] = 1
    return out


def build_training_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        feature_df.groupby(["lat_bin", "lon_bin", "hour", "day_of_week", "month"])
        .size()
        .rename("incident_count")
        .reset_index()
    )

    threshold = grouped["incident_count"].quantile(0.6)
    grouped["target"] = (grouped["incident_count"] >= threshold).astype(int)
    return grouped


def train_model(training_df: pd.DataFrame) -> ModelArtifacts:
    X = training_df[MODEL_FEATURES]
    y = training_df["crime_risk"]
    feature_columns = ["lat_bin", "lon_bin", "hour", "day_of_week", "month"]
    X = training_df[feature_columns]
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
        n_estimators=300,
        max_depth=10,
        random_state=42,
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
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return ModelArtifacts(model=model, feature_columns=feature_columns, auc=auc)


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
    source_df: pd.DataFrame,
    grid_size: float = 0.01,
    prediction_hour: int | None = None,
) -> pd.DataFrame:
    feature_df = engineer_features(source_df, grid_size=grid_size)
    training_df = build_training_frame(feature_df)

    if prediction_hour is not None:
        training_df = training_df[training_df["hour"] == prediction_hour].copy()

    X = training_df[artifacts.feature_columns]
    training_df["risk_score"] = artifacts.model.predict_proba(X)[:, 1]

    training_df["latitude"] = training_df["lat_bin"] * grid_size
    training_df["longitude"] = training_df["lon_bin"] * grid_size

    hotspot_df = (
        training_df.groupby(["latitude", "longitude"], as_index=False)["risk_score"].mean()
    )
    return hotspot_df.sort_values("risk_score", ascending=False)


def create_hotspot_map(hotspot_df: pd.DataFrame, output_file: str) -> None:
    if hotspot_df.empty:
        raise ValueError("No hotspots to map.")

    center_lat = hotspot_df["latitude"].mean()
    center_lon = hotspot_df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    heat_points = hotspot_df[["latitude", "longitude", "risk_score"]].values.tolist()
    HeatMap(heat_points, radius=16, blur=12, min_opacity=0.2).add_to(m)

    for _, row in hotspot_df.head(20).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=f"Risk: {row['risk_score']:.3f}",
        ).add_to(m)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_file)


def run_pipeline(
    input_path: str,
    output_map: str,
    output_grid: str | None,
    grid_size: float,
    prediction_hour: int | None,
) -> tuple[float, int]:
    raw_df = load_and_validate_data(input_path)
    features_df = engineer_features(raw_df, grid_size=grid_size)
    training_df = build_training_frame(features_df)
    artifacts = train_model(training_df)

    hotspots = score_hotspots(
        artifacts,
        raw_df,
        grid_size=grid_size,
        prediction_hour=prediction_hour,
    )
    create_hotspot_map(hotspots, output_map)

    if output_grid:
        Path(output_grid).parent.mkdir(parents=True, exist_ok=True)
        hotspots.to_csv(output_grid, index=False)

    return artifacts.auc, len(hotspots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crime hotspot prediction and mapping tool")
    parser.add_argument("--input", required=True, help="Path to historical crime CSV")
    parser.add_argument("--output-map", required=True, help="Output HTML hotspot map")
    parser.add_argument("--output-grid", default=None, help="Optional output CSV of scored grids")
    parser.add_argument("--grid-size", type=float, default=0.01, help="Spatial grid size in degrees")
    parser.add_argument(
        "--prediction-hour",
        type=int,
        default=None,
        help="Optional hour (0-23) to generate time-specific hotspot predictions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric, hotspot_count, saved_path = run_pipeline(args.input)
    print(f"Model accuracy: {metric:.4f}")
    print(f"Hotspot clusters: {hotspot_count}")
    print(f"Saved model: {saved_path}")
    metric, hotspot_count, saved_path = run_pipeline(args.input, model_path=args.model_path)
    print(f"Model accuracy: {metric:.4f}")
    print(f"Generated hotspots: {hotspot_count}")
    print(f"Saved model: {saved_path}")
    auc, hotspot_count = run_pipeline(
        input_path=args.input,
        output_map=args.output_map,
        output_grid=args.output_grid,
        grid_size=args.grid_size,
        prediction_hour=args.prediction_hour,
    )
    print(f"Model AUC: {auc:.4f}")
    print(f"Generated hotspots: {hotspot_count}")
    print(f"Hotspot map saved to: {args.output_map}")
    if args.output_grid:
        print(f"Grid scores saved to: {args.output_grid}")


if __name__ == "__main__":
    main()
