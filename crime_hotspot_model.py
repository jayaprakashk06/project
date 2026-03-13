import argparse
from dataclasses import dataclass
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}


@dataclass
class ModelArtifacts:
    model: RandomForestClassifier
    feature_columns: list[str]
    auc: float


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

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
        n_estimators=300,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return ModelArtifacts(model=model, feature_columns=feature_columns, auc=auc)


def score_hotspots(
    artifacts: ModelArtifacts,
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
