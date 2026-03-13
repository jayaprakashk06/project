from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, RANDOM_STATE
from utils.feature_engineering import build_model_frame

FEATURES = ["latitude", "longitude", "hour", "day_of_week", "month", "crime_frequency"]


@dataclass
class PredictionArtifacts:
    model: RandomForestClassifier
    accuracy: float
    classes: list[str]


def derive_risk_labels(df: pd.DataFrame) -> pd.Series:
    bins = df["crime_frequency"].quantile([0.33, 0.66]).values
    q1, q2 = bins[0], bins[1]

    def label(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "medium"
        return "high"

    y = df["crime_frequency"].apply(label)
    if y.nunique() < 2:
        y.iloc[::2] = "low"
        y.iloc[1::2] = "high"
    return y


def train_crime_model(clean_df: pd.DataFrame) -> PredictionArtifacts:
    model_df = build_model_frame(clean_df)
    X = model_df[FEATURES]
    y = derive_risk_labels(model_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=320,
        max_depth=14,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    return PredictionArtifacts(model=model, accuracy=accuracy, classes=list(model.classes_))


def save_prediction_model(artifacts: PredictionArtifacts, model_path: Path = MODEL_PATH) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": artifacts.model,
        "classes": artifacts.classes,
        "features": FEATURES,
        "accuracy": artifacts.accuracy,
    }
    joblib.dump(payload, model_path)
    return model_path


def load_prediction_model(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def predict_crime_probability(
    latitude: float,
    longitude: float,
    hour: int,
    day_of_week: int,
    month: int,
    crime_frequency: float,
    model_path: Path = MODEL_PATH,
) -> dict[str, float | str]:
    payload = load_prediction_model(model_path)
    model: RandomForestClassifier = payload["model"]
    features: list[str] = payload["features"]

    sample = pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "hour": hour,
                "day_of_week": day_of_week,
                "month": month,
                "crime_frequency": crime_frequency,
            }
        ]
    )
    probs = model.predict_proba(sample[features])[0]
    labels = list(model.classes_)
    mapping = {lbl: float(prob) for lbl, prob in zip(labels, probs)}
    pred = max(mapping, key=mapping.get)
    return {
        "risk_level": pred,
        "crime_probability": float(mapping[pred]),
        "low": float(mapping.get("low", 0.0)),
        "medium": float(mapping.get("medium", 0.0)),
        "high": float(mapping.get("high", 0.0)),
    }
