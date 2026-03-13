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
COMPAT_MODEL_PATH = Path("crime_risk_model.joblib")


@dataclass
class PredictionArtifacts:
    model: RandomForestClassifier
    accuracy: float
    classes: list[str]


def derive_risk_labels(df: pd.DataFrame) -> pd.Series:
    bins = df["crime_frequency"].quantile([0.33, 0.66]).values
    q1, q2 = float(bins[0]), float(bins[1])

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
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    accuracy = float(model.score(X_test, y_test))
    return PredictionArtifacts(model=model, accuracy=accuracy, classes=list(model.classes_))


def train_demo_model_for_quick_start(model_path: Path = COMPAT_MODEL_PATH) -> Path:
    """Train a tiny RandomForest model using the user-provided sample style data.

    This is a compatibility helper for quick terminal usage when no project-trained
    model exists yet.
    """
    data = {
        "latitude": [13.0827, 11.0168, 9.9252, 10.7905, 11.6643],
        "longitude": [80.2707, 76.9558, 78.1198, 78.7047, 78.1460],
        "hour": [22, 18, 21, 14, 19],
        "day": [5, 3, 6, 2, 4],
        "month": [12, 11, 10, 8, 9],
        "risk": [1, 0, 1, 0, 1],
    }
    df = pd.DataFrame(data)
    X = df[["latitude", "longitude", "hour", "day", "month"]]
    y = df["risk"]

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X, y)

    payload = {
        "compat_model": model,
        "features": ["latitude", "longitude", "hour", "day", "month"],
    }
    joblib.dump(payload, model_path)
    return model_path


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
    if model_path.exists():
        return joblib.load(model_path)
    if COMPAT_MODEL_PATH.exists():
        return joblib.load(COMPAT_MODEL_PATH)
    raise FileNotFoundError(
        f"Model file not found: {model_path}. Train via app/pipeline first, "
        f"or call train_demo_model_for_quick_start()."
    )


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

    if "compat_model" in payload:
        model: RandomForestClassifier = payload["compat_model"]
        sample = pd.DataFrame(
            [{"latitude": latitude, "longitude": longitude, "hour": hour, "day": day_of_week, "month": month}]
        )
        prob_high = float(model.predict_proba(sample)[0][1])
        risk = "high" if prob_high >= 0.5 else "low"
        return {
            "risk_level": risk,
            "crime_probability": prob_high if risk == "high" else 1.0 - prob_high,
            "low": 1.0 - prob_high,
            "medium": 0.0,
            "high": prob_high,
        }

    model = payload["model"]
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


def predict_risk(
    latitude: float,
    longitude: float,
    hour: int,
    day: int,
    month: int,
    crime_frequency: float = 10.0,
    model_path: Path = MODEL_PATH,
) -> str:
    """Simple API compatible with user snippet.

    Returns: "High Crime Risk" or "Low Crime Risk".
    """
    result = predict_crime_probability(
        latitude=latitude,
        longitude=longitude,
        hour=hour,
        day_of_week=day,
        month=month,
        crime_frequency=crime_frequency,
        model_path=model_path,
    )
    return "High Crime Risk" if str(result["risk_level"]).lower() in {"high", "medium"} else "Low Crime Risk"
