from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import MODEL_PATH, RANDOM_STATE
from utils.feature_engineering import build_model_frame

FEATURES = ["latitude", "longitude", "hour", "day_of_week", "month", "crime_frequency"]
RISK_LEVELS = ["low", "medium", "high"]


@dataclass
class PredictionArtifacts:
    model: dict
    model: RandomForestClassifier
    accuracy: float
    classes: list[str]


def derive_risk_labels(df: pd.DataFrame) -> pd.Series:
    bins = df["crime_frequency"].quantile([0.33, 0.66]).values
    q1, q2 = float(bins[0]), float(bins[1])
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


def _build_centroid_model(X: pd.DataFrame, y: pd.Series) -> dict:
    centroids: dict[str, np.ndarray] = {}
    priors: dict[str, float] = {}
    for lbl in sorted(y.unique()):
        mask = y == lbl
        centroids[lbl] = X.loc[mask, FEATURES].mean().to_numpy(dtype=float)
        priors[lbl] = float(mask.mean())

    for lbl in RISK_LEVELS:
        if lbl not in centroids:
            centroids[lbl] = X[FEATURES].mean().to_numpy(dtype=float)
            priors[lbl] = 1e-6

    total_prior = sum(priors.values())
    priors = {k: v / total_prior for k, v in priors.items()}
    return {"centroids": centroids, "priors": priors, "features": FEATURES}


def _predict_proba(model: dict, X: pd.DataFrame) -> np.ndarray:
    rows = X[FEATURES].to_numpy(dtype=float)
    classes = RISK_LEVELS
    centroid_mat = np.vstack([model["centroids"][c] for c in classes])
    priors = np.array([model["priors"][c] for c in classes], dtype=float)

    probs = []
    for r in rows:
        dists = np.linalg.norm(centroid_mat - r, axis=1)
        scores = 1.0 / (dists + 1e-6)
        scores = scores * priors
        scores = scores / scores.sum()
        probs.append(scores)
    return np.array(probs)


def train_crime_model(clean_df: pd.DataFrame) -> PredictionArtifacts:
    model_df = build_model_frame(clean_df)
    y = derive_risk_labels(model_df)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(model_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.75)
    if len(idx) > 3:
        train_idx, test_idx = idx[:split], idx[split:]
    else:
        train_idx, test_idx = idx, idx

    X_train = model_df.iloc[train_idx][FEATURES]
    y_train = y.iloc[train_idx]
    X_test = model_df.iloc[test_idx][FEATURES]
    y_test = y.iloc[test_idx]

    model = _build_centroid_model(X_train, y_train)
    pred_probs = _predict_proba(model, X_test)
    pred_labels = [RISK_LEVELS[int(np.argmax(p))] for p in pred_probs]
    accuracy = float((pd.Series(pred_labels).values == y_test.values).mean())

    return PredictionArtifacts(model=model, accuracy=accuracy, classes=RISK_LEVELS)
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
    model = payload["model"]
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
    probs = _predict_proba(model, sample)[0]
    mapping = {lbl: float(prob) for lbl, prob in zip(RISK_LEVELS, probs)}
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
