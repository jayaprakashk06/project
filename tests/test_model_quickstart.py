from __future__ import annotations

from pathlib import Path

from models.crime_prediction import predict_risk, train_demo_model_for_quick_start


def test_demo_model_train_and_predict(tmp_path: Path) -> None:
    model_path = tmp_path / "crime_risk_model.joblib"
    train_demo_model_for_quick_start(model_path=model_path)
    label = predict_risk(
        latitude=13.0827,
        longitude=80.2707,
        hour=22,
        day=5,
        month=12,
        model_path=model_path,
    )
    assert label in {"High Crime Risk", "Low Crime Risk"}
