from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["month"] = out["timestamp"].dt.month
    return out


def add_crime_frequency(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    freq = (
        out.groupby(["district", "crime_type"]).size().rename("crime_frequency").reset_index()
    )
    out = out.merge(freq, on=["district", "crime_type"], how="left")
    out["crime_frequency"] = out["crime_frequency"].fillna(1)
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = add_temporal_features(df)
    out = add_crime_frequency(out)
    return out
