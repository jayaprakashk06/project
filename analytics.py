"""Analytics helpers for Streamlit dashboard."""

from __future__ import annotations

import pandas as pd


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek
    return out


def crimes_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("crime_type", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )


def crimes_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("hour", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("hour")
    )


def crimes_by_location(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    out = df.copy()
    out["lat_round"] = out["latitude"].round(3)
    out["lon_round"] = out["longitude"].round(3)
    grouped = (
        out.groupby(["lat_round", "lon_round"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
    )
    return grouped.head(top_n)


def crime_trends(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("date")
    )
