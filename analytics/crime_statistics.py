from __future__ import annotations

import pandas as pd


def overview_metrics(df: pd.DataFrame, hotspots: pd.DataFrame) -> dict[str, str | int]:
    total = len(df)
    top_type = str(df["crime_type"].mode().iat[0]) if not df.empty else "N/A"
    district_counts = df.groupby("district").size().sort_values(ascending=False)
    top_district = str(district_counts.index[0]) if not district_counts.empty else "N/A"
    top_cluster = int(hotspots.iloc[0]["cluster_id"]) if not hotspots.empty else -1
    return {
        "total_crimes": total,
        "most_common_crime": top_type,
        "highest_risk_district": top_district,
        "top_cluster_id": top_cluster,
    }


def crimes_by_district(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("district", as_index=False).size().rename(columns={"size": "count"})


def crimes_by_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("crime_type", as_index=False).size().rename(columns={"size": "count"})


def crimes_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = pd.to_datetime(out["timestamp"]).dt.hour
    return out.groupby("hour", as_index=False).size().rename(columns={"size": "count"})
