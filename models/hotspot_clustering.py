from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN


def detect_hotspot_clusters(df: pd.DataFrame, eps: float = 0.03, min_samples: int = 12) -> pd.DataFrame:
    coords = df[["latitude", "longitude"]].copy()
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clusterer.fit_predict(coords)

    out = df.copy()
    out["cluster_id"] = labels

    clustered = out[out["cluster_id"] != -1].copy()
    if clustered.empty:
        return pd.DataFrame(columns=["cluster_id", "latitude", "longitude", "crime_count"])

    hotspots = (
        clustered.groupby("cluster_id", as_index=False)
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), crime_count=("cluster_id", "size"))
        .sort_values("crime_count", ascending=False)
    )
    return hotspots
