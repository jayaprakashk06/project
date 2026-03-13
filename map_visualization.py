"""Folium map utilities for crime hotspot app."""

from __future__ import annotations

import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster


def create_crime_hotspot_map(crime_df: pd.DataFrame, hotspot_df: pd.DataFrame) -> folium.Map:
    center_lat = float(crime_df["latitude"].mean())
    center_lon = float(crime_df["longitude"].mean())

    crime_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron")

    marker_cluster = MarkerCluster(name="Crime Markers").add_to(crime_map)
    for _, row in crime_df.head(1500).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.35,
            popup=f"{row['crime_type']} | {row['timestamp']}",
        ).add_to(marker_cluster)

    heat_data = crime_df[["latitude", "longitude"]].values.tolist()
    HeatMap(heat_data, radius=10, blur=12, name="Crime Heatmap").add_to(crime_map)

    hotspot_cluster = MarkerCluster(name="Hotspot Clusters").add_to(crime_map)
    for _, row in hotspot_df.head(300).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color="red",
            fill=True,
            fill_opacity=0.7,
            popup=f"Risk score: {row['risk_score']:.3f}",
        ).add_to(hotspot_cluster)

    folium.LayerControl().add_to(crime_map)
    return crime_map
