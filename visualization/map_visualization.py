from __future__ import annotations

import folium
from folium.plugins import MarkerCluster

from visualization.heatmap_layer import add_heatmap_layer


def create_interactive_map(df, hotspots):
    center = [float(df["latitude"].mean()), float(df["longitude"].mean())]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB dark_matter")

    marker_cluster = MarkerCluster(name="Crime Markers").add_to(fmap)
    for _, row in df.head(2000).iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="#00bfff",
            fill=True,
            fill_opacity=0.5,
            popup=f"{row['crime_type']} | {row['district']} | {row['timestamp']}",
        ).add_to(marker_cluster)

    add_heatmap_layer(fmap, df)

    hotspot_cluster = MarkerCluster(name="Hotspot Zones").add_to(fmap)
    for _, row in hotspots.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=f"Cluster {int(row['cluster_id'])} | Crimes: {int(row['crime_count'])}",
        ).add_to(hotspot_cluster)

    folium.LayerControl().add_to(fmap)
    return fmap
