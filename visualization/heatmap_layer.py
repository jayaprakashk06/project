from __future__ import annotations

from folium.plugins import HeatMap


def add_heatmap_layer(fmap, df):
    points = df[["latitude", "longitude"]].values.tolist()
    HeatMap(points, radius=10, blur=12, min_opacity=0.25, name="Crime Heatmap").add_to(fmap)
    return fmap
