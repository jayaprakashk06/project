from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import analytics
from crime_hotspot_model import (
    REQUIRED_COLUMNS,
    build_training_frame,
    clean_crime_data,
    engineer_features,
    predict_risk_score,
    risk_label,
    score_hotspots,
    train_model,
)
from map_visualization import create_crime_hotspot_map


st.set_page_config(page_title="Crime Hotspot AI", layout="wide")
st.title("🚓 Crime Hotspot AI Dashboard")
st.caption("Upload historical crime data, analyze trends, map hotspots, and predict risk.")


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    sample_path = Path("data/sample_crime_data.csv")
    if not sample_path.exists():
        raise FileNotFoundError("Sample dataset missing at data/sample_crime_data.csv")
    return pd.read_csv(sample_path)


def check_optional_dependencies() -> list[str]:
    missing = []
    for pkg in ["matplotlib", "seaborn", "numpy", "folium", "streamlit_folium", "sklearn"]:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    return missing


missing_deps = check_optional_dependencies()
if missing_deps:
    st.warning(
        "Some optional dependencies are missing: "
        + ", ".join(missing_deps)
        + ". Core features may still work, but install requirements for full support."
    )

st.header("1) Dataset Upload")
uploaded_file = st.file_uploader("Upload crime CSV", type=["csv"])

try:
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.success("Using uploaded dataset.")
    else:
        raw_df = load_sample_data()
        st.info("No upload detected. Loaded sample dataset automatically.")

    df = clean_crime_data(raw_df)
except FileNotFoundError as exc:
    st.error(f"Dataset error: {exc}")
    st.stop()
except ValueError as exc:
    st.error(f"Data validation error: {exc}")
    st.write(f"Required columns: {sorted(REQUIRED_COLUMNS)}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected dataset error: {exc}")
    st.stop()

st.write(f"Records loaded: **{len(df)}**")
st.dataframe(df.head(10), use_container_width=True)

st.header("2) Crime Analytics")
analytics_df = analytics.add_time_columns(df)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Crimes by Type")
    by_type = analytics.crimes_by_type(analytics_df)
    st.bar_chart(by_type.set_index("crime_type")["count"])

with col2:
    st.subheader("Crimes by Hour")
    by_hour = analytics.crimes_by_hour(analytics_df)
    st.line_chart(by_hour.set_index("hour")["count"])

col3, col4 = st.columns(2)
with col3:
    st.subheader("Crimes by Location (Top Clusters)")
    by_loc = analytics.crimes_by_location(analytics_df)
    st.dataframe(by_loc, use_container_width=True)

with col4:
    st.subheader("Crime Trends Over Time")
    trend_df = analytics.crime_trends(analytics_df)
    trend_plot = trend_df.copy()
    trend_plot["date"] = pd.to_datetime(trend_plot["date"])
    st.line_chart(trend_plot.set_index("date")["count"])

st.header("3) Hotspot Map")
try:
    feature_df = engineer_features(df)
    training_df = build_training_frame(feature_df)
    artifacts = train_model(training_df)
    hotspot_df = score_hotspots(artifacts, feature_df)

    fmap = create_crime_hotspot_map(feature_df, hotspot_df)
    st_folium(fmap, width=1200, height=550)
    st.success(f"Model trained. Validation AUC: {artifacts.auc:.4f}")
except ValueError as exc:
    st.error(f"Model/map error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Unexpected modeling error: {exc}")
    st.stop()

st.header("4) Crime Risk Prediction")

pcol1, pcol2, pcol3, pcol4 = st.columns(4)
with pcol1:
    pred_lat = st.number_input("Latitude", value=float(df["latitude"].median()), format="%.6f")
with pcol2:
    pred_lon = st.number_input("Longitude", value=float(df["longitude"].median()), format="%.6f")
with pcol3:
    pred_hour = st.slider("Hour", 0, 23, 20)
with pcol4:
    pred_dow = st.slider("Day of Week (0=Mon)", 0, 6, 4)

if st.button("Predict Crime Risk"):
    try:
        if not (-90 <= pred_lat <= 90 and -180 <= pred_lon <= 180):
            raise ValueError("Coordinates are out of valid range.")

        score = predict_risk_score(artifacts, pred_lat, pred_lon, pred_hour, pred_dow)
        label = risk_label(score)

        if label == "HIGH":
            st.error(f"Predicted Risk: {label} ({score:.2%})")
        elif label == "MEDIUM":
            st.warning(f"Predicted Risk: {label} ({score:.2%})")
        else:
            st.success(f"Predicted Risk: {label} ({score:.2%})")
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
