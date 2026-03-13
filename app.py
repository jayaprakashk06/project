from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from analytics.crime_statistics import (
    crimes_by_district,
    crimes_by_hour,
    crimes_by_type,
    overview_metrics,
)
from analytics.temporal_analysis import daily_crime_trend, forecast_next_7_days
from config import APP_TITLE, DEFAULT_DATA_PATH
from models.crime_prediction import (
    predict_crime_probability,
    save_prediction_model,
    train_crime_model,
)
from models.hotspot_clustering import detect_hotspot_clusters
from utils.feature_engineering import build_model_frame
from utils.preprocessing import clean_crime_data, generate_synthetic_tn_data
from visualization.map_visualization import create_interactive_map


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    .block-container { padding-top: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🛡️ AI Crime Intelligence System")
st.caption("Production-style analytics, hotspot detection, and risk prediction for Tamil Nadu crime data.")


@st.cache_data
def load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if DEFAULT_DATA_PATH.exists():
        return pd.read_csv(DEFAULT_DATA_PATH)
    return generate_synthetic_tn_data(rows=1800)


@st.cache_data
def get_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    return clean_crime_data(df)


@st.cache_resource
def train_and_cache_model(df: pd.DataFrame):
    artifacts = train_crime_model(df)
    model_path = save_prediction_model(artifacts)
    return artifacts, model_path


st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Crime Overview",
        "Interactive Crime Map",
        "Crime Risk Prediction Tool",
        "Crime Analytics Dashboard",
    ],
)

uploaded = st.sidebar.file_uploader("Upload crime dataset (CSV)", type=["csv"])

try:
    raw_df = load_dataset(uploaded)
    df = get_clean_data(raw_df)
except Exception as exc:
    st.error(f"Dataset error: {exc}")
    st.stop()

# Ensure optional columns for filtering/view
if "crime_id" not in df.columns:
    df = df.copy()
    df["crime_id"] = range(1, len(df) + 1)

crime_types = sorted(df["crime_type"].dropna().unique().tolist())
districts = sorted(df["district"].dropna().unique().tolist())

selected_types = st.sidebar.multiselect("Filter crime type", options=crime_types, default=crime_types)
selected_districts = st.sidebar.multiselect("Filter district", options=districts, default=districts)

filtered = df[df["crime_type"].isin(selected_types) & df["district"].isin(selected_districts)].copy()
if filtered.empty:
    st.warning("No records after filters. Please expand selection.")
    st.stop()

hotspots = detect_hotspot_clusters(filtered)
metrics = overview_metrics(filtered, hotspots)

if section == "Crime Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Crimes", f"{metrics['total_crimes']}")
    c2.metric("Most Common Crime", str(metrics["most_common_crime"]).title())
    c3.metric("Highest Risk District", str(metrics["highest_risk_district"]))
    c4.metric("Top Cluster ID", f"{metrics['top_cluster_id']}")

    st.subheader("Sample Records")
    st.dataframe(filtered.head(20), use_container_width=True)

elif section == "Interactive Crime Map":
    st.subheader("Crime Markers, Heatmap & Hotspot Zones")
    fmap = create_interactive_map(filtered, hotspots)
    st_folium(fmap, width=1200, height=620)

elif section == "Crime Risk Prediction Tool":
    st.subheader("Predict Crime Probability")
    model_df = build_model_frame(filtered)
    artifacts, model_path = train_and_cache_model(filtered)
    st.caption(f"Model accuracy: {artifacts.accuracy:.3f} | Saved model: `{model_path}`")

    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Latitude", value=float(filtered["latitude"].median()), format="%.6f")
    lon = c2.number_input("Longitude", value=float(filtered["longitude"].median()), format="%.6f")
    hour = c3.slider("Hour", 0, 23, 20)

    c4, c5, c6 = st.columns(3)
    day = c4.slider("Day of Week", 0, 6, 4)
    month = c5.slider("Month", 1, 12, int(pd.to_datetime(filtered["timestamp"]).dt.month.mode().iat[0]))
    district = c6.selectbox("District Context", options=districts)

    district_freq = float(model_df[model_df["district"] == district]["crime_frequency"].mean())
    if pd.isna(district_freq):
        district_freq = float(model_df["crime_frequency"].mean())

    if st.button("Predict Risk"):
        try:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid latitude/longitude values.")
            result = predict_crime_probability(
                latitude=lat,
                longitude=lon,
                hour=hour,
                day_of_week=day,
                month=month,
                crime_frequency=district_freq,
            )
            st.success(
                f"Predicted Risk: {str(result['risk_level']).upper()} | Probability: {float(result['crime_probability']):.2%}"
            )
            st.json(result)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

else:
    st.subheader("Interactive Analytics")

    district_df = crimes_by_district(filtered)
    type_df = crimes_by_type(filtered)
    hour_df = crimes_by_hour(filtered)
    trend_df = daily_crime_trend(filtered)
    forecast_df = forecast_next_7_days(filtered)

    fig_district = px.bar(district_df, x="district", y="count", title="Crime by District")
    fig_type = px.pie(type_df, names="crime_type", values="count", title="Crime by Type")
    fig_hour = px.line(hour_df, x="hour", y="count", markers=True, title="Crime by Hour")
    fig_trend = px.line(trend_df, x="date", y="count", title="Historical Crime Trend")
    fig_forecast = px.line(forecast_df, x="date", y="forecast_count", markers=True, title="Forecast: Next 7 Days")

    r1c1, r1c2 = st.columns(2)
    r1c1.plotly_chart(fig_district, use_container_width=True)
    r1c2.plotly_chart(fig_type, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    r2c1.plotly_chart(fig_hour, use_container_width=True)
    r2c2.plotly_chart(fig_trend, use_container_width=True)

    st.plotly_chart(fig_forecast, use_container_width=True)
