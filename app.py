import pandas as pd
import numpy as np

import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SafeCity AI", layout="wide")
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False


REQUIRED_COLUMNS = {
    "district",
    "latitude",
    "longitude",
    "hour",
    "day",
    "month",
    "crime_type",
    "risk",
}


def build_default_dataset() -> pd.DataFrame:
    data = {
        "district": [
            "Chennai",
            "Chennai",
            "Madurai",
            "Coimbatore",
            "Salem",
            "Trichy",
            "Chennai",
            "Madurai",
            "Coimbatore",
            "Salem",
        ],
        "latitude": [13.08, 13.05, 9.92, 11.01, 11.66, 10.79, 13.10, 9.90, 11.00, 11.70],
        "longitude": [80.27, 80.29, 78.11, 76.95, 78.14, 78.70, 80.25, 78.10, 76.97, 78.10],
        "hour": [22, 18, 21, 14, 19, 20, 23, 17, 15, 21],
        "day": [5, 3, 6, 2, 4, 6, 5, 2, 3, 4],
        "month": [12, 11, 10, 8, 9, 7, 6, 5, 4, 3],
        "crime_type": [
            "Theft",
            "Robbery",
            "Assault",
            "Theft",
            "Burglary",
            "Robbery",
            "Theft",
            "Assault",
            "Burglary",
            "Theft",
        ],
        "risk": [1, 0, 1, 0, 1, 1, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out["day"] = pd.to_numeric(out["day"], errors="coerce")
    out["month"] = pd.to_numeric(out["month"], errors="coerce")
    out["risk"] = pd.to_numeric(out["risk"], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude", "hour", "day", "month", "risk", "district", "crime_type"])
    out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
    out = out[(out["hour"].between(0, 23)) & (out["day"].between(1, 7)) & (out["month"].between(1, 12))]
    out["risk"] = out["risk"].astype(int)
    out = out[out["risk"].isin([0, 1])]

    if out.empty:
        raise ValueError("No valid rows remain after cleaning uploaded dataset.")
    return out.reset_index(drop=True)


@st.cache_data
def load_data(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return build_default_dataset()
    return pd.read_csv(uploaded)


@st.cache_resource
def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[["latitude", "longitude", "hour", "day", "month"]]
    y = df["risk"]
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)
    return model


st.title("SafeCity AI: Intelligent Crime Hotspot Mapping System")
st.caption("Reliable Streamlit demo for crime risk prediction and district analytics.")

if not PLOTLY_AVAILABLE:
    st.warning("`plotly` is not installed. Falling back to native Streamlit charts.")

uploaded_file = st.sidebar.file_uploader("Upload crime dataset (CSV)", type=["csv"])



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
if not PLOTLY_AVAILABLE:
    st.warning("Optional dependency `plotly` is not installed. Falling back to built-in Streamlit charts. Run `pip install -r requirements.txt` for full interactive charts.")


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

model = train_model(df)

st.header("Predict Crime Probability")
col1, col2, col3 = st.columns(3)

with col1:
    lat = st.number_input("Latitude", value=float(df["latitude"].median()))
    day = st.slider("Day of Week", 1, 7, int(df["day"].mode().iat[0]))

with col2:
    lon = st.number_input("Longitude", value=float(df["longitude"].median()))
    month = st.slider("Month", 1, 12, int(df["month"].mode().iat[0]))

with col3:
    hour = st.slider("Hour", 0, 23, int(df["hour"].mode().iat[0]))
    district = st.selectbox("District", sorted(df["district"].unique()))

if st.button("Predict Risk"):
    input_data = np.array([[lat, lon, hour, day, month]])
    proba_high = float(model.predict_proba(input_data)[0][1])
    prediction = int(model.predict(input_data)[0])

    st.metric("Predicted High-Risk Probability", f"{proba_high:.2%}")
    if prediction == 1:
        st.error("High Crime Risk Area")
    else:
        st.success("Low Crime Risk Area")

st.header("Crime Dashboard")

# Aggregations
filtered = df[df["district"] == district].copy()
if filtered.empty:
    filtered = df

district_count = df["district"].value_counts().reset_index()
district_count.columns = ["district", "count"]

crime_type_count = filtered["crime_type"].value_counts().reset_index()
crime_type_count.columns = ["crime_type", "count"]

month_count = df.groupby("month").size().reset_index(name="count")
hour_count = df.groupby("hour").size().reset_index(name="count")

c1, c2 = st.columns(2)

if PLOTLY_AVAILABLE:
    fig_district = px.bar(district_count, x="district", y="count", title="Crime Count by District")
    fig_type = px.pie(crime_type_count, names="crime_type", values="count", title="Crime Type Distribution")
    c1.plotly_chart(fig_district, use_container_width=True, key="district_chart")
    c2.plotly_chart(fig_type, use_container_width=True, key="type_chart")

    c3, c4 = st.columns(2)
    fig_month = px.line(month_count, x="month", y="count", title="Crime Trend by Month")
    fig_hour = px.bar(hour_count, x="hour", y="count", title="Crime Occurrence by Hour")
    c3.plotly_chart(fig_month, use_container_width=True, key="month_chart")
    c4.plotly_chart(fig_hour, use_container_width=True, key="hour_chart")
else:
    c1.subheader("Crime Count by District")
    c1.bar_chart(district_count.set_index("district")["count"])
    c2.subheader("Crime Type Distribution")
    c2.bar_chart(crime_type_count.set_index("crime_type")["count"])

    c3, c4 = st.columns(2)
    c3.subheader("Crime Trend by Month")
    c3.line_chart(month_count.set_index("month")["count"])
    c4.subheader("Crime Occurrence by Hour")
    c4.bar_chart(hour_count.set_index("hour")["count"])
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

    if PLOTLY_AVAILABLE:
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
    else:
        r1c1, r1c2 = st.columns(2)
        r1c1.subheader("Crime by District")
        r1c1.bar_chart(district_df.set_index("district")["count"])
        r1c2.subheader("Crime by Type")
        r1c2.bar_chart(type_df.set_index("crime_type")["count"])

        r2c1, r2c2 = st.columns(2)
        r2c1.subheader("Crime by Hour")
        r2c1.line_chart(hour_df.set_index("hour")["count"])
        r2c2.subheader("Historical Crime Trend")
        trend_plot = trend_df.copy()
        trend_plot["date"] = pd.to_datetime(trend_plot["date"])
        r2c2.line_chart(trend_plot.set_index("date")["count"])

        st.subheader("Forecast: Next 7 Days")
        forecast_plot = forecast_df.copy()
        forecast_plot["date"] = pd.to_datetime(forecast_plot["date"])
        st.line_chart(forecast_plot.set_index("date")["forecast_count"])
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
    predict_crime_risk,
    save_model,
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
    saved_model_path = save_model(artifacts)
    st.success(f"Model trained. Validation accuracy: {artifacts.auc:.4f}")
    st.caption(f"Trained model saved to `{saved_model_path}`")
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

        result = predict_crime_risk(
            latitude=pred_lat,
            longitude=pred_lon,
            hour=pred_hour,
            day=pred_dow,
            model_path="models/crime_risk_model.joblib",
        )
        label = str(result["crime_risk"]).upper()
        prob = float(result["prediction_probability"])

        if label == "HIGH":
            st.error(f"Predicted Risk: {label} ({prob:.2%})")
        elif label == "MEDIUM":
            st.warning(f"Predicted Risk: {label} ({prob:.2%})")
        else:
            st.success(f"Predicted Risk: {label} ({prob:.2%})")

        st.write(
            {
                "low_probability": round(float(result["low_probability"]), 4),
                "medium_probability": round(float(result["medium_probability"]), 4),
                "high_probability": round(float(result["high_probability"]), 4),
            }
        )
    except Exception as exc:
        st.error(f"Prediction error: {exc}")
