from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SafeCity AI", layout="wide")

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

try:
    raw_df = load_data(uploaded_file)
    df = validate_dataframe(raw_df)
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
