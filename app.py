from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="SafeCity AI", page_icon="🛡️", layout="wide")

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

FEATURE_COLUMNS = ["latitude", "longitude", "hour", "day", "month"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at 20% 10%, #111827 0%, #060b16 45%, #04070f 100%);
            }
            .main .block-container {
                max-width: 1300px;
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }
            .hero {
                padding: 1.2rem 1.4rem;
                border-radius: 16px;
                background: linear-gradient(135deg, rgba(37,99,235,0.23), rgba(14,165,233,0.15));
                border: 1px solid rgba(148,163,184,0.25);
                box-shadow: 0 8px 26px rgba(15,23,42,0.35);
                margin-bottom: 1rem;
            }
            .hero h1 {
                margin: 0;
                font-size: 2.1rem;
                color: #f8fafc;
                letter-spacing: 0.2px;
            }
            .hero p {
                margin: .35rem 0 0;
                color: #cbd5e1;
                font-size: 1rem;
            }
            .card {
                border: 1px solid rgba(148,163,184,0.22);
                background: linear-gradient(180deg, rgba(15,23,42,0.68), rgba(10,14,26,0.72));
                border-radius: 14px;
                padding: .9rem 1rem;
                box-shadow: 0 8px 20px rgba(2,6,23,0.35);
            }
            .card .label {
                color: #93c5fd;
                font-size: .8rem;
                text-transform: uppercase;
                letter-spacing: .6px;
                margin-bottom: .2rem;
            }
            .card .value {
                color: #f8fafc;
                font-size: 1.35rem;
                font-weight: 700;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1120 0%, #090f1c 100%);
                border-right: 1px solid rgba(148,163,184,0.18);
            }
            .pill {
                display:inline-block;
                padding: .2rem .55rem;
                border-radius: 999px;
                background: rgba(14,165,233,0.18);
                border: 1px solid rgba(56,189,248,0.35);
                color: #bae6fd;
                font-size: .78rem;
                margin-right: .35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    numeric_cols = ["latitude", "longitude", "hour", "day", "month", "risk"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["district"] = out["district"].astype(str).str.strip()
    out["crime_type"] = out["crime_type"].astype(str).str.strip()

    out = out.dropna(subset=numeric_cols + ["district", "crime_type"])
    out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]
    out = out[(out["hour"].between(0, 23)) & (out["day"].between(1, 7)) & (out["month"].between(1, 12))]
    out["risk"] = out["risk"].astype(int)
    out = out[out["risk"].isin([0, 1])]
    out = out[(out["district"] != "") & (out["crime_type"] != "")]

    if out.empty:
        raise ValueError("No valid rows remain after cleaning uploaded dataset.")

    return out.reset_index(drop=True)


@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return build_default_dataset()
    return pd.read_csv(uploaded_file)


@st.cache_resource
def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    y = df["risk"]
    if y.nunique() < 2:
        raise ValueError("Uploaded dataset must contain both risk classes (0 and 1).")

    X = df[FEATURE_COLUMNS]
    model = RandomForestClassifier(n_estimators=250, random_state=42)
    model.fit(X, y)
    return model


def predict_with_probability(
    model: RandomForestClassifier,
    latitude: float,
    longitude: float,
    hour: int,
    day: int,
    month: int,
) -> tuple[int, float]:
    input_df = pd.DataFrame(
        [
            {
                "latitude": latitude,
                "longitude": longitude,
                "hour": hour,
                "day": day,
                "month": month,
            }
        ]
    )
    pred = int(model.predict(input_df)[0])

    classes = list(model.classes_)
    probs = model.predict_proba(input_df)[0]
    prob_high = float(probs[classes.index(1)]) if 1 in classes else (1.0 if pred == 1 else 0.0)
    return pred, prob_high


def render_dashboard(df: pd.DataFrame, district: str) -> None:
    filtered = df[df["district"] == district].copy()
    if filtered.empty:
        filtered = df

    district_count = df["district"].value_counts().reset_index()
    district_count.columns = ["district", "count"]

    crime_type_count = filtered["crime_type"].value_counts().reset_index()
    crime_type_count.columns = ["crime_type", "count"]

    month_count = df.groupby("month", as_index=False).size().rename(columns={"size": "count"})
    hour_count = df.groupby("hour", as_index=False).size().rename(columns={"size": "count"})

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🕒 Time Analysis", "🧾 Data Preview"])

    with tab1:
        c1, c2 = st.columns(2)
        if PLOTLY_AVAILABLE:
            fig_district = px.bar(district_count, x="district", y="count", title="Crime Count by District")
            fig_type = px.pie(crime_type_count, names="crime_type", values="count", title="Crime Type Distribution")
            c1.plotly_chart(fig_district, use_container_width=True, key="district_chart")
            c2.plotly_chart(fig_type, use_container_width=True, key="type_chart")
        else:
            c1.subheader("Crime Count by District")
            c1.bar_chart(district_count.set_index("district")["count"])
            c2.subheader("Crime Type Distribution")
            c2.bar_chart(crime_type_count.set_index("crime_type")["count"])

    with tab2:
        c3, c4 = st.columns(2)
        if PLOTLY_AVAILABLE:
            fig_month = px.line(month_count, x="month", y="count", title="Crime Trend by Month")
            fig_hour = px.bar(hour_count, x="hour", y="count", title="Crime Occurrence by Hour")
            c3.plotly_chart(fig_month, use_container_width=True, key="month_chart")
            c4.plotly_chart(fig_hour, use_container_width=True, key="hour_chart")
        else:
            c3.subheader("Crime Trend by Month")
            c3.line_chart(month_count.set_index("month")["count"])
            c4.subheader("Crime Occurrence by Hour")
            c4.bar_chart(hour_count.set_index("hour")["count"])

    with tab3:
        st.dataframe(filtered.head(50), use_container_width=True)


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ SafeCity AI Intelligence Console</h1>
            <p>High-fidelity crime risk prediction dashboard with clean analytics workflow and interactive controls.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not PLOTLY_AVAILABLE:
        st.warning("`plotly` is not installed. Falling back to native Streamlit charts.")

    st.sidebar.markdown("## ⚙️ Control Center")
    uploaded_file = st.sidebar.file_uploader("Upload crime dataset (CSV)", type=["csv"])
    realtime_mode = st.sidebar.toggle("Realtime monitor mode", value=True)
    show_raw = st.sidebar.toggle("Show raw source dataframe", value=False)

    try:
        raw_df = load_data(uploaded_file)
        df = validate_dataframe(raw_df)
        model = train_model(df)
    except Exception as exc:
        st.error(f"Startup error: {exc}")
        st.stop()

    # Top metric row
    high_risk_rate = (df["risk"] == 1).mean() * 100
    top_district = df["district"].value_counts().index[0]
    top_crime = df["crime_type"].value_counts().index[0]

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="card"><div class="label">Total Records</div><div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="card"><div class="label">High Risk Rate</div><div class="value">{high_risk_rate:.1f}%</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="card"><div class="label">Top District</div><div class="value">{top_district}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="card"><div class="label">Top Crime Type</div><div class="value">{top_crime}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1.8])

    with left:
        st.subheader("🎯 Predict Crime Probability")
        lat = st.number_input("Latitude", value=float(df["latitude"].median()), format="%.6f")
        lon = st.number_input("Longitude", value=float(df["longitude"].median()), format="%.6f")
        hour = st.slider("Hour", 0, 23, int(df["hour"].mode().iat[0]))
        day = st.slider("Day of Week", 1, 7, int(df["day"].mode().iat[0]))
        month = st.slider("Month", 1, 12, int(df["month"].mode().iat[0]))
        district = st.selectbox("District Focus", sorted(df["district"].unique()))

        cta1, cta2 = st.columns(2)
        predict_clicked = cta1.button("🚨 Predict Risk", use_container_width=True)
        cta2.button("🔄 Refresh Stats", use_container_width=True)

        st.markdown('<span class="pill">Realtime: {}</span><span class="pill">Model: RandomForest</span>'.format("ON" if realtime_mode else "OFF"), unsafe_allow_html=True)

        if predict_clicked:
            pred, prob_high = predict_with_probability(
                model=model,
                latitude=lat,
                longitude=lon,
                hour=hour,
                day=day,
                month=month,
            )
            st.metric("Predicted High-Risk Probability", f"{prob_high:.2%}")
            if pred == 1:
                st.error("High Crime Risk Area")
            else:
                st.success("Low Crime Risk Area")

    with right:
        st.subheader("📈 Crime Intelligence Dashboard")
        render_dashboard(df, district)

    if show_raw:
        with st.expander("Raw uploaded/loaded dataframe", expanded=False):
            st.dataframe(raw_df.head(100), use_container_width=True)


if __name__ == "__main__":
    main()
