import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Crime Hotspot Prediction")

st.title("🚓 AI Crime Hotspot Prediction System")

st.write("Upload crime dataset to visualize hotspots")

uploaded_file = st.file_uploader("Upload Crime Dataset (CSV)", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    if "latitude" in df.columns and "longitude" in df.columns:

        st.subheader("Crime Hotspot Map")

        m = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()],
            zoom_start=12
        )

        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                color="red",
                fill=True
            ).add_to(m)

        st_folium(m, width=700, height=500)

    else:
        st.error("Dataset must contain latitude and longitude columns")
