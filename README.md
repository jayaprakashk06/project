# AI Crime Intelligence System

An advanced Streamlit platform for crime analytics, hotspot intelligence, and AI-powered risk prediction.

## What’s New

This repository was upgraded from a basic hotspot demo into a modular, production-style architecture with:

- **AI Crime Risk Prediction** (RandomForestClassifier)
- **Hotspot Clustering** (DBSCAN)
- **Temporal Crime Forecasting** (ARIMA with fallback)
- **Interactive Plotly Analytics Dashboard**
- **Interactive Folium Map** (markers + heatmap + hotspot zones)
- **Dark-theme Streamlit UX** with sidebar navigation and filters
- **Dynamic synthetic Tamil Nadu data generation** when dataset is missing

## Project Structure

```text
project/
├── app.py
├── config.py
├── crime_hotspot_model.py
├── models/
│   ├── crime_prediction.py
│   └── hotspot_clustering.py
├── analytics/
│   ├── crime_statistics.py
│   └── temporal_analysis.py
├── visualization/
│   ├── map_visualization.py
│   └── heatmap_layer.py
├── utils/
│   ├── preprocessing.py
│   └── feature_engineering.py
├── dataset_generator.py
├── data/
│   └── sample_crime_data.csv
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Dataset Schema

Required columns:

- `crime_id`
- `crime_type`
- `timestamp`
- `latitude`
- `longitude`
- `district`

If no dataset is uploaded, the app loads `data/sample_crime_data.csv`.
If that file is missing, synthetic Tamil Nadu city-level crime data is generated automatically.

## AI Components

### 1) Crime Risk Prediction
- Model: lightweight centroid-probability classifier (dependency-stable fallback)
- Features: `latitude`, `longitude`, `hour`, `day_of_week`, `month`, `crime_frequency`
- Output: class probability and risk level (`low`, `medium`, `high`)

### 2) Hotspot Detection
- Method: grid-density clustering (DBSCAN-like behavior without heavy dependency)
- Input: `latitude`, `longitude`
- Output: high-density hotspot clusters

### 3) Temporal Forecast
- Method: lightweight trend extrapolation (NumPy linear forecast)
- Output: next 7-day forecast

## Deployment

The app is Streamlit Cloud compatible:

- Entry point: `app.py`
- Python dependencies: `requirements.txt`
- Default data path: `data/sample_crime_data.csv`


## Troubleshooting

### IndentationError in `crime_hotspot_model.py`
If Streamlit shows an error like:

- `IndentationError: expected an indented block after function definition ...`

Run these checks from project root:

```bash
python -m py_compile app.py crime_hotspot_model.py
pytest -q tests/test_syntax_smoke.py
```

If this fails on your machine, make sure you are running the **latest pulled code** from this repo branch and that local file edits did not introduce tab/space indentation mismatches.


### Streamlit Cloud still shows old error after fix
If Streamlit Cloud keeps showing an old traceback (for example an `IndentationError` from previous code):

1. Confirm deployment branch points to the latest commit.
2. In Streamlit Cloud, open **⋮ menu → Reboot app**.
3. If needed, **Clear cache** and redeploy.
4. Verify startup locally first:

```bash
python -m py_compile app.py crime_hotspot_model.py
pytest -q tests/test_syntax_smoke.py tests/test_runtime_imports.py
```


### Optional `plotly` fallback
If `plotly` is missing, the app now falls back to built-in Streamlit charts instead of crashing.
For full interactive charts, install dependencies:

```bash
pip install -r requirements.txt
```


### Streamlit Cloud dependency stability
This project now uses a lightweight dependency set to reduce deployment failures on Streamlit Cloud (especially around compiled scientific packages).
