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

- `crime_type`
- `timestamp`
- `latitude`
- `longitude`

Optional but recommended:
- `crime_id`
- `district` (auto-derived from nearest Tamil Nadu city if missing)

If no dataset is uploaded, the app loads `data/sample_crime_data.csv`.
If that file is missing, synthetic Tamil Nadu city-level crime data is generated automatically.

## AI Components

### 1) Crime Risk Prediction
- Model: RandomForestClassifier
- Features: `latitude`, `longitude`, `hour`, `day_of_week`, `month`, `crime_frequency`
- Output: class probability and risk level (`low`, `medium`, `high`)

### 2) Hotspot Detection
- Method: grid-density clustering (DBSCAN-like behavior without heavy dependency)
- Input: `latitude`, `longitude`
- Output: high-density hotspot clusters

### 3) Temporal Forecast
- Method: trend + weekly seasonality forecast (NumPy)
- Output: next 7-day forecast with day-of-week variation

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


### Unterminated triple-quoted string error
If you see `SyntaxError: unterminated triple-quoted string literal` in `crime_hotspot_model.py` on your local machine:

1. Delete local stale bytecode and restart from a clean pull.

   **macOS/Linux (bash):**
```bash
find . -name "__pycache__" -type d -prune -exec rm -rf {} +
git fetch --all
git reset --hard origin/main
```

   **Windows PowerShell:**
```powershell
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
git fetch --all
git reset --hard origin/main
```

   **Windows CMD:**
```cmd
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
git fetch --all
git reset --hard origin/main
```

2. Run verification before `streamlit run app.py`:
```bash
pytest -q tests/test_syntax_smoke.py tests/test_runtime_imports.py
```
3. Ensure you are not launching an old extracted folder copy (e.g. `project-main (6)`) with stale files.
