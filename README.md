# Crime Hotspot AI

A Streamlit-based AI web app for urban crime analytics, hotspot mapping, and crime risk prediction using historical incident data.

## Features

- **Dataset Upload**: Upload CSV crime data; if not provided, a sample dataset is loaded automatically.
- **Crime Analytics Dashboard**:
  - crimes by type
  - crimes by hour
  - crimes by location
  - crime trend over time
- **Hotspot Map**:
  - incident markers
  - heatmap layer
  - hotspot clusters
- **Crime Risk Prediction**:
  - inputs: latitude, longitude, hour, day_of_week
  - output: risk score + label (**LOW / MEDIUM / HIGH**)

## Project Structure

```text
crime_hotspot_ai/
├── app.py
├── crime_hotspot_model.py
├── dataset_generator.py
├── map_visualization.py
├── analytics.py
├── requirements.txt
├── README.md
└── data/
    └── sample_crime_data.csv
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Dataset Format

CSV file must include:

- `crime_type` (string)
- `timestamp` (datetime string)
- `latitude` (float)
- `longitude` (float)

### Example

```csv
crime_type,timestamp,latitude,longitude
theft,2024-03-24T14:00:00,40.761776,-73.982323
assault,2024-01-30T16:00:00,40.761942,-73.982541
```

## Notes

- The backend ML model is `RandomForestClassifier`.
- Model features: `hour`, `day_of_week`, `latitude`, `longitude`.
- If uploaded data is invalid (missing columns, bad coordinates), the app shows a friendly error.
