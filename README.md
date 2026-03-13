# Crime Predictive Model & Hotspot Mapping Tool

This repository contains a practical baseline implementation for **urban crime hotspot prediction** using historical crime incident data.

The solution includes:
- A machine learning pipeline to estimate crime risk by location and time features.
- A hotspot scoring workflow for map grids.
- A map generator that visualizes high-risk areas using heatmaps.
- A small command-line interface (CLI) for end-to-end execution.

## Problem Framing

Law enforcement agencies need a data-driven way to answer:
1. **Where** are crimes likely to occur next?
2. **When** is risk elevated in each area?
3. Which zones should receive proactive patrol allocation?

This tool uses historical incident records to train a model and produce a hotspot map that can support operational planning.

## Input Data Schema

The model expects CSV data with these columns:

- `crime_type` (string)
- `timestamp` (ISO datetime string)
- `latitude` (float)
- `longitude` (float)

Optional columns are ignored by the baseline model, but can be incorporated later.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python crime_hotspot_model.py --input data/sample_crime_data.csv --output-map artifacts/hotspots.html
```

## Outputs

- `artifacts/hotspots.html`: Interactive hotspot heatmap.
- Console metrics for predictive performance.
- Optional CSV of scored grid cells (via `--output-grid`).

## Method Summary

1. Parse and clean historical records.
2. Engineer temporal and spatial features:
   - hour of day
   - day of week
   - month
   - spatial bins (`lat_bin`, `lon_bin`)
3. Build a binary target: whether a grid cell experiences an incident in a future time window.
4. Train a `RandomForestClassifier` baseline.
5. Score grid cells and visualize predicted risk as hotspots.

## Example Command

```bash
python crime_hotspot_model.py \
  --input data/sample_crime_data.csv \
  --output-map artifacts/hotspots.html \
  --output-grid artifacts/grid_scores.csv \
  --grid-size 0.01
```

## Notes for Real Deployments

- Replace sample data with city police incident exports.
- Tune time-window definitions for operational shifts.
- Add contextual features (events, weather, holidays, socioeconomic layers).
- Evaluate fairness, bias, and privacy impacts before deployment.
