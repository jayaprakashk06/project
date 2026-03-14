from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import TN_CITY_CENTERS


CRIME_TYPES = ["theft", "burglary", "assault", "vandalism", "robbery", "cybercrime"]
REQUIRED_BASE_COLUMNS = {"crime_type", "timestamp", "latitude", "longitude"}


def _nearest_district(latitude: float, longitude: float) -> str:
    best_name = "Unknown"
    best_dist = float("inf")
    for name, (lat_c, lon_c) in TN_CITY_CENTERS.items():
        d = (latitude - lat_c) ** 2 + (longitude - lon_c) ** 2
        if d < best_dist:
            best_dist = d
            best_name = name
    return best_name


def validate_crime_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_BASE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean_crime_data(df: pd.DataFrame) -> pd.DataFrame:
    validate_crime_data(df)
    out = df.copy()

    out = out.dropna(subset=list(REQUIRED_BASE_COLUMNS))
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["timestamp", "latitude", "longitude"])
    out = out[(out["latitude"].between(-90, 90)) & (out["longitude"].between(-180, 180))]

    if "district" not in out.columns:
        out["district"] = [
            _nearest_district(float(lat), float(lon)) for lat, lon in zip(out["latitude"], out["longitude"])
        ]
    else:
        out["district"] = out["district"].fillna("").astype(str).str.strip()
        missing_mask = out["district"] == ""
        if missing_mask.any():
            out.loc[missing_mask, "district"] = [
                _nearest_district(float(lat), float(lon))
                for lat, lon in zip(out.loc[missing_mask, "latitude"], out.loc[missing_mask, "longitude"])
            ]

    if "crime_id" not in out.columns:
        out["crime_id"] = np.arange(1, len(out) + 1)

    if out.empty:
        raise ValueError("No valid records remain after cleaning.")

    return out.sort_values("timestamp").reset_index(drop=True)


def generate_synthetic_tn_data(rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    base = datetime.now() - timedelta(days=180)
    districts = list(TN_CITY_CENTERS.keys())

    result = []
    for i in range(rows):
        district = random.choice(districts)
        lat_c, lon_c = TN_CITY_CENTERS[district]
        spread = 0.02 if district == "Chennai" else 0.03

        lat = lat_c + random.gauss(0, spread)
        lon = lon_c + random.gauss(0, spread)
        timestamp = base + timedelta(hours=random.randint(0, 24 * 180))
        crime_type = random.choices(CRIME_TYPES, weights=[0.3, 0.16, 0.18, 0.12, 0.14, 0.10], k=1)[0]

        result.append(
            {
                "crime_id": i + 1,
                "crime_type": crime_type,
                "timestamp": timestamp.isoformat(),
                "latitude": lat,
                "longitude": lon,
                "district": district,
            }
        )

    return pd.DataFrame(result)
