from __future__ import annotations

from pathlib import Path

from utils.preprocessing import generate_synthetic_tn_data


def generate_sample_dataset(output_path: str = "data/sample_crime_data.csv", rows: int = 1800) -> None:
    df = generate_synthetic_tn_data(rows=rows)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    generate_sample_dataset()
"""Generate a synthetic crime dataset for demo/testing."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path


def generate_sample_dataset(output_path: str, rows: int = 1000, seed: int = 42) -> None:
    random.seed(seed)

    crime_types = ["theft", "burglary", "assault", "vandalism", "robbery"]
    weights = [0.36, 0.2, 0.2, 0.14, 0.1]
    clusters = [
        (40.7128, -74.0060, 0.004),
        (40.7306, -73.9352, 0.005),
        (40.7580, -73.9855, 0.003),
    ]
    start = datetime(2024, 1, 1)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["crime_type", "timestamp", "latitude", "longitude"])

        for _ in range(rows):
            lat_c, lon_c, spread = random.choice(clusters)
            lat = lat_c + random.gauss(0, spread)
            lon = lon_c + random.gauss(0, spread)
            ts = start + timedelta(hours=random.randint(0, 24 * 180))

            crime_type = random.choices(crime_types, weights=weights, k=1)[0]
            if crime_type == "burglary":
                ts = ts.replace(hour=random.choice([20, 21, 22, 23, 0, 1, 2]))

            writer.writerow([crime_type, ts.isoformat(), f"{lat:.6f}", f"{lon:.6f}"])


if __name__ == "__main__":
    generate_sample_dataset("data/sample_crime_data.csv")
