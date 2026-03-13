"""Synthetic dataset generator for Tamil Nadu crime data."""

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
