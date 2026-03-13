from __future__ import annotations

import numpy as np
import pandas as pd


def daily_crime_trend(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["timestamp"]).dt.date
    return out.groupby("date", as_index=False).size().rename(columns={"size": "count"}).sort_values("date")


def forecast_next_7_days(df: pd.DataFrame) -> pd.DataFrame:
    trend = daily_crime_trend(df)
    ts = trend.copy()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.set_index("date").asfreq("D").fillna(0)

    y = ts["count"].to_numpy(dtype=float)
    if len(y) < 3:
        pred = np.repeat(float(y.mean() if len(y) else 0.0), 7)
    else:
        x = np.arange(len(y), dtype=float)
        coef = np.polyfit(x, y, deg=1)
        future_x = np.arange(len(y), len(y) + 7, dtype=float)
        pred = np.polyval(coef, future_x)
        pred = np.clip(pred, 0, None)

    future_idx = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=7, freq="D")
    return pd.DataFrame({"date": future_idx, "forecast_count": pred})
