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

    if ts.empty:
        future_idx = pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=7, freq="D")
        return pd.DataFrame({"date": future_idx, "forecast_count": np.zeros(7)})

    series = ts["count"].astype(float)
    smoothed = series.rolling(window=7, min_periods=1).mean()

    # Linear trend on smoothed series.
    x = np.arange(len(smoothed), dtype=float)
    if len(smoothed) >= 3:
        slope, intercept = np.polyfit(x, smoothed.to_numpy(), deg=1)
    else:
        slope, intercept = 0.0, float(smoothed.iloc[-1])

    # Weekday seasonality factor from recent data (up to last 8 weeks).
    hist = series.tail(56).copy()
    hist_df = hist.to_frame(name="count")
    hist_df["dow"] = hist_df.index.dayofweek
    overall = float(hist_df["count"].mean()) if len(hist_df) else 1.0
    dow_factor = hist_df.groupby("dow")["count"].mean() / (overall if overall > 0 else 1.0)

    last_idx = len(smoothed) - 1
    future_idx = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=7, freq="D")
    preds: list[float] = []
    for step, dt in enumerate(future_idx, start=1):
        base = intercept + slope * (last_idx + step)
        factor = float(dow_factor.get(dt.dayofweek, 1.0))
        pred = max(0.0, base * factor)
        preds.append(pred)

    # Keep values realistic and smooth.
    pred_arr = np.array(preds, dtype=float)
    pred_arr = np.clip(pred_arr, 0.0, max(1.0, float(series.max()) * 1.5))

    return pd.DataFrame({"date": future_idx, "forecast_count": pred_arr})
