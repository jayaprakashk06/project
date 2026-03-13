from __future__ import annotations

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

    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(ts["count"], order=(1, 1, 1))
        fitted = model.fit()
        forecast = fitted.forecast(steps=7)
    except Exception:
        last_mean = float(ts["count"].tail(14).mean()) if len(ts) else 0.0
        forecast = pd.Series([last_mean] * 7)

    future_idx = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=7, freq="D")
    return pd.DataFrame({"date": future_idx, "forecast_count": forecast.values})
