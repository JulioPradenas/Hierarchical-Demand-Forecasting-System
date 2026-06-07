from __future__ import annotations

import math

import pandas as pd

from .base import BaseForecaster


class SeasonalNaiveForecaster(BaseForecaster):
    """Predicts y_hat(t+h) = y(t + h - m * ceil(h/m)) where m = seasonality.

    For daily data with seasonality=7 and horizon h=1..28:
    - The lookback into training history = m * ceil(h/m) - h steps from end.
    - h=7  → lookback=0 → last training value (same weekday 1 week back)
    - h=28 → lookback=0 → last training value (same weekday 4 weeks back)
    """

    def __init__(self, seasonality: int = 7) -> None:
        self.seasonality = seasonality
        self._history: dict[str, list[float]] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame) -> SeasonalNaiveForecaster:
        self._history = {}
        self._last_dates = {}
        for uid, grp in df.groupby("unique_id"):
            sorted_grp = grp.sort_values("ds")
            self._history[str(uid)] = sorted_grp["y"].tolist()
            self._last_dates[str(uid)] = pd.Timestamp(sorted_grp["ds"].iloc[-1])
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if not self._history:
            raise RuntimeError("Call fit() first.")
        records = []
        for uid, history in self._history.items():
            last_date = self._last_dates[uid]
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
            for h, ds in enumerate(future_dates, start=1):
                lookback = self.seasonality * math.ceil(h / self.seasonality) - h
                idx = -(lookback + 1)
                y_pred = history[idx] if abs(idx) <= len(history) else history[0]
                records.append({"unique_id": uid, "ds": ds, "y_pred": float(y_pred)})
        return pd.DataFrame(records)
