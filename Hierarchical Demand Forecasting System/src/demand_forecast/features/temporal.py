from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBuilder


class TemporalFeatureBuilder(FeatureBuilder):
    """Generates lag, rolling-stat, and Fourier features without leakage.

    All lag offsets are guaranteed >= horizon to prevent look-ahead leakage.
    Rolling windows are computed over the lagged series, not raw sales.
    """

    SAFE_LAGS: list[int] = [28, 29, 30, 35, 42, 49, 56, 91, 119, 182, 364, 365]
    ROLLING_WINDOWS: list[int] = [7, 14, 28, 56, 91, 182]

    def __init__(
        self,
        horizon: int = 28,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        fourier_periods: list[int] | None = None,
        fourier_order: int = 2,
    ) -> None:
        self.horizon = horizon
        self.lags = [lag for lag in (lags or self.SAFE_LAGS) if lag >= horizon]
        self.rolling_windows = rolling_windows or self.ROLLING_WINDOWS
        self.fourier_periods = fourier_periods or [7, 365]
        self.fourier_order = fourier_order

    def fit(self, df: pd.DataFrame) -> TemporalFeatureBuilder:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        required = {"item_id", "store_id", "date", "sales"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        out = df.copy().sort_values(["item_id", "store_id", "date"])
        group_cols = ["item_id", "store_id"]

        # Lag features — all offsets guaranteed >= horizon
        for lag in self.lags:
            out[f"lag_{lag}"] = out.groupby(group_cols)["sales"].shift(lag)

        # Rolling stats over the minimum-safe-lag series (not raw sales)
        if self.lags:
            base_lag = self.lags[0]
            lagged_base = out.groupby(group_cols)["sales"].shift(base_lag)
            groups = out.groupby(group_cols).ngroup()

            def rolling_mean(window: int) -> pd.Series:
                return lagged_base.groupby(groups).transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

            def rolling_std(window: int) -> pd.Series:
                return lagged_base.groupby(groups).transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

            for window in self.rolling_windows:
                out[f"rolling_mean_{window}"] = rolling_mean(window)
                out[f"rolling_std_{window}"] = rolling_std(window)

        out = self._add_fourier_terms(out)
        return out

    def _add_fourier_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        t = (df["date"] - df["date"].min()).dt.days.to_numpy()
        for period in self.fourier_periods:
            for k in range(1, self.fourier_order + 1):
                df[f"fourier_sin_p{period}_k{k}"] = np.sin(2 * np.pi * k * t / period)
                df[f"fourier_cos_p{period}_k{k}"] = np.cos(2 * np.pi * k * t / period)
        return df
