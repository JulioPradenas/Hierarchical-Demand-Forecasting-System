from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS

from .base import BaseForecaster


class ETSForecaster(BaseForecaster):
    """AutoETS via statsforecast — selects error/trend/season automatically.

    Uses statsforecast for speed: >100x faster than statsmodels on many series.
    """

    def __init__(self, season_length: int = 7, n_jobs: int = 1) -> None:
        self.season_length = season_length
        self.n_jobs = n_jobs
        self._sf: StatsForecast | None = None

    def fit(self, df: pd.DataFrame) -> ETSForecaster:
        self._sf = StatsForecast(
            models=[AutoETS(season_length=self.season_length)],
            freq="D",
            n_jobs=self.n_jobs,
        )
        self._sf.fit(df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() first.")
        preds = self._sf.predict(h=horizon).reset_index()
        col = "AutoETS"
        renamed = preds.rename(columns={col: "y_pred"})
        return pd.DataFrame(renamed[["unique_id", "ds", "y_pred"]])


class AutoARIMAForecaster(BaseForecaster):
    """AutoARIMA via statsforecast — selects p/d/q automatically."""

    def __init__(self, season_length: int = 7, n_jobs: int = 1) -> None:
        self.season_length = season_length
        self.n_jobs = n_jobs
        self._sf: StatsForecast | None = None

    def fit(self, df: pd.DataFrame) -> AutoARIMAForecaster:
        self._sf = StatsForecast(
            models=[AutoARIMA(season_length=self.season_length)],
            freq="D",
            n_jobs=self.n_jobs,
        )
        self._sf.fit(df)
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() first.")
        preds = self._sf.predict(h=horizon).reset_index()
        col = "AutoARIMA"
        renamed = preds.rename(columns={col: "y_pred"})
        return pd.DataFrame(renamed[["unique_id", "ds", "y_pred"]])
