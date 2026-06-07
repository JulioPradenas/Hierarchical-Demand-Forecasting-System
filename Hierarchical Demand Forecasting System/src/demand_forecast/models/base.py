from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """Common interface for all forecasting models.

    DataFrames use the Nixtla convention: unique_id (str), ds (datetime), y (float).
    predict() generates future dates automatically from the last training date.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseForecaster:
        """Fit on training data. df must have columns: unique_id, ds, y."""
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Return forecasts. Output columns: unique_id, ds, y_pred."""
        ...

    def fit_predict(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        return self.fit(df).predict(horizon)
