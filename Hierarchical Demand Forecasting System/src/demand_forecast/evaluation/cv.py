from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


class TimeSeriesCV:
    """Expanding-window cross-validation for time series.

    Splits are date-based: all series in the same fold share the same
    train/val cutoff. This is the only correct approach for global models
    where a single model learns from all series simultaneously.

    With n_splits=3 and horizon=28 on T total days:
      fold 0: train [0, T-84), val [T-84, T-56)
      fold 1: train [0, T-56), val [T-56, T-28)
      fold 2: train [0, T-28), val [T-28, T)
    """

    def __init__(self, n_splits: int = 3, horizon: int = 28) -> None:
        self.n_splits = n_splits
        self.horizon = horizon

    def split(
        self, df: pd.DataFrame, date_col: str = "ds"
    ) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Yield (train_dates, val_dates) for each fold, oldest-first."""
        dates = pd.DatetimeIndex(sorted(df[date_col].unique()))
        t = len(dates)
        min_required = (self.n_splits + 1) * self.horizon
        if t < min_required:
            raise ValueError(
                f"Need at least {min_required} dates for {self.n_splits} splits "
                f"with horizon {self.horizon}, got {t}."
            )
        for i in range(self.n_splits):
            # Folds go oldest → newest
            offset = (self.n_splits - 1 - i) * self.horizon
            val_end = t - offset
            val_start = val_end - self.horizon
            yield dates[:val_start], dates[val_start:val_end]

    def filter_fold(
        self, df: pd.DataFrame, fold_dates: pd.DatetimeIndex, date_col: str = "ds"
    ) -> pd.DataFrame:
        """Return rows of df whose date_col is in fold_dates."""
        return df[df[date_col].isin(fold_dates)].reset_index(drop=True)
