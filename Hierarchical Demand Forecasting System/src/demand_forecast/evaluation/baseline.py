from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.naive import SeasonalNaiveForecaster
from ..models.statistical import ETSForecaster
from .cv import TimeSeriesCV
from .metrics import mase


def evaluate_baselines(
    df: pd.DataFrame,
    cv: TimeSeriesCV,
    horizon: int = 28,
    include_ets: bool = False,
) -> pd.DataFrame:
    """Evaluate baseline models using time-series CV.

    Args:
        df: DataFrame with columns unique_id, ds, y (sorted by ds within each series).
        cv: TimeSeriesCV instance.
        horizon: Forecast horizon in days.
        include_ets: Whether to fit ETS (slower; skip for quick checks).

    Returns:
        DataFrame with columns: model, fold, mase — one row per model per fold.
    """
    models: dict[str, SeasonalNaiveForecaster | ETSForecaster] = {
        "SeasonalNaive": SeasonalNaiveForecaster(seasonality=7),
    }
    if include_ets:
        models["ETS"] = ETSForecaster(season_length=7)

    records = []
    for fold_idx, (train_dates, val_dates) in enumerate(cv.split(df)):
        train_df = cv.filter_fold(df, train_dates)
        val_df = cv.filter_fold(df, val_dates)

        series_ids = sorted(df["unique_id"].unique())
        y_true = np.array(
            [
                val_df[val_df["unique_id"] == uid].sort_values("ds")["y"].values
                for uid in series_ids
            ]
        )
        y_train_mat = np.array(
            [
                train_df[train_df["unique_id"] == uid].sort_values("ds")["y"].values
                for uid in series_ids
            ]
        )

        for model_name, model in models.items():
            preds = model.fit(train_df).predict(horizon)
            y_pred = np.array(
                [
                    preds[preds["unique_id"] == uid].sort_values("ds")["y_pred"].values
                    for uid in series_ids
                ]
            )
            h = min(y_true.shape[1], y_pred.shape[1])
            score = mase(y_true[:, :h], y_pred[:, :h], y_train_mat)
            records.append({"model": model_name, "fold": fold_idx, "mase": score})

    return pd.DataFrame(records)


def write_baseline_summary(
    results: pd.DataFrame,
    path: str = "reports/BASELINE_SUMMARY.md",
) -> None:
    """Write a markdown table of mean MASE per model."""
    summary = (
        results.groupby("model")["mase"]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
        .rename(columns={"mean": "MASE (mean)", "std": "MASE (std)"})
    )
    lines = ["# Baseline Summary\n", summary.to_markdown(index=False), "\n"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
