from __future__ import annotations

import numpy as np


def wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    scale: np.ndarray,
) -> float:
    """Weighted Root Mean Squared Scaled Error (M5 competition metric).

    Args:
        y_true: shape (n_series, horizon)
        y_pred: shape (n_series, horizon)
        weights: shape (n_series,), sum to 1. In M5: proportional to dollar sales.
        scale: shape (n_series,), mean absolute diff of naive forecast on train set.
    """
    mse = np.mean((y_true - y_pred) ** 2, axis=1)
    rmsse = np.sqrt(mse / (scale + 1e-8))
    return float(np.sum(weights * rmsse))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error, averaged across series.

    Args:
        y_true: shape (n_series, horizon)
        y_pred: shape (n_series, horizon)
        y_train: shape (n_series, n_train) — used to compute the naive scale.
        seasonality: lag for naive denominator (1 = random walk scale).
    """
    mae = np.mean(np.abs(y_true - y_pred), axis=1)
    naive_scale = np.mean(
        np.abs(y_train[:, seasonality:] - y_train[:, :-seasonality]),
        axis=1,
    )
    return float(np.mean(mae / (naive_scale + 1e-8)))
