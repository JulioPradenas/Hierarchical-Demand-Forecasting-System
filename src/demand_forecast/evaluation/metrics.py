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


def crps(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
) -> float:
    """Continuous Ranked Probability Score for empirical quantile distributions.

    Compares empirical CDF (from quantiles) to observed values.
    CRPS=0 is perfect. CRPS=MAE when using only point forecast.

    Args:
        y_true: shape (n,) or (n_series, horizon) — observed values
        y_quantiles: shape (n, n_quantiles) — quantile predictions at levels
                     [0.01, 0.05, ..., 0.95, 0.99], sorted.

    Returns:
        Mean CRPS across all observations.
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_quantiles.ndim > 2:
        y_quantiles = y_quantiles.reshape(-1, y_quantiles.shape[-1])

    n = len(y_true)
    crps_vals = np.zeros(n)

    # Quantile levels (e.g., 99 quantiles from 0.01 to 0.99)
    n_quantiles = y_quantiles.shape[1]
    quantile_levels = np.linspace(0.01, 0.99, n_quantiles)

    for i in range(n):
        # CRPS integrand: (F(y) - H(y - y_true))^2
        # where H is Heaviside step function and F is empirical CDF
        crps_vals[i] = (
            np.sum((quantile_levels - (y_true[i] <= y_quantiles[i]).astype(float)) ** 2)
            / n_quantiles
        )

    return float(np.mean(crps_vals))


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage: fraction of observations within prediction interval.

    Args:
        y_true: shape (n,) or (n_series, horizon)
        lower: shape (n,) or (n_series, horizon) — lower bound
        upper: shape (n,) or (n_series, horizon) — upper bound

    Returns:
        Coverage rate in [0, 1]. For 80% interval, target is ~0.80.
    """
    y_true_flat = y_true.flatten() if y_true.ndim > 1 else y_true
    lower_flat = lower.flatten() if lower.ndim > 1 else lower
    upper_flat = upper.flatten() if upper.ndim > 1 else upper

    inside = (y_true_flat >= lower_flat) & (y_true_flat <= upper_flat)
    return float(np.mean(inside))


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Mean width of prediction interval.

    Args:
        lower: shape (n,) or (n_series, horizon)
        upper: shape (n,) or (n_series, horizon)

    Returns:
        Mean width (upper - lower).
    """
    lower_flat = lower.flatten() if lower.ndim > 1 else lower
    upper_flat = upper.flatten() if upper.ndim > 1 else upper
    return float(np.mean(upper_flat - lower_flat))
