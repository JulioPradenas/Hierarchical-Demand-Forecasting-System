from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class BusinessMetrics:
    """Business impact metrics for forecast evaluation."""

    total_cost: float  # Total cost of forecast errors
    overforecast_cost: float  # Cost from overestimating demand
    underforecast_cost: float  # Cost from underestimating demand
    avg_cost_per_unit: float  # Average cost per unit error
    cost_reduction_pct: float  # Cost reduction vs baseline (%)


def asymmetric_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_over: float = 1.0,
    cost_under: float = 5.0,
) -> BusinessMetrics:
    """Calculate asymmetric forecast error costs.

    Typical retail scenario:
      - Overforecast (excess inventory): markdown, storage, financing
      - Underforecast (stockout): lost sales, customer dissatisfaction, penalties

    Usually cost_under >> cost_over (ratio 5:1 to 10:1).

    Args:
        y_true: observed values (n,)
        y_pred: forecasted values (n,)
        cost_over: cost per unit of overforecast (default: 1.0)
        cost_under: cost per unit of underforecast (default: 5.0)

    Returns:
        BusinessMetrics with cost breakdown.
    """
    errors = y_pred - y_true

    # Overforecast: predicted > actual
    overforecast = np.maximum(errors, 0.0)
    overforecast_cost = float(np.sum(overforecast * cost_over))

    # Underforecast: predicted < actual
    underforecast = np.maximum(-errors, 0.0)
    underforecast_cost = float(np.sum(underforecast * cost_under))

    total = overforecast_cost + underforecast_cost
    n_units = len(y_true)

    return BusinessMetrics(
        total_cost=total,
        overforecast_cost=overforecast_cost,
        underforecast_cost=underforecast_cost,
        avg_cost_per_unit=total / n_units if n_units > 0 else 0.0,
        cost_reduction_pct=0.0,  # computed separately via baseline comparison
    )


def safety_stock(
    forecast: np.ndarray,
    forecast_std: np.ndarray,
    service_level: float = 0.95,
    lead_time: int = 1,
) -> np.ndarray:
    """Compute optimal safety stock given forecast uncertainty.

    Safety Stock = z * σ_forecast_error * sqrt(lead_time)

    where z is the z-score for the desired service level.

    Args:
        forecast: point forecast (n,)
        forecast_std: std dev of forecast errors (n,)
        service_level: target service level (e.g., 0.95 for 95% fill rate)
        lead_time: replenishment lead time in days (default: 1)

    Returns:
        safety stock levels (n,)
    """
    # Z-score for service level (one-tailed)
    z_score = norm.ppf(service_level)
    ss = z_score * forecast_std * np.sqrt(lead_time)
    return np.maximum(ss, 0.0)  # Non-negative


def cost_comparison(
    y_true: pd.DataFrame,
    forecasts: dict[str, np.ndarray],
    cost_over: float = 1.0,
    cost_under: float = 5.0,
) -> pd.DataFrame:
    """Compare costs across multiple forecast methods.

    Args:
        y_true: observed values (n,)
        forecasts: dict of {method_name: predictions (n,)}
        cost_over: cost per unit overforecast
        cost_under: cost per unit underforecast

    Returns:
        DataFrame with cost metrics per method.
    """
    y_vals = y_true.values if isinstance(y_true, pd.Series) else y_true

    results = []
    baseline_cost = None

    for method_name, pred in forecasts.items():
        metrics = asymmetric_cost(y_vals, pred, cost_over, cost_under)

        if baseline_cost is None:
            baseline_cost = metrics.total_cost
            cost_reduction_pct = 0.0
        else:
            cost_reduction_pct = (
                (baseline_cost - metrics.total_cost) / baseline_cost * 100
                if baseline_cost > 0
                else 0.0
            )

        results.append(
            {
                "method": method_name,
                "total_cost": metrics.total_cost,
                "overforecast_cost": metrics.overforecast_cost,
                "underforecast_cost": metrics.underforecast_cost,
                "avg_cost_per_unit": metrics.avg_cost_per_unit,
                "cost_reduction_pct": cost_reduction_pct,
            }
        )

    return pd.DataFrame(results)


def optimal_order_point(
    forecast: np.ndarray,
    forecast_std: np.ndarray,
    lead_time: int = 1,
    service_level: float = 0.95,
) -> np.ndarray:
    """Compute reorder point = (forecast × lead_time) + safety_stock.

    Args:
        forecast: daily demand forecast (n,)
        forecast_std: daily forecast error std (n,)
        lead_time: replenishment lead time in days
        service_level: target service level

    Returns:
        Reorder point quantity (n,)
    """
    expected_lead_time_demand = forecast * lead_time
    ss = safety_stock(forecast, forecast_std, service_level, lead_time)
    return expected_lead_time_demand + ss
