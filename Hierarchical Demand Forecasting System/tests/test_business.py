from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecast.evaluation.business import (
    asymmetric_cost,
    cost_comparison,
    optimal_order_point,
    safety_stock,
)

# =========================================================================
# Tests: Asymmetric Cost
# =========================================================================


def test_asymmetric_cost_perfect_forecast() -> None:
    """Zero cost when forecast is perfect."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = asymmetric_cost(y_true, y_pred, cost_over=1.0, cost_under=5.0)
    assert metrics.total_cost == 0.0
    assert metrics.overforecast_cost == 0.0
    assert metrics.underforecast_cost == 0.0


def test_asymmetric_cost_overforecast() -> None:
    """Cost increases with overforecast."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])  # All +1
    metrics = asymmetric_cost(y_true, y_pred, cost_over=1.0, cost_under=5.0)
    assert metrics.overforecast_cost == 3.0  # 1 + 1 + 1
    assert metrics.underforecast_cost == 0.0
    assert metrics.total_cost == 3.0


def test_asymmetric_cost_underforecast() -> None:
    """Underforecast cost is higher due to cost ratio."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0])  # All -1
    metrics = asymmetric_cost(y_true, y_pred, cost_over=1.0, cost_under=5.0)
    assert metrics.overforecast_cost == 0.0
    assert metrics.underforecast_cost == 15.0  # 3 × 5
    assert metrics.total_cost == 15.0


def test_asymmetric_cost_mixed() -> None:
    """Mixed forecast errors."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 1.0, 3.0, 4.0])  # [over, under, exact, exact]
    metrics = asymmetric_cost(y_true, y_pred, cost_over=1.0, cost_under=5.0)
    assert metrics.overforecast_cost == 1.0
    assert metrics.underforecast_cost == 5.0
    assert metrics.total_cost == 6.0


def test_asymmetric_cost_asymmetry_ratio() -> None:
    """Same error magnitude but different costs."""
    y_true = np.array([1.0, 1.0])
    y_pred = np.array([2.0, 0.0])  # Same error magnitude, opposite directions

    metrics_over = asymmetric_cost(y_true, np.array([2.0, 1.0]))
    metrics_under = asymmetric_cost(y_true, np.array([1.0, 0.0]))

    # Underforecast cost should be higher (5x in default)
    assert metrics_under.underforecast_cost > metrics_over.overforecast_cost


# =========================================================================
# Tests: Safety Stock
# =========================================================================


def test_safety_stock_zero_std() -> None:
    """Zero std dev leads to zero safety stock."""
    forecast = np.array([10.0, 20.0, 30.0])
    forecast_std = np.array([0.0, 0.0, 0.0])
    ss = safety_stock(forecast, forecast_std, service_level=0.95)
    assert np.allclose(ss, 0.0)


def test_safety_stock_increases_with_std() -> None:
    """Safety stock increases with forecast error std dev."""
    forecast = np.array([10.0, 10.0])
    forecast_std_low = np.array([1.0, 1.0])
    forecast_std_high = np.array([5.0, 5.0])

    ss_low = safety_stock(forecast, forecast_std_low, service_level=0.95)
    ss_high = safety_stock(forecast, forecast_std_high, service_level=0.95)

    assert np.all(ss_high > ss_low)


def test_safety_stock_higher_service_level() -> None:
    """Higher service level leads to higher safety stock."""
    forecast = np.array([10.0, 10.0])
    forecast_std = np.array([2.0, 2.0])

    ss_90 = safety_stock(forecast, forecast_std, service_level=0.90)
    ss_95 = safety_stock(forecast, forecast_std, service_level=0.95)
    ss_99 = safety_stock(forecast, forecast_std, service_level=0.99)

    assert np.all(ss_90 < ss_95)
    assert np.all(ss_95 < ss_99)


def test_safety_stock_lead_time() -> None:
    """Safety stock increases with sqrt(lead_time)."""
    forecast = np.array([10.0, 10.0])
    forecast_std = np.array([2.0, 2.0])

    ss_lt1 = safety_stock(forecast, forecast_std, service_level=0.95, lead_time=1)
    ss_lt4 = safety_stock(forecast, forecast_std, service_level=0.95, lead_time=4)

    # sqrt(4) = 2, so ss_lt4 should be ~2x ss_lt1
    assert np.allclose(ss_lt4, ss_lt1 * 2.0)


# =========================================================================
# Tests: Cost Comparison
# =========================================================================


def test_cost_comparison_shape(synthetic_sales: tuple[np.ndarray, np.ndarray]) -> None:
    """Cost comparison returns DataFrame with expected columns."""
    y_true, y_pred = synthetic_sales[0], synthetic_sales[1]
    y_true_series = pd.Series(y_true)

    forecasts = {"baseline": y_pred, "model": y_pred * 1.1}
    result = cost_comparison(y_true_series, forecasts)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "method" in result.columns
    assert "total_cost" in result.columns
    assert "cost_reduction_pct" in result.columns


def test_cost_comparison_baseline_zero_reduction() -> None:
    """Baseline method (first) has zero cost reduction."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    forecasts = {"baseline": np.array([1.0, 2.0, 3.0])}
    result = cost_comparison(y_true, forecasts)

    assert result[result["method"] == "baseline"]["cost_reduction_pct"].values[0] == 0.0


def test_cost_comparison_better_model() -> None:
    """Better model shows positive cost reduction."""
    y_true = pd.Series([1.0, 2.0, 3.0])
    # Baseline has all errors
    baseline_pred = np.array([0.0, 0.0, 0.0])
    # Better model is closer
    better_pred = np.array([0.9, 1.9, 2.9])

    forecasts = {"baseline": baseline_pred, "better": better_pred}
    result = cost_comparison(y_true, forecasts, cost_over=1.0, cost_under=5.0)

    better_row = result[result["method"] == "better"].iloc[0]
    assert better_row["cost_reduction_pct"] > 0.0


# =========================================================================
# Tests: Optimal Order Point
# =========================================================================


def test_optimal_order_point_includes_demand() -> None:
    """Order point >= expected lead time demand."""
    forecast = np.array([10.0, 20.0, 30.0])
    forecast_std = np.array([1.0, 2.0, 3.0])
    lead_time = 1

    order_point = optimal_order_point(forecast, forecast_std, lead_time=lead_time)

    # Order point should be forecast × lead_time + safety_stock >= forecast
    assert np.all(order_point >= forecast * lead_time)


def test_optimal_order_point_lead_time_scales() -> None:
    """Order point scales with lead time."""
    forecast = np.array([10.0, 10.0])
    forecast_std = np.array([2.0, 2.0])

    op_lt1 = optimal_order_point(forecast, forecast_std, lead_time=1)
    op_lt2 = optimal_order_point(forecast, forecast_std, lead_time=2)

    # With lead time 2, expected demand is 2x, so order point should be much higher
    assert np.all(op_lt2 > op_lt1)


@pytest.fixture
def synthetic_sales() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic sales and predictions."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(1.0, 100.0, 1000)
    y_pred = y_true + rng.normal(0, 5, 1000)
    return y_true, y_pred
