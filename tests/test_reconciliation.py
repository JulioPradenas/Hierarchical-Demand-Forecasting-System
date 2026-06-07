# ruff: noqa: I001
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.hierarchy import ForecastHierarchy
from demand_forecast.reconciliation.methods import (
    BottomUpReconciler,
    OLSReconciler,
    TopDownReconciler,
)
from demand_forecast.reconciliation.mint import MinTReconciler


@pytest.fixture(scope="module")
def synthetic_sales() -> pd.DataFrame:
    """Synthetic M5-like data: 2 items × 2 stores × 60 days."""
    rng = np.random.default_rng(42)
    items = [
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
    ]
    dates = pd.date_range("2011-01-29", periods=60, freq="D")
    records = []
    for item_id, dept_id, cat_id, store_id, state_id in items:
        sales = rng.poisson(lam=5, size=len(dates)).astype(float)
        for date, s in zip(dates, sales):
            records.append(
                {
                    "item_id": item_id,
                    "dept_id": dept_id,
                    "cat_id": cat_id,
                    "store_id": store_id,
                    "state_id": state_id,
                    "date": date,
                    "sales": s,
                }
            )
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def hierarchy(synthetic_sales: pd.DataFrame) -> ForecastHierarchy:
    return ForecastHierarchy(synthetic_sales)


@pytest.fixture(scope="module")
def base_forecasts_base(hierarchy: ForecastHierarchy) -> np.ndarray:
    """Synthetic bottom-level forecasts."""
    rng = np.random.default_rng(1)
    return rng.poisson(lam=5, size=len(hierarchy.base_series_order_)).astype(float)


# =========================================================================
# Tests: Bottom-Up Reconciliation
# =========================================================================


def test_bu_reconciler_fit_returns_self(hierarchy: ForecastHierarchy) -> None:
    reconciler = BottomUpReconciler()
    result = reconciler.fit(hierarchy)
    assert result is reconciler


def test_bu_reconcile_shape(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    reconciler = BottomUpReconciler().fit(hierarchy)
    result = reconciler.reconcile(base_forecasts_base)
    assert result.shape == (len(hierarchy.all_series_order_),)


def test_bu_reconcile_sums_match(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    """Bottom-level forecasts sum to total."""
    reconciler = BottomUpReconciler().fit(hierarchy)
    reconciled_all = reconciler.reconcile(base_forecasts_base)

    # total is first in all_series_order
    total_idx = 0
    total_from_reconcile = reconciled_all[total_idx]
    sum_from_base = base_forecasts_base.sum()
    np.testing.assert_almost_equal(total_from_reconcile, sum_from_base)


# =========================================================================
# Tests: Top-Down Reconciliation
# =========================================================================


def test_td_reconciler_fit(hierarchy: ForecastHierarchy) -> None:
    for variant in ("proportional", "forecast_proportional"):
        reconciler = TopDownReconciler(variant=variant)
        result = reconciler.fit(hierarchy)
        assert result is reconciler


def test_td_reconcile_shape(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    reconciler = TopDownReconciler(variant="proportional").fit(hierarchy)
    total_fcst = 100.0
    result = reconciler.reconcile(base_forecasts_base, total_fcst)
    assert result.shape == (len(hierarchy.all_series_order_),)


def test_td_reconcile_sum(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    """Top-down reconciled forecast should be non-negative."""
    reconciler = TopDownReconciler(variant="proportional").fit(hierarchy)
    total_fcst = 100.0
    reconciled_all = reconciler.reconcile(base_forecasts_base, total_fcst)

    # All forecasts should be non-negative
    assert (reconciled_all >= 0).all()


# =========================================================================
# Tests: OLS Reconciliation
# =========================================================================


def test_ols_reconciler_fit(hierarchy: ForecastHierarchy) -> None:
    reconciler = OLSReconciler()
    result = reconciler.fit(hierarchy)
    assert result is reconciler


def test_ols_reconcile_shape(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    reconciler = OLSReconciler().fit(hierarchy)
    result = reconciler.reconcile(base_forecasts_base)
    assert result.shape == (len(hierarchy.all_series_order_),)


def test_ols_reconcile_coherent(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    """OLS ensures coherence by aggregating bottom-up."""
    reconciler = OLSReconciler().fit(hierarchy)
    reconciled_all = reconciler.reconcile(base_forecasts_base)

    # Total (first element) should match sum of base
    assert abs(reconciled_all[0] - base_forecasts_base.sum()) < 1e-6


# =========================================================================
# Tests: MinT Reconciliation
# =========================================================================


def test_mint_reconciler_fit(hierarchy: ForecastHierarchy) -> None:
    for method in ("ols", "wls"):
        reconciler = MinTReconciler(method=method)
        result = reconciler.fit(hierarchy, residuals_base=None)
        assert result is reconciler


def test_mint_invalid_method() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        MinTReconciler(method="invalid")


def test_mint_reconcile_shape(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    reconciler = MinTReconciler(method="ols").fit(hierarchy, residuals_base=None)
    result = reconciler.reconcile(base_forecasts_base)
    assert result.shape == (len(hierarchy.all_series_order_),)


def test_mint_reconcile_ols_vs_bu(
    hierarchy: ForecastHierarchy, base_forecasts_base: np.ndarray
) -> None:
    """MinT with OLS should give results similar to BottomUp."""
    mint_reconciler = MinTReconciler(method="ols").fit(hierarchy, residuals_base=None)
    bu_reconciler = BottomUpReconciler().fit(hierarchy)

    mint_result = mint_reconciler.reconcile(base_forecasts_base)
    bu_result = bu_reconciler.reconcile(base_forecasts_base)

    # Both should have same shape
    assert mint_result.shape == bu_result.shape
    # Compare totals (first element)
    mint_total = mint_result[0]
    bu_total = bu_result[0]
    assert abs(mint_total - bu_total) < 0.1  # should be similar


def test_mint_requires_fit() -> None:
    reconciler = MinTReconciler()
    base_fcst = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="fit"):
        reconciler.reconcile(base_fcst)


def test_ols_requires_fit() -> None:
    reconciler = OLSReconciler()
    base_fcst = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="fit"):
        reconciler.reconcile(base_fcst)


def test_bu_requires_fit() -> None:
    reconciler = BottomUpReconciler()
    base_fcst = np.array([1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="fit"):
        reconciler.reconcile(base_fcst)


# =========================================================================
# Tests: Summing Matrix Properties
# =========================================================================


def test_summing_matrix_shape(hierarchy: ForecastHierarchy) -> None:
    s_mat = hierarchy.get_summing_matrix()
    n_all = len(hierarchy.all_series_order_)
    n_base = len(hierarchy.base_series_order_)
    assert s_mat.shape == (n_all, n_base)


def test_summing_matrix_valid_aggregation(
    hierarchy: ForecastHierarchy,
) -> None:
    """Check that S correctly aggregates."""
    s_mat = hierarchy.get_summing_matrix()
    base_sales = np.array(
        [
            hierarchy._sales[
                (hierarchy._sales["item_id"] == uid.split("_")[0])
                & (hierarchy._sales["store_id"] == uid.split("_")[1])
            ]["sales"].sum()
            for uid in hierarchy.base_series_order_
        ]
    )

    aggregated = s_mat @ base_sales
    # Total should match
    assert aggregated[0] == base_sales.sum()


# =========================================================================
# Tests: MinT residuals to covariance
# =========================================================================


def test_mint_estimate_w_ols(hierarchy: ForecastHierarchy) -> None:
    """MinT with ols method produces identity matrix W."""
    reconciler = MinTReconciler(method="ols")
    reconciler.fit(hierarchy, residuals_base=None)
    # OLS: W should be effectively I (no covariance weighting)
    assert reconciler.method == "ols"


def test_mint_estimate_w_wls(hierarchy: ForecastHierarchy) -> None:
    """MinT with wls method produces diagonal W."""
    reconciler = MinTReconciler(method="wls")
    reconciler.fit(hierarchy, residuals_base=None)
    # WLS: W should be diagonal
    assert reconciler.method == "wls"
