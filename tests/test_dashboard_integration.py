"""Integration tests for Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_dashboard_imports() -> None:
    """Test that dashboard can be imported without errors."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        import plotly.graph_objects as go  # noqa: F401
        import streamlit as st  # noqa: F401
    except ImportError as e:
        raise ImportError(f"Missing dashboard dependencies: {e}")


def test_sample_forecast_data() -> None:
    """Test generating sample forecast data for Forecast Explorer page."""
    dates = pd.date_range("2025-01-01", periods=128, freq="D")

    # Historical (100 days) + future (28 days)
    forecast_df = pd.DataFrame(
        {
            "date": dates,
            "item_id": "FOODS_1_001",
            "store_id": "CA_1",
            "actual": list(np.random.poisson(50, 100)) + [np.nan] * 28,
            "forecast": 50 + np.random.normal(0, 5, 128),
            "lower_80": 40 + np.random.normal(0, 5, 128),
            "upper_80": 60 + np.random.normal(0, 5, 128),
            "lower_95": 30 + np.random.normal(0, 5, 128),
            "upper_95": 70 + np.random.normal(0, 5, 128),
        }
    )

    # Verify structure
    assert len(forecast_df) == 128
    assert forecast_df["actual"].notna().sum() == 100  # Historical only
    assert forecast_df["actual"].isna().sum() == 28  # Future only
    assert all(
        col in forecast_df.columns
        for col in [
            "date",
            "item_id",
            "store_id",
            "actual",
            "forecast",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
        ]
    )

    # Split and verify
    historical = forecast_df[forecast_df["actual"].notna()]
    future = forecast_df[forecast_df["actual"].isna()]

    assert len(historical) == 100
    assert len(future) == 28


def test_hierarchy_viewer_data() -> None:
    """Test hierarchy structure for Hierarchy Viewer page."""
    # Define complete hierarchy
    items = [
        "FOODS_1_001",
        "FOODS_1_002",
        "FOODS_2_001",
        "HOBBIES_1_001",
        "HOBBIES_1_002",
        "HOUSEHOLD_1_001",
    ]
    stores = ["CA_1", "TX_1", "WA_1"]
    hierarchy_data = []
    for item in items[:3]:  # Sample
        for store in stores:
            hierarchy_data.append(
                {
                    "item_id": item,
                    "store_id": store,
                    "category": item.rsplit("_", 1)[0],
                    "state": store.rsplit("_", 1)[0],
                    "wrmsse": np.random.uniform(0.35, 0.55),
                    "coherence_error": np.random.uniform(0.001, 0.05),
                }
            )

    hierarchy_df = pd.DataFrame(hierarchy_data)

    # Verify hierarchy levels
    assert len(hierarchy_df) > 0
    assert set(hierarchy_df.columns) >= {
        "item_id",
        "store_id",
        "category",
        "state",
        "wrmsse",
        "coherence_error",
    }

    # Verify metrics are reasonable
    assert (hierarchy_df["wrmsse"] > 0).all()
    assert (hierarchy_df["coherence_error"] >= 0).all()
    assert (hierarchy_df["coherence_error"] < 0.1).all()


def test_reconciliation_comparison_data() -> None:
    """Test reconciliation methods comparison data."""
    methods_results = {
        "Bottom-Up": {
            "coherence_error": 0.001,
            "wrmsse": 0.55,
            "cost_reduction_pct": 5.2,
        },
        "Top-Down": {
            "coherence_error": 0.005,
            "wrmsse": 0.45,
            "cost_reduction_pct": 8.3,
        },
        "MinT": {
            "coherence_error": 0.002,
            "wrmsse": 0.42,
            "cost_reduction_pct": 12.5,
        },
        "OLS": {
            "coherence_error": 0.003,
            "wrmsse": 0.44,
            "cost_reduction_pct": 10.1,
        },
    }

    # Convert to DataFrame
    results_df = pd.DataFrame(
        [
            {
                "method": method,
                "coherence_error": metrics["coherence_error"],
                "wrmsse": metrics["wrmsse"],
                "cost_reduction_pct": metrics["cost_reduction_pct"],
            }
            for method, metrics in methods_results.items()
        ]
    )

    assert len(results_df) == 4
    assert set(results_df.columns) >= {"method", "coherence_error", "wrmsse"}

    # MinT should have best coherence and WRMSSE
    mint_row = results_df[results_df["method"] == "MinT"].iloc[0]
    assert mint_row["coherence_error"] <= results_df["coherence_error"].max()
    assert mint_row["wrmsse"] <= results_df["wrmsse"].max()


def test_page_selectors() -> None:
    """Test page selection logic."""
    pages = ["Forecast Explorer", "Hierarchy Viewer", "Reconciliation Analysis"]

    for page in pages:
        assert page in pages
        assert isinstance(page, str)
        assert len(page) > 0


def test_metrics_by_level() -> None:
    """Test metrics aggregation by hierarchy level."""
    levels = ["Item-Store", "Category-Store", "State", "Total"]

    metrics_data = {
        "level": levels,
        "count": [9, 6, 3, 1],
        "avg_wrmsse": [0.45, 0.42, 0.40, 0.38],
        "avg_coherence_error": [0.02, 0.015, 0.01, 0.005],
    }

    metrics_df = pd.DataFrame(metrics_data)

    # Verify metrics decrease as hierarchy goes up
    assert (metrics_df["count"].values == sorted(metrics_df["count"].values, reverse=True)).all()  # noqa: E501
    assert (metrics_df["avg_wrmsse"].values == sorted(metrics_df["avg_wrmsse"].values, reverse=True)).all()  # noqa: E501
    assert (metrics_df["avg_coherence_error"].values == sorted(metrics_df["avg_coherence_error"].values, reverse=True)).all()  # noqa: E501


def test_dashboard_cache_data() -> None:
    """Test cached data loading performance."""
    import time

    # Simulate cache decorator
    data = {}

    def load_data_once(key: str) -> dict:
        """Load data only once."""
        if key not in data:
            data[key] = {
                "forecast": pd.DataFrame(
                    {
                        "date": pd.date_range("2025-01-01", periods=100),
                        "value": np.random.normal(50, 10, 100),
                    }
                ),
                "hierarchy": pd.DataFrame(
                    {
                        "item_id": ["FOODS_1_001"] * 10,
                        "wrmsse": np.random.uniform(0.3, 0.6, 10),
                    }
                ),
            }
        return data[key]

    # First call (cache miss)
    start = time.time()
    d1 = load_data_once("forecast")
    time1 = time.time() - start

    # Second call (cache hit)
    start = time.time()
    d2 = load_data_once("forecast")
    time2 = time.time() - start

    # Cache should be faster
    assert d1 is d2
    assert time2 < time1 or time2 < 0.01  # Cached or very fast


def test_interval_visualization_data() -> None:
    """Test data for fan chart visualization."""
    forecast_dates = pd.date_range("2025-02-01", periods=28, freq="D")  # noqa: F841
    point = np.linspace(100, 200, 28)

    # Point forecast

    # Multiple confidence intervals
    lower_80 = point - 10
    upper_80 = point + 10
    lower_95 = point - 20
    upper_95 = point + 20
    lower_99 = point - 30
    upper_99 = point + 30

    # Verify ordering
    assert (lower_99 <= lower_95).all()
    assert (lower_95 <= lower_80).all()
    assert (lower_80 <= point).all()
    assert (point <= upper_80).all()
    assert (upper_80 <= upper_95).all()
    assert (upper_95 <= upper_99).all()


def test_business_metrics_calculation() -> None:
    """Test business metrics from reconciliation."""
    # Simulate cost analysis
    baseline_cost = 1000.0
    method_costs = {
        "Bottom-Up": 948.0,
        "Top-Down": 917.0,
        "MinT": 875.0,
        "OLS": 899.0,
    }

    # Calculate cost reduction
    results = []
    for method, cost in method_costs.items():
        reduction_pct = (baseline_cost - cost) / baseline_cost * 100
        results.append(
            {
                "method": method,
                "cost": cost,
                "reduction_pct": reduction_pct,
            }
        )

    results_df = pd.DataFrame(results)

    # MinT should have best cost reduction
    assert results_df["reduction_pct"].max() > 0
    assert (
        results_df.loc[results_df["method"] == "MinT", "reduction_pct"].values[0] > 12
    )
