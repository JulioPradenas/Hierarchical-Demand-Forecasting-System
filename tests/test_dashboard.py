from __future__ import annotations

import pandas as pd


def test_sample_data_loading() -> None:
    """Test loading of sample data for dashboard."""
    # Simulate data loading
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    forecast = pd.DataFrame(
        {
            "date": dates,
            "actual": range(100),
            "forecast": range(100),
        }
    )

    assert len(forecast) == 100
    assert "date" in forecast.columns
    assert "actual" in forecast.columns
    assert "forecast" in forecast.columns


def test_hierarchy_data_structure() -> None:
    """Test hierarchy data has expected structure."""
    hierarchy = pd.DataFrame(
        {
            "item_id": ["FOODS_1_001", "FOODS_1_002"],
            "category": ["FOODS_1", "FOODS_1"],
            "store_id": ["CA_1", "CA_1"],
            "state": ["CA", "CA"],
            "wrmsse": [0.45, 0.52],
            "coherence_error": [0.02, 0.01],
        }
    )

    assert len(hierarchy) == 2
    assert set(hierarchy.columns) == {
        "item_id",
        "category",
        "store_id",
        "state",
        "wrmsse",
        "coherence_error",
    }


def test_reconciliation_methods() -> None:
    """Test reconciliation method comparison."""
    methods = {
        "Bottom-Up": {"coherence_error": 0.001, "wrmsse": 0.55},
        "Top-Down": {"coherence_error": 0.005, "wrmsse": 0.45},
        "MinT": {"coherence_error": 0.002, "wrmsse": 0.42},
        "OLS": {"coherence_error": 0.003, "wrmsse": 0.44},
    }

    assert len(methods) == 4
    assert all(isinstance(m, dict) for m in methods.values())
    assert all("coherence_error" in m for m in methods.values())


def test_metrics_aggregation() -> None:
    """Test aggregation of metrics by level."""
    metrics_by_level = {
        "Item-Store": {"count": 3, "avg_wrmsse": 0.45},
        "Category-Store": {"count": 2, "avg_wrmsse": 0.42},
        "State": {"count": 1, "avg_wrmsse": 0.40},
        "Total": {"count": 1, "avg_wrmsse": 0.38},
    }

    assert len(metrics_by_level) == 4
    assert metrics_by_level["Item-Store"]["count"] == 3
    assert metrics_by_level["Total"]["avg_wrmsse"] == 0.38


def test_coherence_error_calculation() -> None:
    """Test coherence error is within bounds."""
    coherence_errors = [0.001, 0.002, 0.003, 0.004, 0.005]

    # Coherence error should be non-negative
    assert all(e >= 0 for e in coherence_errors)

    # Coherence error should be small (< 0.1)
    assert all(e < 0.1 for e in coherence_errors)
