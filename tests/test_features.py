# ruff: noqa: I001
import numpy as np
import pandas as pd
import pytest

from demand_forecast.data.sql_features import DuckDBFeatureRunner
from demand_forecast.features.base import FeatureBuilder
from demand_forecast.features.calendar import CalendarFeatureBuilder
from demand_forecast.features.hierarchical import HierarchicalFeatureBuilder
from demand_forecast.features.temporal import TemporalFeatureBuilder


def test_feature_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        FeatureBuilder()  # type: ignore[abstract]


def test_fit_transform_calls_fit_and_transform() -> None:
    """Verify fit_transform chains fit and transform correctly."""

    class SimpleFeatureBuilder(FeatureBuilder):
        def __init__(self):
            self.fitted = False

        def fit(self, df: pd.DataFrame) -> "SimpleFeatureBuilder":
            self.fitted = True
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            if not self.fitted:
                raise RuntimeError("Not fitted")
            return df.copy()

    builder = SimpleFeatureBuilder()
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = builder.fit_transform(df)

    assert builder.fitted
    assert result.equals(df)


def test_no_future_leakage(synthetic_sales: pd.DataFrame) -> None:
    """No feature column named lag_N may have N < horizon."""
    builder = TemporalFeatureBuilder(horizon=28)
    features = builder.fit_transform(
        synthetic_sales[["item_id", "store_id", "date", "sales"]]
    )
    lag_cols = [c for c in features.columns if c.startswith("lag_")]
    for col in lag_cols:
        lag_n = int(col.split("_")[1])
        assert lag_n >= 28, f"Leakage: {col} has lag {lag_n} < horizon 28"


def test_temporal_features_no_nan_in_valid_rows(synthetic_sales: pd.DataFrame) -> None:
    """After dropping warm-up rows per series, lag cols should be NaN-free."""
    builder = TemporalFeatureBuilder(horizon=28)
    features = builder.fit_transform(
        synthetic_sales[["item_id", "store_id", "date", "sales"]]
    )
    max_lag = max(builder.lags)
    valid = features.groupby(["item_id", "store_id"], group_keys=False).apply(
        lambda g: g.sort_values("date").iloc[max_lag:]
    )
    lag_cols = [c for c in features.columns if c.startswith("lag_")]
    # Only check lags that are <= (200 - max_lag): fixture has 200 days
    checkable_lags = [c for c in lag_cols if int(c.split("_")[1]) <= 200 - max_lag]
    if checkable_lags and not valid.empty:
        assert valid[checkable_lags].isna().sum().sum() == 0


def test_fourier_terms_shape(synthetic_sales: pd.DataFrame) -> None:
    builder = TemporalFeatureBuilder(
        horizon=28, fourier_periods=[7, 365], fourier_order=2
    )
    features = builder.fit_transform(
        synthetic_sales[["item_id", "store_id", "date", "sales"]]
    )
    # 2 periods × 2 orders × 2 (sin+cos) = 8 Fourier columns
    fourier_cols = [c for c in features.columns if "fourier" in c]
    assert len(fourier_cols) == 8


def test_calendar_features_continuous_encoding(
    synthetic_calendar: pd.DataFrame,
) -> None:
    """Event features must be continuous distances, not binary flags."""
    builder = CalendarFeatureBuilder()
    features = builder.fit_transform(synthetic_calendar)
    event_cols = [c for c in features.columns if "days_until" in c or "days_since" in c]
    assert len(event_cols) > 0, "No distance-encoded event features found"
    for col in event_cols:
        assert features[col].dtype in [np.float64, np.float32, float]


def test_snap_columns_present(synthetic_calendar: pd.DataFrame) -> None:
    builder = CalendarFeatureBuilder()
    features = builder.fit_transform(synthetic_calendar)
    assert "snap_CA" in features.columns
    assert "snap_TX" in features.columns


def test_calendar_has_no_nan_on_date(synthetic_calendar: pd.DataFrame) -> None:
    builder = CalendarFeatureBuilder()
    features = builder.fit_transform(synthetic_calendar)
    assert features["date"].isna().sum() == 0


def test_item_share_in_zero_one(synthetic_sales: pd.DataFrame) -> None:
    builder = HierarchicalFeatureBuilder()
    features = builder.fit_transform(synthetic_sales)
    assert (features["item_share"] >= 0).all()
    assert (features["item_share"] <= 1).all()


def test_item_share_sums_to_one_per_cat_store_date(
    synthetic_sales: pd.DataFrame,
) -> None:
    builder = HierarchicalFeatureBuilder()
    features = builder.fit_transform(synthetic_sales)
    # After warm-up (drop rows where item_share == uniform fill), check sums
    # Use only rows where we have enough history for lagged shares
    warm_up = 28
    late = features[
        features.groupby(["item_id", "store_id"])["date"].transform(
            lambda x: x.rank() > warm_up
        )
    ]
    if not late.empty:
        share_sums = late.groupby(["cat_id", "store_id", "date"])["item_share"].sum()
        np.testing.assert_allclose(share_sums.values, 1.0, atol=1e-6)


def test_parent_lag_column_exists(synthetic_sales: pd.DataFrame) -> None:
    builder = HierarchicalFeatureBuilder(parent_lag=28)
    features = builder.fit_transform(synthetic_sales)
    assert "store_sales_lag28" in features.columns


def test_sql_lag_features_match_python(synthetic_sales: pd.DataFrame) -> None:
    """DuckDB SQL lag_28 must match TemporalFeatureBuilder lag_28."""
    runner = DuckDBFeatureRunner()
    sql_result = (
        runner.lag_features(synthetic_sales[["item_id", "store_id", "date", "sales"]])
        .sort_values(["item_id", "store_id", "date"])
        .reset_index(drop=True)
    )

    builder = TemporalFeatureBuilder(horizon=28, lags=[28], rolling_windows=[])
    py_result = (
        builder.fit_transform(synthetic_sales[["item_id", "store_id", "date", "sales"]])
        .sort_values(["item_id", "store_id", "date"])
        .reset_index(drop=True)
    )

    merged = (
        sql_result[["item_id", "store_id", "date", "lag_28"]]
        .merge(
            py_result[["item_id", "store_id", "date", "lag_28"]],
            on=["item_id", "store_id", "date"],
            suffixes=("_sql", "_py"),
        )
        .dropna()
    )

    np.testing.assert_allclose(
        merged["lag_28_sql"].values, merged["lag_28_py"].values, rtol=1e-5
    )
