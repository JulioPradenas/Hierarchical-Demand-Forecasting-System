import pandas as pd
import pytest
from demand_forecast.features.base import FeatureBuilder
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
