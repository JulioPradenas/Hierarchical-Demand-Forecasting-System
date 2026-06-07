import numpy as np
import pandas as pd
import pytest

from demand_forecast.models.base import BaseForecaster
from demand_forecast.models.naive import SeasonalNaiveForecaster
from demand_forecast.models.statistical import AutoARIMAForecaster, ETSForecaster


def test_base_forecaster_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseForecaster()  # type: ignore[abstract]


def _make_df(n_days: int = 60, n_series: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    records = []
    for i in range(n_series):
        uid = f"series_{i}"
        # Deterministic: value = day_of_week (0-6)
        vals = [float(d.dayofweek) for d in dates]
        for d, v in zip(dates, vals):
            records.append({"unique_id": uid, "ds": d, "y": v})
    return pd.DataFrame(records)


def test_seasonal_naive_output_columns() -> None:
    df = _make_df()
    model = SeasonalNaiveForecaster(seasonality=7)
    result = model.fit(df).predict(horizon=7)
    assert set(result.columns) >= {"unique_id", "ds", "y_pred"}


def test_seasonal_naive_output_size() -> None:
    df = _make_df(n_days=60, n_series=3)
    model = SeasonalNaiveForecaster(seasonality=7)
    result = model.fit(df).predict(horizon=28)
    assert len(result) == 3 * 28


def test_seasonal_naive_weekly_pattern() -> None:
    """For h=7, the prediction must equal the last training value (same weekday)."""
    df = _make_df(n_days=60, n_series=1)
    model = SeasonalNaiveForecaster(seasonality=7)
    preds = model.fit(df).predict(horizon=7)
    # Last 7 training values: day_of_week of days 53..59
    last_7 = df[df["unique_id"] == "series_0"].sort_values("ds").tail(7)["y"].values
    pred_vals = (
        preds[preds["unique_id"] == "series_0"].sort_values("ds")["y_pred"].values
    )
    np.testing.assert_array_equal(pred_vals, last_7)


def _make_statsforecast_df(n_days: int = 60, n_series: int = 2) -> pd.DataFrame:
    """Returns DataFrame in statsforecast format: unique_id, ds, y."""
    import numpy as np

    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    records = []
    for i in range(n_series):
        uid = f"item_{i}"
        y = rng.poisson(lam=5, size=n_days).astype(float)
        for d, v in zip(dates, y):
            records.append({"unique_id": uid, "ds": d, "y": v})
    return pd.DataFrame(records)


def test_ets_output_shape() -> None:
    df = _make_statsforecast_df(n_days=60, n_series=2)
    model = ETSForecaster()
    result = model.fit(df).predict(horizon=7)
    assert set(result.columns) >= {"unique_id", "ds", "y_pred"}
    assert len(result) == 2 * 7


def test_ets_predictions_are_positive() -> None:
    """ETS on count data should produce non-negative forecasts."""
    df = _make_statsforecast_df(n_days=60, n_series=3)
    model = ETSForecaster()
    result = model.fit(df).predict(horizon=7)
    assert (result["y_pred"] >= 0).all()


def test_autoarima_output_shape() -> None:
    df = _make_statsforecast_df(n_days=60, n_series=2)
    model = AutoARIMAForecaster()
    result = model.fit(df).predict(horizon=7)
    assert set(result.columns) >= {"unique_id", "ds", "y_pred"}
    assert len(result) == 2 * 7
