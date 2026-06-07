from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand_forecast.models.lgbm_forecaster import LGBMGlobalForecaster

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HORIZON = 28
N_DAYS = 400  # enough history for all lags (max lag 365 + horizon)
N_SERIES = 4


def _make_feature_df(
    n_days: int = N_DAYS,
    n_series: int = N_SERIES,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic DataFrame with unique_id/ds/y + pre-computed lag features.

    Lags are >= HORIZON so no recursion is needed — mirrors the real pipeline.
    """
    rng = np.random.default_rng(seed)
    lags = [28, 35, 42, 56]
    rolling_windows = [7, 14]
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")

    records = []
    for i in range(n_series):
        uid = f"series_{i}"
        y = rng.poisson(lam=5, size=n_days).astype(float)
        for t, (d, v) in enumerate(zip(dates, y)):
            row: dict[str, object] = {"unique_id": uid, "ds": d, "y": v}
            # Lag features (safe: all >= HORIZON)
            for lag in lags:
                row[f"lag_{lag}"] = y[t - lag] if t >= lag else np.nan
            # Rolling stats over lagged base (lag_28)
            base_lag = 28
            for w in rolling_windows:
                start = t - base_lag - w + 1
                end = t - base_lag + 1
                if start >= 0:
                    window_vals = y[start:end]
                    row[f"rolling_mean_{w}"] = float(np.mean(window_vals))
                    row[f"rolling_std_{w}"] = float(np.std(window_vals))
                else:
                    row[f"rolling_mean_{w}"] = np.nan
                    row[f"rolling_std_{w}"] = np.nan
            # Simple calendar features
            row["day_of_week"] = float(d.dayofweek)
            row["month"] = float(d.month)
            records.append(row)

    df = pd.DataFrame(records)
    return df


FEATURE_COLS = (
    ["lag_28", "lag_35", "lag_42", "lag_56"]
    + ["rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14"]
    + ["day_of_week", "month"]
)


@pytest.fixture(scope="module")
def feature_df() -> pd.DataFrame:
    return _make_feature_df()


@pytest.fixture(scope="module")
def trained_model(feature_df: pd.DataFrame) -> LGBMGlobalForecaster:
    """Fitted model on training rows (drop NaN feature rows)."""
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    model = LGBMGlobalForecaster(num_boost_round=50)
    model.fit(train, FEATURE_COLS, target_col="y")
    return model


# ---------------------------------------------------------------------------
# Tests — fit
# ---------------------------------------------------------------------------


def test_fit_returns_self(feature_df: pd.DataFrame) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    model = LGBMGlobalForecaster(num_boost_round=10)
    result = model.fit(train, FEATURE_COLS, target_col="y")
    assert result is model


def test_fit_stores_feature_cols(trained_model: LGBMGlobalForecaster) -> None:
    assert trained_model._feature_cols == FEATURE_COLS


def test_fit_identifies_lag_offsets(trained_model: LGBMGlobalForecaster) -> None:
    assert trained_model._lag_offsets == {
        "lag_28": 28,
        "lag_35": 35,
        "lag_42": 42,
        "lag_56": 56,
    }


# ---------------------------------------------------------------------------
# Tests — predict_recursive
# ---------------------------------------------------------------------------


def _make_future_rows(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Build future feature rows for all series using training history.

    Since all lags >= horizon, every lag value references a historical date.
    """
    future_records = []
    for uid, group in df.groupby("unique_id"):
        group = group.sort_values("ds")
        last_date = group["ds"].max()
        y_series = group.set_index("ds")["y"]

        for h in range(1, horizon + 1):
            future_date = last_date + pd.Timedelta(days=h)
            row: dict[str, object] = {
                "unique_id": uid,
                "ds": future_date,
                "y": np.nan,
            }
            for col in FEATURE_COLS:
                if col.startswith("lag_"):
                    k = int(col.split("_")[1])
                    lookup = future_date - pd.Timedelta(days=k)
                    row[col] = y_series.get(lookup, np.nan)
                elif col.startswith("rolling_mean_"):
                    w = int(col.split("_")[2])
                    base_lag = 28
                    end = future_date - pd.Timedelta(days=base_lag)
                    start = end - pd.Timedelta(days=w - 1)
                    vals = y_series[(y_series.index >= start) & (y_series.index <= end)]
                    row[col] = float(vals.mean()) if len(vals) > 0 else np.nan
                elif col.startswith("rolling_std_"):
                    w = int(col.split("_")[2])
                    base_lag = 28
                    end = future_date - pd.Timedelta(days=base_lag)
                    start = end - pd.Timedelta(days=w - 1)
                    vals = y_series[(y_series.index >= start) & (y_series.index <= end)]
                    row[col] = float(vals.std()) if len(vals) > 1 else 0.0
                elif col == "day_of_week":
                    row[col] = float(future_date.dayofweek)
                elif col == "month":
                    row[col] = float(future_date.month)
                else:
                    row[col] = np.nan
            future_records.append(row)

    return pd.DataFrame(future_records)


def test_predict_recursive_output_columns(
    trained_model: LGBMGlobalForecaster, feature_df: pd.DataFrame
) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    future = _make_future_rows(train)
    combined = pd.concat([train, future], ignore_index=True)

    result = trained_model.predict_recursive(combined, horizon=HORIZON)
    assert set(result.columns) == {"unique_id", "ds", "y_pred"}


def test_predict_recursive_output_size(
    trained_model: LGBMGlobalForecaster, feature_df: pd.DataFrame
) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    future = _make_future_rows(train)
    combined = pd.concat([train, future], ignore_index=True)

    result = trained_model.predict_recursive(combined, horizon=HORIZON)
    assert len(result) == N_SERIES * HORIZON


def test_predict_recursive_non_negative(
    trained_model: LGBMGlobalForecaster, feature_df: pd.DataFrame
) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    future = _make_future_rows(train)
    combined = pd.concat([train, future], ignore_index=True)

    result = trained_model.predict_recursive(combined, horizon=HORIZON)
    assert (result["y_pred"] >= 0).all()


def test_predict_recursive_no_nans(
    trained_model: LGBMGlobalForecaster, feature_df: pd.DataFrame
) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    future = _make_future_rows(train)
    combined = pd.concat([train, future], ignore_index=True)

    result = trained_model.predict_recursive(combined, horizon=HORIZON)
    assert not result["y_pred"].isna().any()


def test_predict_recursive_raises_if_no_future_rows(
    trained_model: LGBMGlobalForecaster, feature_df: pd.DataFrame
) -> None:
    train = feature_df.dropna(subset=FEATURE_COLS).copy()
    with pytest.raises(ValueError, match="No future rows"):
        trained_model.predict_recursive(train, horizon=HORIZON)


def test_predict_recursive_raises_before_fit(
    feature_df: pd.DataFrame,
) -> None:
    model = LGBMGlobalForecaster()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_recursive(feature_df, horizon=HORIZON)


# ---------------------------------------------------------------------------
# Tests — feature importance
# ---------------------------------------------------------------------------


def test_feature_importance_length(trained_model: LGBMGlobalForecaster) -> None:
    imp = trained_model.feature_importance()
    assert len(imp) == len(FEATURE_COLS)


def test_feature_importance_sorted(trained_model: LGBMGlobalForecaster) -> None:
    imp = trained_model.feature_importance()
    assert (imp.values[:-1] >= imp.values[1:]).all()


def test_feature_importance_raises_before_fit() -> None:
    with pytest.raises(RuntimeError):
        LGBMGlobalForecaster().feature_importance()


# ---------------------------------------------------------------------------
# Tests — fit with validation set (early stopping)
# ---------------------------------------------------------------------------


def test_fit_with_val_df_runs(feature_df: pd.DataFrame) -> None:
    clean = feature_df.dropna(subset=FEATURE_COLS)
    split = int(len(clean) * 0.8)
    train = clean.iloc[:split].copy()
    val = clean.iloc[split:].copy()

    model = LGBMGlobalForecaster(num_boost_round=100, early_stopping_rounds=10)
    model.fit(train, FEATURE_COLS, target_col="y", val_df=val)
    assert model._model is not None
