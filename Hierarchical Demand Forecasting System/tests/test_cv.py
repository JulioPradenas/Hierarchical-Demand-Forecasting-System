import pandas as pd
import pytest

from demand_forecast.evaluation.cv import TimeSeriesCV


@pytest.fixture
def daily_df() -> pd.DataFrame:
    """120 days of data for 3 series — enough for 3 folds of horizon=28."""
    dates = pd.date_range("2020-01-01", periods=120, freq="D")
    records = []
    for uid in ["A", "B", "C"]:
        for d in dates:
            records.append({"unique_id": uid, "ds": d, "y": 1.0})
    return pd.DataFrame(records)


def test_cv_yields_correct_n_splits(daily_df: pd.DataFrame) -> None:
    cv = TimeSeriesCV(n_splits=3, horizon=28)
    splits = list(cv.split(daily_df))
    assert len(splits) == 3


def test_cv_no_temporal_leakage(daily_df: pd.DataFrame) -> None:
    """Max train date must be strictly before min val date in every fold."""
    cv = TimeSeriesCV(n_splits=3, horizon=28)
    for train_dates, val_dates in cv.split(daily_df):
        assert train_dates.max() < val_dates.min()


def test_cv_val_window_size(daily_df: pd.DataFrame) -> None:
    """Each validation window must be exactly horizon days."""
    cv = TimeSeriesCV(n_splits=3, horizon=28)
    for _, val_dates in cv.split(daily_df):
        assert len(val_dates) == 28


def test_cv_expanding_train(daily_df: pd.DataFrame) -> None:
    """Training set must grow monotonically across folds."""
    cv = TimeSeriesCV(n_splits=3, horizon=28)
    splits = list(cv.split(daily_df))
    train_sizes = [len(t) for t, _ in splits]
    assert train_sizes == sorted(train_sizes)


def test_cv_filter_df(daily_df: pd.DataFrame) -> None:
    """filter_fold() must return only train or val rows."""
    cv = TimeSeriesCV(n_splits=3, horizon=28)
    train_dates, val_dates = list(cv.split(daily_df))[0]
    train_df = cv.filter_fold(daily_df, train_dates)
    val_df = cv.filter_fold(daily_df, val_dates)
    assert set(train_df["ds"].unique()) == set(train_dates)
    assert set(val_df["ds"].unique()) == set(val_dates)
