import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_sales() -> pd.DataFrame:
    """
    200 days × 10 items × 2 stores = 2,000 rows in long format.
    Schema mirrors the real M5 long-format output of M5DataLoader.load_sales().
    """
    rng = np.random.default_rng(42)
    items = [
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("FOODS_2_001", "FOODS_2", "FOODS", "CA_1", "CA"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "CA_1", "CA"),
        ("FOODS_1_001", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("FOODS_2_001", "FOODS_2", "FOODS", "TX_1", "TX"),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
        ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES", "TX_1", "TX"),
    ]
    dates = pd.date_range("2011-01-29", periods=200, freq="D")
    records = []
    for item_id, dept_id, cat_id, store_id, state_id in items:
        sales = rng.poisson(lam=3, size=len(dates)).astype(float)
        for date, s in zip(dates, sales):
            records.append(
                {
                    "id": f"{item_id}_{store_id}_validation",
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


@pytest.fixture(scope="session")
def synthetic_calendar() -> pd.DataFrame:
    """200-day calendar with a few M5-style event columns."""
    dates = pd.date_range("2011-01-29", periods=200, freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "date": dates,
            "d": [f"d_{i + 1}" for i in range(len(dates))],
            "wm_yr_wk": (
                dates.isocalendar().year * 100 + dates.isocalendar().week
            ).values,
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "event_name_1": pd.array([None] * len(dates), dtype=object),
            "event_type_1": pd.array([None] * len(dates), dtype=object),
            "snap_CA": rng.integers(0, 2, len(dates)).astype(float),
            "snap_TX": rng.integers(0, 2, len(dates)).astype(float),
            "snap_WI": rng.integers(0, 2, len(dates)).astype(float),
        }
    )
    # Add a couple of named events
    df.loc[10, "event_name_1"] = "SuperBowl"
    df.loc[10, "event_type_1"] = "Sporting"
    df.loc[50, "event_name_1"] = "Easter"
    df.loc[50, "event_type_1"] = "Religious"
    return df
