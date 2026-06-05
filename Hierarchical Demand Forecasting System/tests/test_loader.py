import pandas as pd
from demand_forecast.data.loader import M5DataLoader, ValidationReport


def test_load_sales_long_format(tmp_path, synthetic_sales, synthetic_calendar):
    """load_sales() must return a long-format DataFrame with the right columns."""
    # Write synthetic data in wide format (like real M5)
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    pivot = synthetic_sales.pivot_table(
        index=id_cols, columns="date", values="sales", aggfunc="first"
    ).reset_index()
    # Rename date columns to "d_1", "d_2", ...
    dates_sorted = sorted(synthetic_sales["date"].unique())
    d_labels = synthetic_calendar.sort_values("date")["d"].tolist()
    date_map = dict(zip(dates_sorted, d_labels))
    pivot.rename(columns=date_map, inplace=True)
    pivot.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    synthetic_calendar.to_csv(tmp_path / "calendar.csv", index=False)

    loader = M5DataLoader(data_dir=tmp_path)
    sales = loader.load_sales()

    required_cols = {
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "date",
        "sales",
    }
    assert required_cols.issubset(sales.columns)
    assert pd.api.types.is_datetime64_any_dtype(sales["date"])
    assert (sales["sales"] >= 0).all()


def test_validate_reports_no_negatives(tmp_path, synthetic_sales, synthetic_calendar):
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    pivot = synthetic_sales.pivot_table(
        index=id_cols, columns="date", values="sales", aggfunc="first"
    ).reset_index()
    dates_sorted = sorted(synthetic_sales["date"].unique())
    d_labels = synthetic_calendar.sort_values("date")["d"].tolist()
    date_map = dict(zip(dates_sorted, d_labels))
    pivot.rename(columns=date_map, inplace=True)
    pivot.to_csv(tmp_path / "sales_train_validation.csv", index=False)
    synthetic_calendar.to_csv(tmp_path / "calendar.csv", index=False)

    loader = M5DataLoader(data_dir=tmp_path)
    report = loader.validate()

    assert isinstance(report, ValidationReport)
    assert not report.has_negative_sales
    assert report.is_valid()
