from __future__ import annotations

import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class ValidationReport:
    n_series: int
    n_dates: int
    has_negative_sales: bool
    null_counts: dict[str, int] = field(default_factory=dict)
    date_range: tuple[str, str] = field(default=("", ""))

    def is_valid(self) -> bool:
        return not self.has_negative_sales and all(
            v == 0 for v in self.null_counts.values()
        )


class M5DataLoader:
    SALES_FILE = "sales_train_validation.csv"
    CALENDAR_FILE = "calendar.csv"
    PRICES_FILE = "sell_prices.csv"

    ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    def download(self, competition: str = "m5-forecasting-accuracy") -> None:
        """Download and unzip the M5 dataset via the Kaggle CLI.

        Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables
        (or ~/.kaggle/kaggle.json).
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle",
                "competitions",
                "download",
                "-c",
                competition,
                "-p",
                str(self.data_dir),
            ],
            check=True,
        )
        zip_path = self.data_dir / f"{competition}.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)
        zip_path.unlink()

    def load_sales(self) -> pd.DataFrame:
        """Return sales in long format.

        Columns: id, item_id, dept_id, cat_id, store_id, state_id, date, sales.
        """
        wide = pd.read_csv(self.data_dir / self.SALES_FILE)
        day_cols = [c for c in wide.columns if c.startswith("d_")]
        long = wide.melt(
            id_vars=self.ID_COLS,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )
        calendar = self.load_calendar()[["d", "date"]]
        long = long.merge(calendar, on="d").drop(columns="d")
        long["date"] = pd.to_datetime(long["date"])
        return long.reset_index(drop=True)

    def load_calendar(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / self.CALENDAR_FILE, parse_dates=["date"])

    def load_prices(self) -> pd.DataFrame:
        return pd.read_csv(self.data_dir / self.PRICES_FILE)

    def validate(self) -> ValidationReport:
        sales = self.load_sales()
        key_cols = ["date", "sales"] + self.ID_COLS
        null_counts = {c: int(sales[c].isna().sum()) for c in key_cols}
        unique_ids = sales["id"].nunique()
        unique_dates = sales["date"].nunique()
        min_date = str(sales["date"].min().date())
        max_date = str(sales["date"].max().date())
        return ValidationReport(
            n_series=unique_ids,
            n_dates=unique_dates,
            has_negative_sales=bool((sales["sales"] < 0).any()),
            null_counts=null_counts,
            date_range=(min_date, max_date),
        )
