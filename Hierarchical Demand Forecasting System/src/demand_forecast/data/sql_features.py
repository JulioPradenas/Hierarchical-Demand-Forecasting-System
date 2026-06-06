from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).parent.parent.parent.parent / "sql" / "features"


class DuckDBFeatureRunner:
    """Runs SQL feature queries against an in-memory DuckDB instance.

    The input DataFrame is registered as a virtual table named 'sales'.
    All queries in sql/features/ read from this table.
    """

    def __init__(self) -> None:
        self._con = duckdb.connect()

    def _run_query(self, sql_file: str, sales: pd.DataFrame) -> pd.DataFrame:
        self._con.register("sales", sales)
        sql = (SQL_DIR / sql_file).read_text()
        return self._con.execute(sql).df()

    def lag_features(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self._run_query("lag_features.sql", sales)

    def rolling_stats(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self._run_query("rolling_stats.sql", sales)

    def hierarchy_aggregates(self, sales: pd.DataFrame) -> pd.DataFrame:
        return self._run_query("hierarchy_aggregates.sql", sales)
