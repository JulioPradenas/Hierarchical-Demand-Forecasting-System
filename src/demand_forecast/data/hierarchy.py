from __future__ import annotations

import numpy as np
import pandas as pd

from demand_forecast.utils.typing import HIERARCHY_LEVELS, HierarchyLevel

_LEVEL_KEYS: dict[HierarchyLevel, list[str]] = {
    "total": [],
    "state": ["state_id"],
    "store": ["store_id"],
    "category": ["cat_id", "store_id"],
    "department": ["dept_id", "store_id"],
    "item_store": ["item_id", "store_id"],
}


class ForecastHierarchy:
    """Represents the M5 hierarchy and computes the summing matrix S."""

    LEVELS: list[HierarchyLevel] = HIERARCHY_LEVELS

    def __init__(self, sales: pd.DataFrame) -> None:
        self._sales = sales.copy()
        self._sales["date"] = pd.to_datetime(self._sales["date"])
        self._base_info = (
            self._sales[["item_id", "dept_id", "cat_id", "store_id", "state_id"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        self._build_series_index()

    def _build_series_index(self) -> None:
        self._level_ids: dict[HierarchyLevel, list[str]] = {}
        for level in self.LEVELS:
            keys = _LEVEL_KEYS[level]
            if keys:
                combos = self._sales[keys].drop_duplicates().values.tolist()
                ids = sorted("_".join(str(v) for v in combo) for combo in combos)
            else:
                ids = ["total"]
            self._level_ids[level] = ids

        self.base_series_order_: list[str] = self._level_ids["item_store"]
        self.all_series_order_: list[str] = []
        for level in self.LEVELS:
            self.all_series_order_.extend(self._level_ids[level])

    def get_level_series(self, level: HierarchyLevel) -> pd.DataFrame:
        """Return long-format DataFrame [unique_id, date, y] for the given level."""
        keys = _LEVEL_KEYS[level]
        if keys:
            agg = self._sales.groupby(keys + ["date"])["sales"].sum().reset_index()
            agg["unique_id"] = agg[keys].apply(
                lambda r: "_".join(str(r[k]) for k in keys), axis=1
            )
            return agg[["unique_id", "date", "sales"]].rename(columns={"sales": "y"})
        else:
            agg = self._sales.groupby("date")["sales"].sum().reset_index()
            agg["unique_id"] = "total"
            return agg[["unique_id", "date", "sales"]].rename(columns={"sales": "y"})

    def get_summing_matrix(self) -> np.ndarray:
        """
        Build S matrix: shape (n_all_series, n_base_series).
        S[i, j] = 1 if base series j contributes to aggregate series i.
        """
        base_index = {uid: j for j, uid in enumerate(self.base_series_order_)}
        n_base = len(self.base_series_order_)
        n_all = len(self.all_series_order_)
        s_mat = np.zeros((n_all, n_base), dtype=np.int8)

        row = 0
        for level in self.LEVELS:
            keys = _LEVEL_KEYS[level]
            for uid in self._level_ids[level]:
                if not keys:
                    # total: all base series contribute
                    s_mat[row, :] = 1
                else:
                    # Find base_info rows whose key columns concatenate to uid
                    candidate_uids = self._base_info[keys].apply(
                        lambda r: "_".join(str(r[k]) for k in keys), axis=1
                    )
                    matched = self._base_info[candidate_uids == uid]
                    for _, r in matched.iterrows():
                        base_uid = f"{r['item_id']}_{r['store_id']}"
                        if base_uid in base_index:
                            s_mat[row, base_index[base_uid]] = 1
                row += 1
        return s_mat

    def coherence_check(self, forecasts: dict[HierarchyLevel, pd.DataFrame]) -> float:
        """Max relative discrepancy between store and sum of item_store children."""
        item_store_df = forecasts["item_store"].copy()
        store_df = forecasts["store"].copy()

        # For item_store unique_id like "FOODS_1_001_CA_1", the store_id is the
        # last two underscore-separated components ("CA_1"). rsplit("_", n=2) gives
        # ["FOODS_1_001", "CA", "1"] so joining [-2:] yields "CA_1".
        parts = item_store_df["unique_id"].str.rsplit("_", n=2)
        item_store_df["store_id"] = parts.str[-2] + "_" + parts.str[-1]

        implied_store = (
            item_store_df.groupby(["store_id", "date"])["y"].sum().reset_index()
        )
        implied_store["unique_id"] = implied_store["store_id"]

        merged = store_df.merge(
            implied_store[["unique_id", "date", "y"]],
            on=["unique_id", "date"],
            suffixes=("_parent", "_sum_children"),
        )
        if merged.empty:
            return 0.0
        parent: np.ndarray = merged["y_parent"].to_numpy(dtype=float)
        children_sum: np.ndarray = merged["y_sum_children"].to_numpy(dtype=float)
        denom = np.abs(parent) + 1e-8
        return float(np.max(np.abs(parent - children_sum) / denom))
