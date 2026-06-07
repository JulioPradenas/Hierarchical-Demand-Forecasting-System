from __future__ import annotations

import pandas as pd

from .base import FeatureBuilder


class HierarchicalFeatureBuilder(FeatureBuilder):
    """Features that encode cross-level hierarchy relationships.

    item_share: fraction of category sales from this item (using lagged values).
    store_sales_lag{N}: lagged store total as top-down signal.
    """

    def __init__(self, parent_lag: int = 28) -> None:
        self.parent_lag = parent_lag

    def fit(self, df: pd.DataFrame) -> HierarchicalFeatureBuilder:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy().sort_values(["item_id", "store_id", "date"])

        # Category-store daily total
        cat_store_total = (
            out.groupby(["cat_id", "store_id", "date"])["sales"]
            .sum()
            .reset_index()
            .rename(columns={"sales": "_cat_store_total"})
        )
        out = out.merge(cat_store_total, on=["cat_id", "store_id", "date"], how="left")

        # Lag both numerator and denominator before dividing
        out["_cat_store_total_lag"] = out.groupby(["cat_id", "store_id"])[
            "_cat_store_total"
        ].shift(self.parent_lag)
        out["_item_sales_lag"] = out.groupby(["item_id", "store_id"])["sales"].shift(
            self.parent_lag
        )

        # Compute share; clip to [0, 1] to handle edge cases
        out["item_share"] = (
            out["_item_sales_lag"] / out["_cat_store_total_lag"].replace(0.0, 1.0)
        ).clip(0.0, 1.0)

        # Fill warm-up rows with uniform share
        n_items = out.groupby(["cat_id", "store_id"])["item_id"].transform("nunique")
        out["item_share"] = out["item_share"].fillna(1.0 / n_items)

        # Store-level lagged total
        store_total = (
            out.groupby(["store_id", "date"])["sales"]
            .sum()
            .reset_index()
            .rename(columns={"sales": "_store_total"})
        )
        out = out.merge(store_total, on=["store_id", "date"], how="left")
        out[f"store_sales_lag{self.parent_lag}"] = out.groupby("store_id")[
            "_store_total"
        ].shift(self.parent_lag)

        return out.drop(
            columns=[
                "_cat_store_total",
                "_cat_store_total_lag",
                "_item_sales_lag",
                "_store_total",
            ]
        )
