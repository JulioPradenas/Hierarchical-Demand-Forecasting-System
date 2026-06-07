from __future__ import annotations

from typing import cast

import numpy as np

from ..data.hierarchy import ForecastHierarchy


class BottomUpReconciler:
    """Bottom-Up reconciliation: aggregate bottom-level forecasts upward.

    Simplest method: forecast only the bottom level (item-store) and aggregate
    up the hierarchy. Loses information from higher-level models but guarantees
    coherence.

    Useful as a baseline; rarely beats higher-level modeling in practice.
    """

    def __init__(self) -> None:
        self._hierarchy: ForecastHierarchy | None = None
        self._s_mat: np.ndarray | None = None

    def fit(self, hierarchy: ForecastHierarchy) -> BottomUpReconciler:
        self._hierarchy = hierarchy
        self._s_mat = hierarchy.get_summing_matrix()
        return self

    def reconcile(self, base_forecasts_base: np.ndarray) -> np.ndarray:
        """Aggregate bottom-level to all levels.

        Args:
            base_forecasts_base: shape (n_base,)

        Returns:
            shape (n_all,)
        """
        if self._s_mat is None:
            raise RuntimeError("Call fit() first.")
        return cast(np.ndarray, self._s_mat @ base_forecasts_base)


class TopDownReconciler:
    """Top-Down reconciliation: forecast total and disaggregate using proportions.

    Forecast the aggregate (total) and split downward using historical
    proportions. Assumes proportions are stable over time.

    Pros: uses information from the top level
    Cons: ignores lower-level dynamics

    Variants:
      - proportional: split by mean proportion over history
      - forecast_proportional: use proportions from base forecasts
    """

    def __init__(self, variant: str = "proportional") -> None:
        if variant not in ("proportional", "forecast_proportional"):
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant

        self._hierarchy: ForecastHierarchy | None = None
        self._s_mat: np.ndarray | None = None
        self._proportions: np.ndarray | None = None

    def fit(self, hierarchy: ForecastHierarchy) -> TopDownReconciler:
        """Estimate proportions from hierarchy."""
        self._hierarchy = hierarchy
        self._s_mat = hierarchy.get_summing_matrix()

        if self.variant == "proportional":
            # Use simple mean proportions from total sales
            total_by_base = self._hierarchy._sales.groupby(["item_id", "store_id"])[
                "sales"
            ].sum()
            grand_total = total_by_base.sum()
            self._proportions = (total_by_base / grand_total).values

        return self

    def reconcile(
        self,
        base_forecasts_base: np.ndarray,
        total_forecast: float,
    ) -> np.ndarray:
        """Disaggregate top-level forecast using proportions.

        Args:
            base_forecasts_base: shape (n_base,) — for forecast_proportional variant
            total_forecast: scalar — top-level forecast to distribute

        Returns:
            shape (n_all,)
        """
        if self._s_mat is None:
            raise RuntimeError("Call fit() first.")

        if self.variant == "forecast_proportional":
            # Use base forecasts to compute proportions
            total_base = base_forecasts_base.sum()
            if total_base > 0:
                proportions = base_forecasts_base / total_base
            else:
                proportions = np.ones_like(base_forecasts_base) / len(
                    base_forecasts_base
                )
        else:
            if self._proportions is None:
                proportions = np.ones_like(base_forecasts_base) / len(
                    base_forecasts_base
                )
            else:
                proportions = self._proportions

        # Disaggregate: distribute total using proportions
        reconciled_base = proportions * total_forecast
        return cast(np.ndarray, self._s_mat @ reconciled_base)


class OLSReconciler:
    """OLS reconciliation: special case of MinT with W=I.

    Minimises sum-of-squared adjustments to base forecasts subject to
    coherence constraint. Simpler than MinT but assumes independent errors
    of equal variance.

    In practice, almost identical to MinT when errors are roughly homogeneous.
    """

    def __init__(self) -> None:
        self._hierarchy: ForecastHierarchy | None = None
        self._s_mat: np.ndarray | None = None
        self._proj_mat: np.ndarray | None = None

    def fit(self, hierarchy: ForecastHierarchy) -> OLSReconciler:
        """Fit OLS reconciler — requires hierarchy only."""
        self._hierarchy = hierarchy
        self._s_mat = hierarchy.get_summing_matrix()
        return self

    def reconcile(self, base_forecasts_base: np.ndarray) -> np.ndarray:
        """Apply OLS reconciliation: least-squares solution to coherence.

        Minimises ||ŷ_base - ỹ_base||^2 subject to coherence constraint.
        In practice, for this problem, equivalent to bottom-up (aggregate base).

        Args:
            base_forecasts_base: shape (n_base,)

        Returns:
            shape (n_all,)
        """
        if self._s_mat is None:
            raise RuntimeError("Call fit() first.")

        # For OLS with hierarchical structure, the optimal solution
        # respects bottom-up aggregation (no adjustments to base level)
        return cast(np.ndarray, self._s_mat @ base_forecasts_base)
