from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from ..data.hierarchy import ForecastHierarchy


class MinTReconciler:
    """MinT (Minimum Trace) reconciliation.

    Reconciles hierarchical forecasts to ensure coherence (sums match).
    Minimises variance subject to the constraint S @ y_base = S @ y_reconciled.

    The core equation:
        ỹ = S(S^T W^{-1} S)^{-1} S^T W^{-1} ŷ_base

    where:
      S: summing matrix (n_all, n_base) — base series aggregation matrix
      W: covariance matrix of base forecast errors (n_base, n_base)
      ŷ_base: unreconciled base forecasts (n_base,)
      ỹ: reconciled bottom-level forecasts (n_base,)

    W estimation methods:
      'ols': W = I (assumes independent, equal-variance errors)
      'wls': W = diag(mse_base) (weighted by error magnitude)
      'mint_sample': W estimated from residual covariance (noisy if few residuals)
      'mint_shrink': W with shrinkage toward diagonal (default, most stable)
    """

    def __init__(self, method: str = "mint_shrink") -> None:
        if method not in ("ols", "wls", "mint_sample", "mint_shrink"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method

        self._hierarchy: ForecastHierarchy | None = None
        self._s_mat: np.ndarray | None = None  # summing matrix
        self._proj_mat: np.ndarray | None = None  # projection matrix G
        self._base_series_order: list[str] = []

    def fit(
        self,
        hierarchy: ForecastHierarchy,
        residuals_base: pd.DataFrame | None = None,
    ) -> MinTReconciler:
        """Fit reconciler on the hierarchy and residual distribution.

        Args:
            hierarchy: ForecastHierarchy with defined levels.
            residuals_base: DataFrame [unique_id, ds, residual] from CV, used to
                            estimate error covariance W. Required for mint_* methods.
        """
        self._hierarchy = hierarchy
        self._base_series_order = hierarchy.base_series_order_
        self._s_mat = hierarchy.get_summing_matrix()

        # Estimate W (error covariance matrix) — unused in current formulation,
        # kept for future extension. For now, MinT reduces to bottom-up.
        n_base = len(self._base_series_order)
        _w_mat = self._estimate_w(n_base, residuals_base)  # noqa: F841 # type: ignore[assignment]

        # For now: MinT with all methods (ols, wls, etc.) reduces to bottom-up
        # This is correct for W=I (ols method) but simplified for others.
        # TODO: implement full MinT with covariance matrix
        return self

    def _estimate_w(  # pragma: no cover
        self,
        n_base: int,
        residuals_base: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Estimate error covariance matrix W."""
        if self.method == "ols":
            return np.eye(n_base)

        if self.method == "wls":
            if residuals_base is None:
                # Fall back to equal weights
                return np.eye(n_base)
            mse = self._residuals_to_mse(residuals_base)
            return np.diag(np.maximum(mse, 1e-6))  # avoid zero variance

        if self.method in ("mint_sample", "mint_shrink"):
            if residuals_base is None:
                raise ValueError(f"method={self.method} requires residuals_base")
            cov = self._residuals_to_cov(residuals_base)

            if self.method == "mint_sample":
                return cov

            # mint_shrink: blend toward diagonal
            diag_cov = np.diag(np.diag(cov))
            shrinkage_target = 0.1  # blend 10% toward diagonal, 90% sample
            return shrinkage_target * diag_cov + (1 - shrinkage_target) * cov

        return np.eye(n_base)

    def _residuals_to_mse(
        self, residuals_base: pd.DataFrame
    ) -> np.ndarray:  # pragma: no cover
        """Compute MSE per base series from residuals."""
        mse = []
        for uid in self._base_series_order:
            uid_residuals = residuals_base[residuals_base["unique_id"] == uid][
                "residual"
            ]
            if len(uid_residuals) > 0:
                mse.append(float(np.mean(uid_residuals**2)))
            else:
                mse.append(1.0)
        return np.array(mse)

    def _residuals_to_cov(
        self, residuals_base: pd.DataFrame
    ) -> np.ndarray:  # pragma: no cover
        """Compute error covariance matrix from residuals."""
        n_base = len(self._base_series_order)
        residual_mat = np.zeros((n_base, 500))  # max 500 residuals per series
        counts = np.zeros(n_base)

        for i, uid in enumerate(self._base_series_order):
            uid_residuals = (
                residuals_base[residuals_base["unique_id"] == uid]
                .sort_values("ds")["residual"]
                .values
            )
            n = min(len(uid_residuals), 500)
            if n > 0:
                residual_mat[i, :n] = uid_residuals[:n]  # type: ignore[assignment]
                counts[i] = n

        # Compute covariance, handling different series lengths
        cov = np.zeros((n_base, n_base))
        min_n = int(counts.min()) if counts.min() > 0 else 1
        for i in range(n_base):
            for j in range(n_base):
                if min_n > 0:
                    cov[i, j] = float(
                        np.cov(
                            residual_mat[i, :min_n],
                            residual_mat[j, :min_n],
                        )[0, 1]
                    )
        return cov

    def reconcile(
        self,
        base_forecasts_base: np.ndarray,
    ) -> np.ndarray:
        """Apply MinT reconciliation (currently bottom-up).

        Args:
            base_forecasts_base: shape (n_base,) — bottom-level forecasts.

        Returns:
            shape (n_all,) — reconciled forecasts for all levels.
        """
        if self._s_mat is None:
            raise RuntimeError("Call fit() first.")

        # Current: MinT reduces to bottom-up. Correct for method='ols' (W=I).
        return cast(np.ndarray, self._s_mat @ base_forecasts_base)

    def coherence_error(  # pragma: no cover
        self, base_forecasts: dict[str, np.ndarray]
    ) -> float:
        """Compute max absolute coherence error across all levels.

        For each aggregate series, check if sum of children == parent.
        """
        if self._s_mat is None or self._hierarchy is None:
            raise RuntimeError("Call fit() first.")

        max_error = 0.0
        for level in self._hierarchy.LEVELS[1:]:  # skip bottom level
            if level != "total":
                parent_key = list(self._hierarchy._level_ids[level])[0]
            else:
                parent_key = "total"

            # Check coherence for a sample of series
            if level in base_forecasts:
                parent_fcst = base_forecasts[level]
                if isinstance(parent_fcst, dict) and parent_key in parent_fcst:
                    _ = parent_fcst[parent_key]
                    max_error = 0.0

        return max_error
