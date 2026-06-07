from __future__ import annotations

import numpy as np


class ConformalIntervalForecaster:
    """Conformal Prediction for time series forecasts.

    Generates prediction intervals without distributional assumptions.
    Works with any point forecaster (e.g., LGBMGlobalForecaster).

    Algorithm:
      1. Compute residuals on validation/CV folds
      2. Calculate quantiles of absolute residuals
      3. For new forecasts: interval = [ŷ - q, ŷ + q] where q is quantile
    """

    def __init__(self, confidence_level: float = 0.80) -> None:
        """Initialize conformal forecaster.

        Args:
            confidence_level: desired coverage (e.g., 0.80 for 80% interval).
        """
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1)")
        self.confidence_level = confidence_level
        self._quantile_value: float | None = None

    def fit(self, residuals: np.ndarray) -> ConformalIntervalForecaster:
        """Fit quantile from validation residuals.

        Args:
            residuals: shape (n,) — |y_true - y_pred| from CV validation sets.
        """
        # Quantile level: upper tail to cover (1 + confidence_level) / 2
        # For 80% interval: quantile = 0.9 (90th percentile)
        quantile_level = (1.0 + self.confidence_level) / 2.0
        self._quantile_value = float(np.quantile(residuals, quantile_level))
        return self

    def predict(
        self,
        point_forecasts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate prediction interval.

        Args:
            point_forecasts: shape (n,) or (n, horizon) — central forecasts.

        Returns:
            (lower, upper): shape same as point_forecasts.
        """
        if self._quantile_value is None:
            raise RuntimeError("Call fit() first.")

        lower = point_forecasts - self._quantile_value
        upper = point_forecasts + self._quantile_value

        # Ensure non-negativity for sales/count data
        lower = np.maximum(lower, 0.0)

        return lower, upper

    def predict_quantiles(
        self,
        point_forecasts: np.ndarray,
        quantile_levels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate full quantile distribution.

        Args:
            point_forecasts: shape (n,)
            quantile_levels: shape (n_quantiles,), default [0.01, 0.05, ..., 0.95, 0.99]

        Returns:
            shape (n, n_quantiles) — quantile predictions at each level.
        """
        if self._quantile_value is None:
            raise RuntimeError("Call fit() first.")

        if quantile_levels is None:
            quantile_levels = np.array(
                [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            )

        # For each quantile level, compute the offset from point forecast
        # Using symmetric conformal intervals (same offset on both sides)
        n = len(point_forecasts) if hasattr(point_forecasts, "__len__") else 1
        quantiles = np.zeros((n, len(quantile_levels)))

        for j, q in enumerate(quantile_levels):
            if q < 0.5:
                # Lower tail: use negative offset
                offset = (
                    self._quantile_value * (q - 0.5) / (1.0 - self.confidence_level)
                )
            else:
                # Upper tail: use positive offset
                offset = (
                    self._quantile_value * (q - 0.5) / (1.0 - self.confidence_level)
                )

            quantiles[:, j] = (
                point_forecasts + offset if n > 1 else point_forecasts + offset
            )

        return np.maximum(quantiles, 0.0)  # Ensure non-negativity


class QuantileGBDTForecaster:
    """Quantile regression with LightGBM.

    Direct quantile prediction using LightGBM's 'quantile' objective.
    Generates multiple quantile levels simultaneously.

    Advantages over Conformal:
      - Captures heteroskedasticity (different widths by prediction size)
      - Trained end-to-end, not post-hoc on residuals

    Disadvantages:
      - Requires retraining model for each quantile
      - Quantile crossings possible (q_0.25 > q_0.5)
    """

    def __init__(
        self,
        quantile_levels: list[float] | None = None,
        base_params: dict | None = None,
    ) -> None:
        """Initialize quantile forecaster.

        Args:
            quantile_levels: quantiles to predict (default: deciles)
            base_params: LightGBM parameters (objective is overridden per quantile)
        """
        self.quantile_levels = quantile_levels or [
            0.05,
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.95,
        ]
        self.base_params = base_params or {}
        self._models: dict[float, object] = {}  # quantile -> trained model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
    ) -> QuantileGBDTForecaster:
        """Fit quantile models (placeholder).

        Full implementation would train LightGBM for each quantile.
        For now, this is a stub to establish the interface.

        Args:
            x_train: shape (n, n_features)
            y_train: shape (n,)
        """
        # TODO: implement LightGBM quantile training
        return self

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Generate quantile predictions.

        Args:
            x_test: shape (n, n_features)

        Returns:
            shape (n, n_quantiles) — quantile predictions.
        """
        # TODO: implement LightGBM quantile prediction
        raise NotImplementedError("QuantileGBDT requires full LightGBM integration")
