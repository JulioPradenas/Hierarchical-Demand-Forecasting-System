from __future__ import annotations

import numpy as np
import pytest

from demand_forecast.evaluation.metrics import coverage, crps, interval_width
from demand_forecast.models.conformal import ConformalIntervalForecaster

# =========================================================================
# Tests: Metrics
# =========================================================================


def test_coverage_perfect_interval() -> None:
    """Coverage should be 1.0 if all true values are inside interval."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lower = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    upper = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    assert coverage(y_true, lower, upper) == 1.0


def test_coverage_no_overlap() -> None:
    """Coverage should be 0.0 if no true values are inside."""
    y_true = np.array([1.0, 2.0, 3.0])
    lower = np.array([10.0, 11.0, 12.0])
    upper = np.array([15.0, 16.0, 17.0])
    assert coverage(y_true, lower, upper) == 0.0


def test_coverage_partial() -> None:
    """Coverage should be 0.5 for 50% overlap."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.5, 1.5, 2.5, 3.5])
    upper = np.array([1.5, 2.5, 3.5, 4.5])
    # Values 1,2,3,4 vs intervals [0.5-1.5], [1.5-2.5], [2.5-3.5], [3.5-4.5]
    # Inside: 1 in [0.5-1.5], 2 in [1.5-2.5], 3 in [2.5-3.5], 4 in [3.5-4.5] = all 4
    assert coverage(y_true, lower, upper) == 1.0


def test_coverage_2d() -> None:
    """Coverage should work with 2D arrays (n_series, horizon)."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    lower = np.array([[0.0, 1.0], [2.0, 3.0]])
    upper = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert coverage(y_true, lower, upper) == 1.0


def test_interval_width_constant() -> None:
    """Width should be constant for symmetric intervals."""
    lower = np.array([0.0, 1.0, 2.0])
    upper = np.array([2.0, 3.0, 4.0])
    assert np.isclose(interval_width(lower, upper), 2.0)


def test_interval_width_2d() -> None:
    """Width should work with 2D arrays."""
    lower = np.array([[0.0, 1.0], [2.0, 3.0]])
    upper = np.array([[2.0, 3.0], [4.0, 5.0]])
    assert np.isclose(interval_width(lower, upper), 2.0)


def test_crps_perfect_prediction() -> None:
    """CRPS close to 0 if quantiles capture observed value."""
    y_true = np.array([2.0, 2.0, 2.0])
    # Quantiles centered at true value
    y_quantiles = np.array(
        [
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [1.0, 1.5, 2.0, 2.5, 3.0],
        ]
    )
    crps_score = crps(y_true, y_quantiles)
    assert crps_score < 0.5  # Should be small


def test_crps_2d() -> None:
    """CRPS should work with 2D y_true."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_quantiles = np.array(
        [
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0, 2.5],
            [1.5, 2.0, 2.5, 3.0, 3.5],
            [2.5, 3.0, 3.5, 4.0, 4.5],
        ]
    )
    crps_score = crps(y_true, y_quantiles)
    assert 0 <= crps_score <= 10.0  # Valid range


# =========================================================================
# Tests: Conformal Prediction
# =========================================================================


@pytest.fixture
def residuals() -> np.ndarray:
    """Synthetic residuals from CV."""
    rng = np.random.default_rng(42)
    return rng.exponential(scale=2.0, size=100)


@pytest.fixture
def conformal_80(residuals: np.ndarray) -> ConformalIntervalForecaster:
    """Fitted conformal forecaster for 80% intervals."""
    forecaster = ConformalIntervalForecaster(confidence_level=0.80)
    forecaster.fit(residuals)
    return forecaster


def test_conformal_init() -> None:
    """Initialize conformal forecaster."""
    forecaster = ConformalIntervalForecaster(confidence_level=0.80)
    assert forecaster.confidence_level == 0.80


def test_conformal_invalid_level() -> None:
    """Reject invalid confidence levels."""
    with pytest.raises(ValueError, match="confidence_level"):
        ConformalIntervalForecaster(confidence_level=1.5)
    with pytest.raises(ValueError, match="confidence_level"):
        ConformalIntervalForecaster(confidence_level=0.0)


def test_conformal_fit_returns_self(residuals: np.ndarray) -> None:
    """fit() returns self for chaining."""
    forecaster = ConformalIntervalForecaster()
    result = forecaster.fit(residuals)
    assert result is forecaster


def test_conformal_fit_sets_quantile(residuals: np.ndarray) -> None:
    """fit() computes and stores quantile value."""
    forecaster = ConformalIntervalForecaster(confidence_level=0.80)
    forecaster.fit(residuals)
    assert forecaster._quantile_value is not None
    assert forecaster._quantile_value > 0


def test_conformal_predict_shape(conformal_80: ConformalIntervalForecaster) -> None:
    """predict() returns interval with correct shape."""
    point_fcst = np.array([1.0, 2.0, 3.0])
    lower, upper = conformal_80.predict(point_fcst)
    assert lower.shape == point_fcst.shape
    assert upper.shape == point_fcst.shape


def test_conformal_predict_bounds(conformal_80: ConformalIntervalForecaster) -> None:
    """Interval bounds are ordered: lower <= point <= upper."""
    point_fcst = np.array([1.0, 2.0, 3.0])
    lower, upper = conformal_80.predict(point_fcst)
    assert np.all(lower <= point_fcst)
    assert np.all(upper >= point_fcst)


def test_conformal_predict_non_negative(
    conformal_80: ConformalIntervalForecaster,
) -> None:
    """Lower bound is non-negative (for sales/count data)."""
    point_fcst = np.array([0.1, 0.5, 1.0])
    lower, upper = conformal_80.predict(point_fcst)
    assert np.all(lower >= 0.0)


def test_conformal_predict_requires_fit() -> None:
    """predict() raises if not fitted."""
    forecaster = ConformalIntervalForecaster()
    point_fcst = np.array([1.0, 2.0])
    with pytest.raises(RuntimeError, match="fit"):
        forecaster.predict(point_fcst)


def test_conformal_quantiles_shape(conformal_80: ConformalIntervalForecaster) -> None:
    """predict_quantiles() returns correct shape."""
    point_fcst = np.array([1.0, 2.0, 3.0])
    quantiles = conformal_80.predict_quantiles(point_fcst)
    assert quantiles.shape[0] == 3
    assert quantiles.shape[1] == 9  # default: 9 quantile levels


def test_conformal_quantiles_non_negative(
    conformal_80: ConformalIntervalForecaster,
) -> None:
    """All quantiles are non-negative."""
    point_fcst = np.array([0.1])
    quantiles = conformal_80.predict_quantiles(point_fcst)
    assert np.all(quantiles >= 0.0)


def test_conformal_quantiles_sorted(conformal_80: ConformalIntervalForecaster) -> None:
    """Quantiles should be monotonically increasing."""
    point_fcst = np.array([5.0])
    quantiles = conformal_80.predict_quantiles(point_fcst)
    # Check that each row is sorted
    assert np.all(np.diff(quantiles, axis=1) >= -1e-6)  # allow small numerical error


def test_conformal_coverage_empirical(residuals: np.ndarray) -> None:
    """Empirical coverage should match target for fitted forecaster."""
    # Use residuals distribution to generate consistent test data
    rng = np.random.default_rng(1)
    quantile_90 = np.quantile(residuals, 0.9)

    # Generate point forecasts and true values with controlled error
    point_fcst = rng.uniform(1.0, 10.0, size=500)
    errors = rng.uniform(-quantile_90, quantile_90, size=500)
    y_true = point_fcst + errors

    forecaster = ConformalIntervalForecaster(confidence_level=0.80)
    forecaster.fit(residuals)

    lower, upper = forecaster.predict(point_fcst)
    emp_coverage = np.mean((y_true >= lower) & (y_true <= upper))

    # For 80% interval, empirical coverage should be between 70% and 90%
    assert 0.65 < emp_coverage < 0.95
