import numpy as np
import pytest

from demand_forecast.evaluation.metrics import mase, wrmsse


def test_wrmsse_perfect_forecast_is_zero() -> None:
    y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = y_true.copy()
    weights = np.array([0.5, 0.5])
    scale = np.array([1.0, 1.0])
    assert wrmsse(y_true, y_pred, weights, scale) == pytest.approx(0.0)


def test_wrmsse_worse_forecast_is_larger() -> None:
    y_true = np.ones((2, 10))
    y_pred_good = y_true + 0.1
    y_pred_bad = y_true + 1.0
    weights = np.array([0.5, 0.5])
    scale = np.array([1.0, 1.0])
    assert wrmsse(y_true, y_pred_good, weights, scale) < wrmsse(
        y_true, y_pred_bad, weights, scale
    )


def test_mase_perfect_forecast_is_zero() -> None:
    y_train = np.tile(np.arange(50, dtype=float), (3, 1))
    y_true = np.ones((3, 10))
    y_pred = y_true.copy()
    assert mase(y_true, y_pred, y_train) == pytest.approx(0.0)


def test_mase_returns_scalar() -> None:
    rng = np.random.default_rng(0)
    y_train = rng.random((5, 100))
    y_true = rng.random((5, 28))
    y_pred = rng.random((5, 28))
    result = mase(y_true, y_pred, y_train)
    assert isinstance(result, float)
    assert result > 0
