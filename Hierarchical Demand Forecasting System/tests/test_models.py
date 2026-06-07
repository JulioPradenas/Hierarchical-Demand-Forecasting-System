import pytest

from demand_forecast.models.base import BaseForecaster


def test_base_forecaster_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseForecaster()  # type: ignore[abstract]
