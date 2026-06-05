import pandas as pd
import pytest
from demand_forecast.features.base import FeatureBuilder


def test_feature_builder_is_abstract() -> None:
    with pytest.raises(TypeError):
        FeatureBuilder()  # type: ignore[abstract]


def test_fit_transform_calls_fit_and_transform() -> None:
    """Verify fit_transform chains fit and transform correctly."""

    class SimpleFeatureBuilder(FeatureBuilder):
        def __init__(self):
            self.fitted = False

        def fit(self, df: pd.DataFrame) -> "SimpleFeatureBuilder":
            self.fitted = True
            return self

        def transform(self, df: pd.DataFrame) -> pd.DataFrame:
            if not self.fitted:
                raise RuntimeError("Not fitted")
            return df.copy()

    builder = SimpleFeatureBuilder()
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = builder.fit_transform(df)

    assert builder.fitted
    assert result.equals(df)
