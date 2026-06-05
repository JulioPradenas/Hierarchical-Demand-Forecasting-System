from abc import ABC, abstractmethod

import pandas as pd


class FeatureBuilder(ABC):
    """Common interface for all feature builders.

    fit() learns any statistics from training data (e.g. rolling means).
    transform() applies the learned statistics (or pure transformations) to any split.
    Never call transform() on data that was used in fit() if the builder computes
    look-ahead statistics — use fit_transform() only on training data.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """Learn statistics from training data.

        Args:
            df: Training DataFrame.

        Returns:
            self for method chaining.
        """
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned statistics or transformations.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame.
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Safe only for training data; avoids data leakage.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)
