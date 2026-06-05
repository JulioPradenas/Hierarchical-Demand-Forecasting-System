from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    processed_dir: Path = Field(default=Path("data/processed"))
    models_dir: Path = Field(default=Path("models"))
    reports_dir: Path = Field(default=Path("reports"))
    horizon: int = 28
    n_cv_splits: int = 3
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    kaggle_competition: str = "m5-forecasting-accuracy"

    model_config = {"env_prefix": "DEMAND_"}
