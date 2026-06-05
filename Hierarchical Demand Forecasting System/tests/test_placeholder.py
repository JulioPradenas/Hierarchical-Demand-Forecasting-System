from demand_forecast.config.settings import Settings
from demand_forecast.utils.typing import HIERARCHY_LEVELS, HierarchyLevel


def test_settings_defaults() -> None:
    s = Settings()
    assert s.horizon == 28
    assert s.n_cv_splits == 3
    assert s.data_dir.name == "data"


def test_settings_env_override(monkeypatch: object) -> None:
    import os
    os.environ["DEMAND_HORIZON"] = "14"
    s = Settings()
    assert s.horizon == 14
    os.environ.pop("DEMAND_HORIZON")


def test_hierarchy_levels() -> None:
    assert len(HIERARCHY_LEVELS) == 6
    assert "total" in HIERARCHY_LEVELS
    assert "item_store" in HIERARCHY_LEVELS
    level: HierarchyLevel = "state"
    assert level in HIERARCHY_LEVELS
