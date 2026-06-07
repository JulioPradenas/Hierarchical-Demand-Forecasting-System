from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request for item-level forecast."""

    item_id: str
    store_id: str
    horizon: int = Field(28, ge=1, le=28)
    confidence_levels: list[float] = Field([0.80, 0.95])
    include_reconciled: bool = True


class LevelForecast(BaseModel):
    """Forecast for a single hierarchical level."""

    level: str  # "item_store", "category", "state", "total"
    item_id: str | None = None
    store_id: str | None = None
    dates: list[date]
    forecast: list[float]
    lower_80: list[float]
    upper_80: list[float]
    lower_95: list[float]
    upper_95: list[float]


class ForecastResponse(BaseModel):
    """Response containing forecasts at all hierarchy levels."""

    item_id: str
    store_id: str
    horizon: int
    forecasts: dict[str, LevelForecast]
    reconciled: bool
    coherence_error: float
    model_version: str
    inference_time_ms: float


class HierarchyForecastRequest(BaseModel):
    """Request for all forecasts at a given hierarchy level."""

    level: str  # "total", "state", "store", "category", "department", "item_store"
    horizon: int = Field(28, ge=1, le=28)
    confidence_levels: list[float] = Field([0.80, 0.95])


class HealthResponse(BaseModel):
    """Health check response."""

    status: str  # "healthy", "degraded", "unhealthy"
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    tests_passing: bool
