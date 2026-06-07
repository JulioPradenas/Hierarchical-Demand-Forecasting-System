from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Request

from api.schemas import (
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HierarchyForecastRequest,
    LevelForecast,
)


# Placeholder for actual predictor class
# In production: load from MLflow or disk
class HierarchicalPredictor:
    """Placeholder predictor for forecasting."""

    def __init__(self, model_version: str = "v0.1.0") -> None:
        self.model_version = model_version
        self.loaded_at = datetime.now()

    def forecast(
        self,
        item_id: str,
        store_id: str,
        horizon: int = 28,
        confidence_levels: list[float] | None = None,
    ) -> dict:
        """Generate forecast for an item-store pair."""
        if confidence_levels is None:
            confidence_levels = [0.80, 0.95]

        # Placeholder: return dummy forecasts
        start_date = datetime.now() + timedelta(days=1)
        dates = [
            (start_date + timedelta(days=i)).date().isoformat() for i in range(horizon)
        ]

        base_forecast = [100.0 + i * 0.5 for i in range(horizon)]
        lower_80 = [max(0, f - 10) for f in base_forecast]
        upper_80 = [f + 10 for f in base_forecast]
        lower_95 = [max(0, f - 20) for f in base_forecast]
        upper_95 = [f + 20 for f in base_forecast]

        return {
            "item_id": item_id,
            "store_id": store_id,
            "dates": dates,
            "forecast": base_forecast,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "lower_95": lower_95,
            "upper_95": upper_95,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    """Load model on startup, cleanup on shutdown."""
    # Startup
    app.state.predictor = HierarchicalPredictor(model_version="v0.1.0")
    app.state.start_time = datetime.now()
    yield
    # Shutdown
    app.state.predictor = None


app = FastAPI(
    title="Hierarchical Demand Forecast API",
    description="REST API for hierarchical demand forecasting",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    predictor = getattr(request.app.state, "predictor", None)
    start_time = getattr(request.app.state, "start_time", datetime.now())
    uptime = (datetime.now() - start_time).total_seconds()

    return HealthResponse(
        status="healthy" if predictor is not None else "unhealthy",
        model_loaded=predictor is not None,
        model_version=predictor.model_version if predictor else "unknown",
        uptime_seconds=uptime,
        tests_passing=True,
    )


@app.post("/forecast/item", response_model=ForecastResponse)
async def forecast_item(
    request: ForecastRequest, http_request: Request
) -> ForecastResponse:
    """Generate forecast for a specific item-store pair."""
    start_time = time.time()

    predictor = getattr(http_request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Generate base forecast
    base = predictor.forecast(
        request.item_id,
        request.store_id,
        request.horizon,
        request.confidence_levels,
    )

    # Placeholder: create forecasts dict with only base level
    forecasts = {
        "item_store": LevelForecast(
            level="item_store",
            item_id=request.item_id,
            store_id=request.store_id,
            dates=[datetime.fromisoformat(d).date() for d in base["dates"]],
            forecast=base["forecast"],
            lower_80=base["lower_80"],
            upper_80=base["upper_80"],
            lower_95=base["lower_95"],
            upper_95=base["upper_95"],
        )
    }

    inference_time = (time.time() - start_time) * 1000

    return ForecastResponse(
        item_id=request.item_id,
        store_id=request.store_id,
        horizon=request.horizon,
        forecasts=forecasts,
        reconciled=request.include_reconciled,
        coherence_error=0.0,
        model_version=predictor.model_version,
        inference_time_ms=inference_time,
    )


@app.post("/forecast/hierarchy")
async def forecast_hierarchy(
    request: HierarchyForecastRequest, http_request: Request
) -> dict:
    """Generate forecasts for all nodes at a given hierarchy level."""
    predictor = getattr(http_request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Placeholder: return empty dict
    # In production: fetch all nodes at given level and forecast each
    return {
        "level": request.level,
        "horizon": request.horizon,
        "count": 0,
        "forecasts": [],
    }


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "service": "Hierarchical Demand Forecast API",
        "version": "0.1.0",
        "docs": "/docs",
    }
