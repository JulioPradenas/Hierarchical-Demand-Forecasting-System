from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the API with lifespan."""
    with TestClient(app) as test_client:
        yield test_client


# =========================================================================
# Tests: Health Check
# =========================================================================


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["model_loaded"] is True


def test_health_has_required_fields(client: TestClient) -> None:
    """Health response has all required fields."""
    response = client.get("/health")
    data = response.json()
    required = {
        "status",
        "model_loaded",
        "model_version",
        "uptime_seconds",
        "tests_passing",
    }
    assert required.issubset(set(data.keys()))


def test_health_status_healthy(client: TestClient) -> None:
    """Health status is 'healthy'."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"


def test_health_uptime_positive(client: TestClient) -> None:
    """Uptime is positive."""
    response = client.get("/health")
    data = response.json()
    assert data["uptime_seconds"] > 0


# =========================================================================
# Tests: Forecast Item Endpoint
# =========================================================================


def test_forecast_item_valid_request(client: TestClient) -> None:
    """Forecast item endpoint accepts valid request."""
    payload = {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 28}
    response = client.post("/forecast/item", json=payload)
    assert response.status_code == 200


def test_forecast_item_response_structure(client: TestClient) -> None:
    """Forecast response has required fields."""
    payload = {"item_id": "FOODS_1_001", "store_id": "CA_1"}
    response = client.post("/forecast/item", json=payload)
    data = response.json()

    required = {
        "item_id",
        "store_id",
        "horizon",
        "forecasts",
        "reconciled",
        "coherence_error",
        "model_version",
        "inference_time_ms",
    }
    assert required.issubset(set(data.keys()))


def test_forecast_item_horizon_validation(client: TestClient) -> None:
    """Horizon must be between 1 and 28."""
    # Valid: 1
    response = client.post(
        "/forecast/item", json={"item_id": "x", "store_id": "y", "horizon": 1}
    )
    assert response.status_code == 200

    # Valid: 28
    response = client.post(
        "/forecast/item", json={"item_id": "x", "store_id": "y", "horizon": 28}
    )
    assert response.status_code == 200

    # Invalid: 0
    response = client.post(
        "/forecast/item", json={"item_id": "x", "store_id": "y", "horizon": 0}
    )
    assert response.status_code == 422

    # Invalid: 29
    response = client.post(
        "/forecast/item", json={"item_id": "x", "store_id": "y", "horizon": 29}
    )
    assert response.status_code == 422


def test_forecast_item_has_dates(client: TestClient) -> None:
    """Forecast includes date column."""
    payload = {"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 7}
    response = client.post("/forecast/item", json=payload)
    data = response.json()

    # Check base forecast has dates
    base_fcst = data["forecasts"]["item_store"]
    assert "dates" in base_fcst
    assert len(base_fcst["dates"]) == 7


def test_forecast_item_coherence_error(client: TestClient) -> None:
    """Forecast includes coherence error."""
    payload = {"item_id": "FOODS_1_001", "store_id": "CA_1"}
    response = client.post("/forecast/item", json=payload)
    data = response.json()
    assert "coherence_error" in data
    assert isinstance(data["coherence_error"], float)


def test_forecast_item_inference_time(client: TestClient) -> None:
    """Forecast includes inference time in milliseconds."""
    payload = {"item_id": "FOODS_1_001", "store_id": "CA_1"}
    response = client.post("/forecast/item", json=payload)
    data = response.json()
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] > 0


# =========================================================================
# Tests: Hierarchy Forecast Endpoint
# =========================================================================


def test_forecast_hierarchy_valid_request(client: TestClient) -> None:
    """Hierarchy forecast endpoint accepts valid request."""
    payload = {"level": "total", "horizon": 28}
    response = client.post("/forecast/hierarchy", json=payload)
    assert response.status_code == 200


def test_forecast_hierarchy_response_structure(client: TestClient) -> None:
    """Hierarchy forecast response has required fields."""
    payload = {"level": "state"}
    response = client.post("/forecast/hierarchy", json=payload)
    data = response.json()

    required = {"level", "horizon", "count", "forecasts"}
    assert required.issubset(set(data.keys()))


def test_forecast_hierarchy_level_validation(client: TestClient) -> None:
    """Hierarchy level must be one of the valid levels."""
    valid_levels = ["total", "state", "store", "category", "department", "item_store"]

    for level in valid_levels:
        response = client.post("/forecast/hierarchy", json={"level": level})
        # Should not error on the level itself (may be 200 or empty, but not 422)
        assert response.status_code != 422 or "level" not in response.text.lower()


# =========================================================================
# Tests: Root Endpoint
# =========================================================================


def test_root_endpoint(client: TestClient) -> None:
    """Root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data
    assert "docs" in data


# =========================================================================
# Tests: Documentation
# =========================================================================


def test_swagger_docs_available(client: TestClient) -> None:
    """Swagger docs are available at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger" in response.text.lower() or "openapi" in response.text.lower()
