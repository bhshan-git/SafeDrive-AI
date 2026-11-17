"""Integration tests for the Flask REST API."""

import pytest

from src.api import create_app

RISKY = {
    "hour": 2, "visibility_km": 1.0, "traffic_density": 88,
    "avg_speed_kmph": 100, "speed_limit_kmph": 60,
    "accident_history_5yr": 20, "vehicle_count": 900,
    "weather": "Fog", "road_condition": "Wet",
    "road_type": "Highway", "lighting": "Dark",
    "location": "Mahabaleshwar Ghat Section",
}
SAFE = {
    "hour": 11, "visibility_km": 9.5, "traffic_density": 25,
    "avg_speed_kmph": 45, "speed_limit_kmph": 60,
    "accident_history_5yr": 1, "vehicle_count": 150,
    "weather": "Clear", "road_condition": "Dry",
    "road_type": "Urban", "lighting": "Daylight",
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_dashboard_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SafeDrive" in response.data


def test_health_endpoint(client):
    data = client.get("/api/health").get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_model_info_endpoint(client):
    data = client.get("/api/model/info").get_json()
    assert data["best_model"]
    assert data["classes"] == ["Low", "Medium", "High"]
    assert len(data["feature_importance"]) > 0


def test_data_summary_endpoint(client):
    data = client.get("/api/data/summary").get_json()
    assert data["total_records"] > 0
    assert sum(data["risk_distribution"].values()) == data["total_records"]


def test_high_risk_locations_endpoint(client):
    data = client.get("/api/locations/high-risk").get_json()
    assert len(data) > 0
    assert all("risk_pct" in item for item in data)


def test_predict_endpoint_risky_vs_safe(client):
    risky = client.post("/api/predict", json=RISKY)
    safe = client.post("/api/predict", json=SAFE)
    assert risky.status_code == 200
    assert safe.status_code == 200
    assert risky.get_json()["risk_level"] == "High"
    assert safe.get_json()["risk_level"] == "Low"
    assert len(risky.get_json()["recommendations"]) > 0


def test_predict_rejects_invalid_body(client):
    response = client.post("/api/predict", json=["not", "a", "dict"])
    assert response.status_code == 400


def test_batch_predict(client):
    response = client.post("/api/predict/batch", json={"records": [RISKY, SAFE]})
    assert response.status_code == 200
    assert len(response.get_json()["predictions"]) == 2


def test_history_endpoint(client):
    data = client.get("/api/history").get_json()
    assert isinstance(data, list)
