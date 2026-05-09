"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["platform"] == "Chigma"
        assert "detection" in data["modules"]

    def test_health(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestDroneEndpoints:
    """Test drone API endpoints."""

    def test_status(self):
        response = client.get("/api/v1/drone/status")
        assert response.status_code == 200
        data = response.json()
        assert "battery_pct" in data

    def test_emergency_stop(self):
        response = client.post("/api/v1/drone/command/emergency-stop")
        assert response.status_code == 200

    def test_return_home(self):
        response = client.post("/api/v1/drone/command/return-home")
        assert response.status_code == 200

    def test_natural_command(self):
        response = client.post(
            "/api/v1/drone/command/natural",
            json={"command": "fly to the building"},
        )
        assert response.status_code == 200

    def test_model_info(self):
        response = client.get("/api/v1/detection/model/info")
        assert response.status_code == 200
