"""Tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_root(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_health(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_metrics(self):
        response = client.get("/metrics")
        assert response.status_code == 200


class TestDetectionEndpoints:
    """Test detection API endpoints."""

    def test_model_info(self):
        response = client.get("/api/v1/detection/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) > 0

    def test_predict_no_image(self):
        response = client.post("/api/v1/detection/predict")
        assert response.status_code == 422  # validation error: image required


class TestDroneEndpoints:
    """Test drone API endpoints."""

    def test_status(self):
        response = client.get("/api/v1/drone/status")
        assert response.status_code == 200
        data = response.json()
        assert "battery_pct" in data
        assert "mode" in data

    def test_emergency_stop(self):
        response = client.post("/api/v1/drone/command/emergency-stop")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "emergency_stop"

    def test_return_home(self):
        response = client.post("/api/v1/drone/command/return-home")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "returning"

    def test_natural_command_fly(self):
        response = client.post(
            "/api/v1/drone/command/natural",
            json={"command": "fly to the building"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parsed_action"] == "fly_to"

    def test_natural_command_inspect(self):
        response = client.post(
            "/api/v1/drone/command/natural",
            json={"command": "inspect the left wall"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parsed_action"] == "inspect"

    def test_natural_command_hover(self):
        response = client.post(
            "/api/v1/drone/command/natural",
            json={"command": "hover here"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parsed_action"] == "hover"

    def test_natural_command_unknown(self):
        response = client.post(
            "/api/v1/drone/command/natural",
            json={"command": "xyzzy blahblah"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["parsed_action"] == "unknown"


class TestVLMEndpoints:
    """Test VLM API endpoints."""

    def test_query_no_image(self):
        response = client.post("/api/v1/vlm/query?query=test")
        assert response.status_code == 422  # validation error: image required

    def test_report_no_image(self):
        response = client.post("/api/v1/vlm/report")
        assert response.status_code == 422  # validation error: image required
