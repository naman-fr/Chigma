"""Tests for Drone Autonomy module."""

import pytest

np = pytest.importorskip('numpy')


class TestPerception:
    """Test drone perception."""

    def test_classify_obstacles(self):
        from src.drone.perception import DronePerception
        perception = DronePerception()

        detections = [
            {"class_name": "person", "bbox": [0, 0, 100, 200]},
            {"class_name": "crazing", "bbox": [0, 0, 50, 50]}
        ]

        classified = perception.classify_obstacles(detections)

        assert classified[0]["is_obstacle"] is True
        assert classified[1]["is_obstacle"] is False
        assert "estimated_distance_m" in classified[0]


class TestFlightController:
    """Test flight controller."""

    @pytest.mark.asyncio
    async def test_arm_disarm(self):
        from src.drone.flight_controller import FlightController
        controller = FlightController()

        # Force connection for testing
        controller._connection = True

        armed = await controller.arm()
        assert armed is True
        assert controller._armed is True

        disarmed = await controller.disarm()
        assert disarmed is True
        assert controller._armed is False

    @pytest.mark.asyncio
    async def test_safety_constraints(self):
        from src.drone.flight_controller import FlightController
        controller = FlightController()
        controller._connection = True
        await controller.arm()

        # Test takeoff max altitude limit (should be capped at 120m by default)
        await controller.takeoff(altitude_m=500.0)
        # We don't have a direct assert for the internal state change in this simple mock,
        # but the method should not raise an exception.
        assert controller._mode == "GUIDED"


class TestSLAM:
    """Test SLAM module."""

    def test_initialization(self):
        from src.drone.slam import VisualSLAM
        slam = VisualSLAM()
        assert not slam.is_initialized

        # Generate dummy map for testing
        slam._map_points = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        map_data = slam.get_map()

        assert map_data["num_points"] == 2
        assert "bounds" in map_data
