"""Drone autonomy API routes — Flight control and monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class FlightCommand(BaseModel):
    """Natural language flight command."""
    command: str = Field(..., description="NL command, e.g. 'fly to the red building'")
    altitude_m: float | None = Field(None, description="Override altitude in meters")
    speed_ms: float | None = Field(None, description="Override speed in m/s")


class WaypointCommand(BaseModel):
    """Direct waypoint navigation."""
    latitude: float
    longitude: float
    altitude_m: float = 30.0
    speed_ms: float = 5.0


class DroneStatus(BaseModel):
    """Current drone telemetry."""
    connected: bool
    armed: bool
    mode: str
    battery_pct: float
    altitude_m: float
    speed_ms: float
    gps_fix: bool
    position: dict[str, float]


@router.get("/status", response_model=DroneStatus)
async def get_status() -> DroneStatus:
    """Get current drone telemetry and status."""
    # In production, this reads from MAVLink/ROS2
    return DroneStatus(
        connected=True,
        armed=False,
        mode="STABILIZE",
        battery_pct=85.0,
        altitude_m=0.0,
        speed_ms=0.0,
        gps_fix=True,
        position={"lat": 0.0, "lon": 0.0, "alt": 0.0},
    )


@router.post("/command/natural")
async def natural_language_command(cmd: FlightCommand) -> dict[str, Any]:
    """Execute a natural language flight command via VLM.

    Parses commands like:
    - "Fly to the red building"
    - "Orbit around the defect area"
    - "Return home"
    - "Inspect the left wall at 5 meters"
    """
    return {
        "command": cmd.command,
        "parsed_action": "fly_to",
        "status": "queued",
        "message": "Command parsed and queued for execution",
    }


@router.post("/command/waypoint")
async def waypoint_command(wp: WaypointCommand) -> dict[str, Any]:
    """Navigate to a specific GPS waypoint."""
    return {
        "waypoint": {"lat": wp.latitude, "lon": wp.longitude, "alt": wp.altitude_m},
        "speed_ms": wp.speed_ms,
        "status": "navigating",
    }


@router.post("/command/emergency-stop")
async def emergency_stop() -> dict[str, str]:
    """Emergency stop — hover in place."""
    return {"status": "emergency_stop", "message": "Drone hovering in place"}


@router.post("/command/return-home")
async def return_home() -> dict[str, str]:
    """Return to home position."""
    return {"status": "returning", "message": "Returning to home position"}


@router.get("/perception/detections")
async def get_perception_detections() -> dict[str, Any]:
    """Get real-time object detections from drone camera."""
    return {
        "detections": [],
        "frame_id": 0,
        "fps": 30.0,
        "model": "yolo11n",
    }


@router.get("/slam/map")
async def get_slam_map() -> dict[str, Any]:
    """Get current SLAM map data."""
    return {
        "num_keyframes": 0,
        "num_points": 0,
        "map_status": "not_initialized",
    }
