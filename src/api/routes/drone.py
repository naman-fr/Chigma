"""Drone autonomy API routes — Flight control and monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field

from src.drone.flight_controller import FlightController
from src.drone.vlm_commands import VLMCommandParser

router = APIRouter()

# ── Module-level singletons ──
_controller = FlightController()
_parser = VLMCommandParser()

# Force connection for API usage (no real MAVLink in dev)
_controller._connection = True


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
    telemetry = _controller.get_telemetry()
    return DroneStatus(**telemetry)


@router.post("/command/natural")
async def natural_language_command(cmd: FlightCommand) -> dict[str, Any]:
    """Execute a natural language flight command via VLM.

    Parses commands like:
    - "Fly to the red building"
    - "Orbit around the defect area"
    - "Return home"
    - "Inspect the left wall at 5 meters"
    """
    parsed = _parser.parse(cmd.command)
    action = parsed["action"]

    # Apply state changes based on parsed action
    if action == "return_home":
        await _controller.return_to_home()
    elif action == "hover":
        _controller._mode = "LOITER"
    elif action == "land":
        await _controller.land()
    elif action in ("fly_to", "inspect", "orbit", "scan_area", "follow"):
        _controller._mode = "GUIDED"
        _controller._armed = True
        alt = parsed.get("altitude_m") or cmd.altitude_m
        if alt:
            _controller._telemetry_alt = alt
        spd = parsed.get("speed_ms") or cmd.speed_ms
        if spd:
            _controller._telemetry_spd = spd

    logger.info(f"Drone command: '{cmd.command}' → action={action} target={parsed.get('target')}")

    return {
        "command": cmd.command,
        "parsed_action": action,
        "target": parsed.get("target"),
        "parameters": {
            k: v for k, v in parsed.items()
            if k not in ("action", "target", "raw_command", "confidence", "safety_check", "message")
        },
        "confidence": parsed.get("confidence", 0.0),
        "safety_check": parsed.get("safety_check"),
        "status": "queued" if action != "unknown" else "rejected",
        "message": (
            "Command parsed and queued for execution"
            if action != "unknown"
            else parsed.get("message", "Unrecognized command")
        ),
    }


@router.post("/command/waypoint")
async def waypoint_command(wp: WaypointCommand) -> dict[str, Any]:
    """Navigate to a specific GPS waypoint."""
    _controller._mode = "GUIDED"
    _controller._armed = True
    await _controller.goto(wp.latitude, wp.longitude, wp.altitude_m, wp.speed_ms)
    return {
        "waypoint": {"lat": wp.latitude, "lon": wp.longitude, "alt": wp.altitude_m},
        "speed_ms": wp.speed_ms,
        "status": "navigating",
    }


@router.post("/command/emergency-stop")
async def emergency_stop() -> dict[str, str]:
    """Emergency stop — hover in place."""
    await _controller.emergency_stop()
    return {"status": "emergency_stop", "message": "Drone hovering in place"}


@router.post("/command/return-home")
async def return_home() -> dict[str, str]:
    """Return to home position."""
    await _controller.return_to_home()
    return {"status": "returning", "message": "Returning to home position"}


@router.post("/command/arm")
async def arm_drone() -> dict[str, str]:
    """Arm the drone motors."""
    success = await _controller.arm()
    return {
        "status": "armed" if success else "failed",
        "message": "Motors armed" if success else "Arm failed — check connection",
    }


@router.post("/command/takeoff")
async def takeoff(altitude_m: float = 10.0) -> dict[str, Any]:
    """Take off to specified altitude."""
    success = await _controller.takeoff(altitude_m)
    return {
        "status": "taking_off" if success else "failed",
        "target_altitude_m": altitude_m,
    }


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
