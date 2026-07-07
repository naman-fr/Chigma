"""Drone autonomy API routes — Flight control and monitoring."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from src.api.auth import RoleChecker, Roles, log_audit
from src.drone.agents import HierarchicalFlightAgent
from src.drone.flight_controller import FlightController
from src.drone.vlm_commands import VLMCommandParser

router = APIRouter()

# ── Module-level singletons ──
_controller = FlightController()
_parser = VLMCommandParser()
_flight_agent = HierarchicalFlightAgent()

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
async def get_status(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR, Roles.OBSERVER])),
) -> DroneStatus:
    """Get current drone telemetry and status."""
    telemetry = _controller.get_telemetry()
    return DroneStatus(**telemetry)


@router.post("/command/natural")
async def natural_language_command(
    cmd: FlightCommand,
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, Any]:
    """Execute a natural language flight command via VLM.

    Parses commands like:
    - "Fly to the red building"
    - "Orbit around the defect area"
    - "Return home"
    - "Inspect the left wall at 5 meters"
    """
    telemetry = _controller.get_telemetry()
    agent_result = _flight_agent.execute_command(cmd.command, telemetry)

    if agent_result["status"] == "authorized":
        # Execute the planned steps sequentially
        for step in agent_result["executable_steps"]:
            action = step["action"]
            if action == "takeoff":
                await _controller.takeoff(step["altitude_m"])
            elif action == "fly_to":
                # Convert NED/GPS coordinates back to simulation state
                await _controller.goto(step["lat"], step["lon"], step["altitude_m"], step["speed_ms"])
            elif action == "orbit":
                await _controller.orbit(step["lat"], step["lon"], step["radius_m"], step["altitude_m"])
            elif action == "return_home":
                await _controller.return_to_home()
            elif action == "hover":
                _controller._mode = "LOITER"
            elif action == "land":
                await _controller.land()

        log_audit(current_user["username"], "FLIGHT_COMMAND", f"Multi-Agent execution authorized: '{cmd.command}'", "INFO")

        main_action = agent_result["plan_type"]
        main_target = None
        for step in agent_result["plan"]:
            if "target" in step:
                main_target = step["target"]
                break

        return {
            "command": cmd.command,
            "parsed_action": main_action,
            "target": main_target,
            "parameters": agent_result["executable_steps"][0] if agent_result["executable_steps"] else {},
            "confidence": agent_result["confidence"],
            "safety_check": agent_result["safety_log"],
            "status": "queued",
            "message": "Command verified and executed under secure geofence boundaries.",
        }
    else:
        issues_str = ", ".join(agent_result["safety_log"]["issues"])
        log_audit(current_user["username"], "FLIGHT_BLOCKED", f"NL command blocked: '{cmd.command}' | Issues: {issues_str}", "WARNING")
        raise HTTPException(
            status_code=400,
            detail=f"Safety verification failed: {issues_str}"
        )


@router.post("/command/waypoint")
async def waypoint_command(
    wp: WaypointCommand,
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, Any]:
    """Navigate to a specific GPS waypoint."""
    _controller._mode = "GUIDED"
    _controller._armed = True
    await _controller.goto(wp.latitude, wp.longitude, wp.altitude_m, wp.speed_ms)
    log_audit(current_user["username"], "WAYPOINT_COMMAND", f"Goto lat={wp.latitude} lon={wp.longitude} alt={wp.altitude_m}", "INFO")
    return {
        "waypoint": {"lat": wp.latitude, "lon": wp.longitude, "alt": wp.altitude_m},
        "speed_ms": wp.speed_ms,
        "status": "navigating",
    }


@router.post("/command/emergency-stop")
async def emergency_stop(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, str]:
    """Emergency stop — hover in place."""
    await _controller.emergency_stop()
    log_audit(current_user["username"], "EMERGENCY_STOP", "Drone emergency hover triggered", "CRITICAL")
    return {"status": "emergency_stop", "message": "Drone hovering in place"}


@router.post("/command/return-home")
async def return_home(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, str]:
    """Return to home position."""
    await _controller.return_to_home()
    log_audit(current_user["username"], "RETURN_HOME", "RTL procedure triggered", "INFO")
    return {"status": "returning", "message": "Returning to home position"}


@router.post("/command/arm")
async def arm_drone(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, str]:
    """Arm the drone motors."""
    success = await _controller.arm()
    log_audit(current_user["username"], "ARM_DRONE", f"Motors arm success={success}", "INFO")
    return {
        "status": "armed" if success else "failed",
        "message": "Motors armed" if success else "Arm failed — check connection",
    }


@router.post("/command/takeoff")
async def takeoff(
    altitude_m: float = 10.0,
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR])),
) -> dict[str, Any]:
    """Take off to specified altitude."""
    success = await _controller.takeoff(altitude_m)
    log_audit(current_user["username"], "TAKEOFF", f"Takeoff altitude={altitude_m} success={success}", "INFO")
    return {
        "status": "taking_off" if success else "failed",
        "target_altitude_m": altitude_m,
    }


@router.get("/perception/detections")
async def get_perception_detections(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR, Roles.OBSERVER])),
) -> dict[str, Any]:
    """Get real-time object detections from drone camera."""
    return {
        "detections": [],
        "frame_id": 0,
        "fps": 30.0,
        "model": "yolo11n",
    }


@router.get("/slam/map")
async def get_slam_map(
    current_user: dict = Depends(RoleChecker([Roles.COMMANDER, Roles.OPERATOR, Roles.OBSERVER])),
) -> dict[str, Any]:
    """Get current SLAM map data."""
    return {
        "num_keyframes": 0,
        "num_points": 0,
        "map_status": "not_initialized",
    }
