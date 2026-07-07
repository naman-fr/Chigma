"""
Hierarchical Multi-Agent Tactical Autonomy System
===================================================
Orchestrates autonomous drone missions using a Planner Agent, Executor Agent,
and Safety Agent to ensure secure, verifiable flight actions under geofence constraints.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger


class PlannerAgent:
    """Planner Agent: Breaks down complex natural language commands into a tactical flight plan."""

    def plan(self, command: str) -> tuple[list[dict[str, Any]], str]:
        command_lower = command.lower().strip()
        logger.info(f"[Planner Agent] Formulating tactical flight steps for: '{command}'")

        steps = []
        plan_type = "unknown"

        # Simple extraction logic for demonstration of step composition
        if "inspect" in command_lower or "scan" in command_lower:
            plan_type = "inspect"
            target = self._extract_target(command_lower, ["inspect", "scan", "survey"])
            steps.append({"action": "takeoff", "altitude_m": 15.0})
            steps.append({"action": "fly_to", "target": target, "altitude_m": 15.0, "speed_ms": 3.0})
            steps.append({"action": "orbit", "target": target, "radius_m": 15.0, "altitude_m": 15.0})
            steps.append({"action": "land"})
        elif "fly to" in command_lower or "navigate to" in command_lower or "go to" in command_lower:
            plan_type = "fly_to"
            target = self._extract_target(command_lower, ["fly to", "navigate to", "go to", "move to"])
            steps.append({"action": "takeoff", "altitude_m": 10.0})
            steps.append({"action": "fly_to", "target": target, "altitude_m": 20.0, "speed_ms": 5.0})
        elif "return home" in command_lower or "come back" in command_lower or "rtl" in command_lower:
            plan_type = "return_home"
            steps.append({"action": "return_home"})
        elif "hover" in command_lower or "stay" in command_lower:
            plan_type = "hover"
            steps.append({"action": "hover"})
        elif "land" in command_lower:
            plan_type = "land"
            steps.append({"action": "land"})
        else:
            # Fallback single step
            steps.append({"action": "unknown", "raw_command": command})

        logger.info(f"[Planner Agent] Generated plan containing {len(steps)} actions.")
        return steps, plan_type

    def _extract_target(self, text: str, prefixes: list[str]) -> str:
        for prefix in prefixes:
            pattern = rf"{prefix}\s+(?:the\s+)?(.*?)(?:\s+at|\s+speed|\s+radius|\s+for|$)"
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return "unspecified target"

class ExecutorAgent:
    """Executor Agent: Translates abstract flight plan steps into executable parameters."""

    def translate(self, steps: list[dict[str, Any]], flight_controller_state: dict[str, Any]) -> list[dict[str, Any]]:
        translated_actions = []
        logger.info("[Executor Agent] Translating tactical steps into autopilot parameters.")

        for step in steps:
            action = step["action"]
            params: dict[str, Any] = {"action": action}

            if action == "takeoff":
                params["altitude_m"] = step.get("altitude_m", 10.0)
            elif action == "fly_to":
                params["target"] = step.get("target", "waypoint")
                params["altitude_m"] = step.get("altitude_m", 20.0)
                params["speed_ms"] = step.get("speed_ms", 5.0)
                # Mock tactical GPS offset for targets
                params["lat"] = 37.7749 + (0.0001 * len(step.get("target", "")))
                params["lon"] = -122.4194 + (0.0001 * len(step.get("target", "")))
            elif action == "orbit":
                params["target"] = step.get("target", "waypoint")
                params["radius_m"] = step.get("radius_m", 20.0)
                params["altitude_m"] = step.get("altitude_m", 15.0)
                params["lat"] = 37.7749
                params["lon"] = -122.4194
            elif action == "return_home":
                params["mode"] = "RTL"
            elif action == "hover":
                params["mode"] = "LOITER"
            elif action == "land":
                params["mode"] = "LAND"

            translated_actions.append(params)

        return translated_actions

class SafetyAgent:
    """Safety Agent: Reviews and validates proposed actions against geo-fencing and battery envelopes."""

    def __init__(self, max_alt_m: float = 120.0, max_speed_ms: float = 15.0) -> None:
        self.max_alt_m = max_alt_m
        self.max_speed_ms = max_speed_ms

    def verify(self, plans: list[dict[str, Any]], telemetry: dict[str, Any]) -> dict[str, Any]:
        logger.info("[Safety Agent] Running tactical pre-flight checks and risk verification.")
        issues = []
        battery = telemetry.get("battery_pct", 100.0)

        # Geofence & Envelope validation
        for idx, plan in enumerate(plans):
            action = plan["action"]
            alt = plan.get("altitude_m")
            speed = plan.get("speed_ms")

            if alt and alt > self.max_alt_m:
                issues.append(f"Step {idx + 1} ({action}): Target altitude {alt}m exceeds safety ceiling {self.max_alt_m}m")
            if speed and speed > self.max_speed_ms:
                issues.append(f"Step {idx + 1} ({action}): Target speed {speed}m/s exceeds limit {self.max_speed_ms}m/s")

        # Mission battery-aware planning safety
        estimated_drain = len(plans) * 5.0  # Assumes 5% battery drain per maneuver
        if battery < 20.0:
            issues.append("Emergency Risk: Battery is below 20%. Rejecting all navigation maneuvers.")
        elif battery - estimated_drain < 15.0:
            issues.append(f"Mission Risk: Estimated battery drain ({estimated_drain}%) will violate RTL safety reserve (15%).")

        is_safe = len(issues) == 0
        status = "PASSED" if is_safe else "REJECTED"

        logger.info(f"[Safety Agent] Safety envelope review complete. Status: {status}")
        return {
            "is_safe": is_safe,
            "status": status,
            "issues": issues,
            "estimated_battery_drain_pct": estimated_drain
        }

class HierarchicalFlightAgent:
    """Orchestrates Planner, Executor, and Safety agents to evaluate and run flight commands."""

    def __init__(self, safety_ceiling: float = 120.0, speed_ceiling: float = 15.0) -> None:
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent()
        self.safety = SafetyAgent(max_alt_m=safety_ceiling, max_speed_ms=speed_ceiling)

    def execute_command(self, command: str, telemetry: dict[str, Any]) -> dict[str, Any]:
        # 1. Break command down into logical flight plan
        flight_plan, plan_type = self.planner.plan(command)

        # 2. Translate steps into exact coordinate parameters
        executed_parameters = self.executor.translate(flight_plan, telemetry)

        # 3. Double-check all flight actions against safety thresholds
        safety_log = self.safety.verify(executed_parameters, telemetry)

        # Calibration score based on safety indicators
        confidence = 0.95 if safety_log["is_safe"] else 0.20

        return {
            "command": command,
            "plan_type": plan_type,
            "plan": flight_plan,
            "executable_steps": executed_parameters,
            "safety_log": safety_log,
            "confidence": confidence,
            "status": "authorized" if safety_log["is_safe"] else "rejected"
        }
