"""
VLM Flight Commands — Natural Language to Drone Actions
=========================================================
Parses natural language commands into actionable flight plans
using VLM scene understanding.
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger


# Supported command patterns
COMMAND_PATTERNS = {
    "fly_to": [
        r"fly\s+to\s+(?:the\s+)?(.+)",
        r"go\s+to\s+(?:the\s+)?(.+)",
        r"navigate\s+to\s+(?:the\s+)?(.+)",
        r"move\s+to\s+(?:the\s+)?(.+)",
    ],
    "orbit": [
        r"orbit\s+(?:around\s+)?(?:the\s+)?(.+)",
        r"circle\s+(?:around\s+)?(?:the\s+)?(.+)",
    ],
    "inspect": [
        r"inspect\s+(?:the\s+)?(.+)",
        r"examine\s+(?:the\s+)?(.+)",
        r"check\s+(?:the\s+)?(.+)",
    ],
    "follow": [
        r"follow\s+(?:the\s+)?(.+)",
        r"track\s+(?:the\s+)?(.+)",
    ],
    "return_home": [
        r"return\s+(?:to\s+)?home",
        r"come\s+back",
        r"rtl",
        r"go\s+home",
    ],
    "hover": [
        r"hover(?:\s+here)?",
        r"stay(?:\s+here)?",
        r"hold\s+position",
        r"stop",
    ],
    "scan_area": [
        r"scan\s+(?:the\s+)?(.+)",
        r"survey\s+(?:the\s+)?(.+)",
    ],
    "land": [
        r"land(?:\s+here)?",
        r"touch\s+down",
    ],
}


class VLMCommandParser:
    """Parse natural language flight commands.

    Uses regex pattern matching + optional VLM for scene-grounded
    command interpretation.

    Args:
        safety_validation: Enable command safety checks.
        max_altitude_m: Maximum allowed altitude.
    """

    def __init__(
        self,
        safety_validation: bool = True,
        max_altitude_m: float = 120.0,
    ) -> None:
        self.safety_validation = safety_validation
        self.max_altitude_m = max_altitude_m

    def parse(self, command: str) -> dict[str, Any]:
        """Parse a natural language command into a structured action.

        Args:
            command: Natural language command string.

        Returns:
            Parsed command dict with action type and parameters.
        """
        command_lower = command.lower().strip()

        for action, patterns in COMMAND_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, command_lower)
                if match:
                    target = match.group(1) if match.lastindex else None
                    parsed = {
                        "action": action,
                        "target": target,
                        "raw_command": command,
                        "confidence": 0.9,
                    }

                    # Extract numeric parameters
                    parsed.update(self._extract_parameters(command_lower))

                    # Safety validation
                    if self.safety_validation:
                        parsed["safety_check"] = self._validate_safety(parsed)

                    logger.info(f"Parsed command: {action} → target={target}")
                    return parsed

        # Unrecognized command
        return {
            "action": "unknown",
            "raw_command": command,
            "confidence": 0.0,
            "message": "Command not recognized. Try: fly to, orbit, inspect, return home, hover, scan",
        }

    def _extract_parameters(self, command: str) -> dict[str, Any]:
        """Extract numeric parameters from command text."""
        params: dict[str, Any] = {}

        # Altitude: "at 30 meters" / "altitude 50m"
        alt_match = re.search(r"(?:at|altitude)\s+(\d+)\s*(?:m|meters?)", command)
        if alt_match:
            params["altitude_m"] = min(float(alt_match.group(1)), self.max_altitude_m)

        # Speed: "at 5 m/s" / "speed 10"
        speed_match = re.search(r"(?:speed|at)\s+(\d+)\s*(?:m/s|ms)", command)
        if speed_match:
            params["speed_ms"] = float(speed_match.group(1))

        # Radius: "radius 20m"
        radius_match = re.search(r"radius\s+(\d+)\s*(?:m|meters?)", command)
        if radius_match:
            params["radius_m"] = float(radius_match.group(1))

        # Distance: "for 100 meters"
        dist_match = re.search(r"for\s+(\d+)\s*(?:m|meters?)", command)
        if dist_match:
            params["distance_m"] = float(dist_match.group(1))

        return params

    def _validate_safety(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Validate command against safety constraints."""
        issues = []

        alt = parsed.get("altitude_m")
        if alt and alt > self.max_altitude_m:
            issues.append(f"Altitude {alt}m exceeds max {self.max_altitude_m}m")

        speed = parsed.get("speed_ms")
        if speed and speed > 15.0:
            issues.append(f"Speed {speed}m/s exceeds safe limit 15m/s")

        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
        }
