"""
Flight Controller — Autopilot Interface & Kinematics Simulator
================================================================
A physics-based drone kinematics flight simulator with wind turbulence perturbations,
3D waypoint navigation (NED coordinates), dynamic battery depletion, and safety overrides.
"""

from __future__ import annotations

import datetime
import math
import random
import time
from typing import Any

from loguru import logger


class FlightController:
    """MAVLink-based flight controller simulation.

    Simulates high-fidelity multirotor flight dynamics using Euler integration:
    - 3D Position (x, y, z) and Velocity (vx, vy, vz)
    - Heading (yaw) tracking target directions
    - Dynamic battery drainage based on thrust and time
    - Simulates wind gusts and environmental turbulence
    - Restricts altitude and velocity safety envelopes
    """

    def __init__(
        self,
        connection_string: str = "udp:127.0.0.1:14540",
        safety_config: dict[str, Any] | None = None,
    ) -> None:
        self.connection_string = connection_string
        self.safety = safety_config or {
            "max_tilt_deg": 35.0,
            "min_altitude_m": 0.0,
            "geofence_radius_m": 150.0,
            "max_altitude_m": 120.0,
        }
        self._connection = None
        self._armed = False
        self._mode = "STABILIZE"
        self._last_update = time.perf_counter()

        # Physical states (NED coordinates: x=North, y=East, z=Down / Altitude)
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0      # Altitude in meters (upwards)
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._heading = 0.0  # yaw angle in degrees [0-360]
        self._telemetry_bat = 100.0

        # Flight targets
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_z = 0.0
        self._target_speed = 5.0

        logger.info(f"FlightController Simulation loaded | Connection: {connection_string}")

    async def connect(self) -> bool:
        """Establish simulation connection."""
        self._connection = True
        logger.info("MAVLink link established with autopilot simulator.")
        return True

    async def arm(self) -> bool:
        """Arm the drone motors."""
        if not self._connection:
            logger.error("Not connected to autopilot")
            return False
        self._armed = True
        self._mode = "GUIDED"
        logger.info("Motors ARMED. Autopilot mode set to GUIDED.")
        return True

    async def disarm(self) -> bool:
        """Disarm the drone motors."""
        self._armed = False
        self._vx = self._vy = self._vz = 0.0
        self._mode = "STABILIZE"
        logger.info("Motors DISARMED.")
        return True

    async def takeoff(self, altitude_m: float = 10.0) -> bool:
        """Take off to specified altitude."""
        if not self._armed:
            logger.error("Takeoff command rejected: Motors disarmed.")
            return False

        altitude_m = min(altitude_m, self.safety["max_altitude_m"])
        altitude_m = max(altitude_m, 2.0)

        self._mode = "GUIDED"
        self._target_z = altitude_m
        self._target_x = self._x
        self._target_y = self._y
        logger.info(f"Takeoff initiated. Climbing to safety ceiling: {altitude_m}m")
        return True

    async def goto(
        self,
        lat: float,
        lon: float,
        alt: float,
        speed_ms: float = 5.0,
    ) -> bool:
        """Navigate to GPS coordinates (maps to local NED coordinates)."""
        if not self._armed:
            logger.error("Goto rejected: Motors disarmed.")
            return False

        # Convert simple lat/lon displacement to local meters
        # Center reference point: lat 37.7749, lon -122.4194
        dy = (lon - (-122.4194)) * 111320.0 * 0.8  # approximate meters per degree lon
        dx = (lat - 37.7749) * 110574.0          # approximate meters per degree lat

        return await self.goto_local(dx, dy, alt, speed_ms)

    async def goto_local(
        self,
        x: float, y: float, z: float,
        speed_ms: float = 3.0,
    ) -> bool:
        """Navigate to local NED position."""
        if not self._armed:
            logger.error("Goto rejected: Motors disarmed.")
            return False

        # Constrain to geofence and altitude safety ceiling
        dist = (x**2 + y**2)**0.5
        if dist > self.safety["geofence_radius_m"]:
            scale = self.safety["geofence_radius_m"] / dist
            x *= scale
            y *= scale
            logger.warning(f"Waypoint bounded to safety geofence limit ({self.safety['geofence_radius_m']}m)")

        z = min(z, self.safety["max_altitude_m"])
        z = max(z, self.safety["min_altitude_m"])

        self._target_x = x
        self._target_y = y
        self._target_z = z
        self._target_speed = min(speed_ms, 15.0)
        self._mode = "GUIDED"
        logger.info(f"Navigating to NED target: ({x:.1f}, {y:.1f}, {z:.1f}) at {self._target_speed}m/s")
        return True

    async def orbit(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float = 20.0,
        altitude_m: float = 30.0,
    ) -> bool:
        """Orbit around a target coordinate point."""
        # Setup target coordinates to orbit
        dy = (center_lon - (-122.4194)) * 111320.0 * 0.8
        dx = (center_lat - 37.7749) * 110574.0

        # Simple orbit strategy: navigate to radius offset first
        return await self.goto_local(dx + radius_m, dy, altitude_m)

    async def return_to_home(self) -> bool:
        """Return to launch coordinates (0, 0) and land."""
        self._mode = "RTL"
        self._target_x = 0.0
        self._target_y = 0.0
        self._target_z = 15.0  # Return altitude
        self._target_speed = 6.0
        logger.info("Initiating Return To Launch (RTL) mode.")
        return True

    async def emergency_stop(self) -> bool:
        """Emergency stop — immediately brake and hover in place."""
        self._mode = "LOITER"
        self._target_x = self._x
        self._target_y = self._y
        self._target_z = self._z
        self._vx = self._vy = self._vz = 0.0
        logger.warning("EMERGENCY INTERRUPT ACTIVE — hovering in place.")
        return True

    async def land(self) -> bool:
        """Land at current horizontal position."""
        self._mode = "LAND"
        self._target_z = 0.0
        self._target_speed = 1.5
        logger.info("Landing sequence initiated.")
        return True

    def _update_physics(self) -> None:
        """Runs Euler physics integration to update positions and deplete battery."""
        now = time.perf_counter()
        dt = min(now - self._last_update, 1.0)
        self._last_update = now

        if not self._armed:
            return

        # Deplete battery
        base_drain = 0.05  # 0.05% per second base power
        thrust_drain = (self._vx**2 + self._vy**2 + self._vz**2)**0.5 * 0.03
        self._telemetry_bat = max(0.0, self._telemetry_bat - (base_drain + thrust_drain) * dt)

        # Dynamic Wind Turbulence / Perturbations (DRDO rugged environment)
        wind_x = random.uniform(-0.3, 0.3)
        wind_y = random.uniform(-0.3, 0.3)
        wind_z = random.uniform(-0.1, 0.1)

        # Target tracking vectors
        dx = self._target_x - self._x
        dy = self._target_y - self._y
        dz = self._target_z - self._z
        dist_3d = (dx**2 + dy**2 + dz**2)**0.5

        if dist_3d > 0.5:
            # Proportional velocity vector toward waypoint
            vx_ideal = (dx / dist_3d) * self._target_speed
            vy_ideal = (dy / dist_3d) * self._target_speed
            vz_ideal = (dz / dist_3d) * self._target_speed

            # Smooth acceleration (simulate drone mass inertia)
            accel_coeff = 1.8 * dt
            self._vx = self._vx * (1 - accel_coeff) + vx_ideal * accel_coeff + wind_x * dt
            self._vy = self._vy * (1 - accel_coeff) + vy_ideal * accel_coeff + wind_y * dt
            self._vz = self._vz * (1 - accel_coeff) + vz_ideal * accel_coeff + wind_z * dt
        else:
            # Arrived at waypoint, smooth deceleration
            decel_coeff = 2.5 * dt
            self._vx *= (1 - decel_coeff)
            self._vy *= (1 - decel_coeff)
            self._vz *= (1 - decel_coeff)

            # Auto-disarm if landing complete
            if self._mode == "LAND" and self._z <= 0.1:
                self._z = 0.0
                self._armed = False
                self._mode = "STABILIZE"
                logger.info("Landing complete. Autopilot disarmed.")

        # Update position states
        self._x += self._vx * dt
        self._y += self._vy * dt
        self._z = max(0.0, self._z + self._vz * dt)

        # Update heading towards movement velocity
        if (self._vx**2 + self._vy**2) > 0.1:
            target_heading = math.degrees(math.atan2(self._vy, self._vx))
            if target_heading < 0:
                target_heading += 360.0
            # Interpolate heading rotation
            self._heading = (self._heading * 0.9 + target_heading * 0.1) % 360.0

    def get_telemetry(self) -> dict[str, Any]:
        """Update physics dynamics and return current simulated state telemetry."""
        self._update_physics()

        # Map local displacement back to GPS coordinates
        # Center reference point: lat 37.7749, lon -122.4194
        current_lat = 37.7749 + (self._x / 110574.0)
        current_lon = -122.4194 + (self._y / (111320.0 * 0.8))

        return {
            "connected": self._connection is not None,
            "armed": self._armed,
            "mode": self._mode,
            "battery_pct": round(self._telemetry_bat, 2),
            "altitude_m": round(self._z, 2),
            "speed_ms": round((self._vx**2 + self._vy**2 + self._vz**2)**0.5, 2),
            "heading_deg": round(self._heading, 1),
            "gps_fix": True,
            "satellites": 14,
            "position": {
                "lat": current_lat,
                "lon": current_lon,
                "alt": round(self._z, 2),
                "x": round(self._x, 2),
                "y": round(self._y, 2)
            },
        }
