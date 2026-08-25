"""
GPS-Free Sensor Manager Module

Wraps plyer sensor interfaces with fallback mock sensors for desktop testing.
Uses accelerometer for pitch/roll computation with low-pass filtering.
"""

import math
import time
from typing import Dict, Any

import numpy as np


def build_tilt_matrix(pitch_deg: float, roll_deg: float):
    """
    Build 3x3 camera tilt rotation matrix from phone pitch/roll.
    Camera frame convention: x right, y up, -z forward.
    """
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    ry = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    return ry @ rx


class MockAccelerometer:
    def enable(self):
        pass

    def disable(self):
        pass


class MockCompass:
    def enable(self):
        pass

    def disable(self):
        pass


try:
    from plyer import accelerometer, compass
    PLYER_AVAILABLE = True
except Exception:
    PLYER_AVAILABLE = False
    accelerometer = MockAccelerometer()
    compass = MockCompass()


class SensorManager:
    """
    GPS-Free sensor manager using accelerometer for smoothed pitch/roll
    and compass for azimuth heading.
    """

    DEFAULT_TOLERANCE = 3.0  # Practical tolerance in degrees for mobile handheld capture
    DEFAULT_FOV_Y = 65.0

    def __init__(self, fov_y_deg: float = DEFAULT_FOV_Y):
        self.fov_y_deg = fov_y_deg
        self._sensor_available = PLYER_AVAILABLE
        self._accelerometer_enabled = False
        self._compass_enabled = False
        
        self._pitch_deg: float = 0.0
        self._roll_deg: float = 0.0
        self._heading_deg: float = 0.0
        self._last_update: float = 0.0

        # EMA smoothing factor (0.25 removes handheld noise without lag)
        self._alpha = 0.25

    def start(self) -> bool:
        if not self._sensor_available:
            return False

        try:
            accelerometer.enable()
            self._accelerometer_enabled = True
        except Exception:
            self._accelerometer_enabled = False

        try:
            compass.enable()
            self._compass_enabled = True
        except Exception:
            self._compass_enabled = False

        return self._accelerometer_enabled or self._compass_enabled

    def stop(self):
        if not self._sensor_available:
            return

        try:
            if self._accelerometer_enabled:
                accelerometer.disable()
                self._accelerometer_enabled = False
        except Exception:
            pass

        try:
            if self._compass_enabled:
                compass.disable()
                self._compass_enabled = False
        except Exception:
            pass

    def update_sensors(self):
        """Poll latest sensor readings and apply low-pass EMA filter."""
        if not self._sensor_available:
            return

        if self._accelerometer_enabled:
            try:
                acc = accelerometer.acceleration
                if acc and len(acc) >= 3 and None not in acc[:3]:
                    ax, ay, az = acc[0], acc[1], acc[2]
                    raw_pitch, raw_roll = self._compute_pitch_roll(ax, ay, az)
                    
                    # Apply Exponential Moving Average filter to smooth jitter
                    self._pitch_deg = self._alpha * raw_pitch + (1.0 - self._alpha) * self._pitch_deg
                    self._roll_deg = self._alpha * raw_roll + (1.0 - self._alpha) * self._roll_deg
            except Exception:
                pass

        if self._compass_enabled:
            try:
                heading = compass.heading
                if heading is not None:
                    # Filter heading angles properly across 0/360 boundary
                    h_raw = float(heading) % 360.0
                    dh = (h_raw - self._heading_deg + 180) % 360 - 180
                    self._heading_deg = (self._heading_deg + self._alpha * dh) % 360.0
            except Exception:
                pass

        self._last_update = time.time()

    def _compute_pitch_roll(self, ax: float, ay: float, az: float) -> tuple:
        """
        Compute pitch and roll in landscape/portrait hold modes.
        """
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-4:
            return 0.0, 0.0

        ax_n, ay_n, az_n = ax / norm, ay / norm, az / norm

        # Pitch (tilting phone up/down)
        pitch = math.degrees(math.atan2(ay_n, math.sqrt(ax_n * ax_n + az_n * az_n)))
        # Roll (rotating screen left/right)
        roll = math.degrees(math.atan2(-ax_n, az_n)) if abs(az_n) > 1e-4 else 0.0

        return pitch, roll

    def get_pitch(self) -> float:
        return self._pitch_deg

    def get_roll(self) -> float:
        return self._roll_deg

    def get_heading(self) -> float:
        return self._heading_deg % 360.0

    def is_level(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        return abs(self._pitch_deg) <= tolerance and abs(self._roll_deg) <= tolerance

    def get_level_guidance(self) -> str:
        """Get directional text guidance with arrows for easy leveling."""
        p_ok = abs(self._pitch_deg) <= self.DEFAULT_TOLERANCE
        r_ok = abs(self._roll_deg) <= self.DEFAULT_TOLERANCE

        if p_ok and r_ok:
            return "✓ HOLD STEADY — LEVEL"

        actions = []
        if self._pitch_deg > self.DEFAULT_TOLERANCE:
            actions.append("▼ TILT DOWN")
        elif self._pitch_deg < -self.DEFAULT_TOLERANCE:
            actions.append("▲ TILT UP")

        if self._roll_deg > self.DEFAULT_TOLERANCE:
            actions.append("↻ ROTATE RIGHT")
        elif self._roll_deg < -self.DEFAULT_TOLERANCE:
            actions.append("↺ ROTATE LEFT")

        return " | ".join(actions)

    def get_horizon_color(self) -> tuple:
        """Return (r, g, b) 0-255 color for HUD elements."""
        if self.is_level():
            return (0, 255, 102)     # Bright Emerald Green
        elif abs(self._pitch_deg) <= 6.0 and abs(self._roll_deg) <= 6.0:
            return (255, 204, 0)     # Warning Amber
        else:
            return (255, 51, 51)      # Alert Red

    def get_sensor_data(self) -> Dict[str, Any]:
        return {
            "pitch_deg": round(self._pitch_deg, 2),
            "roll_deg": round(self._roll_deg, 2),
            "heading_deg": round(self._heading_deg % 360.0, 1),
            "fov_y_deg": self.fov_y_deg,
            "is_level": self.is_level(),
            "guidance": self.get_level_guidance(),
            "horizon_color": self.get_horizon_color(),
            "timestamp": time.time(),
        }

    def is_hardware_available(self) -> bool:
        return self._sensor_available

    def is_mocking(self) -> bool:
        return not self._sensor_available

    def set_mock_values(self, pitch_deg: float, roll_deg: float, heading_deg: float):
        self._pitch_deg = pitch_deg
        self._roll_deg = roll_deg
        self._heading_deg = heading_deg