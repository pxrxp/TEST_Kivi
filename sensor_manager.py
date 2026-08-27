"""
GPS-Free Sensor Manager Module

Handles accelerometer & compass sensors with Landscape mode axis mapping
and low-pass noise filtering.
"""

import math
import time
from typing import Dict, Any

import numpy as np


def build_tilt_matrix(pitch_deg: float, roll_deg: float):
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
    GPS-Free sensor manager specifically mapped for Landscape holding mode.
    """

    DEFAULT_TOLERANCE = 3.5  # Handheld level tolerance in degrees

    def __init__(self, fov_y_deg: float = 65.0):
        self.fov_y_deg = fov_y_deg
        self._sensor_available = PLYER_AVAILABLE
        self._accelerometer_enabled = False
        self._compass_enabled = False

        self._pitch_deg: float = 0.0
        self._roll_deg: float = 0.0
        self._heading_deg: float = 0.0
        
        # Exponential Moving Average smoothing factor
        self._alpha = 0.20

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
        """Poll latest sensor readings and filter landscape orientation."""
        if not self._sensor_available:
            return

        if self._accelerometer_enabled:
            try:
                acc = accelerometer.acceleration
                if acc and len(acc) >= 3 and None not in acc[:3]:
                    ax, ay, az = acc[0], acc[1], acc[2]
                    raw_pitch, raw_roll = self._compute_landscape_pitch_roll(ax, ay, az)

                    # Exponential Moving Average low-pass filter
                    self._pitch_deg = self._alpha * raw_pitch + (1.0 - self._alpha) * self._pitch_deg
                    self._roll_deg = self._alpha * raw_roll + (1.0 - self._alpha) * self._roll_deg
            except Exception:
                pass

        if self._compass_enabled:
            try:
                heading = compass.heading
                if heading is not None:
                    h_raw = float(heading) % 360.0
                    dh = (h_raw - self._heading_deg + 180) % 360 - 180
                    self._heading_deg = (self._heading_deg + 0.2 * dh) % 360.0
            except Exception:
                pass

    def _compute_landscape_pitch_roll(self, ax: float, ay: float, az: float) -> tuple:
        """
        Exact Landscape Accelerometer Mapping:
        
        AX: Short edge vertical axis when held sideways in landscape.
        AY: Long edge horizontal axis when held sideways in landscape.
        AZ: Normal axis pointing out of screen towards user.
        """
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-4:
            return 0.0, 0.0

        ax_n, ay_n, az_n = ax / norm, ay / norm, az / norm

        # Pitch = tilt forward/backward (dependent ONLY on AZ screen normal & AX vertical)
        pitch = math.degrees(math.atan2(-az_n, ax_n if abs(ax_n) > 1e-3 else 1e-3))

        # Roll = tilt side-to-side (dependent ONLY on AY horizontal & AX vertical)
        roll = math.degrees(math.atan2(ay_n, ax_n if abs(ax_n) > 1e-3 else 1e-3))

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
        """Clear, intuitive direction prompts for the user."""
        p_ok = abs(self._pitch_deg) <= self.DEFAULT_TOLERANCE
        r_ok = abs(self._roll_deg) <= self.DEFAULT_TOLERANCE

        if p_ok and r_ok:
            return "✓ LEVEL - HOLD STEADY"

        prompts = []
        if self._pitch_deg > self.DEFAULT_TOLERANCE:
            prompts.append("TILT DOWN ↓")
        elif self._pitch_deg < -self.DEFAULT_TOLERANCE:
            prompts.append("TILT UP ↑")

        if self._roll_deg > self.DEFAULT_TOLERANCE:
            prompts.append("ROTATE RIGHT ↻")
        elif self._roll_deg < -self.DEFAULT_TOLERANCE:
            prompts.append("ROTATE LEFT ↺")

        return " | ".join(prompts)

    def get_horizon_color(self) -> tuple:
        if self.is_level():
            return (0, 255, 102)    # Emerald Green
        elif abs(self._pitch_deg) <= 7.0 and abs(self._roll_deg) <= 7.0:
            return (255, 204, 0)    # Amber
        else:
            return (255, 51, 51)     # Red

    def get_sensor_data(self) -> Dict[str, Any]:
        return {
            "pitch_deg": round(self._pitch_deg, 1),
            "roll_deg": round(self._roll_deg, 1),
            "heading_deg": round(self._heading_deg % 360.0, 0),
            "fov_y_deg": self.fov_y_deg,
            "is_level": self.is_level(),
            "guidance": self.get_level_guidance(),
            "horizon_color": self.get_horizon_color(),
            "timestamp": time.time(),
        }

    def is_hardware_available(self) -> bool:
        return self._sensor_available

    def set_mock_values(self, pitch_deg: float, roll_deg: float, heading_deg: float):
        self._pitch_deg = pitch_deg
        self._roll_deg = roll_deg
        self._heading_deg = heading_deg