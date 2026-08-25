"""
GPS-Free Sensor Manager Module

Wraps plyer sensor interfaces with fallback mock sensors for desktop testing.
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
    DEFAULT_TOLERANCE = 2.0
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
        if not self._sensor_available:
            return

        if self._accelerometer_enabled:
            try:
                acc = accelerometer.acceleration
                if acc and len(acc) >= 3 and None not in acc[:3]:
                    ax, ay, az = acc[0], acc[1], acc[2]
                    self._pitch_deg, self._roll_deg = self._compute_pitch_roll(ax, ay, az)
            except Exception:
                pass

        if self._compass_enabled:
            try:
                heading = compass.heading
                if heading is not None:
                    self._heading_deg = float(heading)
            except Exception:
                pass

        self._last_update = time.time()

    def _compute_pitch_roll(self, ax: float, ay: float, az: float) -> tuple:
        if az == 0 and ax == 0 and ay == 0:
            return 0.0, 0.0

        total = math.sqrt(ax * ax + az * az)
        if total < 1e-6:
            total = 1e-6

        pitch = math.degrees(math.atan2(ay, total))
        roll = math.degrees(math.atan2(-ax, az)) if abs(az) > 1e-6 else 0.0

        return pitch, roll

    def get_pitch(self) -> float:
        return self._pitch_deg

    def get_roll(self) -> float:
        return self._roll_deg

    def get_heading(self) -> float:
        return self._heading_deg % 360.0

    def is_level(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        return abs(self._pitch_deg) <= tolerance and abs(self._roll_deg) <= tolerance

    def get_level_status(self) -> str:
        threshold = 2.0
        if abs(self._pitch_deg) <= threshold and abs(self._roll_deg) <= threshold:
            return "LEVEL PHONE"
        elif self._pitch_deg > threshold:
            return "TILT UP"
        elif self._pitch_deg < -threshold:
            return "TILT DOWN"
        else:
            return "LEVEL PHONE"

    def get_horizon_color(self) -> tuple:
        if self.is_level():
            return (0, 255, 0)
        elif abs(self._pitch_deg) <= 5.0:
            return (255, 255, 0)
        else:
            return (255, 0, 0)

    def get_sensor_data(self) -> Dict[str, Any]:
        return {
            "pitch_deg": round(self._pitch_deg, 4),
            "roll_deg": round(self._roll_deg, 4),
            "heading_deg": round(self._heading_deg % 360.0, 4),
            "fov_y_deg": self.fov_y_deg,
            "is_level": self.is_level(),
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