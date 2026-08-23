"""
GPS-Free Sensor Manager Module

Wraps plyer sensor interfaces with fallback mock sensors for desktop testing.
Uses accelerometer for pitch/roll computation and magnetometer compass for azimuth heading.
"""

import math
import time
from typing import Dict, Any


class MockAccelerometer:
    """Mock accelerometer for desktop testing."""

    def enable(self):
        pass

    def disable(self):
        pass


class MockCompass:
    """Mock compass for desktop testing."""

    def enable(self):
        pass

    def disable(self):
        pass


try:
    from plyer import accelerometer, compass
    # Test if platform module actually exists
    _ = compass.heading
    PLYER_AVAILABLE = True
except (ImportError, ModuleNotFoundError, NotImplementedError, AttributeError, Exception):
    PLYER_AVAILABLE = False
    accelerometer = MockAccelerometer()
    compass = MockCompass()


class SensorManager:
    """
    GPS-Free sensor manager using accelerometer for pitch/roll
    and magnetometer for heading.
    """

    DEFAULT_TOLERANCE = 2.0  # degrees
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
        """Start sensor services. Returns True if hardware sensors enabled."""
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

        return self._accelerometer_enabled and self._compass_enabled

    def stop(self):
        """Stop sensor services."""
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
        """Poll latest sensor readings."""
        if not self._sensor_available:
            return

        if self._accelerometer_enabled:
            try:
                ax = accelerometer.acceleration[0] or 0.0
                ay = accelerometer.acceleration[1] or 0.0
                az = accelerometer.acceleration[2] or 0.0
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
        """
        Compute pitch and roll from accelerometer readings.

        pitch = arctan2(ay, sqrt(ax^2 + az^2)) * 180/pi
        roll  = arctan2(-ax, az) * 180/pi
        """
        if az == 0 and ax == 0 and ay == 0:
            return 0.0, 0.0

        total = math.sqrt(ax * ax + az * az)
        if total < 1e-6:
            total = 1e-6

        pitch = math.degrees(math.atan2(ay, total))
        roll = math.degrees(math.atan2(-ax, az)) if abs(az) > 1e-6 else 0.0

        return pitch, roll

    def get_pitch(self) -> float:
        """Return current pitch in degrees."""
        return self._pitch_deg

    def get_roll(self) -> float:
        """Return current roll in degrees."""
        return self._roll_deg

    def get_heading(self) -> float:
        """Return current heading in degrees (0-360)."""
        return self._heading_deg % 360.0

    def is_level(self, tolerance: float = DEFAULT_TOLERANCE) -> bool:
        """Check if phone is level within tolerance (both pitch and roll)."""
        return abs(self._pitch_deg) <= tolerance and abs(self._roll_deg) <= tolerance

    def get_level_status(self) -> str:
        """Get human-readable leveling instructions."""
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
        """Return (r, g, b) color (0-255 int) for horizon line based on level status."""
        if self.is_level():
            return (0, 255, 0)  # Green
        elif abs(self._pitch_deg) <= 5.0:
            return (255, 255, 0)  # Yellow
        else:
            return (255, 0, 0)  # Red

    def get_sensor_data(self) -> Dict[str, Any]:
        """Capture structured sensor reading for this snapshot."""
        return {
            "pitch_deg": round(self._pitch_deg, 4),
            "roll_deg": round(self._roll_deg, 4),
            "heading_deg": round(self._heading_deg % 360.0, 4),
            "fov_y_deg": self.fov_y_deg,
            "is_level": self.is_level(),
            "timestamp": time.time(),
        }

    def is_hardware_available(self) -> bool:
        """Check if hardware sensors are available."""
        return self._sensor_available

    def is_mocking(self) -> bool:
        """Returns True if using mock sensors (desktop testing)."""
        return not self._sensor_available

    def set_mock_values(self, pitch_deg: float, roll_deg: float, heading_deg: float):
        """Set mock sensor values for testing."""
        self._pitch_deg = pitch_deg
        self._roll_deg = roll_deg
        self._heading_deg = heading_deg
