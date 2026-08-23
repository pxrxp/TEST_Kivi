"""
Camera Overlay Module

Implements live camera preview with real-time phone level indicators
and horizon line for skyline geolocation capture.
"""

import time
import math
from typing import Optional, Tuple, Any

from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock


class CameraOverlay(FloatLayout):
    """
    Live camera preview overlay with real-time phone level guidance.

    Features:
    - Live camera feed display
    - Artificial horizon / bubble level overlay
    - Pitch/roll threshold detection (±2.0°)
    - Color-changing horizon line (green when level)
    - Visual text prompts for tilt correction
    """

    def __init__(
        self,
        camera: Any,
        screen_width: int,
        screen_height: int,
        fov_y_deg: float = 65.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.camera = camera
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_y_deg = fov_y_deg
        self.fov_y_rad = math.radians(fov_y_deg)

        # Level thresholds
        self.TILT_THRESHOLD = 2.0  # degrees
        self.COLOR_YELLOW_THRESHOLD = 5.0  # degrees
        self.CAPTURE_ENABLED_THRESHOLD = 2.0  # degrees

        # UI dimensions
        self.crosshair_size = min(screen_width, screen_height) // 12
        self.crosshair_line_width = 2.0

        # Horizon line dimensions
        self.horizon_line_width = screen_width // 20
        self.horizon_padding = screen_height // 20

        # Text display
        self.status_text = "HOLD LEVEL"
        self.pitch_text = ""
        self.roll_text = ""

        # Sensor manager reference
        self.sensor_manager = None

        # Drawables
        self._init_drawables()

        # Clock event for updates
        self._update_event = None

    def _init_drawables(self):
        """Initialize all drawable graphics elements (canvas + widgets)."""
        # 1. Canvas Graphics Instructions ONLY (Color, Rectangle, Line)
        with self.canvas:
            self.bg_color = Color(0.0, 0.0, 0.0, 1.0)
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)

            self.crosshair_color = Color(0.0, 1.0, 0.0, 1.0)
            self.crosshair = Line(
                rectangle=(0, 0, 0, 0), width=self.crosshair_line_width
            )

            self.horizon_color = Color(0.0, 1.0, 0.0, 1.0)
            self.horizon_line = Line(points=[], width=self.horizon_line_width)

            self.banner_color = Color(0.0, 0.0, 0.0, 0.7)
            self.banner_rect = Rectangle(
                pos=(0, self.screen_height - 50), size=(self.screen_width, 50)
            )

        # 2. Add Kivy Widgets Separately (NOT inside canvas context)
        self.status_label = Label(
            text=self.status_text,
            color=(0.0, 1.0, 0.0, 1.0),
            font_size="18sp",
            size_hint=(1, None),
            height=30,
            pos=(0, self.screen_height - 45),
        )
        self.pitch_label = Label(
            text="PITCH: 0.0°",
            color=(0.0, 1.0, 0.0, 1.0),
            font_size="14sp",
            size_hint=(0.5, None),
            height=20,
            pos=(self.screen_width * 0.05, self.screen_height - 30),
        )
        self.roll_label = Label(
            text="ROLL: 0.0°",
            color=(0.0, 1.0, 0.0, 1.0),
            font_size="14sp",
            size_hint=(0.5, None),
            height=20,
            pos=(self.screen_width * 0.5, self.screen_height - 30),
        )

        self.add_widget(self.status_label)
        self.add_widget(self.pitch_label)
        self.add_widget(self.roll_label)

        # Bind to resize events
        self.bind(pos=self._update_pos, size=self._update_size)

    def _update_pos(self, instance, value):
        """Update positions on layout change."""
        self._update_pos_size()

    def _update_size(self, instance, value):
        """Update sizes on layout change."""
        self._update_pos_size()

    def _update_pos_size(self):
        """Update all positions and sizes."""
        # Background
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

        # Crosshair center
        cx = self.screen_width // 2
        cy = self.screen_height // 2
        half = self.crosshair_size
        self.crosshair.points = [
            cx - half,
            cy,
            cx,
            cy - half,
            cx + half,
            cy,
            cx,
            cy + half,
            cx - half,
            cy,
        ]

        # Horizon line (spans from left to right at center)
        x1 = 0 + self.horizon_padding
        x2 = self.screen_width - self.horizon_padding
        y = self.screen_height // 2
        self.horizon_line.points = [x1, y, x2, y]

        # Banner
        self.banner_rect.pos = (0, self.screen_height - 50)
        self.banner_rect.size = (self.screen_width, 50)

        # Status label - use absolute pos (NOT pos_hint with pixel y values)
        self.status_label.pos = (0, self.screen_height - 45)

        # Pitch/roll labels - use absolute pos
        self.pitch_label.pos = (self.screen_width * 0.05, self.screen_height - 30)
        self.roll_label.pos = (self.screen_width * 0.5, self.screen_height - 30)

    def update_sensor_data(self, sensor_manager):
        """Update from sensor manager and refresh display."""
        self.sensor_manager = sensor_manager
        if sensor_manager:
            self._update_display(sensor_manager.get_sensor_data())

    def _update_display(self, sensor_data: dict):
        """Update UI display based on sensor data."""
        pitch = sensor_data.get("pitch_deg", 0.0)
        roll = sensor_data.get("roll_deg", 0.0)
        is_level = sensor_data.get("is_level", False)
        timestamp = sensor_data.get("timestamp", time.time())

        # Update pitch/roll text
        self.pitch_label.text = f"PITCH: {pitch:+.1f}°"
        self.roll_label.text = f"ROLL: {roll:+.1f}°"

        # Determine horizon color
        horizon_color = sensor_data.get("horizon_color", (0, 255, 0))
        r, g, b = horizon_color

        # Update canvas colors
        self.crosshair_color.rgb = (r / 255.0, g / 255.0, b / 255.0)
        self.horizon_color.rgb = (r / 255.0, g / 255.0, b / 255.0)

        # Update status text - colors normalized to [0.0, 1.0]
        if is_level:
            self.status_text = "LEVEL PHONE"
            self.status_label.color = (0.0, 1.0, 0.0, 1.0)
            self.pitch_label.color = (0.0, 1.0, 0.0, 1.0)
            self.roll_label.color = (0.0, 1.0, 0.0, 1.0)
        elif abs(pitch) > 5.0:
            self.status_text = "TILT DOWN"
            self.status_label.color = (1.0, 0.0, 0.0, 1.0)
            self.pitch_label.color = (1.0, 0.0, 0.0, 1.0)
            self.roll_label.color = (1.0, 0.0, 0.0, 1.0)
        elif pitch > 2.0:
            self.status_text = "TILT UP"
            self.status_label.color = (1.0, 1.0, 0.0, 1.0)
            self.pitch_label.color = (1.0, 1.0, 0.0, 1.0)
            self.roll_label.color = (1.0, 1.0, 0.0, 1.0)
        elif abs(roll) > 2.0:
            self.status_text = "LEVEL PHONE"
            self.status_label.color = (0.0, 1.0, 0.0, 1.0)
            self.pitch_label.color = (0.0, 1.0, 0.0, 1.0)
            self.roll_label.color = (0.0, 1.0, 0.0, 1.0)
        else:
            self.status_text = "HOLD LEVEL"
            self.status_label.color = (1.0, 1.0, 0.0, 1.0)
            self.pitch_label.color = (1.0, 1.0, 0.0, 1.0)
            self.roll_label.color = (1.0, 1.0, 0.0, 1.0)

        # Store for access
        self._last_sensor_data = sensor_data

    def start_update_loop(self, interval: float = 0.1):
        """Start the update loop for real-time sensor data."""
        if self._update_event is None:
            self._update_event = Clock.schedule_interval(self._update_loop, interval)

    def _update_loop(self, dt):
        """Internal update loop."""
        if self.sensor_manager:
            self._update_display(self.sensor_manager.get_sensor_data())

    def stop_update_loop(self):
        """Stop the update loop."""
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None

    def capture_photo(self) -> Optional[Tuple]:
        """
        Capture the current camera frame.

        Returns tuple of (frame_surface, sensor_data) or None.
        """
        if self.camera and hasattr(self.camera, "texture"):
            texture = self.camera.texture
            if texture:
                # Get pixel data from texture
                # Convert to numpy array for processing
                try:
                    # Convert texture to image-compatible format
                    pixels = texture.pixels
                    width, height = texture.size
                    return {
                        "texture": texture,
                        "width": width,
                        "height": height,
                        "sensor_data": getattr(self, "_last_sensor_data", None),
                        "timestamp": time.time(),
                    }
                except Exception:
                    pass
        return None

    def set_capture_button_enabled(self, enabled: bool):
        """Enable/disable the capture button based on level status."""
        self._capture_enabled = enabled
        if enabled:
            self.status_text = "LEVEL PHONE - CAPTURE READY"
            self.status_label.color = (0.0, 1.0, 0.0, 1.0)
        else:
            self.status_text = "TILT - ADJUST PHONE"
            self.status_label.color = (1.0, 0.0, 0.0, 1.0)


class LevelBanner(BoxLayout):
    """
    Floating level status banner that displays phone orientation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (1, None)
        self.height = 40
        self.pos_hint = {"y": 1}
        self.spacing = 10

        # Level indicator (normalized floats)
        self.indicator_color = (0.0, 1.0, 0.0, 1.0)
        self.indicator_size = 20

        # Text
        self.status_text = "HOLD LEVEL"

        # Add widgets
        self.indicator = Label(text="●", color=self.indicator_color, font_size="24sp")
        self.status_label = Label(
            text=self.status_text, color=self.indicator_color, font_size="16sp"
        )

        self.add_widget(self.indicator)
        self.add_widget(self.status_label)

    def update_status(self, is_level: bool, pitch: float, roll: float):
        """Update the banner status display."""
        if is_level:
            self.indicator_color = (0.0, 1.0, 0.0, 1.0)
            self.status_text = "LEVEL"
        elif abs(pitch) > 5.0:
            self.indicator_color = (1.0, 0.0, 0.0, 1.0)
            self.status_text = "TILT DOWN"
        elif pitch > 2.0:
            self.indicator_color = (1.0, 1.0, 0.0, 1.0)
            self.status_text = "TILT UP"
        elif abs(roll) > 2.0:
            self.indicator_color = (1.0, 0.65, 0.0, 1.0)  # Orange (normalized)
            self.status_text = "LEVEL PHONE"
        else:
            self.indicator_color = (1.0, 1.0, 0.0, 1.0)
            self.status_text = "HOLD LEVEL"

        self.indicator.color = self.indicator_color
        self.status_label.color = self.indicator_color
        self.status_label.text = self.status_text
