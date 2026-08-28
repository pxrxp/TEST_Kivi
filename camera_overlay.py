"""
Camera Overlay Module

Provides a transparent landscape overlay with center target and artificial horizon line.
"""

import math
import time
from typing import Optional, Any

from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock


class CameraOverlay(FloatLayout):
    """
    Transparent Landscape Heads-Up Display overlay.
    """

    def __init__(
        self,
        camera: Any = None,
        fov_y_deg: float = 65.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.camera = camera
        self.fov_y_deg = fov_y_deg

        self.sensor_manager = None
        self._last_sensor_data = {}
        self._update_event = None

        self._init_hud_graphics()
        self._init_hud_widgets()

        self.bind(pos=self._update_pos_size, size=self._update_pos_size)

    def _init_hud_graphics(self):
        """Initialize canvas graphics across full screen (NO solid background!)."""
        with self.canvas:
            self.hud_color = Color(0.0, 1.0, 0.4, 0.9)

            # Center Target Reticle [+]
            self.center_target = Line(points=[], width=2.0)

            # Dynamic Horizon Line
            self.horizon_line = Line(points=[], width=3.5)

            # Top semi-transparent banner background
            self.banner_color = Color(0.0, 0.0, 0.0, 0.6)
            self.banner_rect = Rectangle(pos=(0, 0), size=(0, 0))

    def _init_hud_widgets(self):
        """Create status text labels using relative layout hints."""
        self.status_label = Label(
            text="HOLD PHONE LEVEL",
            color=(0.0, 1.0, 0.4, 1.0),
            font_size="18sp",
            bold=True,
            size_hint=(0.5, 0.08),
            pos_hint={"center_x": 0.5, "top": 0.98},
        )

        self.pitch_label = Label(
            text="PITCH: 0.0°",
            color=(1.0, 1.0, 1.0, 0.9),
            font_size="14sp",
            bold=True,
            size_hint=(0.25, 0.08),
            pos_hint={"x": 0.02, "top": 0.98},
        )

        self.roll_label = Label(
            text="ROLL: 0.0°",
            color=(1.0, 1.0, 1.0, 0.9),
            font_size="14sp",
            bold=True,
            size_hint=(0.25, 0.08),
            pos_hint={"right": 0.98, "top": 0.98},
        )

        self.add_widget(self.status_label)
        self.add_widget(self.pitch_label)
        self.add_widget(self.roll_label)

    def _update_pos_size(self, *args):
        """Position HUD graphics relative to full screen size."""
        w = float(self.width) if self.width > 1 else 1280.0
        h = float(self.height) if self.height > 1 else 720.0

        cx, cy = w / 2.0, h / 2.0

        # Center Crosshair
        r = min(w, h) * 0.035
        self.center_target.points = [
            cx - r, cy, cx + r, cy,
            cx, cy - r, cx, cy + r
        ]

        # Top Banner Overlay
        banner_h = h * 0.08
        self.banner_rect.pos = (0, h - banner_h)
        self.banner_rect.size = (w, banner_h)

        if self._last_sensor_data:
            self._update_display(self._last_sensor_data)

    def update_sensor_data(self, sensor_manager):
        self.sensor_manager = sensor_manager
        if sensor_manager:
            self._update_display(sensor_manager.get_sensor_data())

    def _update_display(self, sensor_data: dict):
        pitch = sensor_data.get("pitch_deg", 0.0)
        roll = sensor_data.get("roll_deg", 0.0)
        guidance = sensor_data.get("guidance", "HOLD LEVEL")
        r, g, b = sensor_data.get("horizon_color", (0, 255, 102))

        # Color updates
        self.hud_color.rgb = (r / 255.0, g / 255.0, b / 255.0)
        self.status_label.text = guidance
        self.status_label.color = (r / 255.0, g / 255.0, b / 255.0, 1.0)

        self.pitch_label.text = f"PITCH: {pitch:+.1f}°"
        self.roll_label.text = f"ROLL: {roll:+.1f}°"

        w = float(self.width) if self.width > 1 else 1280.0
        h = float(self.height) if self.height > 1 else 720.0
        cx, cy = w / 2.0, h / 2.0

        # Vertical translation per degree of pitch
        px_per_deg = h / max(self.fov_y_deg, 30.0)
        y_shift = -pitch * px_per_deg

        # Roll rotation angle
        roll_rad = math.radians(-roll)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        # Horizon line length
        half_w = w * 0.32

        # Rotated endpoints
        x1_loc, y1_loc = -half_w, y_shift
        x2_loc, y2_loc = half_w, y_shift

        x1 = cx + (x1_loc * cos_r - y1_loc * sin_r)
        y1 = cy + (x1_loc * sin_r + y1_loc * cos_r)
        x2 = cx + (x2_loc * cos_r - y2_loc * sin_r)
        y2 = cy + (x2_loc * sin_r + y2_loc * cos_r)

        self.horizon_line.points = [x1, y1, x2, y2]
        self._last_sensor_data = sensor_data

    def start_update_loop(self, interval: float = 0.08):
        if self._update_event is None:
            self._update_event = Clock.schedule_interval(self._update_loop, interval)

    def _update_loop(self, dt):
        if self.sensor_manager:
            self._update_display(self.sensor_manager.get_sensor_data())

    def stop_update_loop(self):
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None

    def capture_photo(self) -> Optional[dict]:
        if self.camera and hasattr(self.camera, "texture") and self.camera.texture:
            try:
                tex = self.camera.texture
                return {
                    "texture": tex,
                    "width": tex.size[0],
                    "height": tex.size[1],
                    "sensor_data": getattr(self, "_last_sensor_data", None),
                    "timestamp": time.time(),
                }
            except Exception as e:
                print(f"[OVERLAY] Texture read error: {e}")
        return None