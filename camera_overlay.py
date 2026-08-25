"""
Camera Overlay Module

Implements live camera preview with real-time phone level indicators
and horizon line for skyline geolocation capture.
"""

import time
import math
from typing import Optional, Tuple, Any

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.clock import Clock


class CameraOverlay(FloatLayout):
    """
    Live camera preview overlay with real-time phone level guidance.
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

        self.TILT_THRESHOLD = 2.0
        self.COLOR_YELLOW_THRESHOLD = 5.0
        self.CAPTURE_ENABLED_THRESHOLD = 2.0

        self.crosshair_size = min(screen_width, screen_height) // 12
        self.crosshair_line_width = 2.0

        self.horizon_line_width = screen_width // 20
        self.horizon_padding = screen_height // 20

        self.status_text = "HOLD LEVEL"
        self.pitch_text = ""
        self.roll_text = ""

        self.sensor_manager = None
        self._init_drawables()
        self._update_event = None

    def _init_drawables(self):
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

        self.bind(pos=self._update_pos, size=self._update_size)

    def _update_pos(self, instance, value):
        self._update_pos_size()

    def _update_size(self, instance, value):
        self._update_pos_size()

    def _update_pos_size(self):
        w = int(self.width) if self.width > 1 else self.screen_width
        h = int(self.height) if self.height > 1 else self.screen_height

        self.bg_rect.pos = self.pos
        self.bg_rect.size = (w, h)

        cx = w // 2
        cy = h // 2
        half = min(w, h) // 12
        self.crosshair.points = [
            cx - half, cy,
            cx, cy - half,
            cx + half, cy,
            cx, cy + half,
            cx - half, cy,
        ]

        padding = w // 20
        x1 = padding
        x2 = w - padding
        y = h // 2
        self.horizon_line.points = [x1, y, x2, y]

        self.banner_rect.pos = (0, h - 50)
        self.banner_rect.size = (w, 50)

        self.status_label.pos = (0, h - 45)
        self.status_label.size = (w, 30)

        self.pitch_label.pos = (w * 0.05, h - 30)
        self.pitch_label.size = (w * 0.4, 20)
        self.roll_label.pos = (w * 0.5, h - 30)
        self.roll_label.size = (w * 0.4, 20)

    def update_sensor_data(self, sensor_manager):
        self.sensor_manager = sensor_manager
        if sensor_manager:
            self._update_display(sensor_manager.get_sensor_data())

    def _update_display(self, sensor_data: dict):
        pitch = sensor_data.get("pitch_deg", 0.0)
        roll = sensor_data.get("roll_deg", 0.0)
        is_level = sensor_data.get("is_level", False)

        self.pitch_label.text = f"PITCH: {pitch:+.1f}°"
        self.roll_label.text = f"ROLL: {roll:+.1f}°"

        horizon_color = sensor_data.get("horizon_color", (0, 255, 0))
        r, g, b = horizon_color

        self.crosshair_color.rgb = (r / 255.0, g / 255.0, b / 255.0)
        self.horizon_color.rgb = (r / 255.0, g / 255.0, b / 255.0)

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

        self._last_sensor_data = sensor_data

    def start_update_loop(self, interval: float = 0.1):
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
        if self.camera and hasattr(self.camera, "texture"):
            texture = self.camera.texture
            if texture:
                try:
                    width, height = texture.size
                    return {
                        "texture": texture,
                        "width": width,
                        "height": height,
                        "sensor_data": getattr(self, "_last_sensor_data", None),
                        "timestamp": time.time(),
                    }
                except Exception as e:
                    print(f"[OVERLAY] Capture texture read failed: {e}")
        return None

    def set_capture_button_enabled(self, enabled: bool):
        self._capture_enabled = enabled
        if enabled:
            self.status_text = "LEVEL PHONE - CAPTURE READY"
            self.status_label.color = (0.0, 1.0, 0.0, 1.0)
        else:
            self.status_text = "TILT - ADJUST PHONE"
            self.status_label.color = (1.0, 0.0, 0.0, 1.0)


class LevelBanner(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (1, None)
        self.height = 40
        self.pos_hint = {"y": 1}
        self.spacing = 10

        self.indicator_color = (0.0, 1.0, 0.0, 1.0)
        self.indicator_size = 20
        self.status_text = "HOLD LEVEL"

        self.indicator = Label(text="●", color=self.indicator_color, font_size="24sp")
        self.status_label = Label(
            text=self.status_text, color=self.indicator_color, font_size="16sp"
        )

        self.add_widget(self.indicator)
        self.add_widget(self.status_label)

    def update_status(self, is_level: bool, pitch: float, roll: float):
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
            self.indicator_color = (1.0, 0.65, 0.0, 1.0)
            self.status_text = "LEVEL PHONE"
        else:
            self.indicator_color = (1.0, 1.0, 0.0, 1.0)
            self.status_text = "HOLD LEVEL"

        self.indicator.color = self.indicator_color
        self.status_label.color = self.indicator_color
        self.status_label.text = self.status_text