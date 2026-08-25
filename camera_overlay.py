"""
Camera Overlay Module

Provides a transparent landscape Heads-Up Display (HUD) with real-time 
artificial horizon, pitch ladder, and tilt correction prompts.
"""

import time
import math
from typing import Optional, Tuple, Any

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle, PushMatrix, PopMatrix, Rotate
from kivy.clock import Clock


class CameraOverlay(FloatLayout):
    """
    Landscape Heads-Up Display (HUD) overlay for live skyline capture.
    """

    def __init__(
        self,
        camera: Any,
        screen_width: int = 1920,
        screen_height: int = 1080,
        fov_y_deg: float = 65.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.camera = camera
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_y_deg = fov_y_deg

        self.sensor_manager = None
        self._last_sensor_data = {}
        self._update_event = None

        self._init_hud_graphics()
        self._init_hud_widgets()

        self.bind(pos=self._update_pos_size, size=self._update_pos_size)

    def _init_hud_graphics(self):
        """Initialize transparent graphics context (NO solid background!)."""
        with self.canvas:
            # HUD Theme Color (Green when level, Amber/Red when tilted)
            self.hud_color = Color(0.0, 1.0, 0.4, 0.9)

            # Central static reticle target (+)
            self.center_target = Line(points=[], width=2.0)

            # Dynamic Artificial Horizon line
            self.horizon_line = Line(points=[], width=3.0)

            # Pitch ladder ticks
            self.pitch_tick_top = Line(points=[], width=1.5)
            self.pitch_tick_bot = Line(points=[], width=1.5)

            # Top Banner dark semi-transparent pill overlay
            self.banner_bg_color = Color(0.0, 0.0, 0.0, 0.5)
            self.banner_bg = Rectangle(pos=(0, 0), size=(0, 0))

    def _init_hud_widgets(self):
        """Initialize UI labels positioned for widescreen landscape layout."""
        self.status_label = Label(
            text="LEVEL PHONE",
            color=(0.0, 1.0, 0.4, 1.0),
            font_size="20sp",
            bold=True,
            size_hint=(None, None),
            height=40,
        )

        self.pitch_label = Label(
            text="PITCH: 0.0°",
            color=(1.0, 1.0, 1.0, 0.9),
            font_size="15sp",
            bold=True,
            size_hint=(None, None),
            height=30,
        )

        self.roll_label = Label(
            text="ROLL: 0.0°",
            color=(1.0, 1.0, 1.0, 0.9),
            font_size="15sp",
            bold=True,
            size_hint=(None, None),
            height=30,
        )

        self.add_widget(self.status_label)
        self.add_widget(self.pitch_label)
        self.add_widget(self.roll_label)

    def _update_pos_size(self, *args):
        """Update graphics coordinates when layout dimensions change."""
        w = float(self.width) if self.width > 1 else float(self.screen_width)
        h = float(self.height) if self.height > 1 else float(self.screen_height)

        cx, cy = w / 2.0, h / 2.0

        # Central Static Crosshair
        r = min(w, h) * 0.04
        self.center_target.points = [
            cx - r, cy, cx + r, cy,
            cx, cy - r, cx, cy + r
        ]

        # Top semi-transparent banner
        banner_h = 55.0
        self.banner_bg.pos = (0, h - banner_h)
        self.banner_bg.size = (w, banner_h)

        # Label positions
        self.status_label.width = w * 0.5
        self.status_label.pos = (w * 0.25, h - 48.0)

        self.pitch_label.width = w * 0.2
        self.pitch_label.pos = (w * 0.03, h - 42.0)

        self.roll_label.width = w * 0.2
        self.roll_label.pos = (w * 0.77, h - 42.0)

        # Force immediate update of artificial horizon lines
        if self._last_sensor_data:
            self._update_display(self._last_sensor_data)

    def update_sensor_data(self, sensor_manager):
        self.sensor_manager = sensor_manager
        if sensor_manager:
            self._update_display(sensor_manager.get_sensor_data())

    def _update_display(self, sensor_data: dict):
        pitch = sensor_data.get("pitch_deg", 0.0)
        roll = sensor_data.get("roll_deg", 0.0)
        is_level = sensor_data.get("is_level", False)
        guidance = sensor_data.get("guidance", "LEVEL PHONE")
        r, g, b = sensor_data.get("horizon_color", (0, 255, 102))

        # Update HUD Theme colors
        self.hud_color.rgb = (r / 255.0, g / 255.0, b / 255.0)

        # Update Status Text & Badges
        self.status_label.text = guidance
        self.status_label.color = (r / 255.0, g / 255.0, b / 255.0, 1.0)

        self.pitch_label.text = f"PITCH: {pitch:+.1f}°"
        self.roll_label.text = f"ROLL: {roll:+.1f}°"

        # Calculate HUD Artificial Horizon Line based on Pitch & Roll
        w = float(self.width) if self.width > 1 else float(self.screen_width)
        h = float(self.height) if self.height > 1 else float(self.screen_height)
        cx, cy = w / 2.0, h / 2.0

        # Pixels shift per degree of pitch angle
        pixels_per_deg = h / max(self.fov_y_deg, 30.0)
        y_offset = -pitch * pixels_per_deg
        
        # Roll angle rotation
        roll_rad = math.radians(-roll)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        # Artificial horizon line length
        line_len = w * 0.35

        # Rotated Horizon endpoints
        x1_local, y1_local = -line_len, y_offset
        x2_local, y2_local = line_len, y_offset

        x1 = cx + (x1_local * cos_r - y1_local * sin_r)
        y1 = cy + (x1_local * sin_r + y1_local * cos_r)
        x2 = cx + (x2_local * cos_r - y2_local * sin_r)
        y2 = cy + (x2_local * sin_r + y2_local * cos_r)

        self.horizon_line.points = [x1, y1, x2, y2]

        # Pitch Ladder reference ticks (+5° and -5°)
        tick_gap = 5.0 * pixels_per_deg
        t_len = w * 0.06

        # Upper +5° tick
        y_top = y_offset + tick_gap
        tx1_top = cx + (-t_len * cos_r - y_top * sin_r)
        ty1_top = cy + (-t_len * sin_r + y_top * cos_r)
        tx2_top = cx + (t_len * cos_r - y_top * sin_r)
        ty2_top = cy + (t_len * sin_r + y_top * cos_r)
        self.pitch_tick_top.points = [tx1_top, ty1_top, tx2_top, ty2_top]

        # Lower -5° tick
        y_bot = y_offset - tick_gap
        tx1_bot = cx + (-t_len * cos_r - y_bot * sin_r)
        ty1_bot = cy + (-t_len * sin_r + y_bot * cos_r)
        tx2_bot = cx + (t_len * cos_r - y_bot * sin_r)
        ty2_bot = cy + (t_len * sin_r + y_bot * cos_r)
        self.pitch_tick_bot.points = [tx1_bot, ty1_bot, tx2_bot, ty2_bot]

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
        """Capture frame texture from live camera."""
        if self.camera and hasattr(self.camera, "texture") and self.camera.texture:
            try:
                texture = self.camera.texture
                return {
                    "texture": texture,
                    "width": texture.size[0],
                    "height": texture.size[1],
                    "sensor_data": getattr(self, "_last_sensor_data", None),
                    "timestamp": time.time(),
                }
            except Exception as e:
                print(f"[OVERLAY] Texture capture error: {e}")
        return None


class LevelBanner(BoxLayout):
    """Floating Level Status Badge."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (None, None)
        self.size = (180, 36)
        self.pos_hint = {"center_x": 0.5, "top": 0.88}
        self.spacing = 8

        self.indicator = Label(
            text="●",
            color=(0.0, 1.0, 0.4, 1.0),
            font_size="20sp",
            size_hint=(None, 1),
            width=24,
        )
        self.status_label = Label(
            text="HOLD LEVEL",
            color=(0.0, 1.0, 0.4, 1.0),
            font_size="15sp",
            bold=True,
        )

        self.add_widget(self.indicator)
        self.add_widget(self.status_label)

    def update_status(self, is_level: bool, pitch: float, roll: float):
        if is_level:
            color = (0.0, 1.0, 0.4, 1.0)
            text = "LEVEL & READY"
        elif abs(pitch) <= 6.0 and abs(roll) <= 6.0:
            color = (1.0, 0.8, 0.0, 1.0)
            text = "ADJUST TILT"
        else:
            color = (1.0, 0.2, 0.2, 1.0)
            text = "UNLEVEL"

        self.indicator.color = color
        self.status_label.color = color
        self.status_label.text = text