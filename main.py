"""
Main Application Module

GPS-Free Skyline Geolocation App with Landscape HUD and Multi-Crop Fusion.
"""

import json
import time
import os
import threading
from typing import Dict, Any, List
import numpy as np

# Local modules
from sensor_manager import SensorManager, build_tilt_matrix
from camera_overlay import CameraOverlay, LevelBanner
from segmentation_engine import SegmentationEngine
from profile_extractor import ProfileExtractor, fuse_profiles_world_frame
from matching import match_query
from db_loader import SkylineDB, find_nearest_known_place

# Kivy imports
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.utils import platform
from kivy.core.window import Window


# Lock window orientation to Landscape
Window.softinput_mode = "below_target"


class MultiCropManager:
    """Manages multi-photo capture for 360° perspective fusion."""

    def __init__(self, max_crops: int = 3):
        self.max_crops = max_crops
        self.crops = []
        self.target_headings = [0, 90, 180]
        self.crop_labels = ["FRONT (0°)", "RIGHT (+90°)", "REAR (+180°)"]

    def add_crop(self, image: Any, sensor_data: Dict, heading: float) -> bool:
        if len(self.crops) >= self.max_crops:
            return False

        crop_data = {
            "image": image,
            "sensor_data": sensor_data,
            "heading": heading,
            "crop_index": len(self.crops),
        }
        self.crops.append(crop_data)
        return True

    def get_crops(self) -> List[Dict]:
        return self.crops

    def reset(self):
        self.crops = []

    def is_complete(self) -> bool:
        return len(self.crops) >= self.max_crops

    def get_current_prompt(self) -> str:
        idx = len(self.crops)
        if idx < self.max_crops:
            return f"CROP {idx + 1} OF {self.max_crops}: {self.crop_labels[idx]}"
        return "SESSION COMPLETE"


class SkylineGeolocationApp(App):
    """Main application class for GPS-Free skyline geolocation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sensor_manager = SensorManager(fov_y_deg=65.0)
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_base_dir, "sky_segmentation_unet_model.onnx")
        self.segmentation_engine = SegmentationEngine(model_path=model_path)
        self.profile_extractor = ProfileExtractor(fov_y_deg=65.0)
        self.crop_manager = MultiCropManager(max_crops=3)

        self.camera = None
        self.capture_enabled = False
        self.output_dir = "./captures"

        self.skyline_db = SkylineDB()
        db_path = self.skyline_db.find_db_path()
        if db_path:
            self.skyline_db.load(db_path)
        else:
            print("[APP] No skyline DB found — matching disabled")

    def build(self):
        """Build widescreen landscape layout."""
        self.sensor_manager.start()
        self.layout = FloatLayout()

        # 1. Full-Screen Camera Feed
        self.init_camera()

        # 2. Transparent Landscape HUD Overlay
        self.overlay = CameraOverlay(
            self.camera,
            screen_width=int(Window.width),
            screen_height=int(Window.height),
            fov_y_deg=65.0,
        )
        self.layout.add_widget(self.overlay)

        # 3. Status Level Banner
        self.banner = LevelBanner()
        self.layout.add_widget(self.banner)

        # 4. Heading & Step Info Pill (Bottom Left)
        self.info_label = Label(
            text="HDG: 0° | CROP 1 OF 3",
            color=(1.0, 1.0, 1.0, 0.9),
            font_size="15sp",
            bold=True,
            size_hint=(None, None),
            size=(220, 36),
            pos_hint={"x": 0.02, "y": 0.04},
        )
        self.layout.add_widget(self.info_label)

        # 5. Capture Button (Bottom Center)
        self.capture_btn = Button(
            text="📸 CAPTURE CROP 1",
            font_size="16sp",
            bold=True,
            size_hint=(0.4, 0.11),
            pos_hint={"center_x": 0.5, "y": 0.04},
            background_color=(0.0, 0.8, 0.4, 1.0),
            disabled=True,
        )
        self.capture_btn.bind(on_press=self.on_capture)
        self.layout.add_widget(self.capture_btn)

        # Start loops
        self.overlay.start_update_loop(interval=0.08)
        self.sensor_update_event = Clock.schedule_interval(self.update_sensors, 0.08)

        return self.layout

    def init_camera(self):
        """Safely initialize camera with full-screen landscape stretch."""
        if self.camera is not None:
            return
        try:
            cam = Camera(
                play=True,
                resolution=(1920, 1080),
                allow_stretch=True,
                keep_ratio=False,
                size_hint=(1, 1),
                pos_hint={"x": 0, "y": 0},
            )
            self.camera = cam
            if hasattr(self, "overlay") and self.overlay:
                self.overlay.camera = cam
            if hasattr(self, "layout") and self.layout:
                self.layout.add_widget(cam, index=len(self.layout.children))
            print("[CAMERA] Camera feed started successfully")
        except Exception as e:
            print(f"[CAMERA] Camera init postponed: {e}")
            self.camera = None

    def on_start(self):
        """Request permissions on Android at startup."""
        self.request_android_permissions()

    def request_android_permissions(self):
        if platform == "android":
            try:
                from android.permissions import request_permissions, Permission
                def callback(permissions, results):
                    if all(results):
                        print("[PERMISSIONS] Permissions granted")
                        Clock.schedule_once(lambda dt: self.init_camera())
                    else:
                        print("[PERMISSIONS] Permissions denied")
                request_permissions([
                    Permission.CAMERA,
                    Permission.READ_EXTERNAL_STORAGE,
                    Permission.WRITE_EXTERNAL_STORAGE,
                ], callback)
            except Exception as e:
                print(f"[PERMISSIONS] Permission request error: {e}")

    def update_sensors(self, dt):
        """Update sensor readings and refresh UI controls."""
        self.sensor_manager.update_sensors()
        sensor_data = self.sensor_manager.get_sensor_data()
        self.overlay.update_sensor_data(self.sensor_manager)

        is_level = sensor_data["is_level"]
        heading = sensor_data["heading_deg"]
        prompt = self.crop_manager.get_current_prompt()

        self.banner.update_status(is_level, sensor_data["pitch_deg"], sensor_data["roll_deg"])
        self.info_label.text = f"HDG: {heading:.0f}° | {prompt}"

        self.capture_enabled = is_level
        self.capture_btn.disabled = not is_level
        if is_level:
            self.capture_btn.background_color = (0.0, 0.8, 0.4, 1.0)
            self.capture_btn.text = f"📸 CAPTURE ({prompt.split(':')[0]})"
        else:
            self.capture_btn.background_color = (0.4, 0.4, 0.4, 0.8)
            self.capture_btn.text = "⏳ HOLD PHONE LEVEL"

    def on_capture(self, instance):
        if not self.capture_enabled:
            return

        if not hasattr(self, "session_start") or self.session_start is None:
            self.session_start = time.time()

        sensor_data = self.sensor_manager.get_sensor_data()
        captured_data = self.overlay.capture_photo()

        if captured_data is None:
            self.show_popup("Error", "Camera texture unavailable")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        crop_idx = len(self.crop_manager.crops)
        img_path = os.path.join(self.output_dir, f"crop_{crop_idx}_{timestamp}.png")

        try:
            self.save_texture(captured_data["texture"], img_path)
        except Exception as e:
            self.show_popup("Error", f"Image save error: {e}")
            return

        self.crop_manager.add_crop(img_path, sensor_data, sensor_data["heading_deg"])
        self.process_crop(img_path, sensor_data, crop_idx)

        if self.crop_manager.is_complete():
            self.finalize_session()

    def save_texture(self, texture, filepath):
        from PIL import Image as PILImage
        width, height = texture.size
        pixels = texture.pixels
        pil_img = PILImage.frombytes("RGBA", (width, height), pixels, "raw", "RGBA", 0, -1)
        pil_img.convert("RGB").save(filepath)

    def process_crop(self, img_path: str, sensor_data: Dict, crop_idx: int):
        def process_thread():
            try:
                r_tilt = build_tilt_matrix(
                    sensor_data["pitch_deg"], sensor_data["roll_deg"]
                )

                result = self.segmentation_engine.extract_horizon_profile(
                    img_path,
                    r_tilt=r_tilt,
                    fov_y_deg=self.profile_extractor.fov_y_deg,
                    bin_deg=self.profile_extractor.bin_deg,
                    profile_extractor=self.profile_extractor,
                )

                crop_data = {
                    "image_path": img_path,
                    "sensor_data": sensor_data,
                    "ok": bool(result["ok"]),
                    "status": result["status"],
                    "reason": result["reason"],
                    "profile": np.asarray(result["profile"]).tolist()
                    if result["profile"] is not None
                    else [],
                    "start_az": result["start_az"],
                    "bin_deg": self.profile_extractor.bin_deg,
                    "heading_deg": sensor_data["heading_deg"],
                    "diagnostics": result["diagnostics"],
                }

                self.crop_manager.crops[crop_idx]["processed"] = crop_data
            except Exception as e:
                print(f"[PROCESS] Processing error: {e}")

        threading.Thread(target=process_thread, daemon=True).start()

    def finalize_session(self):
        try:
            all_crops = self.crop_manager.get_crops()
            processed = [c.get("processed", {}) for c in all_crops]

            fusion = fuse_profiles_world_frame(
                processed, bin_deg=self.profile_extractor.bin_deg
            )
            fused_profile = fusion["profile"]
            valid = fused_profile[np.isfinite(fused_profile)]

            first_sensor = processed[0].get("sensor_data", {}) if processed else {}

            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sensors": {
                    "pitch_deg": first_sensor.get("pitch_deg", 0.0),
                    "roll_deg": first_sensor.get("roll_deg", 0.0),
                    "heading_deg": first_sensor.get("heading_deg", 0.0),
                    "fov_y_deg": first_sensor.get("fov_y_deg", 65.0),
                },
                "diagnostics": {
                    "total_crops": len(processed),
                    "fused_coverage_deg": round(fusion["coverage_deg"], 2),
                    "wide_fov_ok": fusion["wide_fov_ok"],
                },
                "profile": [
                    None if not np.isfinite(v) else round(float(v), 4)
                    for v in fused_profile
                ],
            }

            output_file = os.path.join(
                self.output_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, "w") as f:
                json.dump(payload, f, indent=2)

            match_result = None
            if self.skyline_db.loaded and valid.size >= 30:
                query_for_match = np.where(np.isfinite(fused_profile), fused_profile, 0.0)
                match_result = match_query(
                    self.skyline_db.horizon_matrix,
                    self.skyline_db.lats,
                    self.skyline_db.lons,
                    query_for_match,
                    bin_deg=self.profile_extractor.bin_deg,
                    top_k=5,
                )

            lines = [
                f"360° Fusion Coverage: {fusion['coverage_deg']:.1f}°",
            ]

            if match_result and match_result["ok"]:
                top = match_result["matches"][0]
                place = find_nearest_known_place(top["lat"], top["lon"])
                lines += [
                    "",
                    f"📍 LOCATION MATCHED",
                    f"  Lat: {top['lat']:.4f}°  Lon: {top['lon']:.4f}°",
                    f"  Near: {place[0]} ({place[2]})" if place else "",
                ]

            lines += ["", f"Saved: {output_file}"]
            self.show_popup("Geolocation Complete", "\n".join(lines))

            self.crop_manager.reset()
            self.session_start = None

        except Exception as e:
            self.show_popup("Export Error", str(e))

    def show_popup(self, title: str, message: str):
        popup = Popup(
            title=title,
            content=Label(text=message, font_size="15sp"),
            size_hint=(0.7, 0.5),
        )
        popup.open()

    def on_stop(self):
        self.overlay.stop_update_loop()
        if self.sensor_update_event:
            self.sensor_update_event.cancel()
        self.sensor_manager.stop()


if __name__ == "__main__":
    os.makedirs("./captures", exist_ok=True)
    SkylineGeolocationApp().run()