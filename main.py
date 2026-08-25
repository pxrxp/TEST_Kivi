"""
Main Application Module

Combines all modules into a complete GPS-free skyline geolocation app with:
- Multi-photo perspective fusion
- Sensor-guided capture
- Standalone JSON payload export
"""

import json
import time
import os
import sys
import threading
from typing import Dict, Any, List, Optional
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


class MultiCropManager:
    """Manages multi-photo capture for perspective fusion."""

    def __init__(self, max_crops: int = 3):
        self.max_crops = max_crops
        self.crops = []
        self.current_crop = 0
        self.target_headings = [0, 90, 180]

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
        self.current_crop = len(self.crops)
        return True

    def get_crops(self) -> List[Dict]:
        return self.crops

    def reset(self):
        self.crops = []
        self.current_crop = 0

    def is_complete(self) -> bool:
        return len(self.crops) >= self.max_crops


class SkylineGeolocationApp(App):
    """Main application class for GPS-free skyline geolocation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sensor_manager = SensorManager(fov_y_deg=65.0)
        _base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(_base_dir, "sky_segmentation_unet_model.onnx")
        self.segmentation_engine = SegmentationEngine(model_path=model_path)
        self.profile_extractor = ProfileExtractor(fov_y_deg=65.0)
        self.crop_manager = MultiCropManager(max_crops=3)
        self.current_session = {}
        self.camera = None
        self.capture_enabled = False
        self.output_dir = "./captures"

        # Skyline database
        self.skyline_db = SkylineDB()
        db_path = self.skyline_db.find_db_path()
        if db_path:
            self.skyline_db.load(db_path)
        else:
            print("[APP] No skyline DB found — matching disabled")

    def build(self):
        """Build the main UI layout."""
        self.sensor_manager.start()
        self.layout = FloatLayout()

        # Try initializing camera safely
        self.init_camera()

        # Camera overlay
        self.overlay = CameraOverlay(self.camera, 1920, 1080, fov_y_deg=65.0)
        self.layout.add_widget(self.overlay)

        # Status banner
        self.banner = LevelBanner()
        self.layout.add_widget(self.banner)

        # Multi-crop indicator
        self.crop_label = Label(
            text="Crop: 1 / 3",
            color=(0.0, 1.0, 1.0, 1.0),
            font_size="18sp",
            size_hint=(0.3, 0.1),
            pos_hint={"x": 0.7, "y": 0.9},
        )
        self.layout.add_widget(self.crop_label)

        # Capture button
        self.capture_btn = Button(
            text="📸 CAPTURE SKYLINE CROP",
            size_hint=(0.5, 0.1),
            pos_hint={"x": 0.25, "y": 0.05},
            background_color=(0.0, 0.78, 0.0, 1.0),
            disabled=True,
        )
        self.capture_btn.bind(on_press=self.on_capture)
        self.layout.add_widget(self.capture_btn)

        # Heading display
        self.heading_label = Label(
            text="Heading: 0°",
            color=(1.0, 1.0, 1.0, 1.0),
            font_size="16sp",
            size_hint=(0.4, 0.08),
            pos_hint={"x": 0.05, "y": 0.05},
        )
        self.layout.add_widget(self.heading_label)

        # Status label
        self.status_label = Label(
            text="Initializing sensors...",
            color=(1.0, 1.0, 0.0, 1.0),
            font_size="16sp",
            size_hint=(0.5, 0.1),
            pos_hint={"x": 0.25, "y": 0.85},
        )
        self.layout.add_widget(self.status_label)

        # Sensor loop
        self.overlay.start_update_loop(interval=0.1)
        self.sensor_update_event = Clock.schedule_interval(self.update_sensors, 0.1)

        return self.layout

    def init_camera(self):
        """Safely initialize the Camera widget."""
        if self.camera is not None:
            return
        try:
            cam = Camera(play=True, resolution=(1920, 1080))
            self.camera = cam
            if hasattr(self, "overlay") and self.overlay:
                self.overlay.camera = cam
            if hasattr(self, "layout") and self.layout:
                self.layout.add_widget(cam, index=len(self.layout.children))
            print("[CAMERA] Camera initialized successfully")
        except Exception as e:
            print(f"[CAMERA] Camera init postponed or failed: {e}")
            self.camera = None

    def on_start(self):
        """Request permissions on Android startup."""
        self.request_android_permissions()

    def request_android_permissions(self):
        """Request Android permissions at runtime."""
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
        """Update sensor readings and UI."""
        self.sensor_manager.update_sensors()
        sensor_data = self.sensor_manager.get_sensor_data()
        self.overlay.update_sensor_data(self.sensor_manager)

        is_level = sensor_data["is_level"]
        pitch = sensor_data["pitch_deg"]
        roll = sensor_data["roll_deg"]
        heading = sensor_data["heading_deg"]

        self.banner.update_status(is_level, pitch, roll)
        self.heading_label.text = f"Heading: {heading:.1f}°"

        self.capture_enabled = is_level
        self.capture_btn.disabled = not is_level
        if is_level:
            self.capture_btn.background_color = (0.0, 0.78, 0.0, 1.0)
            self.capture_btn.text = "📸 CAPTURE SKYLINE CROP"
        else:
            self.capture_btn.background_color = (0.59, 0.59, 0.59, 1.0)
            self.capture_btn.text = "⏳ LEVEL PHONE FIRST"

        if is_level:
            self.status_label.text = f"LEVEL: Pitch {pitch:+.1f}° | Roll {roll:+.1f}°"
            self.status_label.color = (0.0, 1.0, 0.0, 1.0)
        else:
            self.status_label.text = f"TILT: Pitch {pitch:+.1f}° | Roll {roll:+.1f}°"
            self.status_label.color = (1.0, 1.0, 0.0, 1.0)

        crops_done = len(self.crop_manager.crops)
        total_crops = self.crop_manager.max_crops
        self.crop_label.text = f"Crop: {crops_done + 1} / {total_crops}"

    def on_capture(self, instance):
        if not self.capture_enabled:
            return

        if not hasattr(self, "session_start") or self.session_start is None:
            self.session_start = time.time()

        sensor_data = self.sensor_manager.get_sensor_data()
        captured_data = self.overlay.capture_photo()

        if captured_data is None:
            self.show_popup("Error", "Failed to capture image")
            return

        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        crop_idx = len(self.crop_manager.crops)
        img_filename = f"crop_{crop_idx}_{timestamp}.png"
        img_path = os.path.join(self.output_dir, img_filename)

        try:
            self.save_texture(captured_data["texture"], img_path)
        except Exception as e:
            self.show_popup("Error", f"Failed to save image: {e}")
            return

        success = self.crop_manager.add_crop(
            img_path, sensor_data, sensor_data["heading_deg"]
        )

        if not success:
            self.show_popup("Error", "Maximum crops reached")
            return

        self.process_crop(img_path, sensor_data, crop_idx)

        if self.crop_manager.is_complete():
            self.finalize_session()
        else:
            next_heading = self.crop_manager.target_headings[crop_idx + 1]
            Clock.schedule_once(
                lambda dt, ci=crop_idx, nh=next_heading: self._update_status_text(
                    f"Crop {ci + 1} captured! Rotate {nh}° for next crop",
                    (0.0, 0.78, 1.0, 1.0),
                )
            )

    def _update_status_text(self, text: str, color):
        self.status_label.text = text
        self.status_label.color = color

    def save_texture(self, texture, filepath):
        from PIL import Image as PILImage

        width, height = texture.size
        pixels = texture.pixels
        pil_img = PILImage.frombytes(
            "RGBA", (width, height), pixels, "raw", "RGBA", 0, -1
        )
        pil_img = pil_img.convert("RGB")
        pil_img.save(filepath)

    def process_crop(self, img_path: str, sensor_data: Dict, crop_idx: int):
        Clock.schedule_once(
            lambda dt, ci=crop_idx: self._update_status_text(
                f"Processing crop {ci + 1}...", (1.0, 0.65, 0.0, 1.0)
            )
        )

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
                Clock.schedule_once(
                    lambda dt: self.on_crop_processed(crop_idx, crop_data)
                )

            except Exception as e:
                err_msg = str(e)
                Clock.schedule_once(
                    lambda dt, msg=err_msg: self.show_popup("Processing Error", msg)
                )

        threading.Thread(target=process_thread, daemon=True).start()

    def on_crop_processed(self, crop_idx: int, crop_data: Dict):
        self.status_label.text = f"Crop {crop_idx + 1} processed - Profile ready"
        self.status_label.color = (0.0, 0.78, 0.0, 1.0)

    def finalize_session(self):
        try:
            all_crops = self.crop_manager.get_crops()
            processed_crops = [c.get("processed", {}) for c in all_crops]

            fusion = fuse_profiles_world_frame(
                processed_crops, bin_deg=self.profile_extractor.bin_deg
            )
            fused_profile = fusion["profile"]
            valid = fused_profile[np.isfinite(fused_profile)]

            first_sensor = (
                processed_crops[0].get("sensor_data", {})
                if processed_crops
                else {}
            )

            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "sensors": {
                    "pitch_deg": first_sensor.get("pitch_deg", 0.0),
                    "roll_deg": first_sensor.get("roll_deg", 0.0),
                    "heading_deg": first_sensor.get("heading_deg", 0.0),
                    "fov_y_deg": first_sensor.get("fov_y_deg", 65.0),
                },
                "diagnostics": {
                    "total_crops": len(processed_crops),
                    "crops_ok": sum(1 for c in processed_crops if c.get("ok")),
                    "fused_coverage_deg": round(fusion["coverage_deg"], 2),
                    "wide_fov_ok": fusion["wide_fov_ok"],
                    "profile_std_deg": round(float(np.std(valid)), 3)
                    if valid.size
                    else 0.0,
                    "profile_max_deg": round(float(np.max(valid)), 3)
                    if valid.size
                    else 0.0,
                    "session_duration": time.time() - self.session_start
                    if hasattr(self, "session_start")
                    else 0,
                },
                "profile": [
                    None if not np.isfinite(v) else round(float(v), 4)
                    for v in fused_profile
                ],
                "crops": [
                    {
                        "image_path": c.get("image_path"),
                        "sensor_data": c.get("sensor_data"),
                        "ok": c.get("ok"),
                        "status": c.get("status"),
                        "reason": c.get("reason"),
                        "start_az": c.get("start_az"),
                        "bin_deg": c.get("bin_deg"),
                        "heading_deg": c.get("heading_deg"),
                        "profile": c.get("profile", []),
                        "diagnostics": c.get("diagnostics", {}),
                    }
                    for c in processed_crops
                ],
            }

            output_file = os.path.join(
                self.output_dir, f"session_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, "w") as f:
                json.dump(payload, f, indent=2)

            coverage_msg = (
                "WIDE-FOV OK"
                if fusion["wide_fov_ok"]
                else f"LOW COVERAGE ({fusion['coverage_deg']:.0f}° < 200°)"
            )

            match_result = None
            if self.skyline_db.loaded and fusion["profile"] is not None:
                fused = fusion["profile"]
                valid_bins = np.isfinite(fused)
                if valid_bins.sum() >= 30:
                    query_for_match = np.where(valid_bins, fused, 0.0)
                    match_result = match_query(
                        self.skyline_db.horizon_matrix,
                        self.skyline_db.lats,
                        self.skyline_db.lons,
                        query_for_match,
                        bin_deg=self.profile_extractor.bin_deg,
                        top_k=10,
                        spatial_stride=2,
                        min_corr=0.1,
                    )

            lines = [
                f"All {len(processed_crops)} crops captured!",
                f"Fused coverage: {fusion['coverage_deg']:.1f}° ({coverage_msg})",
            ]

            if match_result and match_result["ok"]:
                top = match_result["matches"][0]
                place = find_nearest_known_place(top["lat"], top["lon"])
                lines += [
                    "",
                    f"📍 LOCATION FOUND",
                    f"  Lat: {top['lat']:.4f}°  Lon: {top['lon']:.4f}°",
                    f"  Score: {top['score']:.3f}",
                    f"  Near: {place[0]} ({place[2]}), ~{place[1]:.0f} km" if place else "",
                ]
            elif match_result:
                lines += ["", f"⚠ {match_result['status']}: {match_result['reason']}"]
            elif not self.skyline_db.loaded:
                lines += ["", "⚠ No skyline DB for matching"]

            lines += ["", f"Payload saved to:\n{output_file}"]
            self.show_popup("Session Complete", "\n".join(lines))

            self.crop_manager.reset()
            self.session_start = None

        except Exception as e:
            self.show_popup("Export Error", str(e))

    def show_popup(self, title: str, message: str):
        popup = Popup(title=title, content=Label(text=message), size_hint=(0.8, 0.4))
        popup.open()

    def on_stop(self):
        self.overlay.stop_update_loop()
        if self.sensor_update_event:
            self.sensor_update_event.cancel()
        self.sensor_manager.stop()


if __name__ == "__main__":
    os.makedirs("./captures", exist_ok=True)
    SkylineGeolocationApp().run()