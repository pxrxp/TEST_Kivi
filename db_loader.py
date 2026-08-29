"""
On-Device Skyline Database Loader

Loads pre-rendered horizon database (.npz format, converted from parquet
via build_db.py).
"""

import os
from typing import Optional, Dict, Any

import numpy as np

DEG_PER_BIN = 90.0 / 255.0


class SkylineDB:
    def __init__(self):
        self.lats: Optional[np.ndarray] = None
        self.lons: Optional[np.ndarray] = None
        self.elevations: Optional[np.ndarray] = None
        self.horizon_matrix: Optional[np.ndarray] = None
        self.n_rows: int = 0
        self.n_bins: int = 720
        self.loaded: bool = False

    def load(self, npz_path: str, max_rows: Optional[int] = None, quiet: bool = False) -> bool:
        if not os.path.exists(npz_path):
            print(f"[DB] File not found: {npz_path}")
            return False

        try:
            data = np.load(npz_path)
            n_total = len(data["lats"])

            if max_rows is not None:
                n = min(n_total, max_rows)
            else:
                n = n_total

            self.lats = data["lats"][:n].astype(np.float32)
            self.lons = data["lons"][:n].astype(np.float32)
            self.elevations = data["elevations"][:n].astype(np.float32)

            horizon_u8 = data["horizon"][:n]
            self.horizon_matrix = horizon_u8.astype(np.float32) * DEG_PER_BIN

            self.n_rows = n
            self.n_bins = self.horizon_matrix.shape[1]
            self.loaded = True

            if not quiet:
                size_mb = os.path.getsize(npz_path) / (1024 * 1024)
                ram_mb = (self.horizon_matrix.nbytes + self.lats.nbytes + self.lons.nbytes + self.elevations.nbytes) / (1024 * 1024)
                print(f"[DB] Loaded {self.n_rows:,} viewpoints from {npz_path}")
                print(f"[DB] Disk: {size_mb:.1f} MB, RAM: {ram_mb:.1f} MB")
            return True

        except Exception as e:
            print(f"[DB] Load failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        if not self.loaded:
            return {"loaded": False}
        return {
            "loaded": True,
            "n_rows": self.n_rows,
            "n_bins": self.n_bins,
            "lon_range": (float(self.lons.min()), float(self.lons.max())),
            "lat_range": (float(self.lats.min()), float(self.lats.max())),
            "elevation_range": (float(self.elevations.min()), float(self.elevations.max())),
        }

    def find_db_path(self) -> Optional[str]:
        """Search common locations for a bundled skyline DB."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "skyline_db.npz"),
            os.path.join(base_dir, "data", "skyline_db.npz"),
            "skyline_db.npz",
            "data/skyline_db.npz",
            os.path.join(os.path.expanduser("~"), ".skylinegeolocation", "skyline_db.npz"),
        ]

        try:
            from kivy.utils import platform as kivy_platform
            if kivy_platform == "android":
                from jnius import autoclass
                context = autoclass("org.kivy.android.PythonActivity").mActivity
                files_dir = context.getFilesDir().getAbsolutePath()
                candidates.insert(0, os.path.join(files_dir, "app", "skyline_db.npz"))
                candidates.insert(0, os.path.join(files_dir, "skyline_db.npz"))
                ext = context.getExternalFilesDir(None)
                if ext:
                    candidates.insert(0, os.path.join(ext.getAbsolutePath(), "skyline_db.npz"))
        except Exception:
            pass

        for path in candidates:
            if os.path.exists(path):
                return path
        return None


def haversine_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def find_nearest_known_place(lat, lon, max_dist_km=50.0):
    PLACES = [
        ("Kathmandu", 27.7172, 85.3240, "Nepal"),
        ("Pokhara", 28.2096, 83.9856, "Nepal"),
        ("Namche Bazaar", 27.8069, 86.7143, "Nepal"),
        ("Lukla", 27.6870, 86.7310, "Nepal"),
        ("Everest Base Camp", 28.0025, 86.8528, "Nepal"),
        ("Annapurna BC", 28.5308, 83.8781, "Nepal"),
        ("Jomsom", 28.7803, 83.7394, "Nepal"),
        ("Manang", 28.6660, 84.0164, "Nepal"),
        ("Lhasa", 29.6500, 91.1000, "Tibet/China"),
        ("Leh", 34.1526, 77.5771, "India"),
        ("Shimla", 31.1048, 77.1734, "India"),
        ("Darjeeling", 27.0360, 88.2627, "India"),
        ("Gangtok", 27.3389, 88.6065, "India"),
        ("Zermatt", 46.0207, 7.7491, "Switzerland"),
        ("Chamonix", 45.9237, 6.8694, "France"),
        ("Innsbruck", 47.2692, 11.4041, "Austria"),
        ("Interlaken", 46.6863, 7.8632, "Switzerland"),
        ("Banff", 51.1784, -115.5708, "Canada"),
        ("Aspen", 39.1911, -106.8175, "USA"),
        ("Jackson Hole", 43.4799, -110.7624, "USA"),
        ("Queenstown", -45.0312, 168.6626, "New Zealand"),
        ("Cusco", -13.5320, -71.9675, "Peru"),
        ("Nairobi", -1.2921, 36.8219, "Kenya"),
        ("Kilimanjaro", -3.0674, 37.3556, "Tanzania"),
        ("Reykjavik", 64.1466, -21.9426, "Iceland"),
        ("Tromsø", 69.6492, 18.9553, "Norway"),
    ]
    best_name, best_dist, best_country = None, float("inf"), ""
    for name, plat, plon, country in PLACES:
        d = haversine_distance_km(lat, lon, plat, plon)
        if d < best_dist:
            best_name, best_dist, best_country = name, d, country
    return (best_name, best_dist, best_country)