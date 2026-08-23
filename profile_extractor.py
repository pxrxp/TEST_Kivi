"""
Profile Extractor Module

Converts 2D sky masks into 1D elevation profiles using sub-pixel parabolic edge
fitting and camera geometry mapping for precise horizon matching.
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple


class ProfileExtractor:
    """
    Converts binary sky masks into elevation profiles.

    Implements:
    - Closed-form parabolic sub-pixel edge fitting on Sobel-Y gradient
    - 3D world ray rotation via tilt matrix
    - Camera geometry (FOV->elevation angle) conversion
    """

    def __init__(self, fov_y_deg: float = 65.0, bin_deg: float = 0.5):
        self.fov_y_deg = fov_y_deg
        self.fov_y_rad = np.radians(fov_y_deg)
        self.bin_deg = bin_deg

    def find_sky_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        Find sky boundary rows per column.

        Parameters:
        -----------
        mask : np.ndarray
            Refined binary mask (0=SKY, 255=TERRAIN)

        Returns:
        --------
        np.ndarray
            Array of boundary row positions per column (defaults to H-1)
        """
        H, W = mask.shape
        # Convert to binary: 1=SKY, 0=TERRAIN
        binary = (mask < 128).astype(np.uint8)

        skyline_px = np.full(W, H - 1, dtype=np.float32)
        for c in range(W):
            sky_rows = np.where(binary[:, c] == 1)[0]
            if len(sky_rows) > 0:
                skyline_px[c] = sky_rows[-1]  # last sky row

        return skyline_px

    def refine_subpixel_boundary(
        self, gray_img: np.ndarray, boundary: np.ndarray
    ) -> np.ndarray:
        """
        Closed-form parabolic sub-pixel fitting on Sobel-Y gradient.

        For 3-point gradient (g_{-1}, g_0, g_{+1}):
            delta_r = -(g_{+1} - g_{-1}) / (2 * (g_{+1} - 2*g_0 + g_{-1}))

        Parameters:
        -----------
        gray_img : np.ndarray
            Grayscale image (HxW) - source for Sobel-Y gradient
        boundary : np.ndarray
            Integer-row boundary positions (W,)

        Returns:
        --------
        np.ndarray
            Sub-pixel refined boundary positions (W,)
        """
        H, W = gray_img.shape
        gy = cv2.Sobel(gray_img.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
        refined = boundary.copy().astype(np.float64)

        for col in range(W):
            r0 = int(round(boundary[col]))
            if r0 <= 1 or r0 >= H - 2:
                continue
            gm1 = gy[r0 - 1, col]
            g0 = gy[r0, col]
            gp1 = gy[r0 + 1, col]
            denom = 2.0 * (gp1 - 2.0 * g0 + gm1)
            if abs(denom) > 1e-6:
                offset = -(gp1 - gm1) / denom
                refined[col] = float(r0) + float(np.clip(offset, -0.5, 0.5))

        return refined

    def extract_elevation_profile(
        self,
        mask: np.ndarray,
        img_np: np.ndarray,
        r_tilt: Optional[np.ndarray] = None,
        fov_y_deg: float = 65.0,
        bin_deg: float = 0.5,
    ) -> np.ndarray:
        """
        Extract 1D elevation angles projected into 3D world space.

        Parameters:
        -----------
        mask : np.ndarray
            Refined binary mask (0=SKY, 255=TERRAIN)
        img_np : np.ndarray
            Input image (HxWx3 RGB)
        r_tilt : np.ndarray, optional
            3x3 tilt rotation matrix
        fov_y_deg : float
            Vertical FOV (default 65.0)
        bin_deg : float
            Angular bin size in degrees (default 0.5)

        Returns:
        --------
        np.ndarray
            Elevation angles in degrees (W,) - one per column
        """
        H, W = mask.shape

        # Convert mask to binary: 1=SKY, 0=TERRAIN
        binary = (mask < 128).astype(np.uint8)

        # 1. Boundary row detection (sky_rows[-1])
        skyline_px = np.full(W, H - 1, dtype=np.float32)
        for c in range(W):
            sky_rows = np.where(binary[:, c] == 1)[0]
            if len(sky_rows) > 0:
                skyline_px[c] = sky_rows[-1]

        # 2. Sub-pixel refinement via parabolic fit on Sobel-Y gradient
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        skyline_px = self.refine_subpixel_boundary(gray, skyline_px)

        # 3. Build camera rays
        aspect_ratio = W / H
        hfov_deg = np.degrees(
            2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio)
        )
        focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
        x_c, y_c = W / 2.0, H / 2.0

        cols = np.arange(W, dtype=np.float64)
        rays = np.vstack(
            [
                (cols - x_c) / focal_x,
                (y_c - skyline_px.astype(np.float64)) / focal_y,
                -np.ones(W, dtype=np.float64),
            ]
        )
        # Normalize each ray
        rays = rays / np.linalg.norm(rays, axis=0)

        # 4. Apply tilt rotation if provided
        if r_tilt is not None:
            rays = np.asarray(r_tilt) @ rays

        # 5. Elevation angle = arcsin(ray_y)
        elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))
        return elev_deg

    def create_profile(self, mask: np.ndarray, img: np.ndarray) -> Dict[str, Any]:
        """Legacy interface - returns profile dict for backwards compatibility."""
        H, W = mask.shape
        elev = self.extract_elevation_profile(
            mask, img, r_tilt=None, fov_y_deg=self.fov_y_deg, bin_deg=self.bin_deg
        )
        # Bin into profile
        bin_deg = self.bin_deg
        n_bins = int(180.0 / bin_deg) + 1
        angle_bins = np.linspace(-90, 90, n_bins)
        profile = np.zeros(n_bins, dtype=np.float32)
        for a in elev:
            if -90 <= a <= 90:
                idx = int((a + 90) / bin_deg)
                if 0 <= idx < n_bins:
                    profile[idx] += 1.0
        profile = self._smooth_profile(profile)
        profile = self._remove_outliers(profile)
        return {
            "profile_data": profile,
            "bins": angle_bins,
            "bin_deg": bin_deg,
            "total_bins": n_bins,
            "angular_range": (-90, 90),
            "valid_pixels": int(np.sum((mask < 128))),
        }

    def _smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing."""
        if len(profile) < 3:
            return profile
        window = np.ones(5) / 5
        return np.convolve(profile, window, mode="valid")

    def _remove_outliers(self, profile: np.ndarray) -> np.ndarray:
        """Remove extreme outliers."""
        if len(profile) < 3:
            return profile
        median = np.median(profile)
        std = np.std(profile)
        if std < 1e-9:
            return profile
        threshold = 3 * std
        return np.where(np.abs(profile - median) <= threshold, profile, median)
