"""
Segmentation Engine Module

Converts captured RGB photos into refined binary sky masks using MobileNetV3 U-Net
and implements sophisticated edge purification routines for robust horizon extraction.
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional

from profile_extractor import ProfileExtractor, DEFAULT_FOV_Y_DEG, DEFAULT_BIN_DEG


# Mock ONNX Runtime usage - will be replaced with actual inference
class MockSegmentationModel:
    """Mock sky segmentation model placeholder."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        print(f"[Mock] Loading sky segmentation model: {model_path}")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Placeholder prediction - returns mock sky mask."""
        # This would normally run inference with onnxruntime
        height, width = image.shape[:2]
        # Create simplified mock sky mask (top 1/3 sky + some projections)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[: height // 3, :] = 1  # Simple sky region
        return mask


class SegmentationEngine:
    """
    Implements MobileNetV3 U-Net inference for sky segmentation
    with Canny edge refinement and slope filtering.

    Provides robust sky/ground separation for horizon profile extraction.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.30):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Load the sky segmentation model via ModelAdapter (ONNX/TFLite/PyTorch/mock)."""
        try:
            from model_inference import ModelAdapter

            adapter = ModelAdapter(model_path=self.model_path)
            print(f"[SEGMENTATION] ModelAdapter loaded ({adapter.backend_type})")
            return adapter
        except Exception as exc:
            print(f"[SEGMENTATION] ModelAdapter failed: {exc}; using mock")
            return MockSegmentationModel(self.model_path)

    @staticmethod
    def _median_filter_1d(values, kernel_size=7):
        """1-D median filter matching production segmentation.py."""
        values = np.asarray(values, dtype=np.float64)
        if kernel_size <= 1 or len(values) == 0:
            return values
        pad = kernel_size // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        return np.median(sliding_window_view(padded, kernel_size, axis=0), axis=1)

    def refine_sky_mask(
        self, img_np: np.ndarray, raw_unet_mask: np.ndarray
    ) -> np.ndarray:
        """
        Production-grade sky mask refinement ported from
        SkylineGeolocation/src/segmentation.py refine_sky_mask_with_guidance.

        Pipeline:
        1. Top-connected sky filtering with smart fallback
        2. CLAHE-enhanced sky-zone dehazing
        3. Multi-scale Canny edge fusion
        4. Per-column boundary extraction with Canny barrier (±20px)
        5. Outlier rejection via 5-neighbour median (30px threshold)
        6. Two-pass physical slope cap |dr/dc| <= 2.0 px/col
        7. 9-tap median + Gaussian smoothing

        Parameters
        ----------
        img_np : ndarray (H, W, 3) uint8 RGB
        raw_unet_mask : ndarray (H, W) uint8 — 1=SKY, 0=TERRAIN (from U-Net)

        Returns
        -------
        ndarray (H, W) uint8 — 0=SKY, 255=TERRAIN
        """
        H, W = raw_unet_mask.shape
        if H == 0 or W == 0:
            return np.zeros((H, W), dtype=np.uint8)

        sky1 = (raw_unet_mask == 1).astype(np.uint8)

        # --- 1. Top-connected sky region with smart fallback ---
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            sky1, connectivity=8
        )
        top_sky = np.zeros((H, W), dtype=np.uint8)
        top_limit = max(15, int(H * 0.15))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_TOP] <= top_limit and stats[i, cv2.CC_STAT_AREA] > 50:
                top_sky[labels == i] = 1
        if top_sky.sum() == 0 and num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            top_sky[labels == largest_idx] = 1
        if top_sky.sum() == 0:
            top_sky = sky1

        # --- 2. CLAHE-enhanced sky-zone dehazing ---
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(16, 16))
        gray_enhanced = clahe.apply(gray)
        sky_mask_zone = cv2.dilate(top_sky, np.ones((15, 15), np.uint8)) > 0
        gray = np.where(sky_mask_zone, gray_enhanced, gray)

        # --- 3. Multi-scale Canny edge fusion ---
        fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        coarse_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        edges_fine = cv2.Canny(fine_blur, 30, 150)
        edges_coarse = cv2.Canny(coarse_blur, 20, 100)
        canny_edges = (edges_fine > 0) | (edges_coarse > 0)

        # --- 4. Per-column top-down boundary extraction ---
        boundaries = np.full(W, -1, dtype=np.float64)
        for col in range(W):
            sky_rows = np.where(top_sky[:, col] == 1)[0]
            if len(sky_rows) == 0:
                continue
            diffs = np.diff(sky_rows)
            gaps = np.where(diffs > 3)[0]
            max_sky_row = sky_rows[gaps[0]] if len(gaps) > 0 else sky_rows[-1]

            # Canny barrier: ±20px around U-Net boundary (ignores high clouds)
            edge_rows = np.where(canny_edges[:, col])[0]
            valid_mountain_edges = [r for r in edge_rows if abs(r - max_sky_row) <= 20]
            if len(valid_mountain_edges) > 0:
                max_sky_row = valid_mountain_edges[0]

            # Narrow ±10px Canny window refinement
            win_start = max(0, int(max_sky_row) - 10)
            win_end = min(H - 1, int(max_sky_row) + 10)
            canny_in_win = np.where(canny_edges[win_start:win_end + 1, col])[0]
            if len(canny_in_win) > 0:
                candidate_rows = win_start + canny_in_win
                best_r = candidate_rows[np.argmin(np.abs(candidate_rows - max_sky_row))]
                max_sky_row = float(best_r)

            boundaries[col] = float(max_sky_row)

        # --- 5. Outlier rejection + interpolation ---
        valid = boundaries >= 0
        if np.any(valid):
            all_cols = np.arange(W, dtype=np.float64)
            valid_cols = all_cols[valid]
            valid_vals = boundaries[valid]

            # Outlier filter: reject points > 30px from 5-neighbour median
            if len(valid_vals) > 5:
                pad = 2
                padded = np.pad(valid_vals, (pad, pad), mode="edge")
                from numpy.lib.stride_tricks import sliding_window_view
                meds = np.median(sliding_window_view(padded, 5, axis=0), axis=1)
                keep = np.abs(valid_vals - meds) <= 30.0
                if keep.any():
                    valid_cols = valid_cols[keep]
                    valid_vals = valid_vals[keep]

            boundaries = np.interp(all_cols, valid_cols, valid_vals)

            # --- 6. Two-pass physical slope constraint ---
            max_slope = 2.0
            for c in range(1, W):
                delta = boundaries[c] - boundaries[c - 1]
                if abs(delta) > max_slope:
                    boundaries[c] = boundaries[c - 1] + np.sign(delta) * max_slope
            for c in range(W - 2, -1, -1):
                delta = boundaries[c] - boundaries[c + 1]
                if abs(delta) > max_slope:
                    boundaries[c] = boundaries[c + 1] + np.sign(delta) * max_slope

            # --- 7. Median + Gaussian smoothing ---
            boundaries = self._median_filter_1d(boundaries, kernel_size=9)
            boundaries_2d = cv2.GaussianBlur(
                boundaries.reshape(1, -1).astype(np.float32), (7, 1), 0
            )
            boundaries = boundaries_2d.flatten()

        # --- 8. Build refined mask: 0=SKY, 255=TERRAIN ---
        refined = np.zeros((H, W), dtype=np.uint8)
        for col in range(W):
            b = int(np.clip(round(boundaries[col]), 0, H - 1))
            refined[:b, col] = 1

        return np.where(refined == 1, 0, 255).astype(np.uint8)

    def extract_horizon_profile(
        self,
        image_path: str,
        r_tilt: Optional[np.ndarray] = None,
        fov_y_deg: float = DEFAULT_FOV_Y_DEG,
        bin_deg: float = DEFAULT_BIN_DEG,
        profile_extractor: Optional[ProfileExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Full pipeline for one captured crop: U-Net inference -> mask refinement
        -> elevation-profile extraction (production pipeline from
        final/src/query_profile.py via ProfileExtractor).

        Parameters:
        -----------
        image_path : str
            Path to the captured RGB photo
        r_tilt : np.ndarray (3, 3), optional
            Camera tilt rotation built from sensor pitch/roll at capture time
        fov_y_deg : vertical FOV in degrees (default 65.0)
        bin_deg : azimuth bin width (default 0.5)
        profile_extractor : optional shared ProfileExtractor instance

        Returns:
        --------
        dict with keys: mask, ok, status, reason, profile, start_az, diagnostics
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Model inference (ModelAdapter or mock)
        if hasattr(self.model, "predict_sky_mask"):
            raw_unet_mask = self.model.predict_sky_mask(img_rgb)
        else:
            raw_unet_mask = self.model.predict(img_rgb)

        # Refinement: top-connected sky + Canny barrier + slope caps + median
        refined_mask = self.refine_sky_mask(img_rgb, raw_unet_mask)

        # Production-grade 1D profile extraction with quality gates
        extractor = profile_extractor or ProfileExtractor(
            fov_y_deg=fov_y_deg, bin_deg=bin_deg
        )
        result = extractor.extract_elevation_profile(
            refined_mask,
            image=img_rgb,
            r_tilt=r_tilt,
            fov_y_deg=fov_y_deg,
            bin_deg=bin_deg,
        )
        result["mask"] = refined_mask
        return result
