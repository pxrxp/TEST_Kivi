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

    def refine_sky_mask(
        self, img_np: np.ndarray, raw_unet_mask: np.ndarray
    ) -> np.ndarray:
        """
        Refine sky mask using top-connected-sky constraints and Canny edge guides.

        Pipeline (per AGENTS.md spec):
        1. Top-connected sky filtering (flood-fill style from row <= 15,
           fallback to largest region for steep mountain fills)
        2. Canny ridge barrier restricted to +/-10px around U-Net boundary
        3. Two-pass physical slope cap |dr/dc| <= 2.0 px/col
        4. 9-tap 1D median smoothing

        Parameters:
        -----------
        img_np : np.ndarray
            Input RGB image array (HxWx3)
        raw_unet_mask : np.ndarray
            Raw binary sky mask from U-Net (1=SKY, 0=TERRAIN)

        Returns:
        --------
        np.ndarray
            Refined uint8 mask where 0=SKY (black) and 255=TERRAIN (white)
        """
        print("[SEGMENTATION] Starting mask refinement...")

        # 1. Isolate sky pixels (raw_unet_mask: 1=SKY, 0=TERRAIN)
        height, width = raw_unet_mask.shape
        sky1 = (raw_unet_mask == 1).astype(np.uint8)

        # 2. Top-connected sky region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            sky1, connectivity=8
        )
        top_sky = np.zeros((height, width), dtype=np.uint8)
        top_limit = max(15, int(height * 0.15))

        for i in range(1, num_labels):
            if (
                stats[i, cv2.CC_STAT_TOP] <= top_limit
                and stats[i, cv2.CC_STAT_AREA] > 50
            ):
                top_sky[labels == i] = 1

        # Fallback to largest region if no top-connected sky
        if top_sky.sum() == 0 and num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            top_sky[labels == largest_idx] = 1

        # Last resort: use raw mask
        if top_sky.sum() == 0:
            top_sky = sky1

        # 3. Canny edge guidance
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        coarse_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        canny_edges = (cv2.Canny(fine_blur, 30, 150) > 0) | (
            cv2.Canny(coarse_blur, 20, 100) > 0
        )

        # 4. Build boundary array (last sky row per column of contiguous segment)
        boundaries = np.full(width, -1, dtype=np.float64)

        for col in range(width):
            sky_rows = np.where(top_sky[:, col] == 1)[0]
            if len(sky_rows) == 0:
                continue

            # Find largest contiguous sky segment; take its last row
            diffs = np.diff(sky_rows)
            gaps = np.where(diffs > 3)[0]
            max_sky_row = sky_rows[gaps[0]] if len(gaps) > 0 else sky_rows[-1]

            # Restrict Canny search to ±10px window
            if canny_edges is not None:
                edge_rows = np.where(canny_edges[:, col])[0]
                valid_mountain_edges = [
                    r for r in edge_rows if abs(r - max_sky_row) <= 10
                ]
                if len(valid_mountain_edges) > 0:
                    max_sky_row = valid_mountain_edges[0]

            boundaries[col] = float(max_sky_row)

        # 5. Pre-initialize fallback boundaries (all-terrain edge case)
        boundaries_filled = np.full(width, height * 0.5, dtype=np.float64)

        # 6. Outlier interpolation + two-pass slope constraint (|dr/dc| <= 2.0 px/col)
        valid = boundaries >= 0
        if np.any(valid):
            all_cols = np.arange(width, dtype=np.float64)
            boundaries_filled = np.interp(all_cols, all_cols[valid], boundaries[valid])

            max_slope = 2.0
            # Forward pass
            for c in range(1, width):
                delta = boundaries_filled[c] - boundaries_filled[c - 1]
                if abs(delta) > max_slope:
                    boundaries_filled[c] = (
                        boundaries_filled[c - 1] + np.sign(delta) * max_slope
                    )
            # Backward pass
            for c in range(width - 2, -1, -1):
                delta = boundaries_filled[c] - boundaries_filled[c + 1]
                if abs(delta) > max_slope:
                    boundaries_filled[c] = (
                        boundaries_filled[c + 1] + np.sign(delta) * max_slope
                    )

            # 7. Smoothing with 9-tap median filter
            pad = 4
            padded = np.pad(boundaries_filled, (pad, pad), mode="edge")
            from numpy.lib.stride_tricks import sliding_window_view

            meds = np.median(sliding_window_view(padded, 9, axis=0), axis=1)
            boundaries_filled = meds

        # 8. Build refined mask: 0=SKY (black), 255=TERRAIN (white)
        refined_mask = np.zeros((height, width), dtype=np.uint8)
        for col in range(width):
            b = int(np.clip(round(boundaries_filled[col]), 0, height - 1))
            refined_mask[:b, col] = 1

        return np.where(refined_mask == 1, 0, 255).astype(np.uint8)

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
