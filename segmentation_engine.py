"""
Segmentation Engine Module

Converts captured RGB photos into refined binary sky masks using MobileNetV3 U-Net
and implements sophisticated edge purification routines for robust horizon extraction.
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple

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
        mask[:height//3, :] = 1  # Simple sky region
        mask[height//2:, int(width*0.25):int(width*0.75)] = 1  # Valley detection
        return mask

class SegmentationEngine:
    """
    Implements MobileNetV3 U-Net U-Net inference for sky segmentation
    with Canny edge refinement and slope filtering.
    
    Provides robust sky/ground separation for horizon profile extraction.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.7):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
        
    def _load_model(self) -> Any:
        """Load the sky segmentation model."""
        # In real implementation, this would load .onnx model via onnxruntime
        print(f"[SEGMENTATION] Model loaded from: {self.model_path}")
        return MockSegmentationModel(self.model_path)
    
    def refine_sky_mask(self, img_np: np.ndarray, raw_unet_mask: np.ndarray) -> np.ndarray:
        """
        Refine sky mask using plate detection constraints and edge guides.
        
        Parameters:
        -----------
        img_np : np.ndarray
            Input RGB image array (HxWx3)
        raw_unet_mask : np.ndarray  
            Raw binary sky mask from U-Net (0=sky, 255=terrain)
        
        Returns:
        --------
        np.ndarray
            Refined binary mask (0=sky, 255=terrain)
        """
        print("[SEGMENTATION] Starting mask refinement...")
        
        # Work with binary unset value mask
        unset_value = np.uint8(0)
        
        # Convert to binary background 0/1 mask
        raw_mask = (raw_unet_mask == 1).astype(np.uint8)
        print(f"[SEGMENTATION] Raw mask shape: {raw_mask.shape}")
        
        height, width = raw_mask.shape
        
        # 1. Top-connected sky region
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )
        top_sky = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, num_labels):
            top = stats[i, cv2.CC_STAT_TOP]
            area = stats[i, cv2.CC_STAT_AREA]
            if top <= 15 and area > 50:
                top_sky[labels == i] = 1
        
        if top_sky.sum() == 0 and num_labels > 1:
            largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            top_sky[labels == largest_idx] = 1
        
        if top_sky.sum() == 0:
            top_sky = raw_mask
            
        print(f"[SEGMENTATION] Top-connected sky area: {top_sky.sum()}")
        
        # 2. Canny edge guidance confinement
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
        coarse_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        canny_edges = (cv2.Canny(fine_blur, 30, 150) > 0) | (cv2.Canny(coarse_blur, 20, 100) > 0)
        
        # Build boundaries array
        boundaries = np.full(width, -1, dtype=np.float64)
        
        for col in range(width):
            sky_rows = np.where(top_sky[:, col] == 1)[0]
            if len(sky_rows) == 0:
                continue
                
            diffs = np.diff(sky_rows)
            gaps = np.where(diffs > 3)[0]
            max_sky_row = sky_rows[gaps[0]] if len(gaps) > 0 else sky_rows[-1]
            
            if canny_edges.size > 0:
                edge_rows = np.where(canny_edges[:, col])[0]
                valid_mountain_edges = [r for r in edge_rows if abs(r - max_sky_row) <= 10]
                if len(valid_mountain_edges) > 0:
                    max_sky_row = valid_mountain_edges[0]
            
            boundaries[col] = float(max_sky_row)
        
        # 3. Outlier detection and slope constraint (|dr/dc| <= 2.0 px/col)
        valid = boundaries >= 0
        if np.any(valid):
            # Forward pass slope constraint
            all_cols = np.arange(width, dtype=np.float64)
            boundaries_filled = np.interp(all_cols, all_cols[valid], boundaries[valid])
            
            # Apply forward slope constraint
            max_slope = 2.0
            for c in range(1, width):
                delta = boundaries_filled[c] - boundaries_filled[c - 1]
                if abs(delta) > max_slope:
                    boundaries_filled[c] = boundaries_filled[c - 1] + np.sign(delta) * max_slope
            
            # Apply backward slope constraint
            for c in range(width - 2, -1, -1):
                delta = boundaries[c] - boundaries[c + 1]
                if abs(delta) > max_slope:
                    boundaries[c] = boundaries[c + 1] + np.sign(delta) * max_slope
        
        # 4. Smoothing with 9-tap median filter
        pad = 4
        padded = np.pad(boundaries_filled, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 9, axis=0)
        medians = np.median(windows, axis=1)
        smoothed_boundaries = medians[pad:-pad] if len(medians) > pad else np.array(padded)
        
        # 5. Interpolation and cleanup
        if np.any(smoothed_boundaries >= 0):
            all_x = np.arange(width, dtype=np.float64)
            interp_boundaries = np.interp(all_x, all_cols[valid], smoothed_boundaries[valid])
            
            # Backwards pass to refine edges
            for c in range(width - 1, 0, -1):
                delta = interp_boundaries[c] - interp_boundaries[c - 1]
                if abs(delta) > 2.0:
                    interp_boundaries[c] = interp_boundaries[c - 1] + np.sign(delta) * 2.0
        
        # 6. Pixel-level masking
        refined_mask = np.zeros((height, width), dtype=np.uint8)
        for col in range(width):
            boundary_pc = int(np.clip(round(interp_boundaries[col]), 0, height - 1))
            refined_mask[:boundary_pc, col] = 1
        
        return np.where(refined_mask == 1, 0, 255).astype(np.uint8)

    def compute_elevation_profiles(self, mask: np.ndarray, img_np: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to 1D elevation angle profile vector.
        
        Parameters:
        -----------
        mask : np.ndarray
            Refined binary mask (0=sky, 255=terrain)
        img_np : np.ndarray
            Input RGB image array
        
        Returns:
        --------
        np.ndarray
            Elevation angle vector in degrees
        """
        print("[SEGMENTATION] Computing elevation profile...")
        
        height, width = mask.shape
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Boundary row detection with sobel edge detection
        boundary_rows = np.full(width, -1, dtype=np.float64)
        
        for col in range(width):
            col_pixels = mask[:, col]
            if np.any(col_pixels == 1):
                edge_pixels = np.where(col_pixels == 1)[0]
                if len(edge_pixels) > 0:
                    max_edge_row = edge_pixels[-1]
                    
                    # Skip near-top measurements
                    if max_edge_row < height * 0.3:
                        boundary_rows[col] = height * 0.5  # fallback
                    else:
                        boundary_rows[col] = max_edge_row
        
        # Sub-pixel refinement using parabolic interpolation around edge
        refined_boundaries = []
        for col in range(width):
            row_val = boundary_rows[col]
            if row_val < 0 or row_val >= height - 1:
                refined_boundaries.append(row_val)
                continue
                
            # Get neighboring pixel differences to fit parabola
            y_samples = []
            x_samples = []
            for offset in (-1, 0, 1):
                nxt_col = max(0, min(width - 1, col + offset))
                nxt_row_val = boundary_rows[nxt_col]
                if nxt_row_val >= 0 and nxt_row_val <= height - 1:
                    y_samples.append(max_edge_row)
                    x_samples.append(offset)
                    
                    if len(y_samples) >= 3:
                        break
                        
            if len(y_samples) >= 3:
                # Fitting parabola on key points
                x = np.array([1, 0, -1])
                y = np.array([y_samples[0], y_samples[1], y_samples[2]])
                p = np.polyfit(x, y, 2)
                y_refined = p[0] * 0 + p[1] * 0 + p[2]
                refined_boundaries.append(y_refined)
            else:
                refined_boundaries.append(row_val)
        
        refined_boundaries = np.array(refined_boundaries)
        float_boundaries = np.interp(np.arange(width), np.arange(width), refined_boundaries)
        
        # Ray angle processing with FOV mapping
        # Ray vector construction based on camera geometry
        camera_fov_y = 65.0  # degrees
        focal_length = 1.0 / np.tan(np.radians(camera_fov_y / 2))
        rays_y = (refined_boundaries - (height / 2)) / focal_length
        elevation_angles = np.degrees(np.arcsin(rays_y))
        
        # Angular resolution conversion
        pixel_size = height / 1080.0  # Normalized pixel size
        angle_per_pixel = np.degrees(np.arctan2(np.arange(width), focal_length)) - np.degrees(np.arctan2(np.arange(width) - 1, focal_length))
        bin_deg = 0.5  # 0.5 degree bin
        
        # Angle bin interpolation
        n_bins = int(np.ceil(180.0 / bin_deg)) + 1
        angle_profile = np.zeros(n_bins)
        
        for angle in elevation_angles:
            if -90 <= angle <= 90:
                bin_idx = int(angle / bin_deg) + 90 // int(bin_deg)
                if 0 <= bin_idx < n_bins:
                    angle_profile[bin_idx] += 1.0 / width
        
        return angle_profile

    def extract_horizon_profile(self, image_path: str) -> Dict[str, Any]:
        """
        Extract complete 1D elevation profile from a single image.
        
        Parameters:
        -----------
        image_path : str
            Path to input image
            
        Returns:
        --------
        dict
            Contains refined mask, profile data, and diagnostic info
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load model and predict
        raw_unet_mask = self.model.predict(img_rgb)
        refined_mask = self.refine_sky_mask(img_rgb, raw_unet_mask)
        
        # Extract profile
        profile = self.compute_elevation_profiles(refined_mask, img_rgb)
        profile = self._smooth_profile(profile)
        
        profile = self._remove_outliers(profile)
        
        elevation_angles = np.linspace(-90 + 5, 90 - 5, len(profile))
        profile = np.concatenate([elevation_angles, profile])
        
        return {
            "mask": refined_mask,
            "profile": profile,
            "raw_profile": profile,
            "diagnostics": {
                "sky_cover_ratio": "0x",
                "boundary_contrast": "0x",
                "precision_degrees": "0x",
            }
        }
    
    def _smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        """Apply smoothing and normalization to profile."""
        if len(profile) < 3:
            return profile
        
        # 3-point smoothing with outlier removal
        window = np.ones(5) / 5
        smoothed = np.convolve(profile, window, mode='valid')
        
        # Clean outliers
        threshold = 2.5
        filtered = smoothed
        mean = np.mean(filtered)
        std = np.std(filtered)
        
        if std > 0:
            filtered = filtered[np.abs(filtered - mean) <= threshold * std]
        
        # Normalize to unit range
        if len(filtered) > 0:
            filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min() + 1e-6)
        
        return filtered
    
    def _remove_outliers(self, profile: np.ndarray) -> np.ndarray:
        """Remove extreme outliers from profile."""
        if len(profile) < 3:
            return profile
            
        filtered = profile.copy()
        sorted_vals = np.sort(filtered)
        outliers = sorted_vals[(sorted_vals - sorted_vals.min()) > 2 * (sorted_vals.max() - sorted_vals.min())]
        if len(outliers) > 0:
            replacement = np.median(filtered[~np.isin(filtered, outliers)])
            filtered[~np.isin(filtered, outliers)] = replacement
        return filtered
