"""
Profile Extractor Module

Converts 2D sky masks into 1D elevation profiles using sub-pixel edge fitting
and camera geometry mapping for precise horizon matching.
"""

import numpy as np
import cv2
from typing import Any, Dict, Optional, Tuple

class ProfileExtractor:
    """
    Core module for converting binary sky masks into elevation profiles.
    
    Implements sub-pixel parabolic edge fitting and camera geometry mapping.
    """

    def __init__(self, fov_y_deg: float = 65.0):
        self.fov_y_deg = fov_y_deg
        self.fov_y_rad = np.radians(fov_y_deg)
        self.focal_length = 1.0 / np.tan(self.fov_y_rad / 2)
        self.bin_deg = 0.5  # 0.5 degree angular resolution

    def find_sky_boundary(self, mask: np.ndarray) -> np.ndarray:
        """
        Detect boundary between sky (0) and terrain (255)
        Returns array of boundary row indices per column
        
        Parameters:
        -----------
        mask : np.ndarray
            Binary mask (0=sky, 255=terrain)
        
        Returns:
        --------
        np.ndarray
            Array of boundary row positions per column
        """
        height, width = mask.shape
        boundary = np.full(width, -1.0)
        
        # Track sky boundaries using Sobel edge detection
        gray = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)  # Ensure single-channel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        for col in range(width):
            col_mask = mask[:, col]
            if np.any(col_mask == 0):  # Sky pixel found
                edge_pixels = np.where(col_mask == 0)[0]
                if len(edge_pixels) > 0:
                    boundary[col] = edge_pixels[-1]  # Last sky pixel row
        
        return boundary

    def refine_subpixel_boundary(self, boundary: np.ndarray) -> np.ndarray:
        """
        Apply parabolic interpolation for sub-pixel precision
        
        Parameters:
        -----------
        boundary : np.ndarray
            Integer boundary positions
        
        Returns:
        --------
        np.ndarray
            Sub-pixel refined boundary positions
        """
        refined = boundary.copy().astype(float)
        height = len(boundary)
        
        for col in range(1, len(boundary) - 1):
            if boundary[col] < 0 or boundary[col + 1] < 0 or boundary[col - 1] < 0:
                continue
            
            # Fit parabola to three points
            x = np.array([-1, 0, 1])
            y = np.array([boundary[col-1], boundary[col], boundary[col+1]])
            p = np.polyfit(x, y, 2)
            refined[col] = p[0]*col**2 + p[1]*col + p[2]
        
        return refined

    def compute_elevation_angles(self, refined_boundary: np.ndarray, img_height: int) -> np.ndarray:
        """
        Convert boundary rows to elevation angles
        
        Parameters:
        -----------
        refined_boundary : np.ndarray
            Sub-pixel boundary positions
        img_height : int
            Image height in pixels
        
        Returns:
        --------
        np.ndarray
            Elevation angles in degrees
        """
        height = img_height
        focal = self.focal_length
        
        # Convert pixel positions to ray vectors
        ray_ys = (refined_boundary - height/2) / focal
        angles = np.degrees(np.arcsin(ray_ys))
        
        # Handle NaNs and clamp values
        valid_mask = ~np.isnan(angles) & (angles >= -89) & (angles <= 89)
        angles = np.where(valid_mask, angles, 0.0)
        
        return angles

    def create_profile(self, mask: np.ndarray, img: np.ndarray) -> Dict[str, Any]:
        """
        Full profile extraction pipeline
        
        Parameters:
        -----------
        mask : np.ndarray
            Refined binary mask (0=sky, 255=terrain)
        img : np.ndarray
            Input image
        
        Returns:
        --------
        dict
            Contains profile array and diagnostic data
        """
        # Get boundary positions
        raw_boundary = self.find_sky_boundary(mask)
        refined_boundary = self.refine_subpixel_boundary(raw_boundary)
        
        # Compute elevation angles
        angles = self.compute_elevation_angles(refined_boundary, img.shape[0])
        
        # Create angular bins
        n_bins = int(180.0 / self.bin_deg) + 1  # From -90 to 90
        angle_bins = np.linspace(-90, 90, n_bins)
        
        profile = np.zeros(n_bins)
        for angle in angles:
            if -90 <= angle <= 90:
                bin_idx = int((angle + 90) / self.bin_deg)  # Shift from -90 to 0-based
                if 0 <= bin_idx < n_bins:
                    profile[bin_idx] += 1 / len(angles)  # Normalize contribution
        
        # Post-processing
        profile = self._smooth_profile(profile)
        profile = self._remove_outliers(profile)
        
        return {
            "profile_data": profile,
            "bin_deg": self.bin_deg,
            "total_bins": n_bins,
            "angular_range": (-90, 90),
            "valid_pixels": np.sum(mask == 0),
        }
    
    def _smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        """
        Apply moving average smoothing
        """
        if len(profile) < 3:
            return profile
        window = np.ones(5) / 5  # 5-point moving average
        return np.convolve(profile, window, mode='valid')
    
    def _remove_outliers(self, profile: np.ndarray) -> np.ndarray:
        """
        Remove extreme value outliers
        """
        if len(profile) < 3:
            return profile
        median = np.median(profile)
        std = np.std(profile)
        threshold = 3 * std
        return np.where(np.abs(profile - median) <= threshold, profile, median)
