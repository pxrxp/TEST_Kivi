"""
Profile Extractor Module

Converts 2D binary sky masks into 1D elevation-angle skyline profiles,
faithfully mirroring the production pipeline in `SkylineGeolocation/final/src/query_profile.py`:

- Per-column skyline row from the sky/terrain boundary
- 5-tap median filter on the integer skyline (removes sawtooth jitter)
- Sub-pixel parabolic edge fitting on the image Sobel-Y gradient (~0.1 px precision)
- Pin-hole camera geometry (FOV -> focal lengths -> unit rays)
- Optional tilt rotation matrix r_tilt applied to rays before elevation lookup
- Binning onto a uniform azimuth grid (default 0.5 deg) with start_az offset
- Quality gates: boundary coverage, flat-terrain (std) and relief (max) checks

Mask convention accepted: 0=SKY / 255=TERRAIN uint8 (auto-detected either way,
same as the production `sky_is_white` heuristic).
"""

import numpy as np
import cv2
from typing import Any, Dict, List, Optional

# Defaults matching PipelineConfig in final/src/config.py
DEFAULT_FOV_Y_DEG = 65.0
DEFAULT_BIN_DEG = 0.5
DEFAULT_MIN_BOUNDARY_COVERAGE = 0.5
DEFAULT_MIN_STD_DEG = 1.5
DEFAULT_MIN_MAX_ELEV_DEG = 1.0


def is_profile_applicable(
    profile,
    min_std_deg: float = DEFAULT_MIN_STD_DEG,
    min_max_elev_deg: float = DEFAULT_MIN_MAX_ELEV_DEG,
):
    """
    Evaluates whether a horizon profile contains sufficient topographic variation
    and vertical relief to be reliably matched. (Ported verbatim from
    final/src/query_profile.py)
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.size == 0:
        return False, "Empty profile"
    if not np.all(np.isfinite(profile)):
        return False, "Profile contains NaN or Inf values"

    std_val = np.std(profile)
    max_val = np.max(profile)

    if std_val < min_std_deg:
        return False, f"Profile too flat (std={std_val:.2f} deg < {min_std_deg} deg)"

    if max_val < min_max_elev_deg:
        return (
            False,
            f"Insufficient terrain relief above horizontal (max={max_val:.2f} deg < {min_max_elev_deg} deg)",
        )

    return True, "Valid topographic profile"


def median_filter_1d(values: np.ndarray, size: int = 5) -> np.ndarray:
    """1-D median filter (numpy-only replacement for scipy.ndimage.median_filter)."""
    values = np.asarray(values, dtype=np.float64)
    if size < 3 or values.size < size:
        return values.copy()
    pad = size // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, size, axis=0)
    return np.median(windows, axis=1)


def _subpixel_edge_from_image(
    gray: np.ndarray, skyline_px: np.ndarray, half_window: int = 3
) -> np.ndarray:
    """
    Parabolic sub-pixel edge fitting on the image gradient.

    For each column, find the peak of |central-difference Y gradient| near the
    binary-mask skyline boundary and fit a 3-point parabola to extract a
    sub-pixel position. (Ported from final/src/query_profile.py)
    """
    H, W = gray.shape
    gray_f = gray.astype(np.float64)
    gy = np.zeros_like(gray_f)
    gy[1:-1, :] = (gray_f[2:, :] - gray_f[:-2, :]) / 2.0
    gy[0, :] = gray_f[1, :] - gray_f[0, :]
    gy[-1, :] = gray_f[-1, :] - gray_f[-2, :]

    sub_px = skyline_px.copy()
    for c in range(W):
        y0 = int(round(skyline_px[c]))
        y_lo = max(1, y0 - half_window)
        y_hi = min(H - 2, y0 + half_window)
        if y_hi <= y_lo:
            continue
        segment = gy[y_lo : y_hi + 1, c]
        peak = y_lo + int(np.argmax(np.abs(segment)))
        if peak <= 0 or peak >= H - 1:
            continue
        gm1 = gy[peak - 1, c]
        g0 = gy[peak, c]
        gp1 = gy[peak + 1, c]
        # Parabola vertex through (-1, gm1), (0, g0), (+1, gp1):
        #   x* = -(gp1 - gm1) / (2 * (gp1 - 2*g0 + gm1))
        # NOTE: final/src/query_profile.py uses the opposite sign here; the
        # sign below is the mathematically correct vertex (verified against
        # synthetic ground-truth edges) and matches the AGENTS.md spec.
        denom = 2.0 * (gp1 - 2.0 * g0 + gm1)
        if abs(denom) > 1e-6:
            offset = -(gp1 - gm1) / denom
            offset = np.clip(offset, -0.5, 0.5)
            sub_px[c] = peak + offset
    return sub_px


def _invalid(status: str, reason: str, diag: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": False,
        "status": status,
        "reason": reason,
        "profile": None,
        "start_az": None,
        "diagnostics": diag,
    }


class ProfileExtractor:
    """
    Converts refined binary sky masks (0=SKY, 255=TERRAIN) into binned
    elevation-angle profiles ready for horizon matching / multi-crop fusion.
    """

    def __init__(
        self,
        fov_y_deg: float = DEFAULT_FOV_Y_DEG,
        bin_deg: float = DEFAULT_BIN_DEG,
        min_boundary_coverage: float = DEFAULT_MIN_BOUNDARY_COVERAGE,
        min_std_deg: float = DEFAULT_MIN_STD_DEG,
        min_max_elev_deg: float = DEFAULT_MIN_MAX_ELEV_DEG,
    ):
        self.fov_y_deg = fov_y_deg
        self.bin_deg = bin_deg
        self.min_boundary_coverage = min_boundary_coverage
        self.min_std_deg = min_std_deg
        self.min_max_elev_deg = min_max_elev_deg

    def extract_elevation_profile(
        self,
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        r_tilt: Optional[np.ndarray] = None,
        fov_y_deg: Optional[float] = None,
        bin_deg: Optional[float] = None,
        min_boundary_coverage: Optional[float] = None,
        azim_frame: str = "camera",
    ) -> Dict[str, Any]:
        """
        Translate a binary sky-terrain mask into a 1D elevation-angle profile
        projected onto a uniform azimuth grid.

        Parameters
        ----------
        mask : ndarray (H, W) uint8 -- 0=SKY, 255=TERRAIN (convention auto-detected)
        image : ndarray (H, W) or (H, W, 3) uint8, optional
            Original photo. When provided, sub-pixel edge fitting refines each
            column's skyline row via a 3-point parabolic fit on the Y gradient.
        r_tilt : ndarray (3, 3), optional
            Camera tilt rotation applied to rays before elevation lookup.
        fov_y_deg : vertical field of view (default 65.0)
        bin_deg : azimuth bin width (default 0.5)
        azim_frame : "camera" or "world"

        Returns
        -------
        dict with keys: ok, status, reason, profile, start_az, diagnostics
        """
        fov_y_deg = fov_y_deg if fov_y_deg is not None else self.fov_y_deg
        bin_deg = bin_deg if bin_deg is not None else self.bin_deg
        min_boundary_coverage = (
            min_boundary_coverage
            if min_boundary_coverage is not None
            else self.min_boundary_coverage
        )

        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            return _invalid(
                "INVALID_INPUT",
                f"Mask too small ({mask.shape[0]}x{mask.shape[1]})",
                {"width": int(mask.shape[1]), "height": int(mask.shape[0])},
            )

        mask_u8 = mask.astype(np.uint8)
        H, W = mask_u8.shape

        # Auto-detect polarity (same heuristic as production pipeline).
        # In BOTH branches below, binary == 1 marks TERRAIN pixels, matching
        # final/src/query_profile.py exactly.
        sky_is_white = np.mean(mask_u8[:10, :]) > np.mean(mask_u8[-10:, :])
        binary = (
            (mask_u8 < 128).astype(np.uint8)
            if sky_is_white
            else (mask_u8 >= 128).astype(np.uint8)
        )

        sky_ratio = float(binary.sum() / (H * W))
        base_diag = {
            "width": W,
            "height": H,
            "sky_ratio": sky_ratio,
            "sky_is_white": bool(sky_is_white),
        }

        if sky_ratio == 0.0:
            return _invalid("NO_SKYLINE", "No sky pixels found in mask", base_diag)
        if sky_ratio == 1.0:
            return _invalid(
                "NO_SKYLINE", "Mask is all sky, no terrain boundary", base_diag
            )

        # --- 1. Integer skyline: FIRST terrain row per column (= boundary),
        # exactly as in final/src/query_profile.py
        skyline_px = np.full(W, H - 1, dtype=np.float32)
        missing_cols = 0
        for c in range(W):
            terr_rows = np.where(binary[:, c] == 1)[0]
            if len(terr_rows) > 0:
                skyline_px[c] = terr_rows[0]
            else:
                missing_cols += 1
                skyline_px[c] = H - 1

        boundary_coverage = 1.0 - (missing_cols / W)
        skyline_px = median_filter_1d(skyline_px, size=5)

        # --- 2. Sub-pixel refinement from original image gradient
        if image is not None:
            gray = np.asarray(image)
            if gray.ndim == 3:
                gray = (
                    cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
                    if gray.shape[2] == 3
                    else gray[:, :, 0]
                )
            if gray.shape == (H, W):
                skyline_px = _subpixel_edge_from_image(gray, skyline_px)

        # --- 3. Pin-hole ray geometry
        aspect_ratio = W / H
        hfov_deg = np.degrees(
            2.0 * np.arctan(np.tan(np.radians(fov_y_deg) / 2.0) * aspect_ratio)
        )
        focal_x = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
        focal_y = H / (2.0 * np.tan(np.radians(fov_y_deg) / 2.0))
        x_c, y_c = W / 2.0, H / 2.0

        cols = np.arange(W)
        rays = np.vstack(
            [(cols - x_c) / focal_x, (y_c - skyline_px) / focal_y, -np.ones(W)]
        )
        rays /= np.linalg.norm(rays, axis=0)

        # Camera-frame azimuth from unrotated rays (forward = -z), before tilt.
        azim_cam = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))

        if r_tilt is not None:
            rays = np.asarray(r_tilt) @ rays

        elev_deg = np.degrees(np.arcsin(np.clip(rays[1, :], -1.0, 1.0)))

        if azim_frame == "world":
            azim_deg = np.degrees(np.arctan2(rays[0, :], -rays[2, :]))
        else:
            azim_deg = azim_cam

        order = np.argsort(azim_deg)
        azim_deg, elev_deg = azim_deg[order], elev_deg[order]

        # --- 4. Bin onto uniform azimuth grid
        start_az = np.ceil(azim_deg[0] / bin_deg) * bin_deg
        end_az = np.floor(azim_deg[-1] / bin_deg) * bin_deg
        grid = np.arange(start_az, end_az + 1e-6, bin_deg)
        profile = np.interp(grid, azim_deg, elev_deg)

        diagnostics = dict(base_diag)
        diagnostics.update(
            {
                "boundary_coverage": float(boundary_coverage),
                "missing_columns": int(missing_cols),
                "hfov_deg": float(hfov_deg),
                "fov_y_deg": float(fov_y_deg),
                "bin_deg": float(bin_deg),
                "profile_std_deg": float(np.std(profile)),
                "profile_max_deg": float(np.max(profile)),
                "profile_min_deg": float(np.min(profile)),
                "profile_length": int(len(profile)),
            }
        )

        if boundary_coverage < min_boundary_coverage:
            return {
                "ok": False,
                "status": "LOW_CONFIDENCE",
                "reason": f"Boundary coverage too low ({boundary_coverage:.3f} < {min_boundary_coverage})",
                "profile": profile,
                "start_az": float(start_az),
                "diagnostics": diagnostics,
            }

        applicable, msg = is_profile_applicable(
            profile, self.min_std_deg, self.min_max_elev_deg
        )
        if not applicable:
            return {
                "ok": False,
                "status": "LOW_CONFIDENCE",
                "reason": msg,
                "profile": profile,
                "start_az": float(start_az),
                "diagnostics": diagnostics,
            }

        return {
            "ok": True,
            "status": "OK",
            "reason": "Valid profile extracted",
            "profile": profile,
            "start_az": float(start_az),
            "diagnostics": diagnostics,
        }


def fuse_profiles_world_frame(
    crop_entries: List[Dict[str, Any]],
    bin_deg: float = DEFAULT_BIN_DEG,
    n_bins_full: int = 720,
) -> Dict[str, Any]:
    """
    Multi-photo fusion (mirrors final/ METHODOLOGY section 2 step [3]):

    Composite per-crop profiles into ONE wide-field-of-view profile by placing
    each crop's camera-frame azimuth grid onto the world azimuth circle using
    the compass heading captured at snapshot time. Overlapping bins are
    averaged; gaps stay empty and reduce the coverage metric.

    Parameters
    ----------
    crop_entries : list of dicts, each with:
        "profile":   ndarray (L,) elevations in degrees
        "start_az":  float, camera-frame azimuth of profile[0] in degrees
        "bin_deg":   float, azimuth bin width
        "heading_deg": float, compass heading at capture (0-360)

    Returns
    -------
    dict with keys:
        "profile":  ndarray (720,) fused elevations, NaN where no coverage
        "coverage_deg": float, covered arc length in degrees
        "wide_fov_ok":  bool, coverage >= 200 degrees (the key success factor
                        identified in the final project evaluation)
    """
    fused = np.full(n_bins_full, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins_full, dtype=np.int64)

    for entry in crop_entries:
        profile = entry.get("profile")
        if profile is None or len(profile) == 0:
            continue
        profile = np.asarray(profile, dtype=np.float64)
        start_az = float(entry.get("start_az", 0.0))
        crop_bin = float(entry.get("bin_deg", bin_deg))
        heading = float(entry.get("heading_deg", 0.0))

        cam_az = start_az + np.arange(len(profile)) * crop_bin
        world_az = np.mod(cam_az + heading, 360.0)
        idx = np.mod(np.round(world_az / bin_deg).astype(np.int64), n_bins_full)

        valid = np.isfinite(profile)
        for i, v in zip(idx[valid], profile[valid]):
            if np.isfinite(fused[i]):
                fused[i] += v
                counts[i] += 1
            else:
                fused[i] = v
                counts[i] = 1

    covered = counts > 0
    fused[covered] /= counts[covered]

    coverage_deg = float(covered.sum() * bin_deg)
    return {
        "profile": fused,
        "coverage_deg": coverage_deg,
        "wide_fov_ok": bool(coverage_deg >= 200.0),
    }
