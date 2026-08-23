# Build Standalone GPS-Free Kivy Mobile App Foundation for Skyline Geolocation

## 🎯 Goal
Build the mobile app foundation for a **100% Standalone GPS-Free** Skyline Geolocation system using **Kivy / KivyMD**. The app captures mountain skyline photos, guides the user to level and align the phone using mobile IMU/Compass sensors, runs on-device deep learning sky segmentation, and extracts 1D topographic elevation profiles for horizon matching against the global database without needing GPS.

---

## 🏛️ Standalone GPS-Free Architecture
The app operates completely offline without GPS dependencies, using sensor-guided capture:
1. **IMU Accelerometer Pitch ($\pm 2^\circ$):** Measures camera pitch/tilt and guides user to hold the phone upright ($0^\circ$ pitch) and level before capture.
2. **Magnetometer Compass ($\pm 15^\circ$):** Records digital compass heading ($0^\circ - 360^\circ$) at snapshot time.
3. **Multi-Photo Perspective Fusion:** Supports capturing 2–3 perspective crops (e.g. $0^\circ, +90^\circ, +180^\circ$) to build a $220^\circ - 270^\circ$ wide-FOV joint horizon profile that breaks mountain valley symmetry and achieves sub-kilometer localization GPS-free!

---

## 📦 Required Dependencies
- `kivy` / `kivymd` (Mobile UI framework)
- `plyer` (Cross-platform accelerometer and compass sensor access)
- `opencv-python-headless` or `cv2` (Image processing & Canny edge guidance)
- `numpy` & `scipy` (1D profile extraction & signal processing)
- `onnxruntime` or `torch` (On-device MobileNetV3 U-Net sky segmentation inference)
- `PIL` / `Pillow` (Image manipulation)

---

## 🧩 Core Modules to Build

### Module 1: Live Camera Preview & Real-Time Level Guidance (`camera_overlay.py`)
- Display live camera feed on the main screen.
- Overlay a **Real-Time Artificial Horizon / Bubble Level**:
  - Center crosshair with a pitch/roll horizon line.
  - Line turns **GREEN** when phone pitch is within $\pm 2.0^\circ$ of vertical ($0^\circ$ tilt) and roll is within $\pm 2.0^\circ$.
  - Line turns **RED/YELLOW** when tilted, showing visual text prompts: `"TILT UP"`, `"TILT DOWN"`, `"LEVEL PHONE"`.
- **Capture Button:** Enabled when sensors confirm valid phone alignment.

### Module 2: GPS-Free Sensor Manager (`sensor_manager.py`)
Wrap `plyer` sensor interfaces with fallback mock sensors for desktop testing:
- **Pitch & Roll (Accelerometer):**
  $$\text{pitch} = \arctan2(a_y, \sqrt{a_x^2 + a_z^2}) \cdot \frac{180}{\pi}$$
  $$\text{roll} = \arctan2(-a_x, a_z) \cdot \frac{180}{\pi}$$
- **Azimuth Heading (Compass):** $0^\circ - 360^\circ$ North heading.
- Returns a structured dictionary on capture:
  ```python
  {
      "pitch_deg": float,
      "roll_deg": float,
      "heading_deg": float,
      "fov_y_deg": 65.0,  # Camera FOV default
  }

Module 3: On-Device Sky Segmentation Engine (segmentation_engine.py)

Convert captured RGB photo into a refined 2D binary sky mask
(0 = \text{sky}, 255 = \text{terrain}):

1.  Model Inference: Run tu-mobilenetv3_large_100 U-Net model (via PyTorch or
    ONNX Runtime model sky_segmentation_unet_model.onnx).
      - Input shape: (1, 3, 256, 256), ImageNet normalized
        (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
      - Output: Sigmoid probability map P(\text{sky}) \in [0, 1]. Threshold at
        P \le 0.70 for raw sky mask.
2.  Sky Mask Refinement (refine_sky_mask_with_guidance):
      - Top-Connected Sky Filtering: Flood-fill sky starting from top edge (row
        <= 15). Fallback to largest sky region if steep mountain fills top
        frame.
      - Canny Ridge Barrier: Restrict Canny edge search to a narrow
        \pm 10\text{px} window around U-Net boundary to prevent bleeding into
        ground snow fields or high clouds.
      - Two-Pass Physical Slope Cap: Enforce max slope
        |\Delta r| \le 2.0\text{px/col} in forward/backward passes to eliminate
        vertical cloud cliff steps.
      - Smoothing: 9-tap 1D median filter across columns.

def refine_sky_mask_with_guidance(img_np, raw_unet_mask, kernel_size=3):
    H, W = raw_unet_mask.shape
    sky1 = (raw_unet_mask == 1).astype(np.uint8)

    # 1. Top-connected sky region
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sky1, connectivity=8)
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

    # 2. Canny edge guidance
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    fine_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    coarse_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny_edges = (cv2.Canny(fine_blur, 30, 150) > 0) | (cv2.Canny(coarse_blur, 20, 100) > 0)

    boundaries = np.full(W, -1, dtype=np.float64)

    for col in range(W):
        sky_rows = np.where(top_sky[:, col] == 1)[0]
        if len(sky_rows) == 0:
            continue
        diffs = np.diff(sky_rows)
        gaps = np.where(diffs > 3)[0]
        max_sky_row = sky_rows[gaps[0]] if len(gaps) > 0 else sky_rows[-1]

        if canny_edges is not None:
            edge_rows = np.where(canny_edges[:, col])[0]
            valid_mountain_edges = [r for r in edge_rows if abs(r - max_sky_row) <= 10]
            if len(valid_mountain_edges) > 0:
                max_sky_row = valid_mountain_edges[0]

        boundaries[col] = float(max_sky_row)

    # 3. Outlier filter & Two-pass physical slope constraint (|dr/dc| <= 2.0 px/col)
    valid = boundaries >= 0
    if np.any(valid):
        all_cols = np.arange(W, dtype=np.float64)
        boundaries = np.interp(all_cols, all_cols[valid], boundaries[valid])

        max_slope = 2.0
        for c in range(1, W):
            delta = boundaries[c] - boundaries[c - 1]
            if abs(delta) > max_slope:
                boundaries[c] = boundaries[c - 1] + np.sign(delta) * max_slope
        for c in range(W - 2, -1, -1):
            delta = boundaries[c] - boundaries[c + 1]
            if abs(delta) > max_slope:
                boundaries[c] = boundaries[c + 1] + np.sign(delta) * max_slope

        pad = 4
        padded = np.pad(boundaries, (pad, pad), mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        meds = np.median(sliding_window_view(padded, 9, axis=0), axis=1)
        boundaries = meds

    refined = np.zeros((H, W), dtype=np.uint8)
    for col in range(W):
        b = int(np.clip(round(boundaries[col]), 0, H - 1))
        refined[:b, col] = 1

    return np.where(refined == 1, 0, 255).astype(np.uint8)

Module 4: 1D Elevation Profile Extraction (profile_extractor.py)

Convert 2D binary sky mask (0=\text{sky}, 255=\text{terrain}) into a 1D
elevation angle vector \theta(\phi) projected onto camera geometry:

1.  Find top-down sky boundary row r(c) per column c.
2.  Sub-Pixel Parabolic Edge Fitting: Fit 3-point parabola on image Sobel-Y
    gradient around boundary row for 0.1\text{px} precision.
3.  Compute ray vectors from camera focal length (FOV_y = 65.0^\circ):
    \text{rays} = \begin{bmatrix} (c - x_c)/f_x \\ (y_c - r(c))/f_y \\ -1 \end{bmatrix}
4.  Calculate elevation angles \theta = \arcsin(\text{ray}_y).
5.  Interpolate onto uniform angular azimuth grid (\text{bin\_deg} = 0.5^\circ).

Module 5: Standalone Payload Packaging

Package captured session data into structured JSON ready for multi-photo fusion:

{
  "timestamp": "2026-08-19T14:30:00Z",
  "sensors": {
    "pitch_deg": -0.4,
    "roll_deg": 0.1,
    "heading_deg": 184.2,
    "fov_y_deg": 65.0
  },
  "diagnostics": {
    "sky_ratio": 0.38,
    "boundary_coverage": 0.98,
    "profile_std_deg": 4.12,
    "profile_max_deg": 18.5
  },
  "profile": [12.1, 12.3, 12.8, 13.5, "... (1D elevation angle array)"]
}

🖥️ UI / UX Layout Design (KivyMD)

+---------------------------------------------------+
|               [ STATUS BANNER ]                   |
|         "HOLD LEVEL: PITCH -0.4° | ROLL 0.1°"     |
+---------------------------------------------------+
|                                                   |
|                LIVE CAMERA PREVIEW                |
|                                                   |
|                     +-----+                       |
|                     |     |  <-- Crosshair        |
|               ------|--+--|------                 |
|                     |     |  (Turns GREEN)        |
|                     +-----+                       |
|                                                   |
|    Skyline Overlay Line (Cyan) Live Preview       |
|                                                   |
+---------------------------------------------------+
|  Heading: 184° N  |  Multi-Photo Mode: [ 1 / 3 ]  |
+---------------------------------------------------+
|            [ 📸 CAPTURE SKYLINE CROP ]            |
+---------------------------------------------------+

🚀 Execution Steps for Building

1.  Build sensor_manager.py with Plyer accelerometer and compass interfaces +
    desktop mock fallback.
2.  Build camera_overlay.py with Kivy Camera preview & real-time pitch/roll
    green level indicator.
3.  Build segmentation_engine.py with MobileNetV3 U-Net inference & Canny/slope
    refinement routines.
4.  Build profile_extractor.py with sub-pixel parabolic edge fitting & elevation
    angle conversion.
5.  Combine modules in main.py with multi-crop capture support and standalone
    payload JSON export.
