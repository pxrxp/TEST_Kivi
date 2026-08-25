# Skyline Geolocation — Kivy Mobile App

## Goal
GPS-free mountain skyline localization using a Kivy mobile app. Capture mountain photos, run on-device sky segmentation, extract 1D elevation profiles, and match against a pre-rendered horizon database — all without GPS.

## Architecture
```
Camera → Sensor Manager → Segmentation → Profile Extraction → Matching → Result
  │        (plyer)         (OpenCV DNN)    (numpy ray math)   (NCC+DTW)  (nearest city)
```

## Dependencies (all in buildozer.spec)
- `kivy==2.3.0` — UI framework
- `plyer` — accelerometer + compass sensors (mock fallback on desktop)
- `opencv` — DNN inference (ONNX), image processing, Canny edges
- `numpy<2.0.0` — all math, no scipy
- `pillow` — image I/O

**Not used at runtime:** scipy, torch, onnxruntime, pyarrow (dev/export only)

## File Layout

### Core App
| File | Purpose |
|------|---------|
| `main.py` | Kivy app — camera, multi-crop capture, segmentation, matching, JSON export |
| `camera_overlay.py` | Live camera preview with crosshair + horizon line (green/red/yellow) |
| `sensor_manager.py` | Plyer accelerometer (pitch/roll) + compass (heading), desktop mock fallback |

### Segmentation Pipeline
| File | Purpose |
|------|---------|
| `segmentation_engine.py` | U-Net inference → CLAHE dehazing → Canny refinement → slope caps → median smoothing |
| `model_inference.py` | OpenCV DNN backend (primary), ONNX Runtime / TFLite / PyTorch fallbacks |
| `profile_extractor.py` | Binary mask → sub-pixel boundary → pin-hole ray geometry → elevation profile |
| `sky_segmentation_unet_model.onnx` | MobileNetV3 U-Net (41MB, fixed 256×256, opset 11 for OpenCV DNN) |

### Matching & Database
| File | Purpose |
|------|---------|
| `matching.py` | Pure numpy NCC + DTW matching engine |
| `db_loader.py` | Loads `.npz` skyline DB, finds nearest known city |
| `build_db.py` | Converts parquet → `.npz` with region filter (dev tool) |
| `skyline_db.npz` | Khumbu region DB (53k rows, 10.5MB disk, ~138MB RAM) |

### Dev Tools (not in APK)
| File | Purpose |
|------|---------|
| `export_onnx.py` | Re-exports `.pth` → `.onnx` (fixed shapes, legacy tracer, opset 11) |
| `verify_versions.py` | Checks PyPI for latest compatible versions |

## Segmentation Pipeline Detail

1. **Inference** (`model_inference.py`): OpenCV DNN loads `.onnx` with fixed 256×256 input. Aspect-ratio-preserving resize with reflective padding. Output: P(terrain) map.

2. **Threshold**: `raw_mask = (prob <= 0.70)` — model outputs P(terrain), low prob = sky. Convention: 1=SKY, 0=TERRAIN in raw mask.

3. **Refinement** (`segmentation_engine.py`):
   - Top-connected sky region (row ≤ 15% height), fallback to largest component
   - CLAHE dehazing (clipLimit=1.2, tileGridSize=16×16) in sky zone only
   - Multi-scale Canny (30/150 + 20/100) fused, ±20px barrier + ±10px window
   - Outlier rejection: 5-neighbour median, 30px threshold
   - Two-pass slope cap: |Δr/Δc| ≤ 2.0 px/col
   - 9-tap median + Gaussian smoothing
   - Output: 0=SKY, 255=TERRAIN

4. **Profile extraction** (`profile_extractor.py`):
   - Top-down boundary row per column
   - 5-tap median filter on integer boundary
   - Sub-pixel parabolic edge fitting on Sobel-Y gradient (~0.1px precision)
   - Pin-hole ray geometry (FOV_y = 65°)
   - Optional tilt rotation matrix from sensor pitch/roll
   - Interpolation onto 0.5° azimuth grid
   - Quality gates: boundary coverage ≥ 0.5, profile std ≥ 1.5°, max elev ≥ 1.0°

## Database
- Source: `~/SkylineGeolocation/notebooks/02_SkylineDatabase/output/skyline_db.parquet` (1.3M rows, 464MB)
- Converted to `.npz` via `build_db.py` — region filter + compression
- Khumbu subset: 53k rows, 10.5MB on disk, 138MB in RAM
- Format: `lats` (float32), `lons` (float32), `elevations` (float32), `horizon` (uint8, 720 bins, 0.5° each)

## Matching
- **Coarse**: FFT-based Pearson NCC across all DB viewpoints (spatial stride)
- **Fine**: Re-NCC on top-5 coarse hits + neighbours
- **DTW**: Sakoe-Chiba band (window=10) on top-10 fine candidates
  - Pure numpy, O(M×W) where W=window, no extra deps
  - fastdtw is an alternative (pure Python, `pip install fastdtw`) — handles large warps better but adds a build dep for Android
  - Sakoe-Chiba chosen because: zero deps, fast for small warps (W=10 on 720-length profiles), Android-friendly
  - fastdtw could be added as optional with Sakoe-Chiba fallback
- **Score**: NCC − 0.01 × DTW_normalized
- Compass offset masking: narrows search to ±20° of heading
- Nearest city lookup from built-in table (26 cities: Himalaya, Alps, Rockies, etc.)

## Build
- Android build via GitHub Actions (`.github/workflows/build_android.yml`)
- `buildozer.spec` pinned: Python 3.11.5 (p4a v2024.01.21), arm64-v8a, API 33
- **Do not modify buildozer.spec or build_android.yml** — they take very long to debug

## Gotchas
- OpenCV DNN cannot load ONNX models with dynamic shapes — model must be exported with fixed 256×256 shapes and opset ≤ 11
- PyTorch 2.12's default ONNX exporter (dynamo) produces opset-18 nodes OpenCV can't parse — use `dynamo=False` in `torch.onnx.export`
- Model predicts P(terrain), not P(sky) — threshold is `<=` not `>=`
- `plyer.compass` doesn't exist on Linux — falls back to mock gracefully

## Differences from Desktop Pipeline (SkylineGeolocation)

What changed vs the original `~/SkylineGeolocation/src/`:

| Area | Desktop (`src/`) | Mobile (this repo) | Why |
|------|-----------------|-------------------|-----|
| **Inference backend** | PyTorch + `segmentation_models_pytorch` | OpenCV DNN | PyTorch too heavy for Android APK |
| **ONNX export** | Dynamic shapes, opset 13, dynamo exporter | Fixed 256×256, opset 11, legacy tracer (`dynamo=False`) | OpenCV DNN can't parse dynamic shapes or opset-18 nodes |
| **Dependencies** | scipy, torch, smp, albumentations, pyarrow, fastdtw | numpy, opencv, pillow, plyer, kivy only | Build complexity for Android — scipy/pyarrow are nightmarish to cross-compile |
| **DB format** | Parquet (pyarrow) | `.npz` (numpy compressed) | pyarrow can't build for Android; numpy is already required |
| **DTW** | `fastdtw` library | Sakoe-Chiba band (pure numpy) | fastdtw is an extra dep; Sakoe-Chiba is O(M×W), fast enough for W=10 |
| **Refinement methods** | 4 methods: lab_b_subpixel, grayscale_fixed, multichannel_fusion, dynamic_programming | lab_b_subpixel only (the best one) | YAGNI — other methods were ablation/exploration |
| **CLAHE dehazing** | In refinement | Same | Ported faithfully |
| **Column keep mask** | `compute_column_keep_mask()` for fog/haze detection | Not ported yet | Could be added for quality gating |
| **Fog/quality eval** | `evaluate_skyline_quality()` with Sobel gradient check | Not ported yet | Could be added for quality gating |
| **Profile sub-pixel fit** | Parabolic fit sign: `(gp1 - gm1) / denom` | Negated: `-(gp1 - gm1) / denom` | Production sign was verified against synthetic ground truth — mobile version matches the mathematically correct vertex formula |
| **Multi-crop fusion** | `fuse_profiles()` with ground-truth camera tilt | `fuse_profiles_world_frame()` using phone sensor heading/roll | Desktop uses GSV camera poses; mobile uses phone IMU |
| **Matching** | `match_query()` with `fastdtw` | Same logic, pure numpy DTW | Same algorithm, no extra deps |
| **Sky threshold** | `(prob <= 0.70)` → 1=sky | Same | Old mobile code had `(prob >= 0.30)` which was **inverted** — fixed |
| **Preprocessing** | Aspect-ratio resize + reflective padding | Same | Old mobile code did naive squish to 256×256 — fixed to match production |
| **3D rendering** | `mountain_engine.py` (pyrender, trimesh) | Not included | Desktop-only visualization, not needed on mobile |
| **Viewshed analysis** | `viewshed.py` (GDAL) | Not included | Desktop-only analysis, not needed on mobile |
| **Training** | `segmentation.py` training loop, data loaders | Not included | Training happens on desktop, not mobile |
| **DEM/download** | `dem.py`, `download_utils.py` | Not included | Desktop-only data pipeline |
| **Synthetic data** | `synthetic_generator.py` | Not included | Desktop-only data generation |
