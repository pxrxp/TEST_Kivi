#!/usr/bin/env python3
"""
Pipeline verification — matching on known-good panoramas.

Uses pre-computed terrain masks to test profile extraction + matching
against the on-device database format (.npz).

Resources:
  DB subset        ~11 MB  Compressed profiles
  Peak RAM        ~138 MB
"""
import json, os, sys, time
import numpy as np

_src = "/home/admin/SkylineGeolocation/src"
sys.path.insert(0, _src)
from query_profile import extract_elevation_profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matching as mm
from db_loader import SkylineDB

GSV = "/home/admin/SkylineGeolocation/data/street_view"

PANOS = [
    "45TvC0DOQFASM7NjOc-V1A",
    "1Xr_csMd0tcO1RgZfFfKGg",
    "2X37DP_ZxmaRyIb3xM0gLA",
    "3deMQ4aB_kzqrqpKVAr-Ow",
    "-yiHVpEf_kKTG9YGJ-dzsg",
    "0SWYlSUa8TQf7RgTdGijNw",
    "0dZapPlBpQequXJK7PQqjg",
    "3j6nusgkXQ2Xr9uYZdH5pw",
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians([lat1, lat2])
    return R * 2 * np.arcsin(np.sqrt(
        np.sin((p2-p1)/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2))


def main():
    with open(os.path.join(GSV, "ground_truth.json")) as f:
        gt = json.load(f)

    db = SkylineDB()
    db.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "skyline_db.npz"), quiet=True)

    for pid in PANOS:
        g = gt[pid]
        true_lat, true_lon = g["true_lat"], g["true_lon"]
        r_tilt = np.array(g["cam_R_tilt"]) if g.get("cam_R_tilt") else None

        t0 = time.time()
        pr = extract_elevation_profile(
            os.path.join(GSV, "masks", f"{pid}.png"),
            fov_y_deg=g["fov_y_deg"], r_tilt=r_tilt, bin_deg=0.5)
        ext_ms = (time.time() - t0) * 1000

        if not pr["ok"]:
            print(f"{pid}")
            print(f"  actual:     {true_lat:.4f}, {true_lon:.4f}")
            print(f"  profile failed: {pr['status']}")
            print()
            continue

        profile = pr["profile"]
        query = np.where(np.isfinite(profile), profile, 0.0)

        t0 = time.time()
        res = mm.match_query(
            db.horizon_matrix, db.lats, db.lons, query,
            bin_deg=0.5, top_k=10, spatial_stride=3)
        mch_ms = (time.time() - t0) * 1000

        if res["matches"]:
            m = res["matches"][0]
            pred_lat, pred_lon = m["lat"], m["lon"]
            error = haversine(true_lat, true_lon, pred_lat, pred_lon)
            ok = error < 500
            print(f"{pid}")
            print(f"  actual:     {true_lat:.4f}, {true_lon:.4f}")
            print(f"  predicted:  {pred_lat:.4f}, {pred_lon:.4f}")
            print(f"  {'ok' if ok else f'off by {error/1000:.1f}km'}  ({m['score']:.3f})  [{ext_ms:.0f}ms + {mch_ms:.0f}ms]")
        else:
            print(f"{pid}")
            print(f"  actual:     {true_lat:.4f}, {true_lon:.4f}")
            print(f"  predicted:  (no match)  [{ext_ms:.0f}ms + {mch_ms:.0f}ms]")
        print()


if __name__ == "__main__":
    main()
