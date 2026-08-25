"""
Convert parquet skyline database to compressed .npz for on-device use.

Output format:
  - lats: float32 (N,)
  - lons: float32 (N,)
  - elevations: float32 (N,)
  - horizon: uint8 (N, 720) — already encoded as uint8 in parquet

With np.savez_compressed, a 50k-row Himalaya subset is ~4-6MB.

Usage:
    python build_db.py [--input skyline_db.parquet] [--output skyline_db.npz]
                       [--lon-min 86.0] [--lon-max 88.0]
                       [--lat-min 27.0] [--lat-max 29.0]
                       [--max-rows 50000]
"""

import argparse
import os
import sys
import numpy as np


def convert_parquet_to_npz(
    input_path,
    output_path,
    lon_min=None,
    lon_max=None,
    lat_min=None,
    lat_max=None,
    max_rows=None,
):
    """Read parquet and write compressed npz."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow required for conversion (pip install pyarrow)")
        sys.exit(1)

    pf = pq.ParquetFile(input_path)
    n_total = pf.metadata.num_rows
    print(f"Input: {input_path} ({n_total:,} rows)")

    # Column names
    columns = ["lon", "lat", "elevation_m", "raw_horizon_deg"]

    lats_all = []
    lons_all = []
    elev_all = []
    horizons_all = []
    rows_read = 0
    rows_kept = 0

    for rg_idx in range(pf.metadata.num_row_groups):
        if max_rows is not None and rows_kept >= max_rows:
            break

        rg = pf.read_row_group(rg_idx, columns=columns)
        n_rg = len(rg)

        lons = np.array(rg.column("lon"))
        lats = np.array(rg.column("lat"))
        elevs = np.array(rg.column("elevation_m"), dtype=np.float32)

        # Region filter
        mask = np.ones(n_rg, dtype=bool)
        if lon_min is not None:
            mask &= lons >= lon_min
        if lon_max is not None:
            mask &= lons <= lon_max
        if lat_min is not None:
            mask &= lats >= lat_min
        if lat_max is not None:
            mask &= lats <= lat_max

        n_pass = mask.sum()
        if n_pass == 0:
            rows_read += n_rg
            continue

        lons_all.append(lons[mask])
        lats_all.append(lats[mask])
        elev_all.append(elevs[mask])

        # Decode uint8 horizons to uint8 array (already uint8 in parquet)
        for i in np.where(mask)[0]:
            encoded = rg.column("raw_horizon_deg")[i].as_py()
            horizons_all.append(np.array(encoded, dtype=np.uint8))

        rows_kept += n_pass
        rows_read += n_rg

        if rows_read % (pf.metadata.num_row_groups // 10 + 1) == 0:
            print(f"  ... {rows_read:,}/{n_total:,} row groups read, {rows_kept:,} kept")

    if rows_kept == 0:
        print("WARNING: No rows passed filter — writing full DB instead")
        return convert_parquet_to_npz(
            input_path, output_path,
            lon_min=None, lon_max=None,
            lat_min=None, lat_max=None,
            max_rows=max_rows,
        )

    lats = np.concatenate(lats_all).astype(np.float32)
    lons = np.concatenate(lons_all).astype(np.float32)
    elevations = np.concatenate(elev_all).astype(np.float32)
    horizon = np.stack(horizons_all)  # (N, 720) uint8

    # Compress and save
    np.savez_compressed(
        output_path,
        lats=lats,
        lons=lons,
        elevations=elevations,
        horizon=horizon,
    )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nOutput: {output_path}")
    print(f"  Rows: {rows_kept:,}")
    print(f"  Horizon shape: {horizon.shape}")
    print(f"  File size: {size_mb:.1f} MB")
    print(f"  Lon range: [{lons.min():.4f}, {lons.max():.4f}]")
    print(f"  Lat range: [{lats.min():.4f}, {lats.max():.4f}]")

    # RAM estimate for on-device loading
    horizon_mb = horizon.nbytes / (1024 * 1024)
    total_mb = (lats.nbytes + lons.nbytes + elevations.nbytes + horizon.nbytes) / (1024 * 1024)
    print(f"  In-memory estimate: {total_mb:.1f} MB (horizon: {horizon_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert parquet skyline DB to npz")
    parser.add_argument("-i", "--input", default="skyline_db.parquet")
    parser.add_argument("-o", "--output", default="skyline_db.npz")
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    convert_parquet_to_npz(
        args.input, args.output,
        lon_min=args.lon_min, lon_max=args.lon_max,
        lat_min=args.lat_min, lat_max=args.lat_max,
        max_rows=args.max_rows,
    )
