"""
Horizon Matching Engine

Two-stage skyline matching: coarse Pearson NCC + DTW refinement.
Pure numpy — no scipy, no fastdtw. Mobile-friendly.

Coarse stage: FFT-based Pearson NCC across all DB viewpoints.
Fine stage: DTW (Sakoe-Chiba band) on top candidates.
"""

import numpy as np

DEG_PER_BIN = 90.0 / 255.0
DEFAULT_BIN_DEG = 0.5
DEFAULT_N_BINS = 720


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_zscore(x):
    x = np.asarray(x, dtype=np.float64)
    std = x.std()
    if std < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def _safe_zscore_matrix(mat):
    mat = np.asarray(mat, dtype=np.float64)
    means = mat.mean(axis=1, keepdims=True)
    stds = mat.std(axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    return (mat - means) / stds


def _feature_bundle(profile):
    profile = np.asarray(profile, dtype=np.float64)
    value = _safe_zscore(profile)
    d1 = _safe_zscore(np.gradient(value))
    return value, d1


def _feature_bundle_matrix(mat):
    mat = np.asarray(mat, dtype=np.float64)
    value = _safe_zscore_matrix(mat)
    d1 = _safe_zscore_matrix(np.gradient(value, axis=1))
    return value, d1


def _dtw_distance(q, d, window=15):
    """DTW with Sakoe-Chiba band — pure numpy, handles N-D features."""
    q = np.asarray(q, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    M = len(q)
    N = len(d)
    dt = np.full((M + 1, N + 1), np.inf)
    dt[0, 0] = 0.0

    for i in range(1, M + 1):
        j_lo = max(1, i - window)
        j_hi = min(N, i + window)
        qi = q[i - 1]
        for j in range(j_lo, j_hi + 1):
            cost = float(np.linalg.norm(qi - d[j - 1]))
            dt[i, j] = cost + min(dt[i - 1, j], dt[i, j - 1], dt[i - 1, j - 1])

    return dt[M, N] / M if M > 0 else 0.0


# ---------------------------------------------------------------------------
# Pearson NCC via FFT
# ---------------------------------------------------------------------------

def _pearson_ncc_batch(db_ext, query_zm, q_norm):
    N, ext_len = db_ext.shape
    M = len(query_zm)
    L = ext_len - M + 1

    q_pad = np.zeros(L, dtype=np.float64)
    q_pad[:M] = query_zm
    fq = np.fft.rfft(q_pad)
    fdb = np.fft.rfft(db_ext[:, :L], axis=1)
    numer = np.fft.irfft(fdb * np.conj(fq), n=L, axis=1)

    cum = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext, axis=1)], axis=1
    )
    cum_sq = np.concatenate(
        [np.zeros((N, 1), dtype=np.float64), np.cumsum(db_ext ** 2, axis=1)],
        axis=1,
    )

    win_sum = cum[:, M: M + L] - cum[:, :L]
    win_sq_sum = cum_sq[:, M: M + L] - cum_sq[:, :L]
    win_var = win_sq_sum - win_sum ** 2 / M
    win_norm = np.sqrt(np.maximum(win_var, 0.0))

    denom = q_norm * win_norm
    return numer / np.maximum(denom, 1e-12)


def feature_bundle_matrix(db_matrix):
    return _feature_bundle_matrix(np.asarray(db_matrix, dtype=np.float64))


def ncc_scores(
    db_val,
    db_d1,
    query_profile,
    bin_deg=DEFAULT_BIN_DEG,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
):
    query_profile = np.asarray(query_profile, dtype=np.float64)
    N, L = db_val.shape
    M = len(query_profile)

    db_ext_val = np.concatenate([db_val, db_val[:, : M - 1]], axis=1)
    db_ext_d1 = np.concatenate([db_d1, db_d1[:, : M - 1]], axis=1)

    q_val, q_d1 = _feature_bundle(query_profile)
    q_val_zm = q_val - q_val.mean()
    q_d1_zm = q_d1 - q_d1.mean()

    ncc_val = _pearson_ncc_batch(db_ext_val, q_val_zm, np.linalg.norm(q_val_zm))
    ncc_d1 = _pearson_ncc_batch(db_ext_d1, q_d1_zm, np.linalg.norm(q_d1_zm))

    combined = weights[0] * ncc_val + weights[1] * ncc_d1

    if expected_offset_deg is not None and tolerance_deg is not None:
        bins = np.arange(L)
        expected_bin = (expected_offset_deg / bin_deg) % L
        tolerance_bins = tolerance_deg / bin_deg
        circular_dist = np.minimum(
            (bins - expected_bin) % L, (expected_bin - bins) % L
        )
        mask = circular_dist <= tolerance_bins
        combined = np.where(mask[np.newaxis, :], combined, -np.inf)

    best_offset = np.argmax(combined, axis=1).astype(np.int32)
    best_corr = combined[np.arange(N), best_offset]
    return best_corr, best_offset


def fft_prefilter(
    db_matrix,
    query_profile,
    bin_deg=DEFAULT_BIN_DEG,
    weights=(0.5, 0.5),
    expected_offset_deg=None,
    tolerance_deg=None,
):
    db_val, db_d1 = feature_bundle_matrix(db_matrix)
    return ncc_scores(
        db_val,
        db_d1,
        query_profile,
        bin_deg,
        weights=weights,
        expected_offset_deg=expected_offset_deg,
        tolerance_deg=tolerance_deg,
    )


def _compute_confidence(matches, min_score_gap=0.03):
    if not matches:
        return {"best_score": 0.0, "second_score": 0.0, "score_gap": 0.0, "ambiguous": True}
    best = matches[0]["score"]
    second = matches[1]["score"] if len(matches) > 1 else 0.0
    gap = best - second
    return {"best_score": best, "second_score": second, "score_gap": gap, "ambiguous": gap < min_score_gap}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_query(
    db_matrix,
    lats,
    lons,
    query_profile,
    bin_deg=DEFAULT_BIN_DEG,
    top_k=10,
    dtw_window=15,
    expected_offset_deg=None,
    tolerance_deg=20.0,
    weights=(0.5, 0.5),
    spatial_stride=5,
    min_corr=0.3,
    min_score_gap=0.03,
):
    """Two-stage matching: coarse NCC + DTW refinement. Pure numpy."""
    if db_matrix is None or db_matrix.size == 0:
        return {"ok": False, "status": "INVALID_INPUT", "reason": "Empty DB", "matches": [], "confidence": {"ambiguous": True}, "diagnostics": {}}

    profile_length = len(query_profile)
    n_viewpoints = db_matrix.shape[0]

    if profile_length < 10:
        return {"ok": False, "status": "INVALID_QUERY", "reason": f"Profile too short ({profile_length})", "matches": [], "confidence": {"ambiguous": True}, "diagnostics": {}}

    if np.any(~np.isfinite(query_profile)):
        return {"ok": False, "status": "INVALID_QUERY", "reason": "NaN/Inf in profile", "matches": [], "confidence": {"ambiguous": True}, "diagnostics": {}}

    # 1. Coarse search
    coarse_idx = np.arange(0, n_viewpoints, spatial_stride)
    corr, offsets = fft_prefilter(
        db_matrix[coarse_idx], query_profile, bin_deg,
        weights=weights, expected_offset_deg=expected_offset_deg, tolerance_deg=tolerance_deg,
    )

    # 2. Fine refinement around top-5
    top5 = np.argsort(-corr)[:5]
    fine_set = set()
    for idx in top5:
        if corr[idx] == -np.inf:
            continue
        g = coarse_idx[idx]
        fine_set.update(range(max(0, g - spatial_stride), min(n_viewpoints, g + spatial_stride + 1)))
    fine_idx = np.array(sorted(fine_set), dtype=np.int32)
    fine_corr, fine_offsets = fft_prefilter(
        db_matrix[fine_idx], query_profile, bin_deg,
        weights=weights, expected_offset_deg=expected_offset_deg, tolerance_deg=tolerance_deg,
    )

    # 3. DTW on top-k
    query_val, query_d1 = _feature_bundle(query_profile)
    query_feat = np.column_stack([query_val, query_d1])

    candidates = []
    topk = np.argsort(-fine_corr)[:top_k]
    for idx in topk:
        if fine_corr[idx] == -np.inf:
            continue
        vp = fine_idx[idx]
        offset = fine_offsets[idx]
        horizon = db_matrix[vp]
        windowed = horizon[np.arange(offset, offset + profile_length) % len(horizon)]
        d_val, d_d1 = _feature_bundle(windowed)
        db_feat = np.column_stack([d_val, d_d1])
        dtw_cost = _dtw_distance(query_feat, db_feat, window=dtw_window)
        candidates.append({
            "viewpoint_idx": int(vp),
            "fft_corr": float(fine_corr[idx]),
            "dtw_cost": float(dtw_cost),
            "offset_bin": int(offset),
            "score": float(fine_corr[idx] - 0.01 * dtw_cost),
        })

    if not candidates:
        return {"ok": False, "status": "NO_MATCH", "reason": "No candidates", "matches": [], "confidence": {"ambiguous": True}, "diagnostics": {}}

    candidates.sort(key=lambda c: c["score"], reverse=True)

    matches = [
        {
            "row_index": c["viewpoint_idx"],
            "lat": float(lats[c["viewpoint_idx"]]),
            "lon": float(lons[c["viewpoint_idx"]]),
            "score": c["score"],
            "fft_corr": c["fft_corr"],
            "dtw_distance": c["dtw_cost"],
            "offset_deg": float(c["offset_bin"] * bin_deg),
        }
        for c in candidates
    ]

    conf = _compute_confidence(matches, min_score_gap)
    ok = conf["best_score"] >= min_corr and not conf["ambiguous"]

    return {
        "ok": ok,
        "status": "OK" if ok else "LOW_CONFIDENCE",
        "reason": "Match found" if ok else f"Score {conf['best_score']:.3f}, gap {conf['score_gap']:.4f}",
        "matches": matches,
        "confidence": conf,
        "diagnostics": {"n_coarse": len(coarse_idx), "n_fine": len(fine_idx), "n_candidates": len(candidates), "db_size": n_viewpoints},
    }
