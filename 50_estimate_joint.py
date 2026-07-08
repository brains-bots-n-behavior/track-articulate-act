#!/usr/bin/env python
"""Stage 50: per-label joint type + parameter estimation.

For each label in data/<scene>/any4d/<label>/, treat the masked points as
samples on a single rigid movable part of an articulated object and fit a
motion model. Three methods are available:

    --method procrustes  (DEFAULT)
        Per target frame, build the trajectory x_t = pts_ref + scene_flow[t],
        solve the FULL rigid alignment   x_t = R_t @ pts_ref + d_t   via SVD
        Procrustes, extract (axis, angle theta) from R_t, recover an axis
        point from (I - R_t) c = d_t. Filter frames by theta in
        [--theta-min, --theta-max] (so only well-rotated, well-conditioned
        frames contribute) and by RMS within --frame-rms-quantile. Aggregate
        the axis direction and axis-point (re-anchored to the part centroid)
        via sign-aligned median.
        This is the recommended mode — it handles finite rotations exactly,
        unlike the linear-twist models below.

    --method linear      (the slide algorithm)
        Pool (x_i, f_i) across all frames, fit
            prismatic   f_i = t + eps
            revolute    f_i = omega x x_i + b + eps
        and compare residuals. Recover axis from the winning model. Add
        --per-frame to fit each frame independently and median-aggregate.
        These linear models assume infinitesimal motion; they will be biased
        whenever the scene flow encodes a finite rotation.

Reads:
    data/<scene>/any4d/<label>/pts3d_ref.npy
    data/<scene>/any4d/<label>/scene_flow/<frame>.npy

Writes:
    data/<scene>/any4d/joints.json                (summary, all labels)
    data/<scene>/any4d/<label>/joint.json         (per label)

Examples:
    python scripts/50_estimate_joint.py --scene-dir macbook-all
    python scripts/50_estimate_joint.py --scene-dir macbook-all \\
        --labels laptop_up --theta-min 5 --theta-max 60
    python scripts/50_estimate_joint.py --scene-dir macbook-all \\
        --method linear --per-frame --min-flow 0.005
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Path to data/<scene>/ (must contain any4d/<label>/...)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only process these labels (default: all under any4d/)")
    p.add_argument("--method", choices=["procrustes", "linear"], default="procrustes",
                   help="Fit strategy. procrustes (default) handles finite "
                        "rotations exactly; linear is the slide algorithm.")
    p.add_argument("--per-frame", action="store_true",
                   help="(linear only) fit each frame independently and median-aggregate, "
                        "instead of pooling all frames into one LS problem")
    p.add_argument("--theta-min", type=float, default=3.0,
                   help="(procrustes) skip frames with rotation angle below this in degrees "
                        "(default 3); too-small theta is noise-dominated")
    p.add_argument("--theta-max", type=float, default=90.0,
                   help="(procrustes) skip frames with rotation angle above this in degrees "
                        "(default 90); too-large theta risks Procrustes ambiguity")
    p.add_argument("--frame-rms-quantile", type=float, default=0.75,
                   help="(procrustes) keep only frames whose Procrustes RMS is in the "
                        "lower quantile, i.e. the best-fitting frames (default 0.75)")
    p.add_argument("--refine", dest="refine", action="store_true", default=True,
                   help="(procrustes) after coarse fit, run a Levenberg-Marquardt "
                        "refinement on a window of frames temporally adjacent to "
                        "ref_frame (DEFAULT on)")
    p.add_argument("--no-refine", dest="refine", action="store_false",
                   help="Disable the refinement step")
    p.add_argument("--refine-window", type=int, default=8,
                   help="Number of frames closest in time to ref_frame to use in "
                        "the refinement step (default 8). Smaller theta = cleaner "
                        "rigid model, but too few frames = under-constrained axis.")
    p.add_argument("--refine-points", type=int, default=4000,
                   help="Subsample to this many points per label for the LM "
                        "refinement (default 4000). Procrustes is global so a few "
                        "thousand uniformly-random points is plenty.")
    p.add_argument("--refine-side", choices=["both", "prev", "next"], default="both",
                   help="Which temporal side of ref_frame to pull window frames "
                        "from. 'prev' = strictly frame_idx < ref_frame (default 'both').")
    p.add_argument("--refine-theta-min", type=float, default=1.5,
                   help="Minimum |per-frame theta| (degrees) for a frame to enter "
                        "the refinement window (default 1.5). Filters frames where "
                        "the part has barely moved so LM doesn't fit MoGe noise.")
    p.add_argument("--min-flow", type=float, default=0.0,
                   help="Skip frames whose mean flow magnitude is below this "
                        "(filters near-stationary frames that add only noise; default 0)")
    p.add_argument("--max-samples", type=int, default=200_000,
                   help="(linear pooled only) cap on (point, frame) pairs (default 200k)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for point subsampling")
    p.add_argument("--dry-run", action="store_true",
                   help="Print results but do not write JSON files")
    return p.parse_args()


def skew(v: np.ndarray) -> np.ndarray:
    """[v]_x as a (3, 3) matrix. v can be (3,) or (N, 3) -> (N, 3, 3)."""
    if v.ndim == 1:
        x, y, z = v
        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]], dtype=v.dtype)
    # batched
    N = v.shape[0]
    z = np.zeros(N, dtype=v.dtype)
    out = np.stack([
        np.stack([z,         -v[:, 2],   v[:, 1]], axis=1),
        np.stack([v[:, 2],   z,         -v[:, 0]], axis=1),
        np.stack([-v[:, 1],  v[:, 0],   z       ], axis=1),
    ], axis=1)
    return out  # (N, 3, 3)


def fit_prismatic(F: np.ndarray):
    """f_i = t + eps. Returns (t, residual_mse)."""
    t = F.mean(axis=0)
    resid = F - t[None, :]
    mse = float((resid ** 2).sum(axis=1).mean())
    return t.astype(np.float64), mse


def fit_revolute(X: np.ndarray, F: np.ndarray):
    """f_i = omega x x_i + b. Returns (omega, b, residual_mse).

    Stack as A @ theta = f, where theta = [omega; b] in R^6 and
    A_i = [-[x_i]_x | I_3] is (3, 6).
    """
    N = X.shape[0]
    # Build A: (N, 3, 6)
    Sx = skew(X.astype(np.float64))             # (N, 3, 3)
    A = np.concatenate([-Sx, np.broadcast_to(np.eye(3), (N, 3, 3))], axis=2)
    A = A.reshape(N * 3, 6)
    f = F.astype(np.float64).reshape(N * 3)

    theta, _resid_unused, _rank, _sv = np.linalg.lstsq(A, f, rcond=None)
    omega, b = theta[:3], theta[3:]

    pred = A @ theta
    resid = (f - pred).reshape(N, 3)
    mse = float((resid ** 2).sum(axis=1).mean())
    return omega, b, mse


def axis_from_revolute(omega: np.ndarray, b: np.ndarray):
    """axis direction = omega / ||omega||, axis point = -(omega x b) / ||omega||^2."""
    nw = float(np.linalg.norm(omega))
    if nw < 1e-12:
        return None, None
    axis_dir = omega / nw
    axis_pt = -np.cross(omega, b) / (nw ** 2)
    return axis_dir, axis_pt


def classify(E_pris, E_rev):
    return "prismatic" if E_pris < E_rev else "revolute"


def fit_one_label_pooled(X_ref: np.ndarray, flows: list, min_flow: float,
                         max_samples: int, rng: np.random.Generator):
    """Pool (x, f) pairs across frames, fit both models, decide."""
    Xs, Fs, kept_frames = [], [], []
    for frame_idx, F in flows:
        mag = float(np.linalg.norm(F, axis=1).mean())
        if mag < min_flow:
            continue
        Xs.append(X_ref)
        Fs.append(F)
        kept_frames.append(frame_idx)
    if not Xs:
        return None, "no frames passed --min-flow"

    X = np.concatenate(Xs, axis=0)
    F = np.concatenate(Fs, axis=0)
    if X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X = X[idx]
        F = F[idx]

    t_hat, E_pris = fit_prismatic(F)
    omega, b, E_rev = fit_revolute(X, F)
    joint_type = classify(E_pris, E_rev)

    result = {
        "fit_strategy": "pooled",
        "n_samples": int(X.shape[0]),
        "n_frames_used": len(kept_frames),
        "frames_used_first": int(kept_frames[0]),
        "frames_used_last": int(kept_frames[-1]),
        "residual_prismatic": E_pris,
        "residual_revolute": E_rev,
        "type": joint_type,
        "prismatic": {
            "t": t_hat.tolist(),
        },
        "revolute": {
            "omega": omega.tolist(),
            "b": b.tolist(),
        },
    }
    if joint_type == "prismatic":
        n = float(np.linalg.norm(t_hat))
        result["axis_direction"] = (t_hat / max(n, 1e-12)).tolist()
        result["axis_point"] = None
    else:
        a_dir, a_pt = axis_from_revolute(omega, b)
        if a_dir is None:
            result["axis_direction"] = None
            result["axis_point"] = None
            result["note"] = "revolute selected but ||omega|| ~ 0; axis undefined"
        else:
            result["axis_direction"] = a_dir.tolist()
            result["axis_point"] = a_pt.tolist()
    return result, None


def fit_one_label_per_frame(X_ref: np.ndarray, flows: list, min_flow: float):
    """Fit prismatic and revolute on each frame independently. Aggregate by
    voting on type per frame, then take median axis_direction (after sign
    alignment) and median axis_point for revolute."""
    per_frame = []
    for frame_idx, F in flows:
        mag = float(np.linalg.norm(F, axis=1).mean())
        if mag < min_flow:
            continue
        t_hat, E_pris = fit_prismatic(F)
        omega, b, E_rev = fit_revolute(X_ref, F)
        jt = classify(E_pris, E_rev)
        if jt == "prismatic":
            n = float(np.linalg.norm(t_hat))
            axis = (t_hat / max(n, 1e-12)) if n > 1e-12 else None
            axis_pt = None
        else:
            axis, axis_pt = axis_from_revolute(omega, b)
        per_frame.append({
            "frame": int(frame_idx),
            "type": jt,
            "E_pris": E_pris,
            "E_rev": E_rev,
            "axis_dir": None if axis is None else axis.tolist(),
            "axis_point": None if axis_pt is None else axis_pt.tolist(),
            "omega": omega.tolist(),
            "b": b.tolist(),
            "t": t_hat.tolist(),
            "flow_mag_mean": mag,
        })
    if not per_frame:
        return None, "no frames passed --min-flow"

    types = [pf["type"] for pf in per_frame]
    pris_count = types.count("prismatic")
    rev_count = types.count("revolute")
    joint_type = "prismatic" if pris_count >= rev_count else "revolute"

    # Pull axis directions for the winning type, sign-align to the first valid one
    sel = [pf for pf in per_frame
           if pf["type"] == joint_type and pf["axis_dir"] is not None]
    if not sel:
        return {
            "fit_strategy": "per_frame",
            "type": joint_type,
            "n_frames_used": len(per_frame),
            "axis_direction": None,
            "axis_point": None,
            "per_frame_summary": per_frame,
            "note": "no valid axis on winning-type frames",
        }, None

    ref_axis = np.asarray(sel[0]["axis_dir"], dtype=np.float64)
    axes = []
    for pf in sel:
        a = np.asarray(pf["axis_dir"], dtype=np.float64)
        if float(a @ ref_axis) < 0:
            a = -a
        axes.append(a)
    axes = np.stack(axes, axis=0)
    axis_median = np.median(axes, axis=0)
    axis_median /= max(np.linalg.norm(axis_median), 1e-12)

    axis_pt_median = None
    if joint_type == "revolute":
        pts = np.stack([np.asarray(pf["axis_point"], dtype=np.float64)
                        for pf in sel if pf["axis_point"] is not None], axis=0)
        if pts.size:
            axis_pt_median = np.median(pts, axis=0).tolist()

    return {
        "fit_strategy": "per_frame",
        "type": joint_type,
        "n_frames_used": len(per_frame),
        "votes": {"prismatic": pris_count, "revolute": rev_count},
        "axis_direction": axis_median.tolist(),
        "axis_point": axis_pt_median,
        "per_frame_summary": per_frame,
    }, None


def rot_to_axis_angle(R: np.ndarray):
    """Return (axis_unit, theta_radians). theta in [0, pi].

    Returns (None, 0.0) for the identity. Handles theta-near-pi degenerate
    case via eigenvector of R associated with eigenvalue +1.
    """
    cos_t = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-7:
        return None, 0.0
    if abs(theta - np.pi) < 1e-4:
        # Pull axis from the eigenvector with eigenvalue closest to +1
        w, v = np.linalg.eig(R)
        idx = int(np.argmin(np.abs(w - 1.0)))
        a = np.real(v[:, idx])
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            return None, theta
        return (a / n).astype(np.float64), theta
    a = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]], dtype=np.float64)
    a /= (2.0 * np.sin(theta))
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return None, theta
    return a / n, theta


def fit_procrustes_one_frame(X_ref: np.ndarray, X_tgt: np.ndarray):
    """Solve X_tgt = R @ X_ref + d via SVD Procrustes.

    Returns dict with R, d, theta, axis_dir (None if no rotation),
    axis_point_origin (point on the axis closest to world origin; None if
    pure translation), rms (root-mean-square 3D residual).
    """
    mu_ref = X_ref.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    Xc = X_ref - mu_ref
    Yc = X_tgt - mu_tgt
    H = Xc.T @ Yc  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ D @ U.T
    d = mu_tgt - R @ mu_ref

    pred = X_ref @ R.T + d
    rms = float(np.sqrt(((pred - X_tgt) ** 2).sum(axis=1).mean()))

    axis_dir, theta = rot_to_axis_angle(R)
    if axis_dir is None:
        return {"R": R, "d": d, "theta": theta, "axis_dir": None,
                "axis_point_origin": None, "rms": rms}

    # (I - R) c = d, picks closest point on the axis to the world origin.
    # (I - R) has rank 2 (null space = axis_dir); lstsq returns the min-norm
    # solution, which is automatically perpendicular to axis_dir.
    c, *_ = np.linalg.lstsq(np.eye(3) - R, d, rcond=None)
    return {"R": R, "d": d, "theta": theta,
            "axis_dir": axis_dir, "axis_point_origin": c, "rms": rms}


def fit_one_label_procrustes(X_ref: np.ndarray, flows: list, theta_min_deg: float,
                             theta_max_deg: float, rms_quantile: float,
                             min_flow: float):
    """Per-frame Procrustes; filter by theta band + RMS quantile; aggregate."""
    theta_min = np.deg2rad(theta_min_deg)
    theta_max = np.deg2rad(theta_max_deg)
    centroid_part = X_ref.mean(axis=0)

    per_frame = []
    for frame_idx, F in flows:
        mag = float(np.linalg.norm(F, axis=1).mean())
        if mag < min_flow:
            continue
        X_tgt = X_ref + F
        out = fit_procrustes_one_frame(X_ref, X_tgt)
        # Also fit prismatic on this frame for type comparison
        t_hat = F.mean(axis=0)
        E_pris = float(((F - t_hat) ** 2).sum(axis=1).mean())

        entry = {
            "frame": int(frame_idx),
            "flow_mag_mean": mag,
            "theta_deg": float(np.rad2deg(out["theta"])),
            "rms": out["rms"],
            "E_pris": E_pris,
            "E_rev": out["rms"] ** 2,
            "d": out["d"].tolist(),
            "t_prismatic": t_hat.tolist(),
        }
        if out["axis_dir"] is not None:
            entry["axis_dir"] = out["axis_dir"].tolist()
            entry["axis_point_origin"] = out["axis_point_origin"].tolist()
            # Project part centroid onto axis line for a near-part anchor
            c0 = out["axis_point_origin"]
            a = out["axis_dir"]
            lam = float((centroid_part - c0) @ a)
            entry["axis_point_part"] = (c0 + lam * a).tolist()
        else:
            entry["axis_dir"] = None
            entry["axis_point_origin"] = None
            entry["axis_point_part"] = None
        per_frame.append(entry)

    if not per_frame:
        return None, "no frames after --min-flow"

    # Theta band: pick rotation-dominated, well-conditioned frames
    in_band = [e for e in per_frame
               if e["axis_dir"] is not None
               and theta_min <= np.deg2rad(e["theta_deg"]) <= theta_max]

    if not in_band:
        # Could not find any frame with usable rotation. Decide prismatic by
        # voting and aggregate translation direction.
        pris_votes = sum(1 for e in per_frame if e["E_pris"] < e["E_rev"])
        rev_votes = len(per_frame) - pris_votes
        type_ = "prismatic" if pris_votes >= rev_votes else "revolute"
        if type_ == "prismatic":
            t_med = np.median(
                np.stack([np.asarray(e["t_prismatic"]) for e in per_frame]),
                axis=0,
            )
            n = float(np.linalg.norm(t_med))
            return ({
                "method": "procrustes_per_frame",
                "type": "prismatic",
                "n_frames_used": len(per_frame),
                "n_frames_in_theta_band": 0,
                "axis_direction": (t_med / max(n, 1e-12)).tolist() if n > 1e-12 else None,
                "axis_point": None,
                "votes": {"prismatic": pris_votes, "revolute": rev_votes},
                "per_frame_summary": per_frame,
                "note": "no frames passed theta band; classified by prismatic-vs-revolute vote",
            }, None)
        return {
            "method": "procrustes_per_frame",
            "type": "revolute",
            "n_frames_used": len(per_frame),
            "n_frames_in_theta_band": 0,
            "axis_direction": None,
            "axis_point": None,
            "per_frame_summary": per_frame,
            "note": "no frames passed theta band but votes favor revolute; cannot localize axis",
        }, None

    # Filter further by RMS quantile (lower = better fit)
    rms_vals = np.array([e["rms"] for e in in_band])
    cutoff = float(np.quantile(rms_vals, rms_quantile))
    kept = [e for e in in_band if e["rms"] <= cutoff]
    if not kept:
        kept = in_band  # safety

    # Determine type by majority vote among kept frames
    pris_votes = sum(1 for e in kept if e["E_pris"] < e["E_rev"])
    rev_votes = len(kept) - pris_votes
    type_ = "prismatic" if pris_votes > rev_votes else "revolute"

    if type_ == "prismatic":
        ts = np.stack([np.asarray(e["t_prismatic"]) for e in kept])
        # Sign-align toward first
        ref = ts[0] / max(np.linalg.norm(ts[0]), 1e-12)
        aligned = []
        for t in ts:
            n = float(np.linalg.norm(t))
            if n < 1e-12:
                continue
            u = t / n
            if u @ ref < 0:
                u = -u
            aligned.append(u)
        if not aligned:
            return None, "prismatic but all translations are zero"
        axis_med = np.median(np.stack(aligned), axis=0)
        axis_med /= max(np.linalg.norm(axis_med), 1e-12)
        return ({
            "method": "procrustes_per_frame",
            "type": "prismatic",
            "n_frames_used": len(per_frame),
            "n_frames_in_theta_band": len(in_band),
            "n_frames_kept": len(kept),
            "votes": {"prismatic": pris_votes, "revolute": rev_votes},
            "axis_direction": axis_med.tolist(),
            "axis_point": None,
            "rms_cutoff": cutoff,
            "theta_band_deg": [theta_min_deg, theta_max_deg],
            "per_frame_summary": per_frame,
        }, None)

    # Revolute aggregation: sign-align directions, median direction + median axis_point_part
    dirs = np.stack([np.asarray(e["axis_dir"]) for e in kept])
    ref = dirs[0]
    aligned = np.where((dirs @ ref)[:, None] < 0, -dirs, dirs)
    axis_med = np.median(aligned, axis=0)
    axis_med /= max(np.linalg.norm(axis_med), 1e-12)
    pts_part = np.stack([np.asarray(e["axis_point_part"]) for e in kept])
    axis_point_med = np.median(pts_part, axis=0).tolist()
    # Project the median axis_point onto the median direction line through itself
    # (already on the line by construction) — keep as-is.
    return ({
        "method": "procrustes_per_frame",
        "type": "revolute",
        "n_frames_used": len(per_frame),
        "n_frames_in_theta_band": len(in_band),
        "n_frames_kept": len(kept),
        "votes": {"prismatic": pris_votes, "revolute": rev_votes},
        "axis_direction": axis_med.tolist(),
        "axis_point": axis_point_med,
        "rms_cutoff": cutoff,
        "theta_band_deg": [theta_min_deg, theta_max_deg],
        "per_frame_summary": per_frame,
    }, None)


# ---------------------------------------------------------------------------
# Refinement (Levenberg-Marquardt on a temporal window around ref_frame)
# ---------------------------------------------------------------------------


def rodrigues(axis_unit: np.ndarray, theta: float) -> np.ndarray:
    """R = I + sin(theta)[a]_x + (1 - cos(theta))[a]_x^2, axis_unit must be unit."""
    ax, ay, az = axis_unit
    K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=np.float64)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def select_window_frames(flows: list, ref_frame: int, window_size: int, side: str,
                         theta_map: dict = None, theta_min_deg: float = 0.0):
    """Return up to `window_size` (frame_idx, F) entries near ref_frame in time.

    side='both'  → use |frame_idx - ref_frame|
    side='prev'  → only frame_idx < ref_frame
    side='next'  → only frame_idx > ref_frame

    If theta_map (frame_idx -> theta_deg) is given AND theta_min_deg > 0, walk
    outward from ref_frame and keep only frames with |theta| >= theta_min_deg.
    This prevents the LM refinement from fitting MoGe noise on near-stationary
    frames adjacent to ref.
    """
    if side == "prev":
        cand = [(fi, F) for fi, F in flows if fi < ref_frame]
    elif side == "next":
        cand = [(fi, F) for fi, F in flows if fi > ref_frame]
    else:
        cand = list(flows)
    cand.sort(key=lambda kv: abs(kv[0] - ref_frame))

    if theta_map is not None and theta_min_deg > 0:
        filtered = []
        for fi, F in cand:
            th = theta_map.get(int(fi))
            if th is not None and abs(th) >= theta_min_deg:
                filtered.append((fi, F))
                if len(filtered) >= window_size:
                    break
        return filtered
    return cand[:window_size]


def _subsample(X_ref: np.ndarray, flows_window: list, n_sub: int,
               rng: np.random.Generator):
    N = X_ref.shape[0]
    if N > n_sub:
        idx = rng.choice(N, size=n_sub, replace=False)
    else:
        idx = np.arange(N)
    X_sub = X_ref[idx].astype(np.float64)
    targets = np.stack(
        [(X_ref[idx] + F[idx]).astype(np.float64) for _, F in flows_window],
        axis=0,
    )  # (K, N_sub, 3)
    return X_sub, targets, idx


def refine_revolute(X_ref: np.ndarray, flows_window: list,
                    axis_dir_init: np.ndarray, axis_point_init: np.ndarray,
                    n_points_sub: int, max_iters: int, rng: np.random.Generator):
    """Joint LM optimization of (axis_dir, axis_point, {theta_t}) under
    x_t = R(a, theta_t) (x_ref - c) + c. Initial thetas from per-frame Procrustes
    against the initial axis direction (sign-aligned)."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None, "scipy.optimize not available"

    K = len(flows_window)
    if K < 2:
        return None, f"refinement window has only {K} frame(s)"

    X_sub, X_t, _ = _subsample(X_ref, flows_window, n_points_sub, rng)

    # Per-frame Procrustes for initial thetas (sign-aligned to axis_dir_init)
    a0 = axis_dir_init / max(np.linalg.norm(axis_dir_init), 1e-12)
    thetas_init = np.zeros(K, dtype=np.float64)
    for k, (_frame_idx, F) in enumerate(flows_window):
        out = fit_procrustes_one_frame(X_ref.astype(np.float64),
                                        (X_ref + F).astype(np.float64))
        if out["axis_dir"] is None:
            thetas_init[k] = 0.0
        else:
            sign = 1.0 if (out["axis_dir"] @ a0) >= 0 else -1.0
            thetas_init[k] = sign * out["theta"]

    params0 = np.concatenate([a0, axis_point_init.astype(np.float64), thetas_init])

    def residuals(params):
        a = params[:3]
        a_n = max(np.linalg.norm(a), 1e-12)
        a_unit = a / a_n
        c = params[3:6]
        thetas = params[6:6 + K]
        res_per_frame = []
        Xc = X_sub - c
        for k in range(K):
            R = rodrigues(a_unit, thetas[k])
            pred = Xc @ R.T + c
            res_per_frame.append((pred - X_t[k]).ravel())
        return np.concatenate(res_per_frame)

    result = least_squares(residuals, params0, method="lm", max_nfev=max_iters)

    a_ref = result.x[:3]
    a_ref /= max(np.linalg.norm(a_ref), 1e-12)
    c_ref = result.x[3:6]
    thetas_ref = result.x[6:6 + K]

    # Anchor c on the closest point on the axis to the part centroid for display
    centroid_part = X_ref.mean(axis=0).astype(np.float64)
    lam = float((centroid_part - c_ref) @ a_ref)
    c_anchor = c_ref + lam * a_ref
    dist_to_part = float(np.linalg.norm(centroid_part - c_anchor))

    rms_before = float(np.sqrt((residuals(params0) ** 2).mean()))
    rms_after = float(np.sqrt((result.fun ** 2).mean()))

    return {
        "axis_direction": a_ref.tolist(),
        "axis_point": c_anchor.tolist(),
        "axis_point_raw": c_ref.tolist(),
        "thetas_deg": np.rad2deg(thetas_ref).tolist(),
        "frames_used": [int(fi) for fi, _ in flows_window],
        "n_points_used": int(X_sub.shape[0]),
        "n_frames_used": int(K),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "dist_axis_to_part_centroid": dist_to_part,
        "lm_success": bool(result.success),
        "lm_nfev": int(result.nfev),
        "lm_message": str(result.message),
    }, None


def refine_prismatic(X_ref: np.ndarray, flows_window: list,
                     axis_dir_init: np.ndarray,
                     n_points_sub: int, max_iters: int, rng: np.random.Generator):
    """Joint LM optimization of (axis_dir, {alpha_t}) under x_t = x_ref + alpha_t * a."""
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return None, "scipy.optimize not available"

    K = len(flows_window)
    if K < 1:
        return None, "empty refinement window"

    X_sub, X_t, idx = _subsample(X_ref, flows_window, n_points_sub, rng)

    # Initial per-frame alpha from projecting mean flow onto a0
    a0 = axis_dir_init / max(np.linalg.norm(axis_dir_init), 1e-12)
    alphas_init = np.array([
        float(F[idx].mean(axis=0) @ a0) for _, F in flows_window
    ], dtype=np.float64)

    params0 = np.concatenate([a0, alphas_init])

    def residuals(params):
        a = params[:3]
        a_n = max(np.linalg.norm(a), 1e-12)
        a_unit = a / a_n
        alphas = params[3:3 + K]
        res_per_frame = []
        for k in range(K):
            pred = X_sub + alphas[k] * a_unit
            res_per_frame.append((pred - X_t[k]).ravel())
        return np.concatenate(res_per_frame)

    result = least_squares(residuals, params0, method="lm", max_nfev=max_iters)
    a_ref = result.x[:3]
    a_ref /= max(np.linalg.norm(a_ref), 1e-12)
    alphas_ref = result.x[3:3 + K]

    rms_before = float(np.sqrt((residuals(params0) ** 2).mean()))
    rms_after = float(np.sqrt((result.fun ** 2).mean()))

    return {
        "axis_direction": a_ref.tolist(),
        "axis_point": None,
        "alphas": alphas_ref.tolist(),
        "frames_used": [int(fi) for fi, _ in flows_window],
        "n_points_used": int(X_sub.shape[0]),
        "n_frames_used": int(K),
        "rms_before": rms_before,
        "rms_after": rms_after,
        "lm_success": bool(result.success),
        "lm_nfev": int(result.nfev),
        "lm_message": str(result.message),
    }, None


def load_label_data(label_dir: Path):
    pts_path = label_dir / "pts3d_ref.npy"
    flow_dir = label_dir / "scene_flow"
    if not pts_path.is_file():
        return None, f"missing {pts_path}"
    if not flow_dir.is_dir():
        return None, f"missing {flow_dir}"
    pts = np.load(pts_path).astype(np.float32)
    flow_files = sorted(flow_dir.glob("*.npy"))
    if not flow_files:
        return None, f"no .npy in {flow_dir}"
    flows = []
    for fp in flow_files:
        try:
            f = np.load(fp).astype(np.float32)
        except Exception as e:
            print(f"  warning: failed to load {fp}: {e}")
            continue
        if f.shape != pts.shape:
            print(f"  warning: {fp.name} shape {f.shape} != pts {pts.shape}; skipped")
            continue
        try:
            frame_idx = int(fp.stem)
        except ValueError:
            print(f"  warning: non-integer flow filename {fp.name}; skipped")
            continue
        flows.append((frame_idx, f))
    flows.sort(key=lambda kv: kv[0])
    return (pts, flows), None


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    any4d_root = scene_dir / "any4d"
    if not any4d_root.is_dir():
        sys.exit(f"error: {any4d_root} does not exist (run stage 40 first)")

    # Discover labels
    all_label_dirs = [d for d in any4d_root.iterdir()
                      if d.is_dir() and d.name != "moge"
                      and (d / "pts3d_ref.npy").is_file()]
    if not all_label_dirs:
        sys.exit(f"error: no labels with pts3d_ref.npy under {any4d_root}")
    labels = sorted([d.name for d in all_label_dirs])
    if args.labels:
        unknown = [l for l in args.labels if l not in labels]
        if unknown:
            sys.exit(f"error: --labels not present: {unknown}")
        labels = [l for l in labels if l in args.labels]

    if args.method == "procrustes":
        strat = f"procrustes (theta in [{args.theta_min}, {args.theta_max}] deg, " \
                f"rms-quantile={args.frame_rms_quantile})"
        if args.refine:
            strat += f"  +  LM refine (window={args.refine_window} " \
                     f"side={args.refine_side} pts={args.refine_points})"
    else:
        strat = "linear-per-frame" if args.per_frame else "linear-pooled"

    # ref_frame is needed for refinement window selection
    ref_frame = None
    cfg_path = any4d_root / "config.json"
    if cfg_path.is_file():
        with open(cfg_path) as f:
            ref_frame = json.load(f).get("ref_frame")
    if args.refine and args.method == "procrustes" and ref_frame is None:
        print(f"warning: {cfg_path} missing ref_frame; refinement will be skipped")

    print(f"scene:    {scene_dir.name}")
    print(f"strategy: {strat}")
    print(f"ref_frame: {ref_frame}")
    print(f"labels:   {labels}\n")

    rng = np.random.default_rng(args.seed)
    summary = {}

    for label in labels:
        label_dir = any4d_root / label
        loaded, err = load_label_data(label_dir)
        if err:
            print(f"[{label}] SKIP: {err}")
            continue
        pts, flows = loaded
        n_pts, n_frames = pts.shape[0], len(flows)
        print(f"[{label}] n_points={n_pts}, n_frames={n_frames}")

        if args.method == "procrustes":
            result, ferr = fit_one_label_procrustes(
                pts, flows,
                theta_min_deg=args.theta_min,
                theta_max_deg=args.theta_max,
                rms_quantile=args.frame_rms_quantile,
                min_flow=args.min_flow,
            )
        elif args.per_frame:
            result, ferr = fit_one_label_per_frame(pts, flows, args.min_flow)
        else:
            result, ferr = fit_one_label_pooled(
                pts, flows, args.min_flow, args.max_samples, rng
            )
        if ferr:
            print(f"[{label}] FAIL: {ferr}")
            continue

        # ---- Refinement step (revolute or prismatic) ----
        if (args.method == "procrustes" and args.refine and ref_frame is not None
                and result.get("axis_direction") is not None):
            theta_map = {int(e["frame"]): float(e["theta_deg"])
                         for e in result.get("per_frame_summary", [])}
            window = select_window_frames(
                flows, ref_frame, args.refine_window, args.refine_side,
                theta_map=theta_map, theta_min_deg=args.refine_theta_min,
            )
            if len(window) >= 2:
                a_init = np.asarray(result["axis_direction"], dtype=np.float64)
                if result["type"] == "revolute":
                    c_init = np.asarray(result["axis_point"], dtype=np.float64)
                    ref_info, rerr = refine_revolute(
                        pts.astype(np.float64), window,
                        a_init, c_init,
                        args.refine_points, max_iters=200, rng=rng,
                    )
                else:
                    ref_info, rerr = refine_prismatic(
                        pts.astype(np.float64), window,
                        a_init, args.refine_points, max_iters=200, rng=rng,
                    )
                if rerr:
                    print(f"  refine FAIL: {rerr}")
                    result["refine"] = {"error": rerr}
                else:
                    result["coarse_axis_direction"] = result["axis_direction"]
                    result["coarse_axis_point"] = result.get("axis_point")
                    result["axis_direction"] = ref_info["axis_direction"]
                    result["axis_point"] = ref_info["axis_point"]
                    result["refine"] = ref_info
                    print(f"  refine: window={len(window)} frames "
                          f"({ref_info['frames_used'][:6]}{'...' if len(window) > 6 else ''})  "
                          f"rms {ref_info['rms_before']:.5g} -> {ref_info['rms_after']:.5g}  "
                          f"lm_nfev={ref_info['lm_nfev']}")
            else:
                print(f"  refine SKIP: window too small ({len(window)} frame(s))")

        result["label"] = label
        result["n_points"] = int(n_pts)
        result["n_frames_total"] = int(n_frames)

        ad = result.get("axis_direction")
        ap = result.get("axis_point")
        fmt = lambda v: ("[{:+.4f}, {:+.4f}, {:+.4f}]".format(*v)) if v else "None"
        print(f"  type: {result['type']}")
        if "residual_prismatic" in result:
            print(f"  E_pris={result['residual_prismatic']:.6g}  "
                  f"E_rev={result['residual_revolute']:.6g}")
        elif result.get("method") == "procrustes_per_frame":
            print(f"  frames in theta band: {result.get('n_frames_in_theta_band')}  "
                  f"kept after rms-quantile: {result.get('n_frames_kept')}  "
                  f"votes={result.get('votes')}")
            if "rms_cutoff" in result:
                print(f"  rms_cutoff={result['rms_cutoff']:.6g}")
        elif "votes" in result:
            print(f"  per-frame votes: {result['votes']}")
        print(f"  axis_dir   = {fmt(ad)}")
        print(f"  axis_point = {fmt(ap)}")
        if "note" in result:
            print(f"  note: {result['note']}")
        print()

        # Per-label sidecar
        if not args.dry_run:
            sidecar = {k: v for k, v in result.items() if k != "per_frame_summary"}
            with open(label_dir / "joint.json", "w") as fp:
                json.dump(sidecar, fp, indent=2)

        summary[label] = result

    if args.dry_run:
        print("--dry-run: skipping write")
    else:
        out = any4d_root / "joints.json"
        # Trim per_frame_summary out of the scene summary to keep it compact;
        # the full per-frame log stays in <label>/joint.json only if requested.
        compact = {}
        for k, v in summary.items():
            compact[k] = {kk: vv for kk, vv in v.items() if kk != "per_frame_summary"}
        with open(out, "w") as fp:
            json.dump(compact, fp, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
