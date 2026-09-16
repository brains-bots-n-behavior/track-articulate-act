"""Robust revolute/prismatic fitting from sparse, video-derived SE(3) poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (canonicalize_axis, orthogonal_basis, rotation_distance_np,
                       so3_exp_np, so3_log_np)


@dataclass
class JointFit:
    joint_type: str
    axis: np.ndarray
    pivot: np.ndarray | None
    states: np.ndarray
    residual: float
    inliers: np.ndarray
    frame_errors: np.ndarray

    def to_dict(self, frame_ids: list[int] | None = None) -> dict:
        states = self.states
        payload = {
            "type": self.joint_type,
            "axis_direction": self.axis.tolist(),
            "axis_point": None if self.pivot is None else self.pivot.tolist(),
            "pose_fit_residual": float(self.residual),
            "inliers": self.inliers.astype(bool).tolist(),
            "frame_errors": self.frame_errors.tolist(),
        }
        if frame_ids is not None:
            payload["joint_states"] = [
                {"frame": int(frame_id), "q": float(q)}
                for frame_id, q in zip(frame_ids, states)
            ]
        else:
            payload["states"] = states.tolist()
        return payload


def _normalize_transforms(transforms: np.ndarray, reference_index: int) -> np.ndarray:
    base_inv = np.linalg.inv(transforms[reference_index])
    return np.stack([T @ base_inv for T in transforms], axis=0)


def _axis_from_vectors(vectors: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    useful = vectors[np.linalg.norm(vectors, axis=1) > 1e-7]
    if len(useful) == 0:
        axis = np.asarray(fallback, dtype=np.float64)
    else:
        _, _, vt = np.linalg.svd(useful, full_matrices=False)
        axis = vt[0]
    return axis / max(float(np.linalg.norm(axis)), 1e-12)


def _initial_prismatic(transforms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translations = transforms[:, :3, 3]
    axis = _axis_from_vectors(translations, np.array([1.0, 0.0, 0.0]))
    states = translations @ axis
    return canonicalize_axis(axis, states)


def _initial_revolute(transforms: np.ndarray, centroid: np.ndarray,
                      indices: np.ndarray | None = None):
    if indices is None:
        indices = np.arange(len(transforms))
    rotations = transforms[:, :3, :3]
    pairwise = []
    selected = list(map(int, indices))
    for pos, i in enumerate(selected):
        for j in selected[pos + 1:]:
            rv = so3_log_np(rotations[i].T @ rotations[j])
            if np.linalg.norm(rv) > np.deg2rad(0.25):
                pairwise.append(rv)
    direct = np.stack([so3_log_np(rotations[i]) for i in selected], axis=0)
    vectors = np.stack(pairwise, axis=0) if pairwise else direct
    axis = _axis_from_vectors(vectors, np.array([0.0, 0.0, 1.0]))
    states = np.array([axis @ so3_log_np(R) for R in rotations], dtype=np.float64)
    axis, states = canonicalize_axis(axis, states)

    basis = orthogonal_basis(axis)
    A, b = [], []
    for i in selected:
        R, d = rotations[i], transforms[i, :3, 3]
        A.append((np.eye(3) - R) @ basis)
        b.append(d - (np.eye(3) - R) @ centroid)
    A = np.concatenate(A, axis=0)
    b = np.concatenate(b, axis=0)
    coeff, *_ = np.linalg.lstsq(A, b, rcond=None)
    pivot = centroid + basis @ coeff
    return axis, pivot, states


def _project_pivot(raw: np.ndarray, axis: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    delta = raw - centroid
    return centroid + delta - axis * (axis @ delta)


def _predict_transform(joint_type: str, axis: np.ndarray, pivot: np.ndarray | None,
                       state: float) -> tuple[np.ndarray, np.ndarray]:
    if joint_type == "prismatic":
        return np.eye(3), state * axis
    R = so3_exp_np(state * axis)
    return R, (np.eye(3) - R) @ pivot


def _frame_errors(joint_type: str, transforms: np.ndarray, axis: np.ndarray,
                  pivot: np.ndarray | None, states: np.ndarray,
                  diameter: float) -> np.ndarray:
    errors = []
    scale = max(float(diameter), 1e-6)
    for T, q in zip(transforms, states):
        pred_R, pred_t = _predict_transform(joint_type, axis, pivot, float(q))
        rot = rotation_distance_np(pred_R, T[:3, :3])
        trans = np.linalg.norm(pred_t - T[:3, 3]) / scale
        errors.append(np.sqrt(rot * rot + trans * trans))
    return np.asarray(errors, dtype=np.float64)


def _refine(joint_type: str, transforms: np.ndarray, axis: np.ndarray,
            pivot: np.ndarray | None, states: np.ndarray, centroid: np.ndarray,
            diameter: float, confidences: np.ndarray, inliers: np.ndarray) -> JointFit:
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for robust joint fitting") from exc

    n = len(transforms)
    scale = max(float(diameter), 1e-6)
    if joint_type == "prismatic":
        initial = np.concatenate([axis, states])
    else:
        initial = np.concatenate([axis, pivot, states])

    active = np.flatnonzero(inliers)
    if active.size < 2:
        active = np.arange(n)

    def unpack(params):
        a = params[:3]
        a /= max(float(np.linalg.norm(a)), 1e-12)
        if joint_type == "prismatic":
            return a, None, params[3:3 + n]
        c = _project_pivot(params[3:6], a, centroid)
        return a, c, params[6:6 + n]

    def residuals(params):
        a, c, q = unpack(params)
        chunks = []
        for i in active:
            pred_R, pred_t = _predict_transform(joint_type, a, c, q[i])
            rot = so3_log_np(pred_R.T @ transforms[i, :3, :3])
            trans = (pred_t - transforms[i, :3, 3]) / scale
            chunks.append(np.sqrt(max(float(confidences[i]), 1e-3)) *
                          np.concatenate([rot, trans]))
        return np.concatenate(chunks)

    solved = least_squares(residuals, initial, loss="huber", f_scale=0.03,
                           max_nfev=500)
    axis, pivot, states = unpack(solved.x)
    axis, states = canonicalize_axis(axis, states)
    errors = _frame_errors(joint_type, transforms, axis, pivot, states, diameter)
    median = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median)))
    threshold = max(median + 2.5 * 1.4826 * mad, 0.02)
    final_inliers = errors <= threshold
    return JointFit(joint_type, axis, pivot, states, median, final_inliers, errors)


def _fit_one(joint_type: str, transforms: np.ndarray, centroid: np.ndarray,
             diameter: float, confidences: np.ndarray, ransac_iterations: int,
             rng: np.random.Generator) -> JointFit:
    n = len(transforms)
    if joint_type == "prismatic":
        axis, states = _initial_prismatic(transforms)
        pivot = None
    else:
        axis, pivot, states = _initial_revolute(transforms, centroid)
    best = (np.inf, np.ones(n, dtype=bool), axis, pivot, states)
    subset_size = min(n, 4 if joint_type == "revolute" else 3)
    candidates = [np.arange(n)]
    for _ in range(max(0, int(ransac_iterations))):
        candidates.append(np.sort(rng.choice(n, size=subset_size, replace=False)))

    for subset in candidates:
        try:
            if joint_type == "prismatic":
                cand_axis, _ = _initial_prismatic(transforms[subset])
                cand_states = transforms[:, :3, 3] @ cand_axis
                cand_pivot = None
            else:
                cand_axis, cand_pivot, cand_states = _initial_revolute(
                    transforms, centroid, indices=subset)
            errors = _frame_errors(joint_type, transforms, cand_axis, cand_pivot,
                                   cand_states, diameter)
            med = float(np.median(errors))
            mad = float(np.median(np.abs(errors - med)))
            threshold = max(med + 2.5 * 1.4826 * mad, 0.03)
            inliers = errors <= threshold
            score = med + 0.1 * float(np.mean(np.sort(errors)[:max(2, int(0.75 * n))]))
            if score < best[0]:
                best = (score, inliers, cand_axis, cand_pivot, cand_states)
        except (np.linalg.LinAlgError, ValueError):
            continue
    _, inliers, axis, pivot, states = best
    return _refine(joint_type, transforms, axis, pivot, states, centroid,
                   diameter, confidences, inliers)


def fit_joint_hypotheses(transforms: np.ndarray, centroid: np.ndarray,
                         diameter: float, reference_index: int = 0,
                         confidences: np.ndarray | None = None,
                         ransac_iterations: int = 64,
                         seed: int = 0) -> dict[str, JointFit]:
    """Fit and retain both P/R hypotheses; neither is classified at this stage."""
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or len(transforms) < 3:
        raise ValueError("transforms must have shape (N, 4, 4) with N >= 3")
    relative = _normalize_transforms(transforms, int(reference_index))
    if confidences is None:
        confidences = np.ones(len(relative), dtype=np.float64)
    confidences = np.asarray(confidences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        kind: _fit_one(kind, relative, np.asarray(centroid, dtype=np.float64),
                       diameter, confidences, ransac_iterations, rng)
        for kind in ("prismatic", "revolute")
    }
