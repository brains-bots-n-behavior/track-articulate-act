"""Small NumPy/Torch geometry helpers used by the video joint pipeline."""

from __future__ import annotations

import numpy as np


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64)
    return quat_wxyz_to_matrix(np.array([w, x, y, z]))


def skew_np(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp_np(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-10:
        K = skew_np(rotvec)
        return np.eye(3) + K + 0.5 * (K @ K)
    K = skew_np(rotvec / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log_np(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        vals, vecs = np.linalg.eig(R)
        axis = np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return theta * axis
    return theta * vee / (2.0 * np.sin(theta))


def rotation_distance_np(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.linalg.norm(so3_log_np(np.asarray(R_a).T @ np.asarray(R_b))))


def transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def orthogonal_basis(axis: np.ndarray) -> np.ndarray:
    """Return a 3x2 orthonormal basis for the plane normal to ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(axis, seed)
    b1 /= max(float(np.linalg.norm(b1)), 1e-12)
    b2 = np.cross(axis, b1)
    return np.stack([b1, b2], axis=1)


def canonicalize_axis(axis: np.ndarray, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the physically irrelevant (axis, q) sign ambiguity deterministically."""
    axis = np.asarray(axis, dtype=np.float64).copy()
    states = np.asarray(states, dtype=np.float64).copy()
    dominant = int(np.argmax(np.abs(axis)))
    if axis[dominant] < 0:
        axis *= -1.0
        states *= -1.0
    return axis, states


def torch_skew(v):
    import torch

    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], dim=-1),
        torch.stack([v[..., 2], z, -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1], v[..., 0], z], dim=-1),
    ], dim=-2)


def so3_exp_torch(rotvec):
    """Stable differentiable exponential map for (..., 3) rotation vectors."""
    import torch

    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    K = torch_skew(rotvec)
    a = torch.sinc(theta / torch.pi)[..., None]
    b = (0.5 * torch.sinc(theta / (2.0 * torch.pi)).square())[..., None]
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    eye = eye.expand(rotvec.shape[:-1] + (3, 3))
    return eye + a * K + b * (K @ K)


def apply_rigid_torch(vertices, rotation, translation, center=None):
    if center is None:
        return vertices @ rotation.transpose(-1, -2) + translation
    return (vertices - center) @ rotation.transpose(-1, -2) + center + translation
