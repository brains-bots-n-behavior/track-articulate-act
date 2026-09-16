"""World-space point trajectories used only to initialize a joint model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import transform_matrix


@dataclass
class TrackPose:
    transform: np.ndarray
    normalized_rms: float
    confidence: float
    valid_points: int
    inlier_points: int


def _kabsch(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x_mean, y_mean = source.mean(axis=0), target.mean(axis=0)
    u, singular, vt = np.linalg.svd((source - x_mean).T @ (target - y_mean))
    if singular[0] <= 0 or singular[1] <= singular[0] * 1e-10:
        raise ValueError("3D tracks are coincident or collinear; rotation is undefined")
    correction = np.eye(3)
    correction[-1, -1] = 1.0 if np.linalg.det(vt.T @ u.T) >= 0 else -1.0
    rotation = vt.T @ correction @ u.T
    return transform_matrix(rotation, y_mean - rotation @ x_mean)


def rigid_pose_from_tracks(source: np.ndarray, target: np.ndarray) -> TrackPose:
    """Trimmed Kabsch registration; ignore invalid correspondences, retain 80%."""
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape:
        raise ValueError("source and target tracks must have the same (N, 3) shape")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    available = np.flatnonzero(finite)
    if len(available) < 3:
        raise ValueError(f"rigid pose needs at least 3 finite tracks; found {len(available)}")
    source_valid, target_valid = source[available], target[available]
    diameter = float(np.linalg.norm(np.ptp(source_valid, axis=0)))
    if diameter <= 1e-12:
        raise ValueError("3D reference tracks have zero spatial extent")
    active = np.arange(len(available))
    keep = max(3, int(np.ceil(0.8 * len(available))))
    for _ in range(5):
        transform = _kabsch(source_valid[active], target_valid[active])
        residual = np.linalg.norm(
            source_valid @ transform[:3, :3].T + transform[:3, 3] - target_valid, axis=1)
        updated = np.sort(np.argsort(residual, kind="stable")[:keep])
        if np.array_equal(active, updated):
            break
        active = updated
    transform = _kabsch(source_valid[active], target_valid[active])
    residual = np.linalg.norm(
        source_valid[active] @ transform[:3, :3].T + transform[:3, 3]
        - target_valid[active], axis=1)
    normalized_rms = float(np.sqrt(np.mean(residual ** 2)) / diameter)
    confidence = float(np.clip(
        len(available) / len(source) * np.exp(-normalized_rms / 0.01), 0.05, 1.0))
    return TrackPose(transform, normalized_rms, confidence, len(available), len(active))


class PointTrackSequence:
    """Stage-08 tracks: world-space reference points + reference-to-frame flow.

    Frame identifiers are image filename stems, as written by Stage 08. Camera
    record indices must not be guessed per file: adjacent names can both exist.
    """

    def __init__(self, scene_root: Path, label: str, reference_frame: int):
        self.root = Path(scene_root) / "trackcraft" / label
        config_path = self.root.parent / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"missing {config_path}; run Stage 08 to save tracks and their reference frame")
        config = json.loads(config_path.read_text())
        if "ref_frame" not in config:
            raise ValueError(f"{config_path} must specify ref_frame in image filename indices")
        self.track_reference_frame = int(config["ref_frame"])
        self.reference_frame = int(reference_frame)
        points_path = self.root / "pts3d_ref.npy"
        if not points_path.is_file():
            raise FileNotFoundError(f"missing {points_path}; run Stage 08 for moving part {label}")
        self.points = np.asarray(np.load(points_path, allow_pickle=False), dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or len(self.points) < 3:
            raise ValueError(f"{points_path} must have shape (N, 3), N >= 3; got {self.points.shape}")
        self.flow_paths = {}
        for path in sorted((self.root / "scene_flow").glob("*.npy")):
            if path.stem.isdigit():
                frame = int(path.stem)
                if frame in self.flow_paths:
                    raise ValueError(f"duplicate track frame {frame} under {path.parent}")
                self.flow_paths[frame] = path
        self.available_frames = sorted(set(self.flow_paths) | {self.track_reference_frame})
        # Re-anchor correspondences to the mesh's canonical articulation state.
        self.reference_points = self.positions(self.reference_frame)
        finite = self.reference_points[np.isfinite(self.reference_points).all(axis=1)]
        if len(finite) < 3:
            raise ValueError("mesh reference frame has fewer than 3 finite point tracks")
        _kabsch(finite, finite)  # Fail early for unobservable rigid rotations.
        self.centroid = finite.mean(axis=0)
        self.diameter = float(np.linalg.norm(np.ptp(finite, axis=0)))

    def positions(self, frame_id: int) -> np.ndarray:
        if frame_id == self.track_reference_frame:
            return self.points.copy()
        if frame_id not in self.flow_paths:
            raise FileNotFoundError(
                f"no 3D tracks for frame {frame_id} under {self.root / 'scene_flow'}; "
                "Stage 08 must cover the mesh reference frame and at least two target frames")
        path = self.flow_paths[frame_id]
        flow = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        if flow.shape != self.points.shape:
            raise ValueError(f"{path} has shape {flow.shape}; expected {self.points.shape}")
        return self.points + flow

    def estimate_poses(self, frames: list[int]) -> dict[int, TrackPose]:
        """Register training frames only, relative to the mesh reference frame."""
        poses = {}
        for frame in frames:
            try:
                pose = rigid_pose_from_tracks(self.reference_points, self.positions(frame))
            except ValueError as exc:
                raise ValueError(f"invalid 3D tracks for frame {frame}: {exc}") from exc
            if frame == self.reference_frame:
                pose.transform = np.eye(4)
                pose.normalized_rms = 0.0
                pose.confidence = 1.0
            poses[frame] = pose
        return poses
