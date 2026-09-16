"""HaWoR trajectory loading for the legacy Viser replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .overlay import OverlayScene


MANO_BONES = np.asarray([
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
], dtype=np.int32)

FINGERTIP_INDICES = (4, 8, 12, 16, 20)


@dataclass(frozen=True)
class HandSample:
    frame_id: int
    side: str
    vertices: np.ndarray
    joints: np.ndarray
    detected: bool
    coordinate_source: str


@dataclass
class HandTrajectory:
    samples: dict[str, list[HandSample]]
    faces: dict[str, np.ndarray]
    skipped_nonfinite: int = 0
    skipped_without_pose: int = 0

    @property
    def sides(self) -> list[str]:
        return [side for side in ("left", "right") if self.samples.get(side)]

    @property
    def frame_ids(self) -> list[int]:
        return sorted({sample.frame_id for values in self.samples.values()
                       for sample in values})

    @property
    def coordinate_sources(self) -> list[str]:
        return sorted({sample.coordinate_source for values in self.samples.values()
                       for sample in values})

    def nearest(self, side: str, frame_id: int) -> HandSample | None:
        values = self.samples.get(side, [])
        if not values:
            return None
        ids = np.asarray([sample.frame_id for sample in values], dtype=np.int64)
        position = int(np.searchsorted(ids, int(frame_id)))
        choices = [min(position, len(values) - 1)]
        if position > 0:
            choices.append(position - 1)
        index = min(choices, key=lambda item: abs(values[item].frame_id - frame_id))
        return values[index]

    def anchor(self) -> np.ndarray:
        first = min((sample for values in self.samples.values() for sample in values),
                    key=lambda sample: sample.frame_id)
        return first.joints[0].astype(np.float32).copy()

    def all_vertices(self, stride: int = 1) -> np.ndarray:
        values = [sample.vertices for side in self.sides
                  for sample in self.samples[side][::max(1, int(stride))]]
        return np.concatenate(values, axis=0)

    def segments(self, side: str, joint_indices: list[int],
                 max_frame_gap: int) -> tuple[np.ndarray, np.ndarray]:
        """Return trajectory line segments and per-endpoint detection flags."""
        values = self.samples.get(side, [])
        segments, detected = [], []
        for before, after in zip(values[:-1], values[1:]):
            if after.frame_id - before.frame_id > max_frame_gap:
                continue
            for joint_index in joint_indices:
                segments.append([before.joints[joint_index], after.joints[joint_index]])
                detected.append([before.detected, after.detected])
        if not segments:
            return (np.empty((0, 2, 3), dtype=np.float32),
                    np.empty((0, 2), dtype=bool))
        return np.asarray(segments, dtype=np.float32), np.asarray(detected, dtype=bool)


def _camera_to_world(scene: OverlayScene, frame_id: int,
                     points: np.ndarray) -> np.ndarray:
    rotation, translation = scene.camera_to_world(frame_id)
    return (np.asarray(points, dtype=np.float64) @ rotation.T
            + translation).astype(np.float32)


def load_hand_trajectory(scene_dir: Path, coordinate_source: str = "auto",
                         detected_only: bool = False,
                         start_frame: int | None = None,
                         end_frame: int | None = None,
                         hawor_name: str = "hawor") -> HandTrajectory:
    """Load finite Stage-12 hands into the shared DA3 world frame.

    ``auto`` prefers Stage-12 world arrays and otherwise converts its RDF
    camera-frame arrays with the matching DA3 camera pose. Files or individual
    hands containing NaNs are skipped rather than poisoning Viser bounds.
    """
    if coordinate_source not in {"auto", "world", "camera"}:
        raise ValueError(f"unsupported hand coordinate source {coordinate_source!r}")
    if (not hawor_name or Path(hawor_name).name != hawor_name
            or hawor_name in {".", ".."}):
        raise ValueError("hawor_name must be one directory name")
    scene_dir = Path(scene_dir).resolve()
    root = scene_dir / hawor_name
    per_frame = root / "per_frame"
    faces_path = root / "faces.npy"
    if not per_frame.is_dir():
        raise FileNotFoundError(
            f"missing {per_frame}; run Stage 12/13 or select another --hawor-name")
    if not faces_path.is_file():
        raise FileNotFoundError(
            f"missing {faces_path}; run Stage 12/13 or select another --hawor-name")
    right_faces = np.asarray(np.load(faces_path), dtype=np.int32)
    left_path = root / "faces_left.npy"
    left_faces = (np.asarray(np.load(left_path), dtype=np.int32)
                  if left_path.is_file() else right_faces[:, [0, 2, 1]].copy())

    samples: dict[str, dict[int, HandSample]] = {"left": {}, "right": {}}
    camera_scene = None
    skipped_nonfinite = skipped_without_pose = 0
    paths = sorted(per_frame.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no hand frames under {per_frame}")
    for path in paths:
        with np.load(path) as record:
            frame_id = int(record["frame_idx"]) if "frame_idx" in record else int(path.stem)
            if start_frame is not None and frame_id < start_frame:
                continue
            if end_frame is not None and frame_id > end_frame:
                continue
            vertices_camera = np.asarray(record["verts"], dtype=np.float32)
            joints_camera = np.asarray(record["joints"], dtype=np.float32)
            is_right = np.asarray(record["is_right"], dtype=bool)
            valid = np.asarray(record.get("valid", np.ones(len(is_right))), dtype=bool)
            vertices_world = (np.asarray(record["verts_world"], dtype=np.float32)
                              if "verts_world" in record else None)
            joints_world = (np.asarray(record["joints_world"], dtype=np.float32)
                            if "joints_world" in record else None)

            for index in range(len(is_right)):
                if detected_only and not bool(valid[index]):
                    continue
                use_saved_world = (
                    coordinate_source in {"auto", "world"}
                    and vertices_world is not None and joints_world is not None
                    and np.isfinite(vertices_world[index]).all()
                    and np.isfinite(joints_world[index]).all()
                )
                if use_saved_world:
                    vertices = vertices_world[index].copy()
                    joints = joints_world[index].copy()
                    source = "hawor verts_world"
                elif coordinate_source == "world":
                    skipped_nonfinite += 1
                    continue
                else:
                    if (not np.isfinite(vertices_camera[index]).all()
                            or not np.isfinite(joints_camera[index]).all()):
                        skipped_nonfinite += 1
                        continue
                    if camera_scene is None:
                        camera_scene = OverlayScene(scene_dir)
                    if not camera_scene.has_camera_pose(frame_id):
                        skipped_without_pose += 1
                        continue
                    vertices = _camera_to_world(
                        camera_scene, frame_id, vertices_camera[index])
                    joints = _camera_to_world(
                        camera_scene, frame_id, joints_camera[index])
                    source = "hawor camera + DA3 pose"
                side = "right" if bool(is_right[index]) else "left"
                sample = HandSample(
                    frame_id, side, vertices, joints,
                    bool(valid[index]), source)
                previous = samples[side].get(frame_id)
                if previous is None or (sample.detected and not previous.detected):
                    samples[side][frame_id] = sample

    ordered = {side: [by_frame[key] for key in sorted(by_frame)]
               for side, by_frame in samples.items()}
    trajectory = HandTrajectory(
        ordered, {"left": left_faces, "right": right_faces},
        skipped_nonfinite=skipped_nonfinite,
        skipped_without_pose=skipped_without_pose)
    if not trajectory.frame_ids:
        detail = (f"; skipped {skipped_nonfinite} non-finite hands and "
                  f"{skipped_without_pose} hands without DA3 camera poses")
        raise ValueError("no finite hand samples could be placed in the DA3 world frame"
                         + detail)
    return trajectory
