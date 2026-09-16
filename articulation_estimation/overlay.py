"""Geometry and camera helpers for joint-motion overlays."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .data import (Camera, IMAGE_SUFFIXES, VideoScene, _infer_index_offset,
                   _numeric_files)
from .geometry import quat_xyzw_to_matrix, so3_exp_np


@dataclass
class OverlayMesh:
    """One triangle mesh in Stage-09 world coordinates at joint state zero."""

    label: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    source: str
    exact: bool


class OverlayScene:
    """RGB-frame and DA3-camera view without requiring masks or source meshes."""

    def __init__(self, scene_dir: Path):
        self.root = Path(scene_dir).resolve()
        self.frames = _numeric_files(self.root / "frames", IMAGE_SUFFIXES)
        if not self.frames:
            raise FileNotFoundError(f"no numeric RGB frames under {self.root / 'frames'}")

        intrinsics_path = self.root / "da3" / "intrinsics.npz"
        if not intrinsics_path.is_file():
            raise FileNotFoundError(f"missing {intrinsics_path}; run Stage 00 first")
        intrinsics = np.load(intrinsics_path)
        intr_ids = [int(value) for value in intrinsics["frame_indices"].tolist()]
        self.intrinsics = np.asarray(intrinsics["intrinsics"], dtype=np.float64)
        self._intr_pos = {frame_id: index for index, frame_id in enumerate(intr_ids)}
        self.index_offset = _infer_index_offset(set(self.frames), set(intr_ids))

        cameras_path = self.root / "da3" / "cameras.npz"
        self._cameras = np.load(cameras_path) if cameras_path.is_file() else None
        if self._cameras is None:
            self._cam_pos = {}
        else:
            camera_ids = [int(value) for value in
                          self._cameras["frame_indices"].tolist()]
            self._cam_pos = {frame_id: index
                             for index, frame_id in enumerate(camera_ids)}

    def record_id(self, frame_id: int) -> int:
        record_id = int(frame_id) - self.index_offset
        if record_id not in self._intr_pos:
            raise KeyError(f"frame {frame_id} has no matching DA3 camera record")
        return record_id

    def camera(self, frame_id: int) -> Camera:
        record_id = self.record_id(frame_id)
        K = self.intrinsics[self._intr_pos[record_id]].copy()
        if self._cameras is None or record_id not in self._cam_pos:
            return Camera(K, np.eye(3), np.zeros(3))
        index = self._cam_pos[record_id]
        R_camera_to_world = quat_xyzw_to_matrix(
            self._cameras["cam_quats_xyzw"][index])
        t_camera_to_world = np.asarray(
            self._cameras["cam_trans"][index], dtype=np.float64)
        R_world_to_camera = R_camera_to_world.T
        return Camera(K, R_world_to_camera,
                      -R_world_to_camera @ t_camera_to_world)

    def has_camera_pose(self, frame_id: int) -> bool:
        """Whether Stage 00 saved an extrinsic pose for this RGB frame."""
        try:
            record_id = self.record_id(frame_id)
        except KeyError:
            return False
        return self._cameras is not None and record_id in self._cam_pos

    def camera_to_world(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return column-vector camera-to-world rotation and translation."""
        if not self.has_camera_pose(frame_id):
            raise KeyError(f"frame {frame_id} has no matching DA3 camera pose")
        camera = self.camera(frame_id)
        rotation = camera.R_world_to_camera.T
        translation = -rotation @ camera.t_world_to_camera
        return rotation, translation

    def usable_frames(self, start: int | None = None,
                      end: int | None = None, step: int = 1) -> list[int]:
        if step <= 0:
            raise ValueError("frame step must be positive")
        frame_ids = [frame_id for frame_id in sorted(self.frames)
                     if frame_id - self.index_offset in self._intr_pos]
        if start is not None:
            frame_ids = [frame_id for frame_id in frame_ids if frame_id >= start]
        if end is not None:
            frame_ids = [frame_id for frame_id in frame_ids if frame_id <= end]
        return frame_ids[::step]


def load_joint_records(scene_dir: Path,
                       labels: list[str] | None = None) -> dict[str, dict]:
    """Load public Stage-09 results, preferring full per-label sidecars."""
    scene_dir = Path(scene_dir).resolve()
    joints_root = scene_dir / "joints"
    summary_path = joints_root / "joints.json"
    records: dict[str, dict] = {}
    if summary_path.is_file():
        value = json.loads(summary_path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"{summary_path} must contain an object keyed by label")
        records = value
    elif joints_root.is_dir():
        for sidecar in sorted(joints_root.glob("*/joint.json")):
            records[sidecar.parent.name] = json.loads(sidecar.read_text())
    if not records:
        raise FileNotFoundError(
            f"no Stage-09 joints found under {joints_root}; run Stage 09 first")

    requested = set(labels or records)
    missing = sorted(requested - set(records))
    if missing:
        raise ValueError(f"joint labels not found: {missing}; available: {sorted(records)}")
    selected = {}
    for label in records:
        if label not in requested:
            continue
        sidecar = joints_root / label / "joint.json"
        record = json.loads(sidecar.read_text()) if sidecar.is_file() else records[label]
        _validate_joint_record(label, record)
        selected[label] = record
    return selected


def _validate_joint_record(label: str, record: dict):
    joint_type = record.get("type")
    if joint_type not in {"revolute", "prismatic"}:
        raise ValueError(f"joint {label!r} has unsupported type {joint_type!r}")
    axis = np.asarray(record.get("axis_direction"), dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError(f"joint {label!r} has an invalid axis_direction")
    if joint_type == "revolute":
        pivot = np.asarray(record.get("axis_point"), dtype=np.float64)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError(f"revolute joint {label!r} has an invalid axis_point")
    states = record.get("joint_states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"joint {label!r} has no dense joint_states")
    for item in states:
        if "frame" not in item or "q" not in item:
            raise ValueError(f"joint {label!r} has a malformed joint state")


def _load_triangle_mesh(path: Path, label: str, source: str,
                        exact: bool) -> OverlayMesh:
    import trimesh

    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} contains no triangle mesh")
    vertices = np.asarray(loaded.vertices, dtype=np.float32).copy()
    faces = np.asarray(loaded.faces, dtype=np.int32).copy()
    normals = np.asarray(loaded.vertex_normals, dtype=np.float32).copy()
    if normals.shape != vertices.shape or not np.isfinite(normals).all():
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
    return OverlayMesh(label, vertices, faces, normals, source, exact)


def _artifact_path(scene_dir: Path, value: str) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (scene_dir / path).resolve()
    try:
        path.relative_to(scene_dir)
    except ValueError as exc:
        raise ValueError(f"mesh artifact is outside the scene folder: {path}") from exc
    return path


def _apply_static_refinement(vertices: np.ndarray, center: np.ndarray,
                             diagnostics: dict | None) -> np.ndarray:
    if not diagnostics:
        return vertices.copy()
    scale = np.asarray(diagnostics.get("anisotropic_scale", [1.0, 1.0, 1.0]),
                       dtype=np.float64)
    rotvec = np.asarray(diagnostics.get("rotation_vector", [0.0, 0.0, 0.0]),
                        dtype=np.float64)
    translation = np.asarray(diagnostics.get("translation", [0.0, 0.0, 0.0]),
                             dtype=np.float64)
    if scale.shape != (3,) or rotvec.shape != (3,) or translation.shape != (3,):
        raise ValueError("static_pose_refinement has malformed transform fields")
    rotation = so3_exp_np(rotvec)
    scaled = (np.asarray(vertices, dtype=np.float64) - center) * scale + center
    return ((scaled - center) @ rotation.T + center + translation).astype(np.float32)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.vertex_normals, dtype=np.float32).copy()


def _approximate_pair(video_scene: VideoScene, label: str,
                      record: dict) -> tuple[OverlayMesh, OverlayMesh]:
    """Rebuild old Stage-09 outputs from their saved pose/scale diagnostics.

    Old JSON does not contain the final common rotation. Centroid and bounding
    diagonal recover its translation and scale but not that rotation, so these
    assets are deliberately marked approximate.
    """
    static_label = record["static_label"]
    moving_part = video_scene.parts[label]
    static_part = video_scene.parts[static_label]
    center = np.concatenate(
        [moving_part.vertices, static_part.vertices], axis=0).mean(axis=0)
    diagnostics = record.get("static_pose_refinement")
    moving_vertices = _apply_static_refinement(
        moving_part.vertices, center, diagnostics)
    static_vertices = _apply_static_refinement(
        static_part.vertices, center, diagnostics)

    target_center = record.get("part_centroid_world")
    target_diagonal = record.get("part_bbox_diagonal")
    current_center = moving_vertices.mean(axis=0)
    current_diagonal = float(np.linalg.norm(
        moving_vertices.max(axis=0) - moving_vertices.min(axis=0)))
    if target_center is not None and target_diagonal is not None and current_diagonal > 1e-8:
        uniform_scale = float(target_diagonal) / current_diagonal
        target_center = np.asarray(target_center, dtype=np.float32)
        moving_vertices = ((moving_vertices - current_center) * uniform_scale
                           + target_center).astype(np.float32)
        static_vertices = ((static_vertices - current_center) * uniform_scale
                           + target_center).astype(np.float32)

    source = "approximate reconstruction from SegviGen + Stage-09 JSON"
    moving = OverlayMesh(
        label, moving_vertices, moving_part.faces.astype(np.int32).copy(),
        _vertex_normals(moving_vertices, moving_part.faces), source, False)
    static = OverlayMesh(
        static_label, static_vertices, static_part.faces.astype(np.int32).copy(),
        _vertex_normals(static_vertices, static_part.faces), source, False)
    return moving, static


def load_overlay_meshes(scene_dir: Path, records: dict[str, dict],
                        candidate: int = 0) -> tuple[dict[str, OverlayMesh], OverlayMesh]:
    """Load exact Stage-09 q=0 meshes, falling back for pre-artifact results."""
    scene_dir = Path(scene_dir).resolve()
    static_labels = {record.get("static_label") for record in records.values()}
    if None in static_labels or len(static_labels) != 1:
        raise ValueError("selected joints must share one static_label")
    static_label = next(iter(static_labels))

    moving: dict[str, OverlayMesh] = {}
    static_candidates: list[OverlayMesh] = []
    missing = []
    for label, record in records.items():
        descriptor = record.get("canonical_meshes") or {}
        moving_value = descriptor.get("moving")
        static_value = descriptor.get("static")
        default_moving = scene_dir / "joints" / label / "moving_mesh.glb"
        default_static = scene_dir / "joints" / label / "static_mesh.glb"
        moving_path = (_artifact_path(scene_dir, moving_value)
                       if moving_value else default_moving)
        static_path = (_artifact_path(scene_dir, static_value)
                       if static_value else default_static)
        if moving_path.is_file() and static_path.is_file():
            moving[label] = _load_triangle_mesh(
                moving_path, label, str(moving_path.relative_to(scene_dir)), True)
            static_candidates.append(_load_triangle_mesh(
                static_path, static_label, str(static_path.relative_to(scene_dir)), True))
        else:
            missing.append(label)

    if missing:
        initializer_kinds = {
            records[label].get("inputs", {}).get("pose_initializer_kind", "sam3d")
            for label in missing
        }
        if len(initializer_kinds) != 1:
            raise ValueError("fallback joints use inconsistent pose initializers")
        use_any6d = next(iter(initializer_kinds)) == "any6d"
        video_scene = VideoScene(
            scene_dir, static_label, missing, candidate=candidate,
            use_any6d_pose=use_any6d)
        for label in missing:
            approximate_moving, approximate_static = _approximate_pair(
                video_scene, label, records[label])
            moving[label] = approximate_moving
            static_candidates.append(approximate_static)

    if not static_candidates:
        raise FileNotFoundError("no static mesh could be loaded")
    # Each independently estimated moving part stores a copy of the same static
    # mesh. Render it once to avoid duplicate coplanar triangles and z-fighting.
    static_mesh = static_candidates[0]
    return moving, static_mesh


def state_at(record: dict, frame_id: int) -> float:
    states = sorted((int(item["frame"]), float(item["q"]))
                    for item in record["joint_states"])
    frames = np.asarray([item[0] for item in states], dtype=np.float64)
    values = np.asarray([item[1] for item in states], dtype=np.float64)
    value = float(np.interp(float(frame_id), frames, values))
    base_state = float((record.get("canonical_meshes") or {}).get("joint_state", 0.0))
    return value - base_state


def apply_joint_motion(mesh: OverlayMesh, record: dict,
                       state: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply Stage-09's row-vector articulation convention."""
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if record["type"] == "prismatic":
        return ((mesh.vertices + float(state) * axis).astype(np.float32),
                mesh.normals.copy())
    pivot = np.asarray(record["axis_point"], dtype=np.float64)
    rotation = so3_exp_np(float(state) * axis)
    vertices = ((mesh.vertices - pivot) @ rotation.T + pivot).astype(np.float32)
    normals = (mesh.normals @ rotation.T).astype(np.float32)
    return vertices, normals


def joint_axis_segment(mesh: OverlayMesh, record: dict,
                       length_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an object-scaled world-space joint segment and its display center."""
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    centroid = mesh.vertices.mean(axis=0).astype(np.float64)
    if record["type"] == "revolute":
        pivot = np.asarray(record["axis_point"], dtype=np.float64)
        center = pivot + axis * float((centroid - pivot) @ axis)
    else:
        center = centroid
    diagonal = float(np.linalg.norm(
        mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
    half_length = max(float(length_scale) * diagonal, 1e-3)
    return center - half_length * axis, center + half_length * axis, center


def project_world(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    camera_points = points @ camera.R_world_to_camera.T + camera.t_world_to_camera
    depth = camera_points[:, 2]
    safe_depth = np.maximum(depth, 1e-8)
    pixels = np.column_stack([
        camera.K[0, 0] * camera_points[:, 0] / safe_depth + camera.K[0, 2],
        camera.K[1, 1] * camera_points[:, 1] / safe_depth + camera.K[1, 2],
    ])
    return pixels, depth
