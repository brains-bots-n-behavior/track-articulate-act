#!/usr/bin/env python
"""Stage 13: place HaWoR hands at an observed scene surface.

HaWoR's monocular SLAM scale can differ from the reconstructed scene's scale,
and its saved XYZ may use a different focal length.  This stage resolves those
ambiguities without registration.  It first projects the hand with HaWoR's
saved focal length and reinterprets those same pixels with the exact per-frame
DA3 intrinsics.  It then measures a target surface depth under the projected
hand silhouette and scales the hand onto it.

``--scale-source`` selects where that target depth comes from:

``object`` (default)
    The nearest Stage-04 object-mesh candidate is placed in DA3 world
    coordinates and its front surface is rendered into the current hand
    camera.  No depth image is read.  Frames without enough projected
    hand/object overlap fall back to the area-weighted average depth of the
    whole object surface, which temporal regularization treats as unreliable.
``depth``
    The per-frame DA3 depth map ``da3/depth/<frame>.npy`` is read directly and
    reduced over the hand silhouette pixels.  No object mesh is needed, so the
    hand follows whatever surface the depth model sees under it.
``pointmap``
    The dense DA3 reference point map ``da3/pointmap_ref.npy`` is expressed in
    the current hand camera and z-buffered under the hand silhouette.  Like
    ``depth`` it needs no object mesh, but the target is the static scene
    geometry rather than the current frame's monocular depth.

The source depth is the visible MANO depth reduced over the same pixels.
Their ratio is applied uniformly to all hand vertices and joints about the
camera origin, jointly changing hand depth and scale while preserving the image
projection.  ICP, feature matching, and silhouette registration are never used.

The default ``per-frame`` mode measures contact at every usable frame, replaces
unreliable corrections by temporal interpolation, and robustly smooths the
scale trajectory.  ``--no-temporal-regularization`` restores independent-frame
corrections, and ``--scale-mode global`` remains available when a constant
physical hand size is more important than per-frame contact.

Inputs::

    <scene>/hawor/{config.json,faces.npy,faces_left.npy,per_frame/*.npz}
    <scene>/sam3d_scaled/combined/[cand_NN_<frame>/]{mesh.glb,pose.json}
        (--scale-source object)
    <scene>/da3/depth/<frame>.npy         (--scale-source depth)
    <scene>/da3/pointmap_ref.npy          (--scale-source pointmap)
    <scene>/da3/{intrinsics.npz,cameras.npz}

Outputs mirror the consumable Stage-12 files under ``hawor_scaled``.  Camera
arrays (``verts``, ``joints``, and ``cam_t``) are corrected, and world arrays
are regenerated from the DA3 camera poses when available.  The source is never
modified.  A frame containing non-finite hand points or points on/behind the
camera plane is omitted and recorded without interrupting other frames.

Examples::

    python scripts/13_scale_hawor.py --scene-dir data/dryer
    python scripts/13_scale_hawor.py --scene-dir data/dryer --dry-run
    python scripts/13_scale_hawor.py --scene-dir data/dryer \
        --object-label appliance --overwrite
    python scripts/13_scale_hawor.py --scene-dir data/dryer \
        --scale-source depth --overwrite
    python scripts/13_scale_hawor.py --scene-dir data/dryer \
        --scale-source pointmap --overwrite

Run in an environment containing numpy; trimesh is required only by
``--scale-source object``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_CANDIDATE_RE = re.compile(r"cand_(\d+)_(\d+)")
_MODEL_FROM_GLB = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
_PYTORCH3D_TO_RDF = np.diag([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class HandMeasurement:
    frame: int
    hand_index: int
    side: str
    detected: bool
    target_depth: float
    front_depth: float
    ratio: float
    silhouette_pixels: int
    contact_pixels: int
    depth_min: float
    depth_max: float
    source_focal: float | None
    da3_focal: float
    scale_source: str
    depth_support: str
    support_reliable: bool
    surface_sample_count: int | None = None
    object_candidate_frame: int | None = None
    object_mesh: str | None = None


@dataclass(frozen=True)
class ObjectCandidate:
    frame: int
    candidate: int | None
    mesh_path: Path
    pose_path: Path
    samples_world: np.ndarray
    face_centers_world: np.ndarray
    face_areas_world: np.ndarray


@dataclass(frozen=True)
class ContactSupport:
    """Target-surface depth measured under one projected hand silhouette.

    ``front_mask`` selects the silhouette pixels whose MANO front depth forms
    the source of the ratio, so the target and source are always reduced over
    the same pixels unless a whole-surface fallback was used.
    """

    target_depth: float
    depth_min: float
    depth_max: float
    contact_pixels: int
    front_mask: np.ndarray
    support: str
    reliable: bool
    surface_sample_count: int | None = None
    candidate: ObjectCandidate | None = None


@dataclass(frozen=True)
class ScaleCorrection:
    frame: int
    hand_index: int
    side: str
    raw_factor: float
    applied_factor: float
    reliable_anchor: bool
    interpolated: bool
    smoothed: bool
    source: str


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Scene directory, for example data/dryer")
    parser.add_argument("--input-name", default="hawor",
                        help="Stage-12 input directory name within the scene")
    parser.add_argument("--out-name", default="hawor_scaled",
                        help="Output directory name within the scene")
    parser.add_argument("--scale-mode", choices=["global", "per-frame"],
                        default="per-frame",
                        help=("One robust scale per side, or temporally "
                              "regularized measured-frame scales"))
    parser.add_argument("--scale-source", choices=["object", "depth", "pointmap"],
                        default="object",
                        help=("Target surface for the hand: the Stage-04 "
                              "object mesh, the per-frame DA3 depth map, or "
                              "the DA3 reference point map"))
    parser.add_argument("--depth-statistic", choices=["mean", "median"],
                        help=("Reduction of target and MANO front depths over "
                              "the contact pixels; defaults to mean for "
                              "--scale-source object and median for the "
                              "depth-map and point-map sources"))
    parser.add_argument("--object-root", default="sam3d_scaled",
                        help="Object-mesh directory name within the scene")
    parser.add_argument("--object-label", default="combined",
                        help="Stage-03/04 object label used as the contact surface")
    parser.add_argument("--object-candidate-idx", type=int,
                        help="Use only this object candidate instead of nearest keyframe")
    parser.add_argument("--max-object-samples", type=int, default=200000,
                        help=("Maximum object vertices/face centers, or point-map "
                              "points, used for the contact z-buffer"))
    parser.add_argument("--contact-offset", type=float, default=0.0,
                        help="Keep the hand this far in front of the object surface")
    parser.add_argument("--calibration-frames", choices=["detected", "all"],
                        default="detected",
                        help="Frames allowed to estimate the robust global scale")
    parser.add_argument("--start-frame", type=int,
                        help="First frame allowed for contact measurement")
    parser.add_argument("--end-frame", type=int,
                        help="Last frame allowed for contact measurement (inclusive)")
    parser.add_argument("--calibration-step", type=int, default=1,
                        help="Measure every Nth eligible calibration frame")
    parser.add_argument("--min-contact-pixels", "--min-valid-pixels",
                        dest="min_contact_pixels", type=int, default=50,
                        help=("Minimum hand pixels carrying a target-surface "
                              "depth sample"))
    parser.add_argument("--mad-threshold", type=float, default=3.5,
                        help="MAD threshold for rejecting global scale outliers; 0 disables")
    parser.add_argument(
        "--temporal-regularization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("Interpolate unreliable per-frame corrections and smooth the "
              "scale trajectory; use --no-temporal-regularization for the "
              "original independent-frame behavior"),
    )
    parser.add_argument(
        "--temporal-window", type=int, default=9,
        help=("Odd centered window, in frames, for robust temporal scale "
              "smoothing; 1 performs interpolation without smoothing"),
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Measure and report corrections without writing files")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def validate_dir_name(value: str, flag: str):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise RuntimeError(f"{flag} must be one directory name")


def load_intrinsics(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing DA3 intrinsics: {path}")
    try:
        with np.load(path) as archive:
            if not {"frame_indices", "intrinsics"}.issubset(archive.files):
                raise RuntimeError(f"invalid intrinsics archive: {path}")
            frames = np.asarray(archive["frame_indices"]).reshape(-1)
            matrices = np.asarray(archive["intrinsics"])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if len(frames) != len(matrices):
        raise RuntimeError(f"frame/intrinsics row count differs in {path}")
    result = {}
    for frame, value in zip(frames, matrices):
        K = np.asarray(value, dtype=np.float64).reshape(3, 3)
        if (not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
                or abs(np.linalg.det(K)) < 1e-12):
            raise RuntimeError(f"invalid intrinsics for frame {int(frame):06d}")
        result[int(frame)] = K
    return result


def quat_xyzw_to_rotation(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("DA3 camera quaternion is zero or non-finite")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def load_camera_poses(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not path.is_file():
        return {}
    try:
        with np.load(path) as archive:
            required = {"frame_indices", "cam_quats_xyzw", "cam_trans"}
            if not required.issubset(archive.files):
                raise RuntimeError(f"invalid camera archive: {path}")
            frames = np.asarray(archive["frame_indices"]).reshape(-1)
            quaternions = np.asarray(archive["cam_quats_xyzw"])
            translations = np.asarray(archive["cam_trans"])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not (len(frames) == len(quaternions) == len(translations)):
        raise RuntimeError(f"camera row counts differ in {path}")
    result = {}
    for frame, quaternion, translation in zip(frames, quaternions, translations):
        rotation = quat_xyzw_to_rotation(quaternion)
        translation = np.asarray(translation, dtype=np.float64).reshape(3)
        if not np.isfinite(translation).all():
            raise RuntimeError(f"invalid camera translation at frame {int(frame):06d}")
        result[int(frame)] = (rotation, translation)
    return result


def quat_wxyz_to_rotation(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("SAM3D rotation quaternion is zero or non-finite")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def sam3d_pose_to_rdf_camera(pose: dict, path: Path) -> np.ndarray:
    try:
        rotation = quat_wxyz_to_rotation(pose["rotation_quat_wxyz"])
        translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        scale = np.asarray(pose["scale"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid SAM3D pose in {path}") from exc
    if scale.size == 1:
        scale = np.repeat(scale, 3)
    if (scale.size != 3 or not np.isfinite(scale).all()
            or np.any(scale <= 0) or not np.isfinite(translation).all()):
        raise RuntimeError(f"invalid SAM3D scale/translation in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        _PYTORCH3D_TO_RDF
        @ rotation.T
        @ np.diag(scale)
        @ _MODEL_FROM_GLB
    )
    transform[:3, 3] = _PYTORCH3D_TO_RDF @ translation
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=np.float64) @ transform[:3, :3].T
            + transform[:3, 3])


def load_mesh_geometry(path: Path, trimesh) -> tuple[np.ndarray, np.ndarray]:
    try:
        asset = trimesh.load(path, force="scene", process=False)
        meshes = [geometry for geometry in asset.dump(concatenate=False)
                  if isinstance(geometry, trimesh.Trimesh)
                  and len(geometry.vertices) > 0
                  and len(geometry.faces) > 0]
    except Exception as exc:
        raise RuntimeError(f"could not load object mesh {path}: {exc}") from exc
    if not meshes:
        raise RuntimeError(f"object mesh has no triangle geometry: {path}")
    vertex_parts, face_parts = [], []
    offset = 0
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        vertex_parts.append(vertices)
        face_parts.append(faces + offset)
        offset += len(vertices)
    vertices = np.concatenate(vertex_parts, axis=0)
    faces = np.concatenate(face_parts, axis=0)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"object mesh has non-finite vertices: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError(f"object mesh has invalid faces: {path}")
    return vertices, faces


def object_frame_from_dir(path: Path) -> int:
    keyframe_path = path / "keyframe.txt"
    if keyframe_path.is_file():
        try:
            return int(keyframe_path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid object keyframe: {keyframe_path}") from exc
    match = _CANDIDATE_RE.fullmatch(path.name)
    if match:
        return int(match.group(2))
    pose = read_json(path / "pose.json")
    if "keyframe" in pose:
        return int(pose["keyframe"])
    raise RuntimeError(f"cannot determine object keyframe for {path}")


def discover_object_records(root: Path, candidate_idx: int | None):
    if not root.is_dir():
        raise RuntimeError(f"missing object mesh directory: {root}")
    records = []
    has_candidate_dirs = False
    for path in sorted(root.iterdir()):
        match = _CANDIDATE_RE.fullmatch(path.name)
        if not path.is_dir() or match is None:
            continue
        has_candidate_dirs = True
        candidate = int(match.group(1))
        if candidate_idx is not None and candidate != candidate_idx:
            continue
        if not ((path / "mesh.glb").is_file()
                and (path / "pose.json").is_file()):
            continue
        records.append((object_frame_from_dir(path), candidate,
                        path / "mesh.glb", path / "pose.json"))
    if (not has_candidate_dirs and (root / "mesh.glb").is_file()
            and (root / "pose.json").is_file()):
        records.append((object_frame_from_dir(root), None,
                        root / "mesh.glb", root / "pose.json"))
    if not records:
        detail = (f" candidate {candidate_idx}" if candidate_idx is not None
                  else "")
        raise RuntimeError(f"no complete object mesh/pose{detail} under {root}")
    return records


def load_object_candidates(root: Path, candidate_idx: int | None,
                           camera_poses, max_samples: int, trimesh):
    candidates = []
    for frame, candidate, mesh_path, pose_path in discover_object_records(
            root, candidate_idx):
        if frame not in camera_poses:
            raise RuntimeError(
                f"object keyframe {frame:06d} has no DA3 camera pose")
        vertices, faces = load_mesh_geometry(mesh_path, trimesh)
        object_to_camera = sam3d_pose_to_rdf_camera(
            read_json(pose_path), pose_path)
        vertices_camera = transform_points(vertices, object_to_camera)
        rotation, translation = camera_poses[frame]
        vertices_world = vertices_camera @ rotation.T + translation
        triangles_world = vertices_world[faces]
        centers_world = triangles_world.mean(axis=1, dtype=np.float64)
        areas_world = 0.5 * np.linalg.norm(
            np.cross(triangles_world[:, 1] - triangles_world[:, 0],
                     triangles_world[:, 2] - triangles_world[:, 0]), axis=1)
        usable_faces = np.isfinite(areas_world) & (areas_world > 1e-16)
        centers_world = centers_world[usable_faces]
        areas_world = areas_world[usable_faces]
        if len(centers_world) == 0:
            raise RuntimeError(f"object mesh has no nondegenerate faces: {mesh_path}")
        samples_world = np.concatenate((vertices_world, centers_world), axis=0)
        if len(samples_world) > max_samples:
            indices = (np.arange(max_samples, dtype=np.int64)
                       * len(samples_world) // max_samples)
            samples_world = samples_world[indices]
        candidates.append(ObjectCandidate(
            frame=frame,
            candidate=candidate,
            mesh_path=mesh_path,
            pose_path=pose_path,
            samples_world=samples_world,
            face_centers_world=centers_world,
            face_areas_world=areas_world,
        ))
    return sorted(candidates, key=lambda item: (item.frame, item.candidate or -1))


def world_to_camera(points: np.ndarray,
                    camera_pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    rotation, translation = camera_pose
    return (np.asarray(points, dtype=np.float64) - translation) @ rotation


def load_faces(root: Path) -> dict[str, np.ndarray]:
    path = root / "faces.npy"
    if not path.is_file():
        raise RuntimeError(f"missing MANO topology: {path}")
    try:
        right = np.asarray(np.load(path), dtype=np.int64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if right.ndim != 2 or right.shape[1] != 3 or len(right) == 0:
        raise RuntimeError(f"invalid MANO faces in {path}: {right.shape}")
    left_path = root / "faces_left.npy"
    if left_path.is_file():
        try:
            left = np.asarray(np.load(left_path), dtype=np.int64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read {left_path}: {exc}") from exc
        if left.shape != right.shape:
            raise RuntimeError(f"left/right MANO topology differs under {root}")
    else:
        left = right[:, [0, 2, 1]].copy()
    return {"left": left, "right": right}


def load_frame_record(path: Path) -> tuple[int, dict[str, np.ndarray]]:
    try:
        with np.load(path) as archive:
            values = {name: np.asarray(archive[name]).copy()
                      for name in archive.files}
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    required = {"verts", "joints", "is_right"}
    if not required.issubset(values):
        raise RuntimeError(f"{path} is missing {sorted(required - set(values))}")
    vertices = np.asarray(values["verts"])
    joints = np.asarray(values["joints"])
    sides = np.asarray(values["is_right"]).reshape(-1)
    if (vertices.ndim != 3 or vertices.shape[2] != 3
            or joints.ndim != 3 or joints.shape[2] != 3
            or len(vertices) != len(joints) or len(vertices) != len(sides)):
        raise RuntimeError(
            f"invalid hand array shapes in {path}: verts={vertices.shape}, "
            f"joints={joints.shape}, is_right={sides.shape}")
    frame = int(np.asarray(values.get("frame_idx", int(path.stem))).reshape(()))
    if frame < 0:
        raise RuntimeError(f"negative frame index in {path}")
    return frame, values


def camera_points_skip_reason(vertices: np.ndarray,
                              joints: np.ndarray) -> str | None:
    """Explain why a complete HaWoR frame cannot be scaled safely."""
    for name, points in (("vertices", vertices), ("joints", joints)):
        values = np.asarray(points)
        if not np.isfinite(values).all():
            return f"HaWoR {name} contain non-finite camera coordinates"
        behind = int(np.count_nonzero(values[..., 2] <= 1e-8))
        if behind:
            return (
                f"HaWoR {name} are behind the camera "
                f"({behind}/{values[..., 2].size} points have Z <= 0)"
            )
    return None


def rasterize_front_depth(vertices: np.ndarray, faces: np.ndarray,
                          K: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterize a perspective-correct, winding-independent camera z-buffer."""
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise RuntimeError(f"expected Nx3 MANO vertices, got {vertices.shape}")
    if not np.isfinite(vertices).all():
        raise RuntimeError("MANO vertices contain non-finite values")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError("MANO topology contains out-of-range vertex indices")

    homogeneous = vertices @ K.T
    z = vertices[:, 2]
    valid_vertex = np.isfinite(homogeneous).all(axis=1) & (z > 1e-8)
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    uv[valid_vertex] = (
        homogeneous[valid_vertex, :2] / homogeneous[valid_vertex, 2:3])
    z_buffer = np.full((height, width), np.inf, dtype=np.float64)

    for face in faces:
        if not valid_vertex[face].all():
            continue
        triangle = uv[face]
        x0 = max(0, int(np.ceil(np.min(triangle[:, 0]))))
        x1 = min(width - 1, int(np.floor(np.max(triangle[:, 0]))))
        y0 = max(0, int(np.ceil(np.min(triangle[:, 1]))))
        y1 = min(height - 1, int(np.floor(np.max(triangle[:, 1]))))
        if x0 > x1 or y0 > y1:
            continue
        ax, ay = triangle[0]
        bx, by = triangle[1]
        cx, cy = triangle[2]
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            continue
        xs = np.arange(x0, x1 + 1, dtype=np.float64)[None, :]
        ys = np.arange(y0, y1 + 1, dtype=np.float64)[:, None]
        weight_a = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / denominator
        weight_b = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / denominator
        weight_c = 1.0 - weight_a - weight_b
        inside = ((weight_a >= -1e-9) & (weight_b >= -1e-9)
                  & (weight_c >= -1e-9))
        if not inside.any():
            continue
        triangle_z = z[face]
        inverse_depth = (weight_a / triangle_z[0]
                         + weight_b / triangle_z[1]
                         + weight_c / triangle_z[2])
        rendered = np.full(inverse_depth.shape, np.inf, dtype=np.float64)
        positive = inside & np.isfinite(inverse_depth) & (inverse_depth > 0)
        rendered[positive] = 1.0 / inverse_depth[positive]
        patch = z_buffer[y0:y1 + 1, x0:x1 + 1]
        np.minimum(patch, rendered, out=patch)

    return z_buffer


def convert_hawor_points_to_da3(points: np.ndarray,
                                source_focal: float | None,
                                image_size_wh: np.ndarray | None,
                                K: np.ndarray,
                                target_hw: tuple[int, int]) -> np.ndarray:
    """Preserve HaWoR pixels while expressing points on DA3 camera rays.

    Stage 12 uses one scalar focal and an image-centered principal point.
    Older outputs may have been produced before DA3 focal auto-loading existed,
    so treating their XYZ directly under DA3 K can move the projection far from
    the detected hand.  This is an analytic pinhole intrinsics conversion, not
    image or geometry registration; every source pixel remains identical.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RuntimeError(f"expected Nx3 HaWoR points, got {points.shape}")
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 1e-8):
        raise RuntimeError("HaWoR points are non-finite or behind the camera")
    target_height, target_width = target_hw
    if image_size_wh is None:
        source_width, source_height = float(target_width), float(target_height)
    else:
        size = np.asarray(image_size_wh, dtype=np.float64).reshape(-1)
        if (size.size != 2 or not np.isfinite(size).all()
                or np.any(size <= 0)):
            raise RuntimeError(f"invalid HaWoR img_size_wh: {size}")
        source_width, source_height = map(float, size)
    if source_focal is None:
        source_focal = float(np.sqrt(K[0, 0] * K[1, 1]))
    if not np.isfinite(source_focal) or source_focal <= 0:
        raise RuntimeError(f"invalid HaWoR focal length: {source_focal}")

    scale_x = target_width / source_width
    scale_y = target_height / source_height
    source_K = np.array([
        [source_focal * scale_x, 0.0, 0.5 * source_width * scale_x],
        [0.0, source_focal * scale_y, 0.5 * source_height * scale_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    projected = points @ source_K.T
    pixels = projected[:, :2] / projected[:, 2:3]
    pixel_h = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64)))
    rays = pixel_h @ np.linalg.inv(K).T
    if (not np.isfinite(rays).all()
            or np.any(np.abs(rays[:, 2]) < 1e-12)):
        raise RuntimeError("DA3 intrinsics produced invalid hand rays")
    rays /= rays[:, 2:3]
    converted = rays * points[:, 2:3]

    # Verify the analytic conversion independently of the later uniform scale.
    reprojected = converted @ K.T
    reprojected = reprojected[:, :2] / reprojected[:, 2:3]
    if not np.allclose(reprojected, pixels, rtol=1e-8, atol=1e-7):
        raise RuntimeError("intrinsics conversion changed the HaWoR projection")
    return converted


def sampled_surface_zbuffer(samples_camera: np.ndarray, K: np.ndarray,
                            target_mask: np.ndarray) -> np.ndarray:
    """Return nearest sampled surface depth at pixels inside ``target_mask``."""
    samples = np.asarray(samples_camera, dtype=np.float64)
    height, width = target_mask.shape
    projected = samples @ K.T
    finite = np.isfinite(projected).all(axis=1) & (samples[:, 2] > 1e-8)
    uv = np.full((len(samples), 2), np.nan, dtype=np.float64)
    uv[finite] = projected[finite, :2] / projected[finite, 2:3]
    base_cols = np.zeros(len(samples), dtype=np.int64)
    base_rows = np.zeros(len(samples), dtype=np.int64)
    base_cols[finite] = np.floor(uv[finite, 0]).astype(np.int64)
    base_rows[finite] = np.floor(uv[finite, 1]).astype(np.int64)
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)
    for col_offset, row_offset in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cols = base_cols + col_offset
        rows = base_rows + row_offset
        inside = (finite & (cols >= 0) & (cols < width)
                  & (rows >= 0) & (rows < height))
        if not inside.any():
            continue
        rows_inside = rows[inside]
        cols_inside = cols[inside]
        in_target = target_mask[rows_inside, cols_inside]
        if not in_target.any():
            continue
        indices = rows_inside[in_target] * width + cols_inside[in_target]
        np.minimum.at(
            z_buffer, indices, samples[inside][in_target, 2])
    return z_buffer.reshape(height, width)


def image_hw(image_size_wh: np.ndarray | None,
             K: np.ndarray) -> tuple[int, int]:
    if image_size_wh is not None:
        size = np.asarray(image_size_wh, dtype=np.float64).reshape(-1)
        if (size.size == 2 and np.isfinite(size).all()
                and np.all(size > 0)):
            return int(round(size[1])), int(round(size[0]))
    height = int(round(2.0 * K[1, 2]))
    width = int(round(2.0 * K[0, 2]))
    if height <= 0 or width <= 0:
        raise RuntimeError("cannot infer image size from HaWoR record or DA3 K")
    return height, width


def reduce_depths(values: np.ndarray, statistic: str,
                  weights: np.ndarray | None = None) -> float:
    """Reduce contact depths with the requested robust or plain statistic."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise RuntimeError("cannot reduce an empty depth sample")
    if statistic == "median":
        return float(np.median(values))
    if statistic != "mean":
        raise RuntimeError(f"unknown depth statistic: {statistic}")
    if weights is None:
        return float(np.mean(values, dtype=np.float64))
    return float(np.average(values, weights=weights))


def resample_nearest(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize so a DA3 raster matches the hand raster."""
    height, width = target_hw
    if image.shape[:2] == (height, width):
        return image
    source_height, source_width = image.shape[:2]
    rows = np.clip(
        ((np.arange(height) + 0.5) * source_height / height).astype(np.int64),
        0, source_height - 1)
    cols = np.clip(
        ((np.arange(width) + 0.5) * source_width / width).astype(np.int64),
        0, source_width - 1)
    return image[np.ix_(rows, cols)]


class ObjectMeshSource:
    """Front depth of the Stage-04 object mesh under the hand silhouette."""

    kind = "object"
    needs_camera_pose = True
    description = (
        "nearest object-mesh sample depth under the projected MANO "
        "silhouette, area-weighted whole-mesh depth when overlap is sparse"
    )

    def __init__(self, candidates: list[ObjectCandidate]):
        if not candidates:
            raise RuntimeError("no object mesh candidates are available")
        self.candidates = candidates

    def support(self, frame: int, silhouette: np.ndarray, K: np.ndarray,
                camera_pose, min_contact_pixels: int,
                statistic: str) -> ContactSupport:
        candidate = min(
            self.candidates,
            key=lambda item: (abs(item.frame - frame), item.frame))
        samples_camera = world_to_camera(candidate.samples_world, camera_pose)
        object_z_buffer = sampled_surface_zbuffer(samples_camera, K, silhouette)
        contact = silhouette & np.isfinite(object_z_buffer)
        contact_count = int(contact.sum())
        if contact_count >= min_contact_pixels:
            depths = object_z_buffer[contact]
            return ContactSupport(
                target_depth=reduce_depths(depths, statistic),
                depth_min=float(np.min(depths)),
                depth_max=float(np.max(depths)),
                contact_pixels=contact_count,
                front_mask=contact,
                support="object_front_samples_under_hand_silhouette",
                reliable=True,
                surface_sample_count=len(candidate.samples_world),
                candidate=candidate,
            )
        centers_camera = world_to_camera(
            candidate.face_centers_world, camera_pose)
        visible = np.isfinite(centers_camera).all(axis=1) & (
            centers_camera[:, 2] > 1e-8)
        if not visible.any():
            raise RuntimeError("object mesh has no surface in front of the camera")
        depths = centers_camera[visible, 2]
        return ContactSupport(
            target_depth=reduce_depths(
                depths, statistic, candidate.face_areas_world[visible]),
            depth_min=float(np.min(depths)),
            depth_max=float(np.max(depths)),
            contact_pixels=contact_count,
            front_mask=silhouette,
            support="area_weighted_whole_object_surface_fallback",
            reliable=False,
            surface_sample_count=len(candidate.samples_world),
            candidate=candidate,
        )


class DepthMapSource:
    """Per-frame DA3 z-depth read directly under the hand silhouette.

    DA3 writes full-resolution z-depth in the same RDF camera frame the hand
    is expressed in, so no reprojection is needed: the depth already lives on
    the pixels the MANO silhouette covers.  Non-positive samples mark invalid
    DA3 pixels and are excluded.
    """

    kind = "depth"
    needs_camera_pose = False
    description = (
        "per-frame DA3 depth-map samples under the projected MANO silhouette"
    )

    def __init__(self, scene_dir: Path):
        self.depth_dir = scene_dir / "da3" / "depth"
        if not self.depth_dir.is_dir():
            raise RuntimeError(
                f"missing DA3 depth directory: {self.depth_dir}; run stage 00")
        self._cache_key = None
        self._cache_depth = None

    def depth_map(self, frame: int, target_hw: tuple[int, int]) -> np.ndarray:
        key = (frame, target_hw)
        if key == self._cache_key:
            return self._cache_depth
        depth_path = self.depth_dir / f"{frame:06d}.npy"
        if not depth_path.is_file():
            raise RuntimeError(f"missing DA3 depth map: {depth_path}")
        try:
            depth = np.asarray(np.load(depth_path), dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read {depth_path}: {exc}") from exc
        if depth.ndim != 2 or depth.size == 0:
            raise RuntimeError(
                f"expected HxW depth in {depth_path}, got {depth.shape}")
        depth = resample_nearest(depth, target_hw)
        self._cache_key = key
        self._cache_depth = depth
        return depth

    def support(self, frame: int, silhouette: np.ndarray, K: np.ndarray,
                camera_pose, min_contact_pixels: int,
                statistic: str) -> ContactSupport:
        depth = self.depth_map(frame, silhouette.shape)
        contact = silhouette & np.isfinite(depth) & (depth > 1e-8)
        contact_count = int(contact.sum())
        if contact_count < min_contact_pixels:
            raise RuntimeError(
                f"only {contact_count} valid DA3 depth pixels under the hand "
                f"silhouette; need at least {min_contact_pixels}")
        depths = depth[contact]
        return ContactSupport(
            target_depth=reduce_depths(depths, statistic),
            depth_min=float(np.min(depths)),
            depth_max=float(np.max(depths)),
            contact_pixels=contact_count,
            front_mask=contact,
            support="da3_depth_map_under_hand_silhouette",
            reliable=True,
            surface_sample_count=contact_count,
        )


class PointmapSource:
    """DA3 reference point map z-buffered into the current hand camera.

    ``pointmap_ref.npy`` is a dense per-pixel world point for the DA3 reference
    frame.  Expressing it in the current camera and keeping the nearest sample
    per silhouette pixel yields the static scene surface the hand covers, which
    stays consistent across frames even where the per-frame depth map drifts.
    """

    kind = "pointmap"
    needs_camera_pose = True
    description = (
        "nearest DA3 reference point-map sample depth under the projected "
        "MANO silhouette"
    )

    def __init__(self, scene_dir: Path, max_samples: int):
        path = scene_dir / "da3" / "pointmap_ref.npy"
        if not path.is_file():
            raise RuntimeError(
                f"missing DA3 point map: {path}; run stage 00")
        try:
            pointmap = np.asarray(np.load(path), dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read {path}: {exc}") from exc
        if pointmap.ndim != 3 or pointmap.shape[2] != 3 or pointmap.size == 0:
            raise RuntimeError(
                f"expected HxWx3 point map in {path}, got {pointmap.shape}")
        points = pointmap.reshape(-1, 3)
        usable = np.isfinite(points).all(axis=1)
        reference_depth = self._reference_depth(scene_dir, pointmap.shape[:2])
        if reference_depth is not None:
            usable &= (np.isfinite(reference_depth)
                       & (reference_depth > 1e-8)).reshape(-1)
        points = points[usable]
        if len(points) == 0:
            raise RuntimeError(f"point map has no valid points: {path}")
        if len(points) > max_samples:
            indices = (np.arange(max_samples, dtype=np.int64)
                       * len(points) // max_samples)
            points = points[indices]
        self.path = path
        self.samples_world = points
        self.reference_depth_used = reference_depth is not None

    @staticmethod
    def _reference_depth(scene_dir: Path, target_hw) -> np.ndarray | None:
        """Return the reference frame's depth map, used only for validity."""
        config = read_json(scene_dir / "da3" / "config.json")
        reference = config.get("ref_frame")
        if reference is None:
            return None
        depth_path = (scene_dir / "da3" / "depth"
                      / f"{int(reference):06d}.npy")
        if not depth_path.is_file():
            return None
        try:
            depth = np.asarray(np.load(depth_path), dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read {depth_path}: {exc}") from exc
        if depth.shape != tuple(target_hw):
            return None
        return depth

    def support(self, frame: int, silhouette: np.ndarray, K: np.ndarray,
                camera_pose, min_contact_pixels: int,
                statistic: str) -> ContactSupport:
        samples_camera = world_to_camera(self.samples_world, camera_pose)
        z_buffer = sampled_surface_zbuffer(samples_camera, K, silhouette)
        contact = silhouette & np.isfinite(z_buffer)
        contact_count = int(contact.sum())
        if contact_count < min_contact_pixels:
            raise RuntimeError(
                f"only {contact_count} DA3 point-map samples land under the "
                f"hand silhouette; need at least {min_contact_pixels}")
        depths = z_buffer[contact]
        return ContactSupport(
            target_depth=reduce_depths(depths, statistic),
            depth_min=float(np.min(depths)),
            depth_max=float(np.max(depths)),
            contact_pixels=contact_count,
            front_mask=contact,
            support="da3_reference_pointmap_under_hand_silhouette",
            reliable=True,
            surface_sample_count=len(self.samples_world),
        )


def build_contact_source(scene_dir: Path, args, camera_poses):
    """Create the target-surface source selected by ``--scale-source``."""
    if args.scale_source == "depth":
        return DepthMapSource(scene_dir), None, []
    if args.scale_source == "pointmap":
        return PointmapSource(scene_dir, args.max_object_samples), None, []
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required to load the object mesh; use "
            "--scale-source depth or pointmap to scale without it") from exc
    object_root = scene_dir / args.object_root / args.object_label
    candidates = load_object_candidates(
        object_root, args.object_candidate_idx, camera_poses,
        args.max_object_samples, trimesh)
    return ObjectMeshSource(candidates), object_root, candidates


def measure_hand(frame: int, hand_index: int, side: str,
                 detected: bool, vertices: np.ndarray, faces: np.ndarray,
                 K: np.ndarray, min_contact_pixels: int,
                 source_focal: float | None,
                 image_size_wh: np.ndarray | None,
                 contact_source,
                 camera_pose: tuple[np.ndarray, np.ndarray] | None = None,
                 contact_offset: float = 0.0,
                 statistic: str = "mean") -> HandMeasurement:
    if camera_pose is None and contact_source.needs_camera_pose:
        raise RuntimeError(f"frame {frame:06d} has no DA3 camera pose")
    height, width = image_hw(image_size_wh, K)
    vertices_da3 = convert_hawor_points_to_da3(
        vertices, source_focal, image_size_wh, K, (height, width))
    z_buffer = rasterize_front_depth(vertices_da3, faces, K, height, width)
    silhouette = np.isfinite(z_buffer)
    silhouette_count = int(silhouette.sum())
    if silhouette_count == 0:
        raise RuntimeError("projected hand silhouette is empty")

    support = contact_source.support(
        frame, silhouette, K, camera_pose, min_contact_pixels, statistic)
    front_depth = reduce_depths(z_buffer[support.front_mask], statistic)
    target_depth = support.target_depth - contact_offset
    if (not np.isfinite(target_depth) or target_depth <= 0
            or not np.isfinite(front_depth) or front_depth <= 0):
        raise RuntimeError(
            f"invalid target/hand depths: {target_depth}, {front_depth}")
    ratio = target_depth / front_depth
    if not np.isfinite(ratio) or ratio <= 0:
        raise RuntimeError(f"invalid depth scale factor: {ratio}")
    da3_focal = float(np.sqrt(K[0, 0] * K[1, 1]))
    candidate = support.candidate
    return HandMeasurement(
        frame=frame,
        hand_index=hand_index,
        side=side,
        detected=detected,
        target_depth=target_depth,
        front_depth=front_depth,
        ratio=float(ratio),
        silhouette_pixels=silhouette_count,
        contact_pixels=support.contact_pixels,
        depth_min=support.depth_min,
        depth_max=support.depth_max,
        source_focal=source_focal,
        da3_focal=da3_focal,
        scale_source=contact_source.kind,
        depth_support=support.support,
        support_reliable=support.reliable,
        surface_sample_count=support.surface_sample_count,
        object_candidate_frame=None if candidate is None else candidate.frame,
        object_mesh=None if candidate is None else str(candidate.mesh_path),
    )


def robust_scale(measurements: list[HandMeasurement], mad_threshold: float):
    ratios = np.asarray([item.ratio for item in measurements], dtype=np.float64)
    if len(ratios) == 0:
        raise RuntimeError("cannot estimate a scale without measurements")
    median = float(np.median(ratios))
    deviations = np.abs(ratios - median)
    mad = float(np.median(deviations))
    keep = np.ones(len(ratios), dtype=bool)
    if mad_threshold > 0 and len(ratios) >= 3 and mad > 1e-12:
        robust_sigma = 1.4826 * mad
        keep = deviations <= mad_threshold * robust_sigma
        if not keep.any():
            keep[:] = True
    factor = float(np.median(ratios[keep]))
    return factor, keep, median, mad


def measurement_is_temporally_reliable(item: HandMeasurement) -> bool:
    """Return whether a measurement is suitable as a temporal anchor.

    Every source marks a measurement unreliable when the target depth did not
    come from samples under the hand silhouette itself, such as the object
    source's whole-mesh fallback.
    """
    return bool(
        item.detected
        and item.contact_pixels > 0
        and item.support_reliable
    )


def smooth_scale_trajectory(frames: np.ndarray, factors: np.ndarray,
                            window: int) -> np.ndarray:
    """Robustly smooth positive factors without crossing long frame gaps.

    A centered median removes isolated contact-depth spikes.  A centered
    triangular average then softens the remaining frame-to-frame variation.
    Both operations happen in log space so equal relative scale changes receive
    equal weight and the returned factors remain positive.
    """
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    factors = np.asarray(factors, dtype=np.float64).reshape(-1)
    if len(frames) != len(factors):
        raise RuntimeError("temporal frame/factor row counts differ")
    if (not np.isfinite(factors).all()) or np.any(factors <= 0):
        raise RuntimeError("temporal scale factors must be finite and positive")
    if window <= 1 or len(factors) <= 1:
        return factors.copy()

    radius = window // 2
    log_factors = np.log(factors)
    median_filtered = np.empty_like(log_factors)
    for index, frame in enumerate(frames):
        neighborhood = np.abs(frames - frame) <= radius
        median_filtered[index] = np.median(log_factors[neighborhood])

    smoothed = np.empty_like(median_filtered)
    for index, frame in enumerate(frames):
        distances = np.abs(frames - frame)
        neighborhood = distances <= radius
        weights = (radius + 1 - distances[neighborhood]).astype(np.float64)
        smoothed[index] = np.average(
            median_filtered[neighborhood], weights=weights)
    return np.exp(smoothed)


def build_scale_corrections(
        instances: dict[tuple[int, int], str],
        measurements: dict[tuple[int, int], HandMeasurement],
        side_scales: dict[str, float], scale_mode: str,
        temporal_regularization: bool,
        temporal_window: int) -> dict[tuple[int, int], ScaleCorrection]:
    """Resolve raw, interpolated, and temporally regularized hand scales."""
    corrections: dict[tuple[int, int], ScaleCorrection] = {}
    grouped: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for key, side in instances.items():
        frame, hand_index = key
        grouped.setdefault((side, hand_index), []).append(key)

    for (side, _), keys in grouped.items():
        keys.sort()
        if side not in side_scales:
            raise RuntimeError(f"missing global scale for {side} hand")
        global_factor = float(side_scales[side])
        if not np.isfinite(global_factor) or global_factor <= 0:
            raise RuntimeError(f"invalid global scale for {side} hand")

        frames = np.asarray([key[0] for key in keys], dtype=np.int64)
        raw = np.asarray([
            measurements[key].ratio if key in measurements else global_factor
            for key in keys
        ], dtype=np.float64)
        reliable = np.asarray([
            key in measurements
            and measurement_is_temporally_reliable(measurements[key])
            for key in keys
        ], dtype=bool)

        if scale_mode == "global":
            base = np.full(len(keys), global_factor, dtype=np.float64)
            applied = base.copy()
            interpolated = np.zeros(len(keys), dtype=bool)
            sources = ["global_scale"] * len(keys)
        elif not temporal_regularization:
            base = raw.copy()
            applied = base.copy()
            interpolated = np.asarray(
                [key not in measurements for key in keys], dtype=bool)
            sources = [
                "raw_measurement" if key in measurements
                else "global_fallback"
                for key in keys
            ]
        elif reliable.any():
            anchor_frames = frames[reliable]
            anchor_log_factors = np.log(raw[reliable])
            base = np.exp(np.interp(frames, anchor_frames, anchor_log_factors))
            interpolated = ~reliable
            applied = smooth_scale_trajectory(
                frames, base, temporal_window)
            first_anchor = int(anchor_frames[0])
            last_anchor = int(anchor_frames[-1])
            sources = []
            for key, frame, is_reliable in zip(keys, frames, reliable):
                if is_reliable:
                    sources.append("reliable_measurement")
                elif frame < first_anchor or frame > last_anchor:
                    sources.append("nearest_reliable_edge_fill")
                elif key in measurements:
                    sources.append("unreliable_measurement_interpolated")
                else:
                    sources.append("missing_measurement_interpolated")
        else:
            base = np.full(len(keys), global_factor, dtype=np.float64)
            applied = base.copy()
            interpolated = np.ones(len(keys), dtype=bool)
            sources = ["global_fallback_no_reliable_anchors"] * len(keys)

        for index, key in enumerate(keys):
            factor = float(applied[index])
            raw_factor = float(raw[index])
            corrections[key] = ScaleCorrection(
                frame=key[0],
                hand_index=key[1],
                side=side,
                raw_factor=raw_factor,
                applied_factor=factor,
                reliable_anchor=bool(reliable[index]),
                interpolated=bool(interpolated[index]),
                smoothed=not np.isclose(
                    factor, float(base[index]),
                    rtol=1e-10, atol=1e-12,
                ),
                source=sources[index],
            )
    return corrections


def project_bbox(vertices: np.ndarray, K: np.ndarray,
                 height: int, width: int) -> np.ndarray:
    projected = np.asarray(vertices, dtype=np.float64) @ K.T
    valid = np.isfinite(projected).all(axis=1) & (vertices[:, 2] > 1e-8)
    if not valid.any():
        return np.zeros(4, dtype=np.float32)
    uv = projected[valid, :2] / projected[valid, 2:3]
    return np.asarray([
        np.clip(np.min(uv[:, 0]), 0, width),
        np.clip(np.min(uv[:, 1]), 0, height),
        np.clip(np.max(uv[:, 0]), 0, width),
        np.clip(np.max(uv[:, 1]), 0, height),
    ], dtype=np.float32)


def json_measurement(item: HandMeasurement, used_for_global: bool | None = None):
    value = {
        "frame": item.frame,
        "hand_index": item.hand_index,
        "side": item.side,
        "detected": item.detected,
        "scale_source": item.scale_source,
        "target_surface_depth": item.target_depth,
        "original_average_hand_front_surface_depth": item.front_depth,
        "depth_scale_factor": item.ratio,
        "hand_silhouette_pixels": item.silhouette_pixels,
        "contact_pixels": item.contact_pixels,
        "contact_coverage": (
            item.contact_pixels / item.silhouette_pixels),
        "target_depth_min": item.depth_min,
        "target_depth_max": item.depth_max,
        "target_depth_support": item.depth_support,
        "target_depth_support_reliable": item.support_reliable,
        "target_surface_sample_count": item.surface_sample_count,
        "object_candidate_frame": item.object_candidate_frame,
        "object_mesh": item.object_mesh,
        "source_hawor_focal": item.source_focal,
        "da3_geometric_mean_focal": item.da3_focal,
    }
    if used_for_global is not None:
        value["used_for_global_scale"] = bool(used_for_global)
    return value


def json_scale_correction(item: ScaleCorrection):
    return {
        "frame": item.frame,
        "hand_index": item.hand_index,
        "side": item.side,
        "raw_depth_scale_factor": item.raw_factor,
        "applied_depth_scale_factor": item.applied_factor,
        "reliable_temporal_anchor": item.reliable_anchor,
        "interpolated": item.interpolated,
        "smoothed": item.smoothed,
        "source": item.source,
    }


def relative_path(path: Path, scene_dir: Path) -> str:
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def write_scaled_output(scene_dir: Path, input_root: Path, output_root: Path,
                        frame_paths: list[Path], intrinsics,
                        camera_poses, faces_by_side, measurements,
                        side_scales, scale_corrections, args, source_config,
                        metadata):
    if output_root.exists():
        if not args.overwrite:
            raise RuntimeError(f"{output_root} exists; pass --overwrite to replace")
        if not output_root.is_dir():
            raise RuntimeError(f"refusing to replace non-directory: {output_root}")
        shutil.rmtree(output_root)
    per_frame_out = output_root / "per_frame"
    per_frame_out.mkdir(parents=True)

    for name in ("faces.npy", "faces_left.npy"):
        source = input_root / name
        if source.is_file():
            shutil.copy2(source, output_root / name)
    if not (output_root / "faces_left.npy").is_file():
        np.save(output_root / "faces_left.npy", faces_by_side["left"].astype(np.int32))

    files_written = 0
    instances_written = 0
    per_frame_exact = 0
    per_frame_without_measurement = 0
    per_frame_global_fallback = 0
    world_frames_written = 0
    output_skips = []
    for path in frame_paths:
        try:
            frame, values = load_frame_record(path)
        except RuntimeError as exc:
            output_skips.append({"file": str(path), "error": str(exc)})
            print(f"[{path.stem}] OMIT output: {exc}")
            continue
        if frame not in intrinsics:
            message = f"frame {frame:06d} is absent from da3/intrinsics.npz"
            output_skips.append({"file": str(path), "frame": frame,
                                 "error": message})
            print(f"[{frame:06d}] OMIT output: {message}")
            continue
        K = intrinsics[frame]
        vertices = np.asarray(values["verts"], dtype=np.float64)
        joints = np.asarray(values["joints"], dtype=np.float64)
        skip_reason = camera_points_skip_reason(vertices, joints)
        if skip_reason is not None:
            output_skips.append({
                "file": str(path),
                "frame": frame,
                "reason": "behind_camera_or_nonfinite_hand",
                "error": skip_reason,
            })
            print(f"[{frame:06d}] OMIT output: {skip_reason}")
            continue
        is_right = np.asarray(values["is_right"], dtype=bool).reshape(-1)
        focal_value = values.get("focal_length")
        source_focal = None
        if focal_value is not None:
            candidate = float(np.asarray(focal_value).reshape(()))
            if np.isfinite(candidate) and candidate > 0:
                source_focal = candidate
        image_size_wh = values.get("img_size_wh")
        height, width = image_hw(image_size_wh, K)
        corrected_vertices = np.empty_like(vertices, dtype=np.float32)
        corrected_joints = np.empty_like(joints, dtype=np.float32)
        factors = np.empty(len(vertices), dtype=np.float32)
        raw_factors = np.empty(len(vertices), dtype=np.float32)
        target_depths = np.full(len(vertices), np.nan, dtype=np.float32)
        front_depths = np.full(len(vertices), np.nan, dtype=np.float32)
        measured_flags = np.zeros(len(vertices), dtype=bool)
        reliable_flags = np.zeros(len(vertices), dtype=bool)
        interpolated_flags = np.zeros(len(vertices), dtype=bool)
        smoothed_flags = np.zeros(len(vertices), dtype=bool)

        for index in range(len(vertices)):
            measurement = measurements.get((frame, index))
            correction = scale_corrections.get((frame, index))
            if correction is None:
                raise RuntimeError(
                    f"frame {frame:06d} hand {index} has no scale correction")
            factor = correction.applied_factor
            if args.scale_mode == "per-frame" and measurement is not None:
                per_frame_exact += 1
            elif args.scale_mode == "per-frame":
                per_frame_without_measurement += 1
            if correction.source.startswith("global_fallback"):
                per_frame_global_fallback += 1
            vertices_da3 = convert_hawor_points_to_da3(
                vertices[index], source_focal, image_size_wh, K,
                (height, width))
            joints_da3 = convert_hawor_points_to_da3(
                joints[index], source_focal, image_size_wh, K,
                (height, width))
            corrected_vertices[index] = (
                vertices_da3 * factor).astype(np.float32)
            corrected_joints[index] = (
                joints_da3 * factor).astype(np.float32)
            factors[index] = factor
            raw_factors[index] = correction.raw_factor
            reliable_flags[index] = correction.reliable_anchor
            interpolated_flags[index] = correction.interpolated
            smoothed_flags[index] = correction.smoothed
            if measurement is not None:
                target_depths[index] = measurement.target_depth
                front_depths[index] = measurement.front_depth
                measured_flags[index] = True

        values["verts"] = corrected_vertices
        values["joints"] = corrected_joints
        values["cam_t"] = corrected_joints[:, 0].copy()
        values["depth_scale_factor"] = factors
        values["raw_depth_scale_factor"] = raw_factors
        values["target_surface_depth"] = target_depths
        values["hand_front_surface_depth_mean"] = front_depths
        values["contact_scale_measured"] = measured_flags
        values["temporal_scale_reliable_anchor"] = reliable_flags
        values["temporal_scale_interpolated"] = interpolated_flags
        values["temporal_scale_smoothed"] = smoothed_flags
        values["da3_intrinsics"] = K.astype(np.float32)
        values["depth_scale_mode"] = np.asarray(args.scale_mode)
        values["depth_scale_source"] = np.asarray(args.scale_source)
        values["temporal_regularization"] = np.bool_(
            args.temporal_regularization and args.scale_mode == "per-frame")
        if focal_value is not None:
            values["source_focal_length"] = np.asarray(focal_value).copy()
        if image_size_wh is not None:
            values["source_img_size_wh"] = np.asarray(image_size_wh).copy()
        values["focal_length"] = np.float32(np.sqrt(K[0, 0] * K[1, 1]))
        values["img_size_wh"] = np.asarray([width, height], dtype=np.int32)

        values["bbox"] = np.stack([
            project_bbox(hand, K, height, width)
            for hand in corrected_vertices
        ]).astype(np.float32)

        values.pop("verts_world", None)
        values.pop("joints_world", None)
        if frame in camera_poses:
            rotation, translation = camera_poses[frame]
            values["verts_world"] = (
                corrected_vertices.astype(np.float64) @ rotation.T
                + translation).astype(np.float32)
            values["joints_world"] = (
                corrected_joints.astype(np.float64) @ rotation.T
                + translation).astype(np.float32)
            world_frames_written += 1

        np.savez_compressed(per_frame_out / path.name, **values)
        files_written += 1
        instances_written += len(vertices)

    config = dict(source_config)
    da3_focals = [float(np.sqrt(K[0, 0] * K[1, 1]))
                  for K in intrinsics.values()]
    config.update({
        "stage": "13_scale_hawor",
        "source_stage": source_config.get("stage", "12_hawor_hands"),
        "source_hawor": relative_path(input_root, scene_dir),
        "depth_scale_metadata": relative_path(
            output_root / "depth_scale.json", scene_dir),
        "depth_scale_mode": args.scale_mode,
        "depth_scale_factors_by_side": side_scales,
        "temporal_regularization": bool(
            args.temporal_regularization and args.scale_mode == "per-frame"),
        "temporal_window": args.temporal_window,
        "depth_calibration_frames": args.calibration_frames,
        "intrinsics_source": "da3/intrinsics.npz",
        "scale_source": args.scale_source,
        "depth_statistic": args.depth_statistic,
        "object_mesh_source": (
            f"{args.object_root}/{args.object_label}"
            if args.scale_source == "object" else None),
        "object_candidate_index": (
            args.object_candidate_idx if args.scale_source == "object"
            else None),
        "contact_offset": args.contact_offset,
        "source_hawor_img_focal": source_config.get("img_focal"),
        "img_focal": float(np.median(da3_focals)),
        "focal_source": "da3_per_frame_intrinsics_geometric_mean_median",
        "world_frame_baked": bool(
            files_written and world_frames_written == files_written),
        "overlay_video": None,
        "n_frames_with_hands": files_written,
        "total_hands": instances_written,
    })
    with (output_root / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    metadata["files_written"] = files_written
    metadata["hand_instances_written"] = instances_written
    metadata["per_frame_exact_corrections"] = per_frame_exact
    metadata["per_frame_without_measurement"] = per_frame_without_measurement
    metadata["per_frame_global_fallbacks"] = per_frame_global_fallback
    metadata["temporal_interpolated_corrections"] = sum(
        item.interpolated for item in scale_corrections.values())
    metadata["temporal_smoothed_corrections"] = sum(
        item.smoothed for item in scale_corrections.values())
    metadata["world_frames_written"] = world_frames_written
    metadata["world_frame_baked_for_all_output_frames"] = bool(
        files_written and world_frames_written == files_written)
    metadata["output_skips"] = output_skips
    with (output_root / "depth_scale.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def run(args):
    scene_dir = args.scene_dir.expanduser().resolve()
    validate_dir_name(args.input_name, "--input-name")
    validate_dir_name(args.out_name, "--out-name")
    if args.scale_source == "object":
        validate_dir_name(args.object_root, "--object-root")
        validate_dir_name(args.object_label, "--object-label")
    if args.depth_statistic is None:
        # The object mesh is a clean surface, so its mean keeps the historical
        # behaviour.  Depth maps and point maps carry outliers where the hand
        # silhouette overlaps background, so they default to the median.
        args.depth_statistic = (
            "mean" if args.scale_source == "object" else "median")
    if args.input_name == args.out_name:
        raise RuntimeError("input and output names must differ (source is preserved)")
    if args.min_contact_pixels <= 0:
        raise RuntimeError("--min-contact-pixels must be positive")
    if args.max_object_samples <= 0:
        raise RuntimeError("--max-object-samples must be positive")
    if args.object_candidate_idx is not None and args.object_candidate_idx < 0:
        raise RuntimeError("--object-candidate-idx must be non-negative")
    if not np.isfinite(args.contact_offset) or args.contact_offset < 0:
        raise RuntimeError("--contact-offset must be finite and non-negative")
    if args.calibration_step <= 0:
        raise RuntimeError("--calibration-step must be positive")
    if args.mad_threshold < 0:
        raise RuntimeError("--mad-threshold cannot be negative")
    if args.temporal_window <= 0 or args.temporal_window % 2 == 0:
        raise RuntimeError("--temporal-window must be a positive odd integer")
    if (args.start_frame is not None and args.end_frame is not None
            and args.start_frame > args.end_frame):
        raise RuntimeError("--start-frame cannot exceed --end-frame")

    input_root = scene_dir / args.input_name
    output_root = scene_dir / args.out_name
    per_frame = input_root / "per_frame"
    if not per_frame.is_dir():
        raise RuntimeError(f"missing Stage-12 per-frame directory: {per_frame}")
    frame_paths = sorted(per_frame.glob("*.npz"))
    if not frame_paths:
        raise RuntimeError(f"no hand frames under {per_frame}")
    if output_root.exists() and not args.overwrite and not args.dry_run:
        raise RuntimeError(f"{output_root} exists; pass --overwrite to replace")

    intrinsics = load_intrinsics(scene_dir / "da3" / "intrinsics.npz")
    camera_poses = load_camera_poses(scene_dir / "da3" / "cameras.npz")
    if not camera_poses:
        raise RuntimeError(
            "DA3 cameras.npz is required to relate object and hand frames")
    faces_by_side = load_faces(input_root)
    source_config = read_json(input_root / "config.json")
    contact_source, object_root, object_candidates = build_contact_source(
        scene_dir, args, camera_poses)

    print(f"scene:              {scene_dir.name}")
    print(f"hand frames:        {len(frame_paths)}")
    print(f"scale mode:         {args.scale_mode}")
    print(f"scale source:       {args.scale_source}")
    print(f"depth statistic:    {args.depth_statistic}")
    print(f"calibration frames: {args.calibration_frames}")
    if args.scale_source == "object":
        print(f"object mesh:        {args.object_root}/{args.object_label} "
              f"({len(object_candidates)} candidate(s))")
    elif args.scale_source == "depth":
        print("depth maps:         da3/depth/<frame>.npy")
    else:
        print(f"point map:          da3/pointmap_ref.npy "
              f"({len(contact_source.samples_world)} samples)")
    print(f"method: {contact_source.description}, reduced by "
          f"{args.depth_statistic}, applied as a coupled camera-ray scale "
          "(no registration)")
    if args.dry_run:
        print("dry run: no files will be written")

    measurements: dict[tuple[int, int], HandMeasurement] = {}
    failures = []
    eligible_counter = {"left": 0, "right": 0}
    calibration_keys = set()
    for path in frame_paths:
        try:
            frame, values = load_frame_record(path)
        except RuntimeError as exc:
            failures.append({"file": str(path), "error": str(exc)})
            print(f"[{path.stem}] SKIP: {exc}")
            continue
        if frame not in intrinsics:
            failures.append({"frame": frame, "error": "missing DA3 intrinsics"})
            print(f"[{frame:06d}] SKIP: missing DA3 intrinsics")
            continue
        if args.start_frame is not None and frame < args.start_frame:
            continue
        if args.end_frame is not None and frame > args.end_frame:
            continue
        vertices = np.asarray(values["verts"], dtype=np.float64)
        is_right = np.asarray(values["is_right"], dtype=bool).reshape(-1)
        valid = np.asarray(
            values.get("valid", np.ones(len(vertices), dtype=bool)),
            dtype=bool).reshape(-1)
        if len(valid) != len(vertices):
            failures.append({"frame": frame, "error": "invalid valid array length"})
            print(f"[{frame:06d}] SKIP: invalid valid array length")
            continue
        focal_value = values.get("focal_length")
        source_focal = None
        if focal_value is not None:
            candidate = float(np.asarray(focal_value).reshape(()))
            if np.isfinite(candidate) and candidate > 0:
                source_focal = candidate
        image_size_wh = values.get("img_size_wh")

        for index in range(len(vertices)):
            side = "right" if bool(is_right[index]) else "left"
            calibration_eligible = (
                args.calibration_frames == "all" or bool(valid[index]))
            selected_for_calibration = False
            if calibration_eligible:
                eligible_counter[side] += 1
                selected_for_calibration = (
                    (eligible_counter[side] - 1) % args.calibration_step == 0)
            should_measure = (
                args.scale_mode == "per-frame" or selected_for_calibration)
            if not should_measure:
                continue
            try:
                measurement = measure_hand(
                    frame, index, side, bool(valid[index]),
                    vertices[index], faces_by_side[side], intrinsics[frame],
                    args.min_contact_pixels, source_focal, image_size_wh,
                    contact_source, camera_poses.get(frame),
                    args.contact_offset, args.depth_statistic)
                measurements[(frame, index)] = measurement
                if selected_for_calibration:
                    calibration_keys.add((frame, index))
                detail = (
                    f"object_frame={measurement.object_candidate_frame:06d}"
                    if measurement.object_candidate_frame is not None
                    else f"support={measurement.depth_support}")
                print(
                    f"[{frame:06d}] {side}[{index}] "
                    f"z_hand={measurement.front_depth:.6g} "
                    f"z_target={measurement.target_depth:.6g} "
                    f"scale={measurement.ratio:.6g} "
                    f"contact_pixels={measurement.contact_pixels} "
                    f"{detail}")
            except Exception as exc:
                failures.append({
                    "frame": frame,
                    "hand_index": index,
                    "side": side,
                    "detected": bool(valid[index]),
                    "error": str(exc),
                })
                print(f"[{frame:06d}] {side}[{index}] SKIP: {exc}")

    instances: dict[tuple[int, int], str] = {}
    for path in frame_paths:
        try:
            frame, values = load_frame_record(path)
        except RuntimeError:
            continue
        if frame not in intrinsics:
            continue
        if camera_points_skip_reason(
                np.asarray(values["verts"]),
                np.asarray(values["joints"])) is not None:
            continue
        for index, value in enumerate(
                np.asarray(values["is_right"], dtype=bool).reshape(-1)):
            key = (frame, index)
            if key in instances:
                raise RuntimeError(
                    f"duplicate hand instance at frame {frame:06d} index {index}")
            instances[key] = "right" if value else "left"
    sides_present = set(instances.values())
    side_scales = {}
    global_details = {}
    used_keys = set()
    for side in sorted(sides_present):
        calibration = [
            item for item in measurements.values()
            if item.side == side
            and (item.frame, item.hand_index) in calibration_keys
        ]
        if not calibration:
            # Per-frame mode may have measured only in-filled frames while the
            # requested detected calibration range was empty.  Falling back to
            # every usable measurement is preferable to silently leaving a side
            # unscaled, and is recorded in the metadata.
            calibration = [item for item in measurements.values()
                           if item.side == side]
            fallback = True
        else:
            fallback = False
        if not calibration:
            raise RuntimeError(
                f"no usable object-contact measurements for {side} hand; "
                "inspect the projected hand/object coverage")
        factor, keep, raw_median, mad = robust_scale(
            calibration, args.mad_threshold)
        side_scales[side] = factor
        kept = [item for item, flag in zip(calibration, keep) if flag]
        used_keys.update((item.frame, item.hand_index) for item in kept)
        global_details[side] = {
            "depth_scale_factor": factor,
            "measurement_count": len(calibration),
            "inlier_count": int(keep.sum()),
            "raw_ratio_median": raw_median,
            "ratio_mad": mad,
            "calibration_fell_back_to_all_measured_frames": fallback,
            "frame_min": min(item.frame for item in calibration),
            "frame_max": max(item.frame for item in calibration),
        }
        print(
            f"{side} global scale: {factor:.6g} "
            f"({int(keep.sum())}/{len(calibration)} inlier measurements)")

    scale_corrections = build_scale_corrections(
        instances, measurements, side_scales, args.scale_mode,
        args.temporal_regularization, args.temporal_window)
    temporal_applied = bool(
        args.temporal_regularization and args.scale_mode == "per-frame")
    interpolated_count = sum(
        item.interpolated for item in scale_corrections.values())
    smoothed_count = sum(
        item.smoothed for item in scale_corrections.values())
    if args.scale_mode == "per-frame":
        print(
            "temporal correction: "
            f"{'enabled' if temporal_applied else 'disabled'} "
            f"({interpolated_count} interpolated/edge-filled, "
            f"{smoothed_count} smoothed)")

    metadata = {
        "stage": "13_scale_hawor",
        "method": (
            f"mano_to_{args.scale_source}_surface_contact_"
            "pinhole_uniform_scale"),
        "registration_used": False,
        "source_hawor": relative_path(input_root, scene_dir),
        "output_hawor": relative_path(output_root, scene_dir),
        "intrinsics_source": "da3/intrinsics.npz",
        "scale_source": args.scale_source,
        "target_surface": contact_source.description,
        "depth_statistic": args.depth_statistic,
        "depth_map_used": args.scale_source == "depth",
        "pointmap_used": args.scale_source == "pointmap",
        "depth_map_source": (
            "da3/depth/<frame>.npy" if args.scale_source == "depth" else None),
        "pointmap_source": (
            "da3/pointmap_ref.npy" if args.scale_source == "pointmap"
            else None),
        "pointmap_samples": (
            len(contact_source.samples_world)
            if args.scale_source == "pointmap" else None),
        "object_mesh_root": (
            None if object_root is None
            else relative_path(object_root, scene_dir)),
        "object_candidate_index": (
            args.object_candidate_idx if args.scale_source == "object"
            else None),
        "object_candidates": [
            {
                "frame": item.frame,
                "candidate": item.candidate,
                "mesh": relative_path(item.mesh_path, scene_dir),
                "pose": relative_path(item.pose_path, scene_dir),
                "zbuffer_samples": len(item.samples_world),
                "surface_faces": len(item.face_centers_world),
            }
            for item in object_candidates
        ],
        "target_depth_statistic": (
            f"{args.depth_statistic} of {contact_source.description}"
        ),
        "hand_front_surface_definition": (
            "nearest perspective-correct MANO triangle depth per projected pixel"
        ),
        "intrinsics_conversion": (
            "HaWoR focal/image-center pixels reinterpreted as per-frame DA3 rays"
        ),
        "scale_mode": args.scale_mode,
        "temporal_regularization_requested": bool(
            args.temporal_regularization),
        "temporal_regularization_applied": temporal_applied,
        "temporal_window": args.temporal_window,
        "temporal_anchor_definition": (
            "detected hand with object front samples under its silhouette"
        ),
        "temporal_interpolation": (
            "linear in log scale between reliable anchors; nearest reliable "
            "anchor outside their frame range"
        ),
        "temporal_smoothing": (
            "centered median followed by centered triangular mean in log scale"
        ),
        "calibration_frames": args.calibration_frames,
        "calibration_step": args.calibration_step,
        "calibration_start_frame": args.start_frame,
        "calibration_end_frame": args.end_frame,
        "minimum_contact_pixels": args.min_contact_pixels,
        "maximum_object_samples": args.max_object_samples,
        "contact_offset": args.contact_offset,
        "mad_threshold": args.mad_threshold,
        "global_scale_by_side": global_details,
        "measurements": [
            json_measurement(item, (item.frame, item.hand_index) in used_keys)
            for item in sorted(
                measurements.values(),
                key=lambda value: (value.frame, value.hand_index))
        ],
        "scale_corrections": [
            json_scale_correction(item)
            for item in sorted(
                scale_corrections.values(),
                key=lambda value: (value.frame, value.hand_index))
        ],
        "temporal_interpolated_corrections": interpolated_count,
        "temporal_smoothed_corrections": smoothed_count,
        "measurement_failures": failures,
        "camera_arrays_scaled_about_origin": True,
        "source_projection_preserved_under_da3_intrinsics": True,
        "uniform_scaling_projection_unchanged": True,
        "da3_camera_poses_available": bool(camera_poses),
    }

    if args.dry_run:
        print(f"measurements:       {len(measurements)}")
        print(f"measurement skips: {len(failures)}")
        return
    write_scaled_output(
        scene_dir, input_root, output_root, frame_paths, intrinsics,
        camera_poses, faces_by_side, measurements, side_scales,
        scale_corrections,
        args, source_config, metadata)
    print(f"wrote {output_root}")


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        if getattr(args, "dry_run", False):
            traceback.print_exc()
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
