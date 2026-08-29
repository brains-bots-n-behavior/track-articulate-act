#!/usr/bin/env python
"""Stage 61: render HaWoR hands and hand-driven SegviGen articulation.

This Viser viewer combines three outputs after converting the HaWoR world
trajectory into Stage 31's registered reference-mesh coordinate frame:

* the dense HaWoR MANO hand trajectory (``hawor`` or ``hawor_scaled``),
* registered Stage-31 SegviGen static and moving meshes, and
* the sparse prismatic/revolute estimate from ``simple_joint/joints.json``.

The current DA3 camera is also shown as an animated OpenCV/RDF frustum.  With
``--export-video``, the first connected Viser browser renders every animation
frame from the exact DA3 camera pose and full pinhole intrinsics while the
interactive server remains live.  A small overscan-and-remap step preserves
``fx``, ``fy``, skew, and an off-center principal point even though Viser's
native browser camera has a centered, square-pixel projection.

Stage 31 only contains moving meshes at a few reconstruction frames.  The
viewer therefore uses either the first or last registered moving mesh as a
canonical asset and drives it at every video frame.  Sparse Stage-35 joint
states remain exact anchors.  Between anchors, motion timing comes from the
HaWoR wrist velocity relative to the joint:

* prismatic: linear wrist velocity projected onto the joint axis;
* revolute: signed orbital angular velocity of the wrist about the joint axis.

The velocity is normalized independently between neighboring sparse anchors,
which preserves every native joint state while using the hand motion to fill
the missing frames.  Before the first anchor and after the last anchor, the
nearest native state is held.  The static mesh is loaded once and never moved.

Examples::

    python scripts/61_render_all.py --scene-dir data/trashbin
    python scripts/61_render_all.py --scene-dir data/trashbin --mesh-frame last
    python scripts/61_render_all.py --scene-dir data/trashbin --play
    python scripts/61_render_all.py --scene-dir data/trashbin --export-video
    python scripts/61_render_all.py --scene-dir data/trashbin --dry-run

Run the live viewer in an environment containing numpy, Pillow, trimesh, and
viser; ``--export-video`` additionally needs OpenCV. ``--dry-run`` does not
import viser or OpenCV.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from articulation_estimation.hand_replay import (  # noqa: E402
    HandSample,
    HandTrajectory,
    MANO_BONES,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JOINT_COLORS = {
    "revolute": (35, 180, 75),
    "prismatic": (50, 110, 230),
}
HAND_COLORS = {
    "left": (75, 155, 245),
    "right": (245, 145, 65),
}
HAND_INFILL_COLORS = {
    "left": (110, 135, 165),
    "right": (165, 135, 105),
}
MOVING_COLORS = [
    (235, 145, 65),
    (85, 155, 235),
    (150, 105, 210),
    (65, 185, 145),
    (225, 90, 120),
]


@dataclass(frozen=True)
class MeshData:
    """One registered triangle mesh with all GLB node transforms baked in."""

    label: str
    frame_id: int | None
    vertices: np.ndarray
    faces: np.ndarray
    source: str
    geometry: str

    @property
    def diagonal(self) -> float:
        if len(self.vertices) == 0:
            return 0.0
        return float(np.linalg.norm(
            self.vertices.max(axis=0) - self.vertices.min(axis=0)))


@dataclass(frozen=True)
class RegisteredMeshes:
    """Shared static mesh plus first/last moving meshes for every label."""

    static: MeshData
    moving: dict[str, dict[str, MeshData]]
    available_frames: dict[str, tuple[int, ...]]
    registered_mesh_path: Path
    metadata_path: Path
    reference_to_world: np.ndarray


@dataclass(frozen=True)
class DenseJointMotion:
    """A scalar joint trajectory evaluated on the viewer timeline."""

    frame_ids: np.ndarray
    states: np.ndarray
    relative_velocity: np.ndarray
    driver_side: str
    velocity_kind: str
    segment_modes: tuple[tuple[int, int, str], ...]

    def state_at(self, frame_id: int) -> float:
        return float(np.interp(
            float(frame_id), self.frame_ids.astype(np.float64), self.states))


@dataclass(frozen=True)
class CameraSample:
    """One DA3 pinhole camera expressed in Stage 31 reference coordinates."""

    frame_id: int
    camera_to_reference: np.ndarray
    intrinsics: np.ndarray
    native: bool


@dataclass(frozen=True)
class Da3CameraTrajectory:
    """Direct-index DA3 camera/intrinsics records for the Viser timeline."""

    frame_ids: np.ndarray
    camera_to_reference: np.ndarray
    intrinsics: np.ndarray


@dataclass(frozen=True)
class CameraRenderSpec:
    """Centered Viser render plus remap implementing one full DA3 K matrix."""

    output_hw: tuple[int, int]
    render_hw: tuple[int, int]
    vertical_fov: float
    map_x: np.ndarray
    map_y: np.ndarray
    scaled_intrinsics: np.ndarray


@dataclass(frozen=True)
class PreparedScene:
    scene_dir: Path
    hawor_name: str
    hands: HandTrajectory
    joints: dict[str, dict]
    meshes: RegisteredMeshes
    frame_ids: np.ndarray
    driver_side: str
    motions: dict[str, DenseJointMotion]
    playback_fps: float
    cameras: Da3CameraTrajectory
    frame_paths: dict[int, Path]
    image_hw: tuple[int, int]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--scene-dir", type=Path, required=True,
        help="Scene directory, for example data/trashbin",
    )
    parser.add_argument(
        "--hawor-name", default="auto",
        help=("Hand directory within the scene. 'auto' prefers hawor_scaled "
              "when present, then hawor"),
    )
    parser.add_argument(
        "--mesh-frame", choices=("first", "last"), default="first",
        help="Initial canonical registered moving mesh (also switchable in Viser)",
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Moving-label subset (default: every simple_joint label)",
    )
    parser.add_argument(
        "--driver-hand", choices=("auto", "left", "right"), default="auto",
        help="HaWoR hand side whose relative velocity drives every moving part",
    )
    parser.add_argument(
        "--hand-coordinate-source", choices=("auto", "world", "camera"),
        default="auto",
        help="Prefer saved world hands or force DA3 camera-to-world conversion",
    )
    parser.add_argument(
        "--detected-only", action="store_true",
        help="Exclude HaWoR in-filled samples when constructing the hand track",
    )
    parser.add_argument("--start-frame", type=int, help="First displayed frame")
    parser.add_argument("--end-frame", type=int, help="Last displayed frame")
    parser.add_argument(
        "--velocity-smoothing", type=int, default=5,
        help="Centered moving-average width for relative interval velocities",
    )
    parser.add_argument(
        "--playback-fps", type=float,
        help="Playback rate; defaults to HaWoR overlay FPS, then 24",
    )
    parser.add_argument("--play", action="store_true",
                        help="Start playback immediately")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop rather than wrap at the final frame")
    parser.add_argument("--host", default="0.0.0.0", help="Viser bind address")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument(
        "--axis-length-scale", type=float, default=1.25,
        help="Joint-axis half-length in moving-part diagonals",
    )
    parser.add_argument(
        "--axis-radius-scale", type=float, default=0.025,
        help="Joint-axis radius in moving-part diagonals",
    )
    parser.add_argument(
        "--camera-scale", type=float, default=0.18,
        help="DA3 frustum depth in scene diagonals",
    )
    parser.add_argument(
        "--export-video", type=Path, nargs="?",
        const=Path("render_all/da3_camera.mp4"),
        metavar="PATH",
        help=("Capture an MP4 through the DA3 camera when the first browser "
              "connects; without PATH, write render_all/da3_camera.mp4 "
              "inside the scene"),
    )
    parser.add_argument(
        "--video-fps", type=float,
        help="Export FPS (default: the playback FPS)",
    )
    parser.add_argument(
        "--video-max-side", type=int, default=720,
        help="Maximum exported width or height; DA3 intrinsics are rescaled exactly",
    )
    parser.add_argument(
        "--video-codec", default="mp4v",
        help="Four-character OpenCV codec for --export-video",
    )
    parser.add_argument(
        "--render-timeout", type=float, default=30.0,
        help="Seconds to wait for each browser-rendered video frame",
    )
    parser.add_argument(
        "--overwrite-video", action="store_true",
        help="Atomically replace an existing --export-video output",
    )
    parser.add_argument("--no-hands", action="store_true",
                        help="Hide hand surfaces initially")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Hide hand skeletons initially")
    parser.add_argument("--no-meshes", action="store_true",
                        help="Hide registered object meshes initially")
    parser.add_argument("--no-joints", action="store_true",
                        help="Hide simple_joint axes initially")
    parser.add_argument("--no-camera", action="store_true",
                        help="Hide the animated DA3 camera frustum initially")
    parser.add_argument("--no-grid", action="store_true",
                        help="Hide the ground grid initially")
    parser.add_argument(
        "--viser-timeout", type=float, default=0.0,
        help="Exit after this many seconds; zero waits for Ctrl-C",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and summarize all inputs without starting Viser",
    )
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _safe_directory_name(value: str, flag: str):
    if not value or Path(value).name != value or value in {".", ".."}:
        raise ValueError(f"{flag} must be one directory name")


def _resolve_scene_path(scene_dir: Path, value: str | None,
                        fallback: Path, description: str) -> Path:
    path = fallback if not value else Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (scene_dir / path).resolve()
    try:
        path.relative_to(scene_dir)
    except ValueError as exc:
        raise ValueError(f"{description} is outside the scene directory: {path}") from exc
    return path


def _quat_wxyz_to_rotation(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("SAM3D quaternion is zero or non-finite")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _quat_xyzw_to_rotation(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("DA3 camera quaternion is zero or non-finite")
    x, y, z, w = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def _sam3d_pose_to_rdf_camera(path: Path) -> np.ndarray:
    """Reproduce Stage 31's raw-SAM3D-GLB to RDF-camera transform."""
    pose = _read_json(path)
    try:
        rotation = _quat_wxyz_to_rotation(pose["rotation_quat_wxyz"])
        translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        scales = np.asarray(pose["scale"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid SAM3D pose in {path}") from exc
    if len(scales) not in (1, 3):
        raise ValueError(f"invalid SAM3D scale in {path}: {scales}")
    if len(scales) == 1:
        scales = np.repeat(scales, 3)
    if (not np.isfinite(translation).all() or not np.isfinite(scales).all()
            or np.any(scales <= 0)):
        raise ValueError(f"invalid SAM3D translation/scale in {path}")
    model_from_glb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    pytorch3d_to_rdf = np.diag([-1.0, -1.0, 1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        pytorch3d_to_rdf @ rotation.T @ np.diag(scales) @ model_from_glb)
    transform[:3, 3] = pytorch3d_to_rdf @ translation
    return transform


def _load_camera_to_world_direct(scene_dir: Path) -> dict[int, np.ndarray]:
    """Load DA3 poses with Stage 31/60a's direct numeric-frame convention."""
    path = scene_dir / "da3" / "cameras.npz"
    if not path.is_file():
        return {}
    try:
        with np.load(path) as cameras:
            required = {"frame_indices", "cam_quats_xyzw", "cam_trans"}
            if not required.issubset(cameras.files):
                raise ValueError(f"invalid DA3 camera archive: {path}")
            frame_ids = np.asarray(cameras["frame_indices"]).reshape(-1)
            quaternions = np.asarray(cameras["cam_quats_xyzw"])
            translations = np.asarray(cameras["cam_trans"])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not (len(frame_ids) == len(quaternions) == len(translations)):
        raise ValueError(f"camera array lengths differ in {path}")
    result = {}
    for frame_id, quaternion, translation in zip(
            frame_ids, quaternions, translations):
        frame = int(frame_id)
        if frame in result:
            raise ValueError(f"duplicate DA3 camera frame {frame:06d} in {path}")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _quat_xyzw_to_rotation(quaternion)
        transform[:3, 3] = np.asarray(
            translation, dtype=np.float64).reshape(3)
        if not np.isfinite(transform).all():
            raise ValueError(f"non-finite DA3 pose for frame {frame:06d}")
        result[frame] = transform
    return result


def _load_intrinsics_direct(scene_dir: Path) -> dict[int, np.ndarray]:
    """Load DA3 intrinsics with the same direct numeric-frame convention."""
    path = scene_dir / "da3" / "intrinsics.npz"
    if not path.is_file():
        raise FileNotFoundError(f"missing DA3 intrinsics: {path}")
    try:
        with np.load(path) as archive:
            required = {"frame_indices", "intrinsics"}
            if not required.issubset(archive.files):
                raise ValueError(f"invalid DA3 intrinsics archive: {path}")
            frame_ids = np.asarray(archive["frame_indices"]).reshape(-1)
            matrices = np.asarray(archive["intrinsics"])
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if len(frame_ids) != len(matrices):
        raise ValueError(f"intrinsics array lengths differ in {path}")
    result = {}
    for frame_id, value in zip(frame_ids, matrices):
        K = np.asarray(value, dtype=np.float64).reshape(3, 3)
        linear = K[:2, :2]
        if (not np.isfinite(K).all()
                or K[0, 0] <= 0.0 or K[1, 1] <= 0.0
                or abs(float(np.linalg.det(linear))) < 1e-12
                or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-5)):
            raise ValueError(
                f"invalid DA3 intrinsics for frame {int(frame_id):06d}")
        frame = int(frame_id)
        if frame in result:
            raise ValueError(f"duplicate DA3 intrinsics frame {frame:06d}")
        result[frame] = K
    return result


def _rotation_to_wxyz(value: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=2e-5)):
        raise ValueError("camera rotation is not a proper orthonormal matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array([
            0.25 * root,
            (matrix[2, 1] - matrix[1, 2]) / root,
            (matrix[0, 2] - matrix[2, 0]) / root,
            (matrix[1, 0] - matrix[0, 1]) / root,
        ])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            root = np.sqrt(1.0 + matrix[0, 0]
                           - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                (matrix[2, 1] - matrix[1, 2]) / root,
                0.25 * root,
                (matrix[0, 1] + matrix[1, 0]) / root,
                (matrix[0, 2] + matrix[2, 0]) / root,
            ])
        elif index == 1:
            root = np.sqrt(1.0 + matrix[1, 1]
                           - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array([
                (matrix[0, 2] - matrix[2, 0]) / root,
                (matrix[0, 1] + matrix[1, 0]) / root,
                0.25 * root,
                (matrix[1, 2] + matrix[2, 1]) / root,
            ])
        else:
            root = np.sqrt(1.0 + matrix[2, 2]
                           - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array([
                (matrix[1, 0] - matrix[0, 1]) / root,
                (matrix[0, 2] + matrix[2, 0]) / root,
                (matrix[1, 2] + matrix[2, 1]) / root,
                0.25 * root,
            ])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def _slerp_wxyz(first: np.ndarray, second: np.ndarray,
                 alpha: float) -> np.ndarray:
    """Shortest-path spherical interpolation between WXYZ quaternions."""
    before = np.asarray(first, dtype=np.float64).reshape(4)
    after = np.asarray(second, dtype=np.float64).reshape(4)
    before /= np.linalg.norm(before)
    after /= np.linalg.norm(after)
    cosine = float(before @ after)
    if cosine < 0.0:
        after *= -1.0
        cosine *= -1.0
    cosine = float(np.clip(cosine, -1.0, 1.0))
    if cosine > 0.9995:
        result = (1.0 - alpha) * before + alpha * after
    else:
        angle = np.arccos(cosine)
        result = (
            np.sin((1.0 - alpha) * angle) / np.sin(angle) * before
            + np.sin(alpha * angle) / np.sin(angle) * after
        )
    result /= np.linalg.norm(result)
    return result.astype(np.float32)


def _reference_similarity(reference_to_world: np.ndarray
                          ) -> tuple[float, np.ndarray, np.ndarray]:
    """Decompose Stage 31's reference-to-world affine as uniform s*R + t.

    A rigid Viser camera can reproduce the DA3 projection in the registered
    reference frame exactly only when this transform is a similarity.  SAM3D
    emits a three-vector scale, so validate instead of silently discarding a
    possible anisotropic component.
    """
    transform = np.asarray(reference_to_world, dtype=np.float64).reshape(4, 4)
    linear = transform[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(scales))
    if (not np.isfinite(scale) or scale <= 1e-12
            or not np.allclose(scales, scale, rtol=2e-5, atol=1e-8)):
        raise ValueError(
            "DA3 camera replay requires the Stage-31 reference transform to "
            f"have uniform scale; column scales are {scales.tolist()}")
    rotation = linear / scale
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-5)):
        raise ValueError(
            "DA3 camera replay requires a proper similarity reference transform")
    return scale, rotation, transform[:3, 3].copy()


def load_da3_cameras(scene_dir: Path,
                     reference_to_world: np.ndarray) -> Da3CameraTrajectory:
    """Load direct-index DA3 cameras in Stage-31 reference coordinates."""
    camera_to_world = _load_camera_to_world_direct(scene_dir)
    if not camera_to_world:
        raise FileNotFoundError(
            f"missing or empty DA3 cameras: {scene_dir / 'da3' / 'cameras.npz'}")
    intrinsics = _load_intrinsics_direct(scene_dir)
    frame_ids = sorted(set(camera_to_world) & set(intrinsics))
    if not frame_ids:
        raise ValueError("DA3 cameras and intrinsics have no common frame indices")
    scale, reference_rotation, reference_translation = _reference_similarity(
        reference_to_world)
    transforms = []
    matrices = []
    for frame_id in frame_ids:
        camera_world = camera_to_world[frame_id]
        camera_reference = np.eye(4, dtype=np.float64)
        camera_reference[:3, :3] = (
            reference_rotation.T @ camera_world[:3, :3])
        camera_reference[:3, 3] = (
            reference_rotation.T
            @ (camera_world[:3, 3] - reference_translation)
        ) / scale
        # Validate and remove harmless archive round-off before interpolation.
        u, _, vt = np.linalg.svd(camera_reference[:3, :3])
        camera_reference[:3, :3] = u @ vt
        if np.linalg.det(camera_reference[:3, :3]) < 0.0:
            u[:, -1] *= -1.0
            camera_reference[:3, :3] = u @ vt
        _rotation_to_wxyz(camera_reference[:3, :3])
        transforms.append(camera_reference)
        matrices.append(intrinsics[frame_id])
    return Da3CameraTrajectory(
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        camera_to_reference=np.stack(transforms),
        intrinsics=np.stack(matrices),
    )


def camera_sample_at(cameras: Da3CameraTrajectory,
                     frame_id: int) -> CameraSample:
    """Return an exact DA3 sample or interpolate/hold a missing pose and K."""
    requested = int(frame_id)
    ids = cameras.frame_ids
    position = int(np.searchsorted(ids, requested))
    if position < len(ids) and int(ids[position]) == requested:
        return CameraSample(
            frame_id=requested,
            camera_to_reference=cameras.camera_to_reference[position].copy(),
            intrinsics=cameras.intrinsics[position].copy(),
            native=True,
        )
    if position == 0 or position == len(ids):
        index = 0 if position == 0 else len(ids) - 1
        return CameraSample(
            frame_id=requested,
            camera_to_reference=cameras.camera_to_reference[index].copy(),
            intrinsics=cameras.intrinsics[index].copy(),
            native=False,
        )
    first_id, last_id = int(ids[position - 1]), int(ids[position])
    alpha = (requested - first_id) / float(last_id - first_id)
    first = cameras.camera_to_reference[position - 1]
    last = cameras.camera_to_reference[position]
    quaternion = _slerp_wxyz(
        _rotation_to_wxyz(first[:3, :3]),
        _rotation_to_wxyz(last[:3, :3]),
        alpha,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_wxyz_to_rotation(quaternion)
    transform[:3, 3] = (
        (1.0 - alpha) * first[:3, 3] + alpha * last[:3, 3])
    K = ((1.0 - alpha) * cameras.intrinsics[position - 1]
         + alpha * cameras.intrinsics[position])
    K[2] = [0.0, 0.0, 1.0]
    return CameraSample(
        frame_id=requested,
        camera_to_reference=transform,
        intrinsics=K,
        native=False,
    )


def _stage31_reference_to_world(scene_dir: Path, metadata: dict) -> np.ndarray:
    """Recover the transform Stage 31 used for its reference SAM3D GLB."""
    try:
        reference_frame = int(metadata["reference_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("registered metadata has no valid reference_frame") from exc
    reference_record = next(
        (record for record in metadata.get("frames", [])
         if isinstance(record, dict)
         and int(record.get("frame", -1)) == reference_frame),
        None,
    )
    if reference_record is None or not reference_record.get("pose"):
        raise ValueError(
            "registered metadata has no SAM3D pose for its reference frame")
    pose_path = _resolve_scene_path(
        scene_dir, str(reference_record["pose"]),
        scene_dir / "__missing_reference_pose__", "reference SAM3D pose")
    if not pose_path.is_file():
        raise FileNotFoundError(f"missing reference SAM3D pose: {pose_path}")
    reference_to_world = _sam3d_pose_to_rdf_camera(pose_path)
    if metadata.get("camera_to_world_used"):
        cameras = _load_camera_to_world_direct(scene_dir)
        if reference_frame not in cameras:
            raise ValueError(
                f"DA3 cameras lack Stage-31 reference frame {reference_frame:06d}")
        reference_to_world = cameras[reference_frame] @ reference_to_world
    determinant = float(np.linalg.det(reference_to_world[:3, :3]))
    if (not np.isfinite(reference_to_world).all()
            or not np.isfinite(determinant) or abs(determinant) < 1e-12):
        raise ValueError("Stage-31 reference-to-world transform is singular")
    return reference_to_world


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return (points @ transform[:3, :3].T
            + transform[:3, 3]).astype(np.float32)


def _validate_joint(label: str, record: dict):
    joint_type = record.get("type")
    if joint_type not in {"prismatic", "revolute"}:
        raise ValueError(
            f"simple_joint label {label!r} has unsupported type {joint_type!r}")
    axis = np.asarray(record.get("axis_direction"), dtype=np.float64)
    if (axis.shape != (3,) or not np.isfinite(axis).all()
            or np.linalg.norm(axis) < 1e-10):
        raise ValueError(f"simple_joint label {label!r} has an invalid axis")
    if joint_type == "revolute":
        pivot = np.asarray(record.get("axis_point"), dtype=np.float64)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise ValueError(
                f"revolute simple_joint label {label!r} has an invalid axis_point")
    states = record.get("joint_states")
    if not isinstance(states, list) or len(states) < 2:
        raise ValueError(
            f"simple_joint label {label!r} needs at least two sparse states")
    parsed = []
    for item in states:
        if not isinstance(item, dict):
            raise ValueError(f"simple_joint label {label!r} has a malformed state")
        try:
            frame = int(item["frame"])
            state = float(item["q"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"simple_joint label {label!r} has a malformed state") from exc
        if not np.isfinite(state):
            raise ValueError(
                f"simple_joint label {label!r} has a non-finite state")
        parsed.append((frame, state))
    parsed.sort()
    if len({frame for frame, _ in parsed}) != len(parsed):
        raise ValueError(f"simple_joint label {label!r} has duplicate state frames")


def load_simple_joints(scene_dir: Path,
                       labels: list[str] | None = None) -> tuple[dict, dict[str, dict]]:
    """Load and validate the public Stage-35 joint payload."""
    path = scene_dir / "simple_joint" / "joints.json"
    payload = _read_json(path)
    if payload.get("stage") not in {None, "35_simple_joint"}:
        raise ValueError(f"unexpected stage in {path}: {payload.get('stage')!r}")
    if payload.get("coordinate_frame") not in {None, "stage31_reference_mesh"}:
        raise ValueError(
            f"{path} is not expressed in the Stage-31 reference frame")
    records = payload.get("joints")
    if not isinstance(records, dict) or not records:
        raise ValueError(f"{path} has no joints")
    requested = list(dict.fromkeys(labels if labels is not None else records))
    if not requested:
        raise ValueError("--labels selected no moving joints")
    missing = sorted(set(requested) - set(records))
    if missing:
        raise ValueError(
            f"simple_joint labels not found: {missing}; available: {sorted(records)}")
    selected = {}
    for label in requested:
        record = records[label]
        if not isinstance(record, dict):
            raise ValueError(f"simple_joint record {label!r} is not an object")
        _validate_joint(label, record)
        selected[label] = record
    return payload, selected


def load_hawor_trajectory(
        scene_dir: Path,
        hawor_name: str,
        coordinate_source: str = "auto",
        detected_only: bool = False,
        start_frame: int | None = None,
        end_frame: int | None = None) -> HandTrajectory:
    """Load HaWoR hands in DA3 world using the Stage-31 direct frame mapping.

    This intentionally differs from generic filename-offset inference: Stage
    31, Stage 60, and Stage 60a all index ``da3/cameras.npz`` directly by the
    numeric frame saved in their records.  Keeping that convention is required
    before converting the hands into ``stage31_reference_mesh`` coordinates.
    """
    if coordinate_source not in {"auto", "world", "camera"}:
        raise ValueError(
            f"unsupported --hand-coordinate-source {coordinate_source!r}")
    _safe_directory_name(hawor_name, "--hawor-name")
    root = scene_dir / hawor_name
    per_frame = root / "per_frame"
    faces_path = root / "faces.npy"
    if not per_frame.is_dir():
        raise FileNotFoundError(f"missing HaWoR frames: {per_frame}")
    if not faces_path.is_file():
        raise FileNotFoundError(f"missing MANO topology: {faces_path}")
    try:
        right_faces = np.asarray(np.load(faces_path), dtype=np.int32)
        left_path = root / "faces_left.npy"
        left_faces = (np.asarray(np.load(left_path), dtype=np.int32)
                      if left_path.is_file()
                      else right_faces[:, [0, 2, 1]].copy())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not load MANO topology under {root}: {exc}") from exc
    for side, faces in (("right", right_faces), ("left", left_faces)):
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f"invalid {side} MANO topology under {root}")

    camera_to_world = _load_camera_to_world_direct(scene_dir)
    samples: dict[str, dict[int, HandSample]] = {"left": {}, "right": {}}
    skipped_nonfinite = 0
    skipped_without_pose = 0
    paths = sorted(per_frame.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no HaWoR NPZ files under {per_frame}")
    for path in paths:
        try:
            with np.load(path) as record:
                frame_id = int(record["frame_idx"]) \
                    if "frame_idx" in record else int(path.stem)
                if start_frame is not None and frame_id < start_frame:
                    continue
                if end_frame is not None and frame_id > end_frame:
                    continue
                required = {"verts", "joints", "is_right"}
                if not required.issubset(record.files):
                    raise ValueError(
                        f"missing {sorted(required - set(record.files))}")
                vertices_camera = np.asarray(record["verts"], dtype=np.float32)
                joints_camera = np.asarray(record["joints"], dtype=np.float32)
                is_right = np.asarray(record["is_right"], dtype=bool).reshape(-1)
                valid = np.asarray(
                    record["valid"] if "valid" in record
                    else np.ones(len(is_right), dtype=bool),
                    dtype=bool,
                ).reshape(-1)
                vertices_world = (np.asarray(
                    record["verts_world"], dtype=np.float32)
                    if "verts_world" in record else None)
                joints_world = (np.asarray(
                    record["joints_world"], dtype=np.float32)
                    if "joints_world" in record else None)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read {path}: {exc}") from exc
        if (vertices_camera.ndim != 3 or vertices_camera.shape[2] != 3
                or joints_camera.ndim != 3 or joints_camera.shape[2] != 3
                or len(vertices_camera) != len(is_right)
                or len(joints_camera) != len(is_right)
                or len(valid) != len(is_right)):
            raise ValueError(f"invalid HaWoR array shapes in {path}")
        for hand_index in range(len(is_right)):
            if detected_only and not bool(valid[hand_index]):
                continue
            saved_world_ok = (
                vertices_world is not None and joints_world is not None
                and len(vertices_world) == len(is_right)
                and len(joints_world) == len(is_right)
                and np.isfinite(vertices_world[hand_index]).all()
                and np.isfinite(joints_world[hand_index]).all()
            )
            if coordinate_source in {"auto", "world"} and saved_world_ok:
                vertices = vertices_world[hand_index].copy()
                joints = joints_world[hand_index].copy()
                source = "HaWoR saved world"
            elif coordinate_source == "world":
                skipped_nonfinite += 1
                continue
            else:
                if (not np.isfinite(vertices_camera[hand_index]).all()
                        or not np.isfinite(joints_camera[hand_index]).all()):
                    skipped_nonfinite += 1
                    continue
                transform = camera_to_world.get(frame_id)
                if transform is None:
                    skipped_without_pose += 1
                    continue
                vertices = _transform_points(
                    vertices_camera[hand_index], transform)
                joints = _transform_points(joints_camera[hand_index], transform)
                source = "HaWoR camera + direct DA3 pose"
            side = "right" if bool(is_right[hand_index]) else "left"
            sample = HandSample(
                frame_id=frame_id,
                side=side,
                vertices=vertices,
                joints=joints,
                detected=bool(valid[hand_index]),
                coordinate_source=source,
            )
            previous = samples[side].get(frame_id)
            if previous is None or (sample.detected and not previous.detected):
                samples[side][frame_id] = sample
    ordered = {
        side: [by_frame[frame] for frame in sorted(by_frame)]
        for side, by_frame in samples.items()
    }
    result = HandTrajectory(
        samples=ordered,
        faces={"left": left_faces, "right": right_faces},
        skipped_nonfinite=skipped_nonfinite,
        skipped_without_pose=skipped_without_pose,
    )
    if not result.frame_ids:
        raise ValueError(
            "no finite HaWoR hands could be placed in DA3 world; "
            f"skipped {skipped_nonfinite} non-finite and "
            f"{skipped_without_pose} without direct DA3 poses")
    return result


def hands_world_to_reference(hands: HandTrajectory,
                             reference_to_world: np.ndarray) -> HandTrajectory:
    """Express all HaWoR world arrays in Stage 31's reference-mesh frame."""
    world_to_reference = np.linalg.inv(
        np.asarray(reference_to_world, dtype=np.float64).reshape(4, 4))
    converted = {}
    for side, samples in hands.samples.items():
        converted[side] = [
            HandSample(
                frame_id=sample.frame_id,
                side=sample.side,
                vertices=_transform_points(sample.vertices, world_to_reference),
                joints=_transform_points(sample.joints, world_to_reference),
                detected=sample.detected,
                coordinate_source=(sample.coordinate_source
                                   + " -> Stage-31 reference"),
            )
            for sample in samples
        ]
    return HandTrajectory(
        samples=converted,
        faces=hands.faces,
        skipped_nonfinite=hands.skipped_nonfinite,
        skipped_without_pose=hands.skipped_without_pose,
    )


def _mesh_arrays(mesh, path: Path, geometry: str) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
    faces = np.asarray(mesh.faces, dtype=np.int32).copy()
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0
            or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0):
        raise ValueError(f"{path} geometry {geometry!r} is not a triangle mesh")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path} geometry {geometry!r} has non-finite vertices")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(f"{path} geometry {geometry!r} has invalid face indices")
    return vertices, faces


def _load_named_geometry(scene, geometry_name: str, trimesh, path: Path):
    """Load one named GLB geometry with all scene-node transforms baked in."""
    instances = []
    for node_name in scene.graph.nodes_geometry:
        transform, node_geometry = scene.graph[node_name]
        if node_geometry != geometry_name:
            continue
        mesh = scene.geometry[node_geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            continue
        mesh.apply_transform(transform)
        instances.append(mesh)
    if not instances and geometry_name in scene.geometry:
        mesh = scene.geometry[geometry_name].copy()
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            instances.append(mesh)
    if not instances:
        raise ValueError(
            f"registered GLB {path} lacks geometry {geometry_name!r}")
    return (instances[0] if len(instances) == 1
            else trimesh.util.concatenate(instances))


def _moving_records(metadata: dict, label: str) -> list[dict]:
    records = []
    for frame_record in metadata.get("frames", []):
        if not isinstance(frame_record, dict) or "frame" not in frame_record:
            continue
        frame_id = int(frame_record["frame"])
        for moving in frame_record.get("moving_meshes", []):
            if isinstance(moving, dict) and moving.get("label") == label:
                geometry = moving.get("geometry")
                if not geometry:
                    raise ValueError(
                        f"registered metadata has no geometry for {label!r} "
                        f"at frame {frame_id}")
                records.append({
                    "frame": frame_id,
                    "geometry": str(geometry),
                    "source": moving.get("source"),
                })
                break
    records.sort(key=lambda value: value["frame"])
    if len({record["frame"] for record in records}) != len(records):
        raise ValueError(f"registered metadata has duplicate frames for {label!r}")
    if len(records) < 2:
        raise ValueError(
            f"registered metadata has only {len(records)} moving mesh(es) for "
            f"{label!r}; need at least two")
    return records


def load_registered_meshes(scene_dir: Path, simple_payload: dict,
                           joints: dict[str, dict]) -> RegisteredMeshes:
    """Load the registered static mesh and first/last moving SegviGen pieces."""
    metadata_path = _resolve_scene_path(
        scene_dir,
        simple_payload.get("registered_metadata"),
        scene_dir / "registered_static" / "metadata.json",
        "registered metadata",
    )
    metadata = _read_json(metadata_path)
    if metadata.get("mesh_source") not in {None, "segvigen"}:
        raise ValueError(
            "Stage-31 registration was not produced from SegviGen meshes")
    if metadata.get("camera_to_world_used") is not True:
        raise ValueError(
            "registered_static/metadata.json does not confirm camera_to_world_used; "
            "the Stage-31 mesh frame cannot be related safely to HaWoR world hands")
    reference_to_world = _stage31_reference_to_world(scene_dir, metadata)
    mesh_path = _resolve_scene_path(
        scene_dir,
        simple_payload.get("registered_mesh") or metadata.get("output_mesh"),
        scene_dir / "registered_static" / "registered_meshes.glb",
        "registered mesh",
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(f"missing registered Stage-31 mesh: {mesh_path}")
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required; run Stage 61 in the TrackCraft3R/trellis2 "
            "environment") from exc
    try:
        scene = trimesh.load(str(mesh_path), force="scene", process=False)
    except Exception as exc:
        raise RuntimeError(f"could not load {mesh_path}: {exc}") from exc

    static_geometry = metadata.get("static_geometry")
    if not static_geometry:
        candidates = [name for name in scene.geometry
                      if str(name).startswith("static_reference_")]
        if len(candidates) != 1:
            raise ValueError(
                "registered metadata has no unambiguous static_geometry")
        static_geometry = candidates[0]
    static_mesh = _load_named_geometry(
        scene, str(static_geometry), trimesh, mesh_path)
    static_vertices, static_faces = _mesh_arrays(
        static_mesh, mesh_path, str(static_geometry))
    static_label = "+".join(metadata.get("static_labels") or ["static"])
    static = MeshData(
        label=static_label,
        frame_id=int(metadata.get("reference_frame", 0)),
        vertices=static_vertices,
        faces=static_faces,
        source=str(mesh_path.relative_to(scene_dir)),
        geometry=str(static_geometry),
    )

    moving: dict[str, dict[str, MeshData]] = {}
    available: dict[str, tuple[int, ...]] = {}
    for label in joints:
        records = _moving_records(metadata, label)
        available[label] = tuple(record["frame"] for record in records)
        chosen_records = {"first": records[0], "last": records[-1]}
        choices: dict[str, MeshData] = {}
        loaded_by_geometry: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for choice, record in chosen_records.items():
            geometry = record["geometry"]
            if geometry not in loaded_by_geometry:
                mesh = _load_named_geometry(scene, geometry, trimesh, mesh_path)
                loaded_by_geometry[geometry] = _mesh_arrays(
                    mesh, mesh_path, geometry)
            vertices, faces = loaded_by_geometry[geometry]
            choices[choice] = MeshData(
                label=label,
                frame_id=int(record["frame"]),
                vertices=vertices.copy(),
                faces=faces.copy(),
                source=str(record.get("source") or mesh_path.relative_to(scene_dir)),
                geometry=geometry,
            )
        moving[label] = choices
    return RegisteredMeshes(
        static=static,
        moving=moving,
        available_frames=available,
        registered_mesh_path=mesh_path,
        metadata_path=metadata_path,
        reference_to_world=reference_to_world,
    )


def _sparse_states(record: dict) -> tuple[np.ndarray, np.ndarray]:
    states = sorted((int(item["frame"]), float(item["q"]))
                    for item in record["joint_states"])
    return (
        np.asarray([value[0] for value in states], dtype=np.int64),
        np.asarray([value[1] for value in states], dtype=np.float64),
    )


def sparse_state_at(record: dict, frame_id: int) -> float:
    """Interpolate the Stage-35 sparse scalar state at one native frame."""
    frames, states = _sparse_states(record)
    return float(np.interp(float(frame_id), frames.astype(np.float64), states))


def _interpolate_positions(sample_frames: np.ndarray, positions: np.ndarray,
                           query_frames: np.ndarray) -> np.ndarray:
    sample_frames = np.asarray(sample_frames, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    query_frames = np.asarray(query_frames, dtype=np.float64).reshape(-1)
    if (len(sample_frames) == 0 or positions.shape != (len(sample_frames), 3)
            or not np.isfinite(positions).all()):
        raise ValueError("hand driver positions must be finite Nx3 values")
    if len(sample_frames) > 1 and np.any(np.diff(sample_frames) <= 0):
        raise ValueError("hand driver frames must be strictly increasing")
    return np.column_stack([
        np.interp(query_frames, sample_frames, positions[:, dimension])
        for dimension in range(3)
    ])


def _fill_nonfinite_rates(mid_frames: np.ndarray, rates: np.ndarray) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64).copy()
    finite = np.isfinite(rates)
    if finite.all():
        return rates
    if not finite.any():
        return np.zeros_like(rates)
    rates[~finite] = np.interp(
        mid_frames[~finite], mid_frames[finite], rates[finite])
    return rates


def _smooth_rates(rates: np.ndarray, window: int) -> np.ndarray:
    rates = np.asarray(rates, dtype=np.float64)
    if len(rates) < 2 or window <= 1:
        return rates.copy()
    width = min(int(window), len(rates))
    if width % 2 == 0 and width > 2:
        width -= 1
    if width <= 1:
        return rates.copy()
    left = width // 2
    right = width - 1 - left
    padded = np.pad(rates, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def relative_interval_velocity(frame_ids: np.ndarray, positions: np.ndarray,
                               record: dict) -> tuple[np.ndarray, str]:
    """Measure the hand's relative scalar velocity for one Stage-35 joint.

    The returned values live on intervals ``[frame_ids[i], frame_ids[i+1]]``.
    Revolute velocity is the signed change of the wrist's radial direction
    around the axis, divided by the frame-index interval.
    """
    frame_ids = np.asarray(frame_ids, dtype=np.float64).reshape(-1)
    positions = np.asarray(positions, dtype=np.float64)
    if len(frame_ids) < 2 or positions.shape != (len(frame_ids), 3):
        raise ValueError("relative velocity requires at least two Nx3 samples")
    deltas_t = np.diff(frame_ids)
    if np.any(deltas_t <= 0):
        raise ValueError("relative velocity frames must be strictly increasing")
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if record["type"] == "prismatic":
        rates = (np.diff(positions, axis=0) @ axis) / deltas_t
        return rates, "linear velocity along axis"

    pivot = np.asarray(record["axis_point"], dtype=np.float64)
    radial = positions - pivot
    radial -= np.outer(radial @ axis, axis)
    norms = np.linalg.norm(radial, axis=1)
    positive = norms[norms > 0]
    radius_floor = max(
        1e-10,
        (1e-4 * float(np.median(positive))) if len(positive) else 1e-10,
    )
    rates = np.full(len(deltas_t), np.nan, dtype=np.float64)
    for index in range(len(deltas_t)):
        if norms[index] <= radius_floor or norms[index + 1] <= radius_floor:
            continue
        before = radial[index] / norms[index]
        after = radial[index + 1] / norms[index + 1]
        sine = float(axis @ np.cross(before, after))
        cosine = float(np.clip(before @ after, -1.0, 1.0))
        rates[index] = np.arctan2(sine, cosine) / deltas_t[index]
    mid_frames = 0.5 * (frame_ids[:-1] + frame_ids[1:])
    rates = _fill_nonfinite_rates(mid_frames, rates)
    return rates, "angular velocity about axis"


def infer_dense_joint_motion(
        output_frames: np.ndarray,
        hand_frames: np.ndarray,
        hand_positions: np.ndarray,
        record: dict,
        driver_side: str,
        smoothing_window: int = 5) -> DenseJointMotion:
    """Fill sparse Stage-35 states using relative hand-velocity timing.

    Every sparse anchor is included in an internal work timeline.  On each
    anchor interval, signed relative velocity is integrated and rescaled to the
    known state delta.  If the signed signal cancels almost completely, the
    absolute speed profile is used; if there is no usable motion signal, only
    that interval falls back to linear time interpolation.
    """
    output_frames = np.asarray(output_frames, dtype=np.int64).reshape(-1)
    if len(output_frames) == 0 or np.any(np.diff(output_frames) <= 0):
        raise ValueError("output frames must be a non-empty increasing sequence")
    anchor_frames, anchor_states = _sparse_states(record)
    lower = min(int(output_frames[0]), int(anchor_frames[0]))
    upper = max(int(output_frames[-1]), int(anchor_frames[-1]))
    relevant_hand_frames = {
        int(frame) for frame in np.asarray(hand_frames).reshape(-1)
        if lower <= int(frame) <= upper
    }
    work_frames = np.asarray(sorted(
        set(output_frames.tolist())
        | set(anchor_frames.tolist())
        | relevant_hand_frames
    ), dtype=np.int64)
    work_positions = _interpolate_positions(
        hand_frames, hand_positions, work_frames)
    interval_rates, velocity_kind = relative_interval_velocity(
        work_frames, work_positions, record)
    interval_rates = _smooth_rates(interval_rates, smoothing_window)
    deltas_t = np.diff(work_frames.astype(np.float64))
    increments = interval_rates * deltas_t
    raw = np.concatenate([[0.0], np.cumsum(increments)])

    states = np.full(len(work_frames), np.nan, dtype=np.float64)
    states[work_frames <= anchor_frames[0]] = anchor_states[0]
    states[work_frames >= anchor_frames[-1]] = anchor_states[-1]
    work_index = {int(frame): index for index, frame in enumerate(work_frames)}
    segment_modes = []
    for first_frame, last_frame, first_state, last_state in zip(
            anchor_frames[:-1], anchor_frames[1:],
            anchor_states[:-1], anchor_states[1:]):
        first_index = work_index[int(first_frame)]
        last_index = work_index[int(last_frame)]
        local_raw = raw[first_index:last_index + 1] - raw[first_index]
        signed_total = float(local_raw[-1])
        path = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(local_raw)))])
        path_total = float(path[-1])
        signed_floor = max(1e-10, 0.05 * path_total)
        signed_progress = (local_raw / signed_total
                           if abs(signed_total) > signed_floor else None)
        signed_is_stable = (
            signed_progress is not None
            and float(np.min(signed_progress)) >= -0.25
            and float(np.max(signed_progress)) <= 1.25
        )
        if signed_is_stable:
            progress = signed_progress
            mode = f"signed {velocity_kind}"
        elif path_total > 1e-10:
            progress = path / path_total
            mode = (
                f"absolute {velocity_kind} "
                "(signed motion cancelled or was unstable)"
            )
        else:
            local_frames = work_frames[first_index:last_index + 1]
            progress = ((local_frames - first_frame)
                        / float(last_frame - first_frame))
            mode = "linear-time fallback (no usable hand velocity)"
        states[first_index:last_index + 1] = (
            first_state + (last_state - first_state) * progress)
        states[first_index] = first_state
        states[last_index] = last_state
        segment_modes.append((int(first_frame), int(last_frame), mode))
    if not np.isfinite(states).all():
        raise RuntimeError("dense joint interpolation left non-finite states")

    frame_rates = np.empty(len(work_frames), dtype=np.float64)
    frame_rates[0] = interval_rates[0]
    frame_rates[-1] = interval_rates[-1]
    if len(work_frames) > 2:
        frame_rates[1:-1] = 0.5 * (interval_rates[:-1] + interval_rates[1:])
    output_positions = np.searchsorted(work_frames, output_frames)
    return DenseJointMotion(
        frame_ids=output_frames.copy(),
        states=states[output_positions].astype(np.float32),
        relative_velocity=frame_rates[output_positions].astype(np.float32),
        driver_side=driver_side,
        velocity_kind=velocity_kind,
        segment_modes=tuple(segment_modes),
    )


def subset_dense_motion(motion: DenseJointMotion,
                        frame_ids: np.ndarray) -> DenseJointMotion:
    """Select exact frames from a motion inferred on the complete timeline."""
    frame_ids = np.asarray(frame_ids, dtype=np.int64).reshape(-1)
    positions = np.searchsorted(motion.frame_ids, frame_ids)
    if (len(frame_ids) == 0 or np.any(positions >= len(motion.frame_ids))
            or not np.array_equal(motion.frame_ids[positions], frame_ids)):
        raise ValueError("motion subset frames are absent from the full timeline")
    return DenseJointMotion(
        frame_ids=frame_ids.copy(),
        states=motion.states[positions].copy(),
        relative_velocity=motion.relative_velocity[positions].copy(),
        driver_side=motion.driver_side,
        velocity_kind=motion.velocity_kind,
        segment_modes=motion.segment_modes,
    )


def _numeric_frame_paths(scene_dir: Path) -> dict[int, Path]:
    root = scene_dir / "frames"
    result: dict[int, Path] = {}
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    frame_id = int(path.stem)
                except ValueError:
                    continue
                previous = result.get(frame_id)
                if previous is not None:
                    raise ValueError(
                        f"duplicate numeric RGB frame {frame_id}: "
                        f"{previous.name} and {path.name}")
                result[frame_id] = path.resolve()
    return result


def _numeric_frame_ids(scene_dir: Path) -> list[int]:
    """Compatibility wrapper returning sorted numeric RGB frame IDs."""
    return sorted(_numeric_frame_paths(scene_dir))


def _image_size(path: Path) -> tuple[int, int]:
    """Read an RGB frame header and return ``(height, width)``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to read the DA3 camera image dimensions") from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read RGB frame dimensions from {path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid RGB frame dimensions in {path}: {width}x{height}")
    return int(height), int(width)


def _consistent_image_size(frame_paths: dict[int, Path]) -> tuple[int, int]:
    """Validate the fixed-resolution RGB sequence assumed by one DA3 K stream."""
    expected = None
    expected_path = None
    for frame_id, path in sorted(frame_paths.items()):
        size = _image_size(path)
        if expected is None:
            expected, expected_path = size, path
        elif size != expected:
            raise ValueError(
                f"RGB frame {frame_id:06d} has size {size[1]}x{size[0]}, "
                f"but {expected_path.name} has {expected[1]}x{expected[0]}; "
                "DA3 camera replay requires one fixed image resolution")
    if expected is None:
        raise ValueError("cannot determine image size from an empty frame sequence")
    return expected


def _resolve_hawor_name(scene_dir: Path, requested: str) -> str:
    if requested != "auto":
        _safe_directory_name(requested, "--hawor-name")
        return requested
    for candidate in ("hawor_scaled", "hawor"):
        root = scene_dir / candidate
        if (root / "per_frame").is_dir() and (root / "faces.npy").is_file():
            return candidate
    raise FileNotFoundError(
        f"no usable hawor_scaled/ or hawor/ trajectory under {scene_dir}")


def _choose_driver_side(hands: HandTrajectory, requested: str) -> str:
    if requested != "auto":
        if requested not in hands.sides:
            raise ValueError(
                f"--driver-hand {requested!r} is unavailable; loaded {hands.sides}")
        return requested
    return max(hands.sides, key=lambda side: len(hands.samples[side]))


def _load_playback_fps(scene_dir: Path, hawor_name: str,
                       requested: float | None) -> float:
    if requested is not None:
        if not np.isfinite(requested) or requested <= 0:
            raise ValueError("--playback-fps must be positive")
        return float(requested)
    config_path = scene_dir / hawor_name / "config.json"
    if config_path.is_file():
        config = _read_json(config_path)
        for key in ("overlay_fps", "fps"):
            try:
                value = float(config.get(key, np.nan))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                return value
    return 24.0


def prepare_scene(args) -> PreparedScene:
    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory does not exist: {scene_dir}")
    if args.start_frame is not None and args.end_frame is not None:
        if args.start_frame > args.end_frame:
            raise ValueError("--start-frame cannot exceed --end-frame")
    if args.velocity_smoothing <= 0:
        raise ValueError("--velocity-smoothing must be positive")
    if (args.axis_length_scale <= 0 or args.axis_radius_scale <= 0
            or args.camera_scale <= 0):
        raise ValueError("axis and camera scale values must be positive")
    if args.port <= 0 or args.port > 65535:
        raise ValueError("--port must be between 1 and 65535")
    if args.viser_timeout < 0:
        raise ValueError("--viser-timeout cannot be negative")
    if args.video_fps is not None:
        if not np.isfinite(args.video_fps) or args.video_fps <= 0:
            raise ValueError("--video-fps must be positive")
    if args.video_max_side <= 0:
        raise ValueError("--video-max-side must be positive")
    if not np.isfinite(args.render_timeout) or args.render_timeout <= 0:
        raise ValueError("--render-timeout must be positive")
    if len(args.video_codec) != 4:
        raise ValueError("--video-codec must contain exactly four characters")
    if args.export_video is not None:
        if args.export_video.suffix.lower() != ".mp4":
            raise ValueError("--export-video PATH must end in .mp4")

    simple_payload, joints = load_simple_joints(scene_dir, args.labels)
    meshes = load_registered_meshes(scene_dir, simple_payload, joints)
    cameras = load_da3_cameras(scene_dir, meshes.reference_to_world)
    hawor_name = _resolve_hawor_name(scene_dir, args.hawor_name)
    hands_world = load_hawor_trajectory(
        scene_dir,
        hawor_name=hawor_name,
        coordinate_source=args.hand_coordinate_source,
        detected_only=args.detected_only,
    )
    hands = hands_world_to_reference(
        hands_world, meshes.reference_to_world)
    driver_side = _choose_driver_side(hands, args.driver_hand)

    frame_paths = _numeric_frame_paths(scene_dir)
    if not frame_paths:
        raise FileNotFoundError(
            f"no numeric RGB frames found under {scene_dir / 'frames'}")
    image_hw = _consistent_image_size(frame_paths)
    complete_timeline = set(frame_paths) | set(hands.frame_ids)
    displayed = set(complete_timeline)
    if args.start_frame is not None:
        displayed = {frame for frame in displayed if frame >= args.start_frame}
    if args.end_frame is not None:
        displayed = {frame for frame in displayed if frame <= args.end_frame}
    if not displayed:
        raise ValueError("no video/hand frames remain after filtering")
    full_frame_ids = np.asarray(sorted(complete_timeline), dtype=np.int64)
    frame_ids = np.asarray(sorted(displayed), dtype=np.int64)

    driver_samples = hands.samples[driver_side]
    driver_frames = np.asarray(
        [sample.frame_id for sample in driver_samples], dtype=np.int64)
    driver_positions = np.stack(
        [sample.joints[0] for sample in driver_samples]).astype(np.float64)
    full_motions = {
        label: infer_dense_joint_motion(
            output_frames=full_frame_ids,
            hand_frames=driver_frames,
            hand_positions=driver_positions,
            record=record,
            driver_side=driver_side,
            smoothing_window=args.velocity_smoothing,
        )
        for label, record in joints.items()
    }
    motions = {
        label: subset_dense_motion(motion, frame_ids)
        for label, motion in full_motions.items()
    }
    playback_fps = _load_playback_fps(
        scene_dir, hawor_name, args.playback_fps)
    return PreparedScene(
        scene_dir=scene_dir,
        hawor_name=hawor_name,
        hands=hands,
        joints=joints,
        meshes=meshes,
        frame_ids=frame_ids,
        driver_side=driver_side,
        motions=motions,
        playback_fps=playback_fps,
        cameras=cameras,
        frame_paths=frame_paths,
        image_hw=image_hw,
    )


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    skew = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return (np.eye(3) + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew))


def _axis_angle_wxyz(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * float(angle)
    return np.concatenate(
        [[np.cos(half)], np.sin(half) * axis]).astype(np.float32)


def joint_handle_transform(record: dict, state_delta: float,
                           display_offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Viser quaternion/translation for a canonical moving mesh."""
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if record["type"] == "prismatic":
        return (np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (float(state_delta) * axis).astype(np.float32))
    pivot = np.asarray(record["axis_point"], dtype=np.float64)
    pivot_display = pivot + np.asarray(display_offset, dtype=np.float64)
    rotation = _axis_angle_rotation(axis, float(state_delta))
    position = pivot_display - rotation @ pivot_display
    return (_axis_angle_wxyz(axis, float(state_delta)),
            position.astype(np.float32))


def apply_joint_motion(vertices: np.ndarray, record: dict,
                       state_delta: float) -> np.ndarray:
    """Apply a simple_joint delta directly; useful for validation and tests."""
    vertices = np.asarray(vertices, dtype=np.float64)
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    if record["type"] == "prismatic":
        return (vertices + float(state_delta) * axis).astype(np.float32)
    pivot = np.asarray(record["axis_point"], dtype=np.float64)
    rotation = _axis_angle_rotation(axis, float(state_delta))
    return ((vertices - pivot) @ rotation.T + pivot).astype(np.float32)


def _hand_sample_at(samples: list[HandSample], frame_id: int):
    """Linearly interpolate a same-side MANO sample on a missing video frame."""
    ids = np.asarray([sample.frame_id for sample in samples], dtype=np.int64)
    position = int(np.searchsorted(ids, int(frame_id)))
    if position < len(samples) and samples[position].frame_id == frame_id:
        sample = samples[position]
        return sample.vertices, sample.joints, sample.detected
    if position == 0:
        sample = samples[0]
        return sample.vertices, sample.joints, False
    if position == len(samples):
        sample = samples[-1]
        return sample.vertices, sample.joints, False
    before, after = samples[position - 1], samples[position]
    alpha = ((frame_id - before.frame_id)
             / float(after.frame_id - before.frame_id))
    vertices = ((1.0 - alpha) * before.vertices
                + alpha * after.vertices).astype(np.float32)
    joints = ((1.0 - alpha) * before.joints
              + alpha * after.joints).astype(np.float32)
    return vertices, joints, False


def _hand_skeleton(joints: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints, dtype=np.float32)
    if joints.ndim != 2 or joints.shape[1] != 3 or len(joints) < 21:
        raise ValueError(f"expected at least 21 MANO joints, got {joints.shape}")
    return joints[MANO_BONES]


def _orthogonal_basis(axis: np.ndarray):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ seed)) > 0.8:
        seed = np.array([0.0, 1.0, 0.0])
    first = np.cross(axis, seed)
    first /= np.linalg.norm(first)
    return first, np.cross(axis, first)


def _add_joint_visuals(server, trimesh, prepared: PreparedScene,
                       display_offset: np.ndarray, scene_diagonal: float,
                       length_scale: float, radius_scale: float,
                       visible: bool):
    handles = []
    for label, record in prepared.joints.items():
        mesh = prepared.meshes.moving[label]["first"]
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        center = np.asarray(record.get("position"), dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            center = mesh.vertices.mean(axis=0).astype(np.float64)
        center += display_offset
        part_diagonal = max(mesh.diagonal, 1e-6)
        half_length = max(
            length_scale * part_diagonal,
            0.35 * scene_diagonal,
        )
        radius = max(
            radius_scale * part_diagonal,
            0.006 * scene_diagonal,
            1e-5,
        )
        color = JOINT_COLORS[record["type"]]
        head_height = min(
            0.45 * half_length,
            max(0.16 * half_length, 4.0 * radius),
        )
        start = center - half_length * axis
        tip = center + half_length * axis
        head_base = tip - head_height * axis
        shaft = trimesh.creation.cylinder(
            radius=radius, segment=np.stack([start, head_base]), sections=32)
        cone = trimesh.creation.cone(
            radius=2.8 * radius, height=head_height, sections=32)
        cone.apply_transform(trimesh.geometry.align_vectors(
            [0.0, 0.0, 1.0], axis))
        cone.apply_translation(head_base)
        shaft_handle = server.scene.add_mesh_simple(
            f"/simple_joint/{label}/shaft",
            np.asarray(shaft.vertices, dtype=np.float32),
            np.asarray(shaft.faces, dtype=np.int32),
            color=color, material="toon5", side="double",
            flat_shading=False, cast_shadow=False, visible=visible,
        )
        cone_handle = server.scene.add_mesh_simple(
            f"/simple_joint/{label}/direction",
            np.asarray(cone.vertices, dtype=np.float32),
            np.asarray(cone.faces, dtype=np.int32),
            color=color, material="toon5", side="double",
            flat_shading=False, cast_shadow=False, visible=visible,
        )
        point_handle = server.scene.add_icosphere(
            f"/simple_joint/{label}/position",
            radius=2.2 * radius, color=color, subdivisions=2,
            material="toon5", position=center, cast_shadow=False,
            visible=visible,
        )
        label_handle = server.scene.add_label(
            f"/simple_joint/{label}/label",
            text=f"{label}: {record['type']}",
            position=center + np.array([0.0, 0.0, 5.0 * radius]),
            font_size_mode="scene",
            font_scene_height=max(5.0 * radius, 0.04 * scene_diagonal),
            depth_test=False, anchor="bottom-center", visible=visible,
        )
        handles.extend([shaft_handle, cone_handle, point_handle, label_handle])
        if record["type"] == "revolute":
            first, second = _orthogonal_basis(axis)
            angles = np.linspace(0.0, 2.0 * np.pi, 65)
            ring = center + 0.32 * part_diagonal * (
                np.cos(angles)[:, None] * first
                + np.sin(angles)[:, None] * second)
            segments = np.stack([ring[:-1], ring[1:]], axis=1).astype(np.float32)
            ring_handle = server.scene.add_line_segments(
                f"/simple_joint/{label}/rotation",
                segments, color,
                line_width=6.0, visible=visible,
            )
            handles.append(ring_handle)
    return handles


def _scene_bounds(prepared: PreparedScene) -> tuple[np.ndarray, np.ndarray]:
    values = [prepared.meshes.static.vertices]
    for choices in prepared.meshes.moving.values():
        values.extend(mesh.vertices for mesh in choices.values())
    for side in prepared.hands.sides:
        samples = prepared.hands.samples[side]
        stride = max(1, len(samples) // 30)
        values.extend(sample.vertices for sample in samples[::stride])
    vertices = np.concatenate(values, axis=0)
    return vertices.min(axis=0), vertices.max(axis=0)


def _nearest_frame_index(frame_ids: np.ndarray, requested: int) -> int:
    position = int(np.searchsorted(frame_ids, int(requested)))
    candidates = [min(position, len(frame_ids) - 1)]
    if position > 0:
        candidates.append(position - 1)
    return min(candidates, key=lambda index: abs(int(frame_ids[index]) - requested))


def camera_frustum_segments(sample: CameraSample,
                            image_hw: tuple[int, int], depth: float,
                            display_offset: np.ndarray) -> np.ndarray:
    """Build an exact full-intrinsics DA3 frustum as Viser line segments."""
    height, width = image_hw
    if height <= 0 or width <= 0 or depth <= 0:
        raise ValueError("camera frustum dimensions and depth must be positive")
    pixels = np.array([
        [0.0, 0.0, 1.0],
        [float(width), 0.0, 1.0],
        [float(width), float(height), 1.0],
        [0.0, float(height), 1.0],
    ])
    rays = pixels @ np.linalg.inv(sample.intrinsics).T
    if np.any(np.abs(rays[:, 2]) < 1e-10) or not np.isfinite(rays).all():
        raise ValueError("DA3 intrinsics produced an invalid frustum ray")
    corners_camera = rays / rays[:, 2:3] * float(depth)
    transform = sample.camera_to_reference
    corners = _transform_points(corners_camera, transform).astype(np.float64)
    origin = transform[:3, 3].astype(np.float64)
    offset = np.asarray(display_offset, dtype=np.float64).reshape(3)
    corners += offset
    origin += offset
    segments = [[origin, corner] for corner in corners]
    segments.extend([
        [corners[0], corners[1]],
        [corners[1], corners[2]],
        [corners[2], corners[3]],
        [corners[3], corners[0]],
    ])
    return np.asarray(segments, dtype=np.float32)


def camera_render_spec(K: np.ndarray, image_hw: tuple[int, int],
                       max_side: int) -> CameraRenderSpec:
    """Create an exact DA3-K remap around Viser's centered square-pixel camera.

    The browser first renders an overscanned centered pinhole image with one
    focal length.  ``map_x``/``map_y`` then resample it into the requested DA3
    output pixels.  Solving the full upper-left 2x2 block preserves unequal
    focal lengths and skew; subtracting the last column preserves ``cx/cy``.
    """
    source_h, source_w = map(int, image_hw)
    if source_h <= 0 or source_w <= 0 or max_side <= 0:
        raise ValueError("image dimensions and max_side must be positive")
    scale = min(1.0, float(max_side) / max(source_h, source_w))
    output_h = max(1, int(round(source_h * scale)))
    output_w = max(1, int(round(source_w * scale)))
    sx, sy = output_w / float(source_w), output_h / float(source_h)
    scaled = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    scaled[0, :] *= sx
    scaled[1, :] *= sy
    linear = scaled[:2, :2]
    if (not np.isfinite(scaled).all()
            or abs(float(np.linalg.det(linear))) < 1e-12
            or not np.allclose(scaled[2], [0.0, 0.0, 1.0], atol=1e-5)):
        raise ValueError("invalid intrinsics for camera video rendering")
    inverse_linear = np.linalg.inv(linear)
    principal = scaled[:2, 2]
    focal = float(np.sqrt(abs(np.linalg.det(linear))))
    if not np.isfinite(focal) or focal <= 1e-8:
        raise ValueError("camera intrinsics have no usable focal scale")

    # Use image-boundary corners to size a centered overscan canvas that covers
    # every desired output ray, including off-center/skewed calibrations.
    boundary_pixels = np.array([
        [0.0, 0.0], [float(output_w), 0.0],
        [float(output_w), float(output_h)], [0.0, float(output_h)],
    ])
    boundary_rays = (boundary_pixels - principal) @ inverse_linear.T
    max_x = max(float(np.max(np.abs(boundary_rays[:, 0]))), 1e-8)
    max_y = max(float(np.max(np.abs(boundary_rays[:, 1]))), 1e-8)
    render_w = max(2, int(np.ceil(2.0 * focal * max_x + 4.0)))
    render_h = max(2, int(np.ceil(2.0 * focal * max_y + 4.0)))
    vertical_fov = float(2.0 * np.arctan(render_h / (2.0 * focal)))
    if not 0.0 < vertical_fov < np.pi:
        raise ValueError("DA3 intrinsics produced an invalid Viser field of view")

    x_pixels, y_pixels = np.meshgrid(
        np.arange(output_w, dtype=np.float64),
        np.arange(output_h, dtype=np.float64),
    )
    desired = np.stack([x_pixels, y_pixels], axis=-1)
    normalized = (desired - principal) @ inverse_linear.T
    map_x = (focal * normalized[..., 0] + render_w / 2.0).astype(np.float32)
    map_y = (focal * normalized[..., 1] + render_h / 2.0).astype(np.float32)
    return CameraRenderSpec(
        output_hw=(output_h, output_w),
        render_hw=(render_h, render_w),
        vertical_fov=vertical_fov,
        map_x=map_x,
        map_y=map_y,
        scaled_intrinsics=scaled.astype(np.float32),
    )


def _centered_camera_fov(K: np.ndarray, image_hw: tuple[int, int]) -> float:
    """Centered vertical-FOV approximation used only for interactive follow."""
    height = int(image_hw[0])
    fy = abs(float(np.asarray(K)[1, 1]))
    if height <= 0 or not np.isfinite(fy) or fy <= 1e-8:
        raise ValueError("invalid DA3 fy/image height")
    return float(2.0 * np.arctan(height / (2.0 * fy)))


def _video_output_path(scene_dir: Path, requested: Path) -> Path:
    path = requested.expanduser()
    return path.resolve() if path.is_absolute() else (scene_dir / path).resolve()


def _client_render_with_timeout(client, timeout: float, **kwargs) -> np.ndarray:
    """Bound Viser 1.0's otherwise-unbounded browser render request.

    Viser 1.1 has a native timeout, but 1.0.30 (the repository environment)
    waits forever if a browser disappears.  Isolating that call in a daemon
    thread lets the exporter fail, clean its temporary file, and stay usable.
    """
    finished = threading.Event()
    result: dict[str, Any] = {}

    def request():
        try:
            result["image"] = client.get_render(**kwargs)
        except Exception as exc:
            result["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(
        target=request, name="stage61-viser-render-request", daemon=True)
    thread.start()
    if not finished.wait(float(timeout)):
        raise TimeoutError(
            f"Viser browser did not return a frame within {timeout:.3g}s; "
            "keep the export browser connected or increase --render-timeout")
    if "error" in result:
        raise RuntimeError(f"Viser browser render failed: {result['error']}") \
            from result["error"]
    return np.asarray(result["image"])


def write_camera_video(args, prepared: PreparedScene, frame_provider,
                       status_callback=None, cancel_event=None,
                       capture_settings: dict | None = None
                       ) -> tuple[Path, Path]:
    """Atomically encode browser-rendered RGB frames and provenance metadata."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--export-video needs OpenCV in the Viser environment") from exc
    if args.export_video is None:
        raise ValueError("write_camera_video called without --export-video")
    output_path = _video_output_path(prepared.scene_dir, args.export_video)
    metadata_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite_video:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite-video to replace it")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.tmp-",
        suffix=output_path.suffix,
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    first_sample = camera_sample_at(prepared.cameras, int(prepared.frame_ids[0]))
    first_spec = camera_render_spec(
        first_sample.intrinsics, prepared.image_hw, args.video_max_side)
    output_h, output_w = first_spec.output_hw
    fps = float(args.video_fps or prepared.playback_fps)
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*args.video_codec),
        fps,
        (output_w, output_h),
    )
    if not writer.isOpened():
        writer.release()
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError(
            f"OpenCV could not open {temporary_path} with codec "
            f"{args.video_codec!r}")

    native_frames = []
    inferred_frames = []
    next_progress = 0
    try:
        for index, frame_id_value in enumerate(prepared.frame_ids):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("camera video export was cancelled")
            frame_id = int(frame_id_value)
            sample = camera_sample_at(prepared.cameras, frame_id)
            (native_frames if sample.native else inferred_frames).append(frame_id)
            rgb = np.asarray(frame_provider(frame_id, sample))
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise ValueError(
                    f"browser render for frame {frame_id:06d} has shape {rgb.shape}")
            if rgb.shape[:2] != (output_h, output_w):
                raise ValueError(
                    f"browser render for frame {frame_id:06d} is "
                    f"{rgb.shape[1]}x{rgb.shape[0]}, expected {output_w}x{output_h}")
            writer.write(np.ascontiguousarray(rgb[..., :3][..., ::-1]))
            percent = int(100 * (index + 1) / len(prepared.frame_ids))
            if percent >= next_progress:
                message = (
                    f"camera video {index + 1}/{len(prepared.frame_ids)} "
                    f"({percent}%)")
                print(f"  {message}")
                if status_callback is not None:
                    status_callback(message)
                next_progress += 10
    except Exception:
        writer.release()
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    writer.release()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        if temporary_path.exists():
            temporary_path.unlink()
        raise RuntimeError("camera video writer produced no output")
    settings = capture_settings or {
        "canonical_moving_mesh": args.mesh_frame,
        "visibility": {
            "registered_meshes": not args.no_meshes,
            "hawor_surfaces": not args.no_hands,
            "hawor_skeletons": not args.no_skeleton,
            "simple_joint": not args.no_joints,
            "ground_grid": not args.no_grid,
        },
    }

    metadata = {
        "version": 1,
        "stage": "61_render_all",
        "method": "viser_browser_da3_full_intrinsics",
        "video": str(output_path),
        "frame_count": int(len(prepared.frame_ids)),
        "frame_ids": [int(value) for value in prepared.frame_ids],
        "fps": fps,
        "codec": args.video_codec,
        "output_size": [output_w, output_h],
        "source_image_size": [prepared.image_hw[1], prepared.image_hw[0]],
        "camera_pose_source": "da3/cameras.npz (camera-to-world, XYZW)",
        "intrinsics_source": "da3/intrinsics.npz",
        "frame_mapping": "direct numeric frame index (Stage 31/60/60a convention)",
        "native_camera_frames": native_frames,
        "interpolated_or_held_camera_frames": inferred_frames,
        "projection": (
            "centered Viser overscan followed by exact full-K pixel remap"
        ),
        "coordinate_frame": "stage31_reference_mesh",
        "canonical_moving_mesh": settings["canonical_moving_mesh"],
        "includes": {
            **settings["visibility"],
            "camera_frustum": False,
        },
    }
    temporary_metadata = None
    try:
        metadata_descriptor, metadata_name = tempfile.mkstemp(
            prefix=f".{metadata_path.stem}.tmp-",
            suffix=metadata_path.suffix,
            dir=metadata_path.parent,
        )
        os.close(metadata_descriptor)
        temporary_metadata = Path(metadata_name)
        temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        # Both complete artifacts exist before either public path changes.
        os.replace(temporary_path, output_path)
        os.replace(temporary_metadata, metadata_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        if temporary_metadata is not None and temporary_metadata.exists():
            temporary_metadata.unlink()
        raise
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")
    if status_callback is not None:
        status_callback(f"wrote {output_path.name}")
    return output_path, metadata_path


def run_viewer(args, prepared: PreparedScene):
    try:
        import trimesh
        import viser
    except ImportError as exc:
        raise RuntimeError(
            "the live viewer needs viser and trimesh; activate the "
            "TrackCraft3R environment") from exc

    bounds_min, bounds_max = _scene_bounds(prepared)
    world_center = 0.5 * (bounds_min + bounds_max)
    scene_diagonal = max(float(np.linalg.norm(bounds_max - bounds_min)), 1e-3)
    display_offset = -world_center
    planned_video_path = None
    if args.export_video is not None:
        planned_video_path = _video_output_path(
            prepared.scene_dir, args.export_video)
        if planned_video_path.exists() and not args.overwrite_video:
            raise FileExistsError(
                f"{planned_video_path} exists; pass --overwrite-video to replace it")

    server = viser.ViserServer(
        host=args.host,
        port=args.port,
        label=f"Hand-driven articulation: {prepared.scene_dir.name}",
    )
    server.gui.configure_theme(
        control_layout="floating", control_width="large",
        dark_mode=True, show_logo=False,
    )
    server.scene.set_up_direction("+z")
    server.scene.add_frame(
        "/world/origin", axes_length=0.15 * scene_diagonal,
        axes_radius=0.006 * scene_diagonal,
    )
    ground_z = float(bounds_min[2] - world_center[2] - 0.04 * scene_diagonal)
    grid = server.scene.add_grid(
        "/world/grid",
        width=3.0 * scene_diagonal,
        height=3.0 * scene_diagonal,
        plane="xy",
        cell_size=max(scene_diagonal / 10.0, 1e-3),
        section_size=max(scene_diagonal / 2.0, 1e-3),
        position=(0.0, 0.0, ground_z),
        plane_opacity=0.25,
        cell_color=(85, 90, 100),
        section_color=(130, 135, 145),
        visible=not args.no_grid,
    )

    static_handle = server.scene.add_mesh_simple(
        f"/registered_mesh/static/{prepared.meshes.static.label}",
        prepared.meshes.static.vertices + display_offset,
        prepared.meshes.static.faces,
        color=(165, 170, 180), material="toon3", flat_shading=False,
        side="double", cast_shadow=True, receive_shadow=True,
        visible=not args.no_meshes,
    )
    moving_handles: dict[str, dict[str, Any]] = {}
    for label_index, (label, choices) in enumerate(prepared.meshes.moving.items()):
        moving_handles[label] = {}
        color = MOVING_COLORS[label_index % len(MOVING_COLORS)]
        for choice, mesh in choices.items():
            handle = server.scene.add_mesh_simple(
                f"/registered_mesh/moving/{label}/{choice}_{mesh.frame_id:06d}",
                mesh.vertices + display_offset,
                mesh.faces,
                color=color, material="toon3", flat_shading=False,
                side="double", cast_shadow=True, receive_shadow=True,
                visible=(not args.no_meshes and choice == args.mesh_frame),
            )
            moving_handles[label][choice] = handle

    joint_handles = _add_joint_visuals(
        server, trimesh, prepared, display_offset, scene_diagonal,
        args.axis_length_scale, args.axis_radius_scale,
        visible=not args.no_joints,
    )

    hand_handles = {}
    skeleton_handles = {}
    initial_frame = int(prepared.frame_ids[0])
    for side in prepared.hands.sides:
        vertices, joints, detected = _hand_sample_at(
            prepared.hands.samples[side], initial_frame)
        color = HAND_COLORS[side] if detected else HAND_INFILL_COLORS[side]
        hand_handles[side] = server.scene.add_mesh_simple(
            f"/hawor/{side}/surface",
            vertices + display_offset,
            prepared.hands.faces[side],
            color=color, material="toon3", flat_shading=False,
            side="double", cast_shadow=True, receive_shadow=True,
            visible=not args.no_hands,
        )
        skeleton_handles[side] = server.scene.add_line_segments(
            f"/hawor/{side}/skeleton",
            _hand_skeleton(joints + display_offset),
            color,
            line_width=4.0,
            visible=not args.no_skeleton,
        )

    initial_camera = camera_sample_at(prepared.cameras, initial_frame)
    camera_depth = args.camera_scale * scene_diagonal
    initial_camera_position = (
        initial_camera.camera_to_reference[:3, 3] + display_offset)
    initial_camera_wxyz = _rotation_to_wxyz(
        initial_camera.camera_to_reference[:3, :3])
    camera_frustum_handle = server.scene.add_line_segments(
        "/da3_camera/current/frustum",
        camera_frustum_segments(
            initial_camera, prepared.image_hw, camera_depth, display_offset),
        (60, 225, 235), line_width=3.0,
        visible=not args.no_camera,
    )
    camera_pose_handle = server.scene.add_frame(
        "/da3_camera/current/pose",
        wxyz=initial_camera_wxyz,
        position=initial_camera_position,
        axes_length=0.28 * camera_depth,
        axes_radius=max(0.012 * camera_depth, 1e-6),
        visible=not args.no_camera,
    )
    camera_origin_handle = server.scene.add_icosphere(
        "/da3_camera/current/center",
        radius=max(0.035 * camera_depth, 1e-6),
        color=(60, 225, 235), subdivisions=2, material="toon5",
        position=initial_camera_position, cast_shadow=False,
        visible=not args.no_camera,
    )
    camera_positions = np.stack([
        camera_sample_at(prepared.cameras, int(frame_id))
        .camera_to_reference[:3, 3] + display_offset
        for frame_id in prepared.frame_ids
    ])
    if len(camera_positions) > 1:
        camera_path_points = np.stack(
            [camera_positions[:-1], camera_positions[1:]], axis=1)
    else:
        camera_path_points = np.stack(
            [camera_positions, camera_positions], axis=1)
    camera_path_handle = server.scene.add_line_segments(
        "/da3_camera/path",
        camera_path_points.astype(np.float32),
        (45, 135, 145), line_width=2.0,
        visible=not args.no_camera,
    )
    camera_handles = [
        camera_frustum_handle, camera_pose_handle,
        camera_origin_handle, camera_path_handle,
    ]

    native_text = ", ".join(
        f"{label}: {frames[0]:06d}/{frames[-1]:06d}"
        for label, frames in prepared.meshes.available_frames.items())
    server.gui.add_markdown(
        "## HaWoR + registered SegviGen\n"
        f"**Driver:** {prepared.driver_side} wrist  \n"
        f"**Native first/last:** {native_text}  \n"
        "**Camera:** direct-index DA3 pose + full intrinsics  \n"
        "Sparse simple-joint states are exact; hand-relative velocity fills "
        "the intervening video frames."
    )
    canonical_gui = server.gui.add_dropdown(
        "Canonical moving mesh", ("first", "last"),
        initial_value=args.mesh_frame,
        hint="Use each label's first or last registered SegviGen moving mesh",
    )
    frame_gui = server.gui.add_slider(
        "Frame",
        min=int(prepared.frame_ids[0]),
        max=int(prepared.frame_ids[-1]),
        step=1,
        initial_value=initial_frame,
    )
    play_gui = server.gui.add_checkbox("Play", initial_value=args.play)
    loop_gui = server.gui.add_checkbox("Loop", initial_value=not args.no_loop)
    fps_gui = server.gui.add_slider(
        "Playback FPS", min=1.0, max=60.0, step=0.5,
        initial_value=float(np.clip(prepared.playback_fps, 1.0, 60.0)),
    )
    previous_gui = server.gui.add_button("Previous frame")
    next_gui = server.gui.add_button("Next frame")
    show_mesh_gui = server.gui.add_checkbox(
        "Show registered meshes", initial_value=not args.no_meshes)
    show_hand_gui = server.gui.add_checkbox(
        "Show HaWoR surfaces", initial_value=not args.no_hands)
    show_skeleton_gui = server.gui.add_checkbox(
        "Show hand skeletons", initial_value=not args.no_skeleton)
    show_joint_gui = server.gui.add_checkbox(
        "Show simple_joint axes", initial_value=not args.no_joints)
    show_camera_gui = server.gui.add_checkbox(
        "Show DA3 camera + path", initial_value=not args.no_camera)
    follow_camera_gui = server.gui.add_checkbox(
        "Follow DA3 camera", initial_value=False,
        hint=("Use the DA3 pose and centered vertical FOV in the live viewport; "
              "MP4 export additionally applies the exact full-K remap"),
    )
    view_camera_gui = server.gui.add_button("View through current DA3 camera")
    show_grid_gui = server.gui.add_checkbox(
        "Show ground grid", initial_value=not args.no_grid)
    status_gui = server.gui.add_markdown("Initializing frame...")
    video_status_gui = server.gui.add_markdown(
        ("Camera video: waiting for the first browser connection..."
         if args.export_video is not None else
         "Camera video: disabled (start with `--export-video`).")
    )

    update_lock = threading.RLock()

    def set_client_to_camera(client, sample: CameraSample):
        client.camera.position = (
            sample.camera_to_reference[:3, 3] + display_offset)
        client.camera.wxyz = _rotation_to_wxyz(
            sample.camera_to_reference[:3, :3])
        client.camera.fov = _centered_camera_fov(
            sample.intrinsics, prepared.image_hw)

    def render_frame(requested_frame: int, canonical_override: str | None = None,
                     update_follow_clients: bool = True,
                     visibility_override: dict | None = None):
        with update_lock:
            frame_index = _nearest_frame_index(
                prepared.frame_ids, int(requested_frame))
            frame_id = int(prepared.frame_ids[frame_index])
            canonical = (str(canonical_override) if canonical_override is not None
                         else str(canonical_gui.value))
            visibility = visibility_override or {
                "registered_meshes": bool(show_mesh_gui.value),
                "hawor_surfaces": bool(show_hand_gui.value),
                "hawor_skeletons": bool(show_skeleton_gui.value),
                "simple_joint": bool(show_joint_gui.value),
                "da3_camera": bool(show_camera_gui.value),
                "ground_grid": bool(show_grid_gui.value),
            }
            camera_sample = camera_sample_at(prepared.cameras, frame_id)
            context = server.atomic() if hasattr(server, "atomic") else nullcontext()
            with context:
                object_states = []
                for label, record in prepared.joints.items():
                    state = float(prepared.motions[label].states[frame_index])
                    object_states.append(f"{label} q={state:+.4f}")
                    for choice, handle in moving_handles[label].items():
                        mesh_frame = prepared.meshes.moving[label][choice].frame_id
                        canonical_state = sparse_state_at(record, int(mesh_frame))
                        wxyz, position = joint_handle_transform(
                            record, state - canonical_state, display_offset)
                        handle.wxyz = wxyz
                        handle.position = position
                        handle.visible = bool(
                            visibility["registered_meshes"]
                            and choice == canonical)
                static_handle.visible = bool(visibility["registered_meshes"])

                hand_status = []
                for side in prepared.hands.sides:
                    vertices, joints, detected = _hand_sample_at(
                        prepared.hands.samples[side], frame_id)
                    color = (HAND_COLORS[side] if detected
                             else HAND_INFILL_COLORS[side])
                    hand_handles[side].vertices = (
                        vertices + display_offset).astype(np.float32)
                    hand_handles[side].color = color
                    hand_handles[side].visible = bool(
                        visibility["hawor_surfaces"])
                    skeleton_handles[side].points = _hand_skeleton(
                        joints + display_offset)
                    skeleton_handles[side].colors = np.asarray(
                        color, dtype=np.uint8)
                    skeleton_handles[side].visible = bool(
                        visibility["hawor_skeletons"])
                    hand_status.append(
                        f"{side}: {'detected' if detected else 'in-filled/interpolated'}")

                camera_position = (
                    camera_sample.camera_to_reference[:3, 3] + display_offset)
                camera_frustum_handle.points = camera_frustum_segments(
                    camera_sample, prepared.image_hw,
                    camera_depth, display_offset)
                camera_pose_handle.position = camera_position
                camera_pose_handle.wxyz = _rotation_to_wxyz(
                    camera_sample.camera_to_reference[:3, :3])
                camera_origin_handle.position = camera_position
                for handle in camera_handles:
                    handle.visible = bool(visibility["da3_camera"])
                for handle in joint_handles:
                    handle.visible = bool(visibility["simple_joint"])
                grid.visible = bool(visibility["ground_grid"])
                status_gui.content = (
                    f"**Frame {frame_id:06d}**  \n"
                    + " · ".join(object_states) + "  \n"
                    + " · ".join(hand_status) + "  \n"
                    + ("DA3 camera: native pose/K" if camera_sample.native
                       else "DA3 camera: interpolated/endpoint-held pose/K")
                )
            if update_follow_clients and bool(follow_camera_gui.value):
                for client in list(server.get_clients().values()):
                    set_client_to_camera(client, camera_sample)
            return frame_index

    @frame_gui.on_update
    def _(_event):
        render_frame(int(frame_gui.value))

    @canonical_gui.on_update
    def _(_event):
        render_frame(int(frame_gui.value))

    @previous_gui.on_click
    def _(_event):
        index = _nearest_frame_index(prepared.frame_ids, int(frame_gui.value))
        index = max(0, index - 1)
        frame_gui.value = int(prepared.frame_ids[index])
        render_frame(int(prepared.frame_ids[index]))

    @next_gui.on_click
    def _(_event):
        index = _nearest_frame_index(prepared.frame_ids, int(frame_gui.value))
        index = min(len(prepared.frame_ids) - 1, index + 1)
        frame_gui.value = int(prepared.frame_ids[index])
        render_frame(int(prepared.frame_ids[index]))

    @show_mesh_gui.on_update
    def _(_event):
        render_frame(int(frame_gui.value))

    @show_hand_gui.on_update
    def _(_event):
        with update_lock:
            for handle in hand_handles.values():
                handle.visible = bool(show_hand_gui.value)

    @show_skeleton_gui.on_update
    def _(_event):
        with update_lock:
            for handle in skeleton_handles.values():
                handle.visible = bool(show_skeleton_gui.value)

    @show_joint_gui.on_update
    def _(_event):
        with update_lock:
            for handle in joint_handles:
                handle.visible = bool(show_joint_gui.value)

    @show_camera_gui.on_update
    def _(_event):
        with update_lock:
            for handle in camera_handles:
                handle.visible = bool(show_camera_gui.value)

    @follow_camera_gui.on_update
    def _(_event):
        if bool(follow_camera_gui.value):
            sample = camera_sample_at(
                prepared.cameras, int(frame_gui.value))
            for client in list(server.get_clients().values()):
                set_client_to_camera(client, sample)

    @view_camera_gui.on_click
    def _(event):
        sample = camera_sample_at(
            prepared.cameras, int(frame_gui.value))
        client = getattr(event, "client", None)
        if client is not None:
            set_client_to_camera(client, sample)
        else:
            for connected in list(server.get_clients().values()):
                set_client_to_camera(connected, sample)

    @show_grid_gui.on_update
    def _(_event):
        with update_lock:
            grid.visible = bool(show_grid_gui.value)

    export_start_lock = threading.Lock()
    export_started = False
    export_thread = None
    export_cancel = threading.Event()

    def start_camera_export(client):
        nonlocal export_started, export_thread
        if args.export_video is None:
            return
        with export_start_lock:
            if export_started:
                return
            export_started = True
        if not hasattr(client, "get_render"):
            message = "connected Viser client does not support browser rendering"
            video_status_gui.content = f"Camera video failed: {message}"
            print(f"error: {message}", file=sys.stderr)
            return
        with update_lock:
            export_canonical = str(canonical_gui.value)
            export_visibility = {
                "registered_meshes": bool(show_mesh_gui.value),
                "hawor_surfaces": bool(show_hand_gui.value),
                "hawor_skeletons": bool(show_skeleton_gui.value),
                "simple_joint": bool(show_joint_gui.value),
                "da3_camera": bool(show_camera_gui.value),
                "ground_grid": bool(show_grid_gui.value),
            }
        capture_settings = {
            "canonical_moving_mesh": export_canonical,
            "visibility": {
                key: value for key, value in export_visibility.items()
                if key != "da3_camera"
            },
        }

        def update_video_status(message: str):
            video_status_gui.content = f"Camera video: {message}"

        def frame_provider(frame_id: int, sample: CameraSample):
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError(
                    "--export-video needs OpenCV in the Viser environment") from exc
            spec = camera_render_spec(
                sample.intrinsics, prepared.image_hw, args.video_max_side)
            viewport_h = int(getattr(client.camera, "image_height", 0) or 0)
            viewport_w = int(getattr(client.camera, "image_width", 0) or 0)
            if ((viewport_h > 0 and spec.render_hw[0] > viewport_h)
                    or (viewport_w > 0 and spec.render_hw[1] > viewport_w)):
                raise RuntimeError(
                    f"calibrated overscan {spec.render_hw[1]}x"
                    f"{spec.render_hw[0]} exceeds the connected browser viewport "
                    f"{viewport_w}x{viewport_h}; enlarge the browser or lower "
                    "--video-max-side")
            with update_lock:
                render_frame(
                    frame_id, canonical_override=export_canonical,
                    update_follow_clients=False,
                    visibility_override=export_visibility)
                for handle in camera_handles:
                    handle.visible = False
                if hasattr(server, "flush"):
                    server.flush()
                pose = sample.camera_to_reference
                try:
                    raw = _client_render_with_timeout(
                        client, args.render_timeout,
                        height=spec.render_hw[0],
                        width=spec.render_hw[1],
                        wxyz=_rotation_to_wxyz(pose[:3, :3]),
                        position=pose[:3, 3] + display_offset,
                        fov=spec.vertical_fov,
                        transport_format="jpeg",
                    )
                finally:
                    for handle in camera_handles:
                        handle.visible = bool(show_camera_gui.value)
                    if hasattr(server, "flush"):
                        server.flush()
                raw = np.asarray(raw)
                if raw.shape[:2] != spec.render_hw:
                    raise ValueError(
                        f"Viser returned {raw.shape[:2]} for requested "
                        f"{spec.render_hw}")
                return cv2.remap(
                    raw[..., :3], spec.map_x, spec.map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(25, 25, 28),
                )

        def worker():
            nonlocal export_started
            was_playing = bool(play_gui.value)
            play_gui.value = False
            completed = False
            try:
                update_video_status("exporting from the connected browser...")
                write_camera_video(
                    args, prepared, frame_provider,
                    status_callback=update_video_status,
                    cancel_event=export_cancel,
                    capture_settings=capture_settings)
                completed = True
            except Exception as exc:
                update_video_status(f"failed: {exc}")
                print(f"camera video export failed: {exc}", file=sys.stderr)
            finally:
                try:
                    with update_lock:
                        render_frame(
                            int(frame_gui.value), update_follow_clients=False)
                except Exception as exc:
                    print(f"warning: could not restore Viser frame: {exc}",
                          file=sys.stderr)
                finally:
                    play_gui.value = was_playing
                    if not completed and not export_cancel.is_set():
                        with export_start_lock:
                            export_started = False

        export_thread = threading.Thread(
            target=worker,
            name="stage61-da3-camera-video",
            daemon=True,
        )
        export_thread.start()

    @server.on_client_connect
    def _(client):
        if bool(follow_camera_gui.value):
            set_client_to_camera(
                client, camera_sample_at(
                    prepared.cameras, int(frame_gui.value)))
        else:
            client.camera.position = (
                np.array([1.35, -1.7, 1.15]) * scene_diagonal)
            client.camera.look_at = np.zeros(3)
            client.camera.up_direction = np.array([0.0, 0.0, 1.0])
        start_camera_export(client)

    render_frame(initial_frame)
    print(f"Viser scene: http://localhost:{args.port}")
    print("Use Play/Frame, camera, and Canonical moving mesh controls. Ctrl-C to exit.")
    if planned_video_path is not None:
        print(f"Camera video: connect a browser to start {planned_video_path}")
    started = last_step = time.monotonic()
    try:
        while (args.viser_timeout <= 0
               or time.monotonic() - started < args.viser_timeout
               or (export_thread is not None and export_thread.is_alive())):
            now = time.monotonic()
            fps = max(float(fps_gui.value), 1e-3)
            if bool(play_gui.value) and now - last_step >= 1.0 / fps:
                steps = max(1, int((now - last_step) * fps))
                last_step += steps / fps
                index = _nearest_frame_index(
                    prepared.frame_ids, int(frame_gui.value))
                next_index = index + steps
                if next_index >= len(prepared.frame_ids):
                    if bool(loop_gui.value):
                        next_index %= len(prepared.frame_ids)
                    else:
                        next_index = len(prepared.frame_ids) - 1
                        play_gui.value = False
                frame_id = int(prepared.frame_ids[next_index])
                frame_gui.value = frame_id
                render_frame(frame_id)
            elif not bool(play_gui.value):
                last_step = now
            time.sleep(0.01)
    except KeyboardInterrupt:
        export_cancel.set()
        if export_thread is not None and export_thread.is_alive():
            print("\nCancelling camera video export...")
            export_thread.join(timeout=args.render_timeout + 2.0)
        print("\nStopped.")
    return server


def print_summary(prepared: PreparedScene, initial_mesh_frame: str):
    print(f"scene:              {prepared.scene_dir.name}")
    print(f"HaWoR input:        {prepared.hawor_name}")
    print(f"hand sides:         {', '.join(prepared.hands.sides)}")
    print(f"hand coordinates:   {', '.join(prepared.hands.coordinate_sources)}")
    print(f"driver hand:        {prepared.driver_side} wrist")
    print(f"animation frames:   {len(prepared.frame_ids)} "
          f"({prepared.frame_ids[0]:06d}..{prepared.frame_ids[-1]:06d})")
    native_camera_count = sum(
        camera_sample_at(prepared.cameras, int(frame)).native
        for frame in prepared.frame_ids)
    print(f"DA3 cameras:        {len(prepared.cameras.frame_ids)} native records, "
          f"{native_camera_count}/{len(prepared.frame_ids)} displayed frames native")
    print(f"camera mapping:     direct numeric frame IDs; missing IDs interpolate/hold")
    print(f"camera image size:  {prepared.image_hw[1]}x{prepared.image_hw[0]}")
    print(f"registered mesh:    "
          f"{prepared.meshes.registered_mesh_path.relative_to(prepared.scene_dir)}")
    print(f"static geometry:    {prepared.meshes.static.geometry} (shared, unchanged)")
    print(f"initial mesh frame: {initial_mesh_frame}")
    for label, record in prepared.joints.items():
        first = prepared.meshes.moving[label]["first"]
        last = prepared.meshes.moving[label]["last"]
        motion = prepared.motions[label]
        anchor_frames, anchor_states = _sparse_states(record)
        anchor_error = max(
            abs(motion.state_at(int(frame)) - float(state))
            for frame, state in zip(anchor_frames, anchor_states)
            if prepared.frame_ids[0] <= frame <= prepared.frame_ids[-1]
        ) if np.any((anchor_frames >= prepared.frame_ids[0])
                    & (anchor_frames <= prepared.frame_ids[-1])) else 0.0
        print(
            f"[{label}] {record['type']}: native={first.frame_id:06d}/"
            f"{last.frame_id:06d}, q={motion.states.min():+.5g}.."
            f"{motion.states.max():+.5g}, anchor_error={anchor_error:.3g}")
        print(f"  driver signal: {motion.velocity_kind}")
        for start, end, mode in motion.segment_modes:
            print(f"  {start:06d}..{end:06d}: {mode}")
    if prepared.hands.skipped_nonfinite:
        print(f"skipped hands:      {prepared.hands.skipped_nonfinite} non-finite")
    if prepared.hands.skipped_without_pose:
        print(f"skipped hands:      {prepared.hands.skipped_without_pose} without DA3 pose")


def run(args):
    prepared = prepare_scene(args)
    print_summary(prepared, args.mesh_frame)
    if args.export_video is not None:
        print(f"camera video:       "
              f"{_video_output_path(prepared.scene_dir, args.export_video)}")
    if args.dry_run:
        print("dry run: validated inputs, DA3 cameras, and dense motion; "
              "Viser/video export were not started")
        return prepared
    return run_viewer(args, prepared)


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
