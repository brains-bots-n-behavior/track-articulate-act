#!/usr/bin/env python
"""Stage 63: test whether HaWoR hand contact can drive an estimated joint.

This viewer loads the registered object meshes and ``simple_joint/joints.json``
in the same Stage-31 reference coordinate frame used by Stages 61 and 62.  The
selected first/last moving mesh is attached to the estimated MuJoCo hinge or
slide joint, but its joint trajectory is *not* prescribed.  There are no
actuators and gravity is zero.  The moving part can therefore be driven only by
contact with a replayed ``hawor_scaled`` hand.

Two drive modes exist.  The default ``--drive-mode velocity`` is a grip model:
while a hand surface intersects the moving part, the hand's measured velocity is
projected onto the estimated joint -- orbital angular velocity about a revolute
axis, or linear velocity along a prismatic axis -- and the part is advanced at
exactly that rate.  The instant contact is lost the part stops dead, even while
the hand keeps moving.  ``--drive-mode contact-force`` restores the original
behaviour, where contact impulses accelerate a free 1-DOF inertia that then
coasts.

HaWoR is a deforming MANO surface.  To retain the measured surface at every
frame, this script creates one rigid, non-convex MuJoCo flex surface per hand
sample beneath a kinematic (mocap) palm body.  Only the nearest sample's flex
is visible and collision-enabled; the palm pose is interpolated at the physics
rate.  The moving object also uses a rigid flex collider, avoiding the convex
hull approximation of ordinary MuJoCo mesh geoms.

Collision masks allow only hand-to-moving-part contacts.  The registered
static mesh and simple-joint axes are visual-only, so they cannot accidentally
drive or block the moving part.

Examples::

    python scripts/63_mujoco_retarget.py --scene-dir data/trashbin --play
    python scripts/63_mujoco_retarget.py --scene-dir data/dryer \
        --label dryer_door --play
    python scripts/63_mujoco_retarget.py --scene-dir data/trashbin --dry-run
    python scripts/63_mujoco_retarget.py --scene-dir data/trashbin --export-video

``--export-video`` skips the viewer entirely and renders the same replay
offscreen through the DA3 camera, placing the MuJoCo camera from
``da3/cameras.npz`` and reproducing the full ``da3/intrinsics.npz`` pinhole --
unequal focal lengths, skew and an off-center principal point included -- by
rendering an overscanned centered image and remapping it, reusing Stage 61's
``camera_render_spec``.  It additionally needs OpenCV and Pillow.  The
finished MP4 is also mirrored to ``<scene>/render_all/mujoco-retarget.mp4``
alongside the other Stage-6x renders; pass ``--no-render-all-copy`` to skip
that copy.

In the viewer, Space pauses/resumes and R resets the object and hand replay.
The replay loops by default; pass ``--no-loop`` to stop after one pass.  MuJoCo
contact points and force arrows are hidden by default because the flex colliders
generate enough of them to hide the meshes; pass ``--show-contacts`` to draw
them.  On macOS, launch with ``mjpython`` rather than ``python``.  Runtime dependencies
are numpy, trimesh, and mujoco 3.x (rigid flex collision is required).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
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


RENDER_ALL_COPY = Path("render_all") / "mujoco-retarget.mp4"

STATIC_RGBA = (0.64, 0.67, 0.72, 1.0)
MOVING_RGBA = (0.95, 0.50, 0.18, 1.0)
MOVING_CONTACT_RGBA = (0.95, 0.16, 0.10, 1.0)
HAND_RGBA = {
    "left": (0.28, 0.62, 0.96, 0.58),
    "right": (0.18, 0.42, 0.88, 0.58),
}
JOINT_RGBA = {
    "revolute": (0.14, 0.72, 0.30, 1.0),
    "prismatic": (0.20, 0.46, 0.95, 1.0),
}

# A pair is checked when either geom's contype intersects the other's
# conaffinity.  No other geom receives either of these bit combinations.
MOVING_CONTACT_BIT = 1
HAND_CONTACT_BIT = 2

# Fixed camera whose pose/fovy are rewritten per frame from da3/cameras.npz.
DA3_CAMERA_NAME = "da3_camera"


def _load_stage61():
    """Load Stage 61 without importing Viser or OpenCV."""
    path = SCRIPT_DIR / "61_render_all.py"
    spec = importlib.util.spec_from_file_location(
        "articulate4d_stage61_for_mujoco", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Stage 61 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE61 = _load_stage61()


@dataclass(frozen=True)
class HandMeshFrame:
    """One MANO surface expressed relative to a tracked palm body."""

    frame_id: int
    side: str
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    vertices_local: np.ndarray
    faces: np.ndarray
    detected: bool


@dataclass(frozen=True)
class PreparedScene:
    scene_dir: Path
    hawor_name: str
    meshes: Any
    joints: dict[str, dict]
    hand_frames: dict[str, tuple[HandMeshFrame, ...]]
    start_frame: int
    end_frame: int
    playback_fps: float
    scene_bounds: np.ndarray

    @property
    def duration(self) -> float:
        return (self.end_frame - self.start_frame) / self.playback_fps


@dataclass(frozen=True)
class JointBuildSpec:
    label: str
    safe_label: str
    joint_name: str
    geom_name: str
    flex_name: str
    range_min: float
    range_max: float
    canonical_state: float


@dataclass(frozen=True)
class ModelBundle:
    xml: str
    assets: dict[str, bytes]
    joints: dict[str, JointBuildSpec]
    moving_flex_names: tuple[str, ...]
    hand_flex_names: dict[str, tuple[str, ...]]
    hand_body_names: dict[str, str]


@dataclass(frozen=True)
class RuntimeIds:
    joint_qpos: dict[str, int]
    joint_dof: dict[str, int]
    moving_geoms: np.ndarray
    moving_geom_by_label: dict[str, int]
    moving_flexes: np.ndarray
    moving_flex_by_label: dict[str, int]
    hand_flexes: dict[str, np.ndarray]
    hand_mocap: dict[str, int]
    hand_side_by_flex: dict[int, str]


@dataclass(frozen=True)
class HandMotion:
    """Finite-difference rigid motion of one mocap palm over a physics step."""

    position: np.ndarray
    linear: np.ndarray
    angular: np.ndarray

    def velocity_at(self, point: np.ndarray) -> np.ndarray:
        """Velocity of a point that moves rigidly with the palm."""
        offset = np.asarray(point, dtype=np.float64).reshape(3) - self.position
        return self.linear + np.cross(self.angular, offset)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Scene directory, for example data/trashbin",
    )
    parser.add_argument(
        "--hawor-name",
        default="hawor_scaled",
        help="HaWoR directory inside the scene",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        help="Moving label to simulate; repeat for multiple labels (default: all)",
    )
    parser.add_argument(
        "--mesh-frame",
        choices=("first", "last"),
        default="first",
        help="Canonical registered moving mesh attached to each joint",
    )
    parser.add_argument(
        "--hands",
        choices=("all", "left", "right"),
        default="all",
        help="HaWoR hand surfaces that can contact the moving part",
    )
    parser.add_argument(
        "--hand-coordinate-source",
        choices=("auto", "world", "camera"),
        default="auto",
        help="Prefer saved world hands or force DA3 camera conversion",
    )
    parser.add_argument(
        "--detected-only",
        action="store_true",
        help="Exclude HaWoR in-filled hand samples",
    )
    parser.add_argument("--start-frame", type=int, help="First replayed HaWoR frame")
    parser.add_argument("--end-frame", type=int, help="Last replayed HaWoR frame")
    parser.add_argument(
        "--playback-fps",
        type=float,
        help="Trajectory FPS; defaults to hawor_scaled config, then 24",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=240.0,
        help="MuJoCo integration frequency",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wall-clock replay speed",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Start replay immediately instead of paused",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset the physics state and repeat after the final hand frame",
    )
    parser.add_argument(
        "--show-contacts",
        action="store_true",
        help="Draw MuJoCo contact points and force arrows, which are hidden "
        "by default because dense flex contacts obscure the scene",
    )
    parser.add_argument(
        "--drive-mode",
        choices=("velocity", "contact-force"),
        default="velocity",
        help=(
            "velocity: the part tracks the contacting hand's joint-relative "
            "speed and stops when contact ends; contact-force: the part is a "
            "free inertia pushed by contact impulses"
        ),
    )
    parser.add_argument(
        "--drive-point",
        choices=("wrist", "contact"),
        default="wrist",
        help=(
            "Hand point whose velocity is projected onto the joint: the mocap "
            "palm origin, or the mean hand/part contact position"
        ),
    )
    parser.add_argument(
        "--contact-hold",
        type=float,
        default=0.0,
        help=(
            "Seconds of grace after contact is lost during which the last "
            "rate is held; zero stops the part immediately"
        ),
    )
    parser.add_argument(
        "--object-mass",
        type=float,
        default=1.0,
        help="Mass assigned to each moving part",
    )
    parser.add_argument(
        "--joint-damping",
        type=float,
        default=0.0,
        help="Optional passive joint damping; zero keeps contact as the only drive",
    )
    parser.add_argument(
        "--joint-friction",
        type=float,
        default=0.0,
        help="Optional joint friction loss; zero keeps contact as the only drive",
    )
    parser.add_argument(
        "--joint-armature",
        type=float,
        default=1e-4,
        help="Small numerical rotor inertia (not a drive force)",
    )
    parser.add_argument(
        "--range-min",
        type=float,
        help=(
            "Joint offset lower limit: degrees for hinge, Stage-31 units "
            "for slide; only valid with one --label"
        ),
    )
    parser.add_argument(
        "--range-max",
        type=float,
        help=(
            "Joint offset upper limit: degrees for hinge, Stage-31 units "
            "for slide; only valid with one --label"
        ),
    )
    parser.add_argument(
        "--range-margin",
        type=float,
        default=0.05,
        help="Fraction of observed state span added to inferred limits",
    )
    parser.add_argument(
        "--contact-friction",
        type=float,
        nargs=3,
        default=(1.0, 0.01, 0.001),
        metavar=("SLIDE", "TORSION", "ROLL"),
        help="Hand/moving mesh contact friction",
    )
    parser.add_argument(
        "--contact-softness",
        type=float,
        default=0.01,
        help="Contact time constant used as the first solref value",
    )
    parser.add_argument(
        "--contact-margin-scale",
        type=float,
        default=0.0,
        help=(
            "Contact tolerance as a fraction of each moving-mesh diagonal; "
            "use zero for strict surface intersection"
        ),
    )
    parser.add_argument(
        "--contact-thickness-scale",
        type=float,
        default=0.001,
        help=(
            "Radius of the hand/object flex surfaces as a fraction of the "
            "moving-mesh diagonal"
        ),
    )
    parser.add_argument(
        "--max-hull-vertices",
        type=int,
        default=96,
        help="Maximum convex-hull vertices compiled for visual mesh assets",
    )
    parser.add_argument(
        "--axis-length-scale",
        type=float,
        default=0.75,
        help="Joint-axis half-length in moving-mesh diagonals",
    )
    parser.add_argument(
        "--axis-radius-scale",
        type=float,
        default=0.012,
        help="Joint-axis radius in moving-mesh diagonals",
    )
    parser.add_argument(
        "--export-video",
        type=Path,
        nargs="?",
        const=Path("mujoco_retarget/da3_camera.mp4"),
        metavar="PATH",
        help=(
            "Render an MP4 offscreen through the DA3 camera instead of "
            "opening the viewer; without PATH, write "
            "mujoco_retarget/da3_camera.mp4 inside the scene"
        ),
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        help="Export FPS (default: the playback FPS)",
    )
    parser.add_argument(
        "--video-max-side",
        type=int,
        default=720,
        help="Maximum exported width or height; DA3 intrinsics are rescaled exactly",
    )
    parser.add_argument(
        "--video-codec",
        default="mp4v",
        help="Four-character OpenCV codec for --export-video",
    )
    parser.add_argument(
        "--overwrite-video",
        action="store_true",
        help="Atomically replace an existing --export-video output",
    )
    parser.add_argument(
        "--no-render-all-copy",
        action="store_true",
        help=(
            f"Skip the extra copy written to <scene>/"
            f"{RENDER_ALL_COPY.as_posix()}, which is replaced on every export"
        ),
    )
    parser.add_argument("--azimuth", type=float, default=35.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument(
        "--distance",
        type=float,
        help="Viewer camera distance (default: fit object and hand trajectory)",
    )
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Viewer camera target (default: center of scene bounds)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize inputs without importing MuJoCo",
    )
    return parser.parse_args(argv)


def _safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in str(value)
    )
    return result or "unnamed"


def _normalized(value: np.ndarray, description: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError(
            f"cannot construct hand palm pose: {description} is degenerate"
        )
    return vector / norm


def palm_pose(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable palm position and body-to-reference rotation."""
    points = np.asarray(joints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 18:
        raise ValueError("a MANO hand needs at least 18 finite 3D joints")
    if not np.isfinite(points).all():
        raise ValueError("MANO joints contain non-finite values")

    origin = points[0].copy()
    forward = _normalized(points[[5, 9, 13, 17]].mean(axis=0) - origin, "palm forward")
    across = points[5] - points[17]
    across -= float(across @ forward) * forward
    across = _normalized(across, "palm width")
    normal = _normalized(np.cross(across, forward), "palm normal")
    # Re-orthogonalize so the columns form a proper body-to-world rotation.
    across = _normalized(np.cross(forward, normal), "palm width")
    rotation = np.column_stack((across, forward, normal))
    quaternion = STAGE61._rotation_to_wxyz(rotation)
    return origin.astype(np.float32), quaternion.astype(np.float32)


def _quaternion_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return STAGE61._quat_wxyz_to_rotation(quaternion_wxyz)


def prepare_hand_frames(hands, selected: str) -> dict[str, tuple[HandMeshFrame, ...]]:
    sides = hands.sides if selected == "all" else [selected]
    missing = [side for side in sides if side not in hands.sides]
    if missing:
        raise ValueError(
            f"requested HaWoR hand(s) unavailable: {missing}; loaded {hands.sides}"
        )
    result: dict[str, tuple[HandMeshFrame, ...]] = {}
    for side in sides:
        frames = []
        faces = np.asarray(hands.faces[side], dtype=np.int32)
        for sample in hands.samples[side]:
            position, quaternion = palm_pose(sample.joints)
            rotation = _quaternion_to_rotation(quaternion)
            local = (
                (np.asarray(sample.vertices, dtype=np.float64) - position) @ rotation
            ).astype(np.float32)
            frames.append(
                HandMeshFrame(
                    frame_id=int(sample.frame_id),
                    side=side,
                    position=position,
                    quaternion_wxyz=quaternion,
                    vertices_local=local,
                    faces=faces,
                    detected=bool(sample.detected),
                )
            )
        result[side] = tuple(frames)
    return result


def _scene_bounds(meshes, hand_frames) -> np.ndarray:
    arrays = [meshes.static.vertices]
    arrays.extend(
        choice.vertices
        for choices in meshes.moving.values()
        for choice in choices.values()
    )
    # Sampling every tenth hand frame is sufficient for camera fitting while
    # avoiding a large temporary concatenation on long videos.
    for frames in hand_frames.values():
        for frame in frames[::10]:
            rotation = _quaternion_to_rotation(frame.quaternion_wxyz)
            arrays.append(frame.vertices_local @ rotation.T + frame.position)
        if frames and (len(frames) - 1) % 10:
            frame = frames[-1]
            rotation = _quaternion_to_rotation(frame.quaternion_wxyz)
            arrays.append(frame.vertices_local @ rotation.T + frame.position)
    low = np.min([np.asarray(values).min(axis=0) for values in arrays], axis=0)
    high = np.max([np.asarray(values).max(axis=0) for values in arrays], axis=0)
    bounds = np.stack((low, high)).astype(np.float64)
    if not np.isfinite(bounds).all() or np.linalg.norm(high - low) < 1e-9:
        raise ValueError("object/hand scene bounds are degenerate")
    return bounds


def validate_args(args):
    if args.start_frame is not None and args.end_frame is not None:
        if args.start_frame > args.end_frame:
            raise ValueError("--start-frame cannot exceed --end-frame")
    positive = {
        "--physics-hz": args.physics_hz,
        "--speed": args.speed,
        "--object-mass": args.object_mass,
        "--contact-softness": args.contact_softness,
        "--axis-length-scale": args.axis_length_scale,
        "--axis-radius-scale": args.axis_radius_scale,
        "--contact-thickness-scale": args.contact_thickness_scale,
    }
    for flag, value in positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{flag} must be positive")
    nonnegative = {
        "--joint-damping": args.joint_damping,
        "--joint-friction": args.joint_friction,
        "--joint-armature": args.joint_armature,
        "--range-margin": args.range_margin,
        "--contact-hold": args.contact_hold,
        "--contact-margin-scale": args.contact_margin_scale,
    }
    for flag, value in nonnegative.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{flag} cannot be negative")
    if args.max_hull_vertices < 4:
        raise ValueError("--max-hull-vertices must be at least 4")
    friction = np.asarray(args.contact_friction, dtype=np.float64)
    if not np.isfinite(friction).all() or np.any(friction < 0):
        raise ValueError("--contact-friction values must be finite and non-negative")
    if (args.range_min is None) != (args.range_max is None):
        raise ValueError("--range-min and --range-max must be supplied together")
    if args.range_min is not None:
        if args.labels is None or len(args.labels) != 1:
            raise ValueError("custom joint limits require exactly one --label")
        if (
            not np.isfinite([args.range_min, args.range_max]).all()
            or args.range_min >= args.range_max
        ):
            raise ValueError("custom joint limits require finite MIN < MAX")
    if args.distance is not None:
        if not np.isfinite(args.distance) or args.distance <= 0:
            raise ValueError("--distance must be positive")
    if args.video_fps is not None:
        if not np.isfinite(args.video_fps) or args.video_fps <= 0:
            raise ValueError("--video-fps must be positive")
    if args.video_max_side <= 0:
        raise ValueError("--video-max-side must be positive")
    if len(args.video_codec) != 4:
        raise ValueError("--video-codec must contain exactly four characters")
    if args.export_video is not None:
        if args.export_video.suffix.lower() != ".mp4":
            raise ValueError("--export-video PATH must end in .mp4")


def prepare_scene(args) -> PreparedScene:
    validate_args(args)
    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene directory does not exist: {scene_dir}")

    simple_payload, joints = STAGE61.load_simple_joints(scene_dir, args.labels)
    meshes = STAGE61.load_registered_meshes(scene_dir, simple_payload, joints)
    hands_world = STAGE61.load_hawor_trajectory(
        scene_dir,
        hawor_name=args.hawor_name,
        coordinate_source=args.hand_coordinate_source,
        detected_only=args.detected_only,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    hands = STAGE61.hands_world_to_reference(hands_world, meshes.reference_to_world)
    hand_frames = prepare_hand_frames(hands, args.hands)
    frame_ids = sorted(
        {frame.frame_id for frames in hand_frames.values() for frame in frames}
    )
    if not frame_ids:
        raise ValueError("no selected HaWoR hand frames remain")
    playback_fps = STAGE61._load_playback_fps(
        scene_dir, args.hawor_name, args.playback_fps
    )
    bounds = _scene_bounds(meshes, hand_frames)
    return PreparedScene(
        scene_dir=scene_dir,
        hawor_name=args.hawor_name,
        meshes=meshes,
        joints=joints,
        hand_frames=hand_frames,
        start_frame=int(frame_ids[0]),
        end_frame=int(frame_ids[-1]),
        playback_fps=float(playback_fps),
        scene_bounds=bounds,
    )


def resolve_joint_range(
    record: dict, canonical_frame: int, args
) -> tuple[float, float, float]:
    """Return limits and canonical q, all as offsets from the mesh pose."""
    canonical = STAGE61.sparse_state_at(record, canonical_frame)
    if args.range_min is not None:
        low, high = float(args.range_min), float(args.range_max)
        if record["type"] == "revolute":
            low, high = map(float, np.deg2rad([low, high]))
        return low, high, canonical

    states = (
        np.asarray(
            [float(item["q"]) for item in record["joint_states"]],
            dtype=np.float64,
        )
        - canonical
    )
    low, high = float(states.min()), float(states.max())
    span = high - low
    if span < 1e-8:
        fallback = float(np.deg2rad(90.0)) if record["type"] == "revolute" else 0.25
        low, high, span = 0.0, fallback, fallback
    margin = args.range_margin * span
    low -= margin
    high += margin
    # The canonical mesh is qpos=0.  Include it despite small interpolation
    # error in the sparse state record.
    low = min(low, 0.0)
    high = max(high, 0.0)
    if not np.isfinite([low, high]).all() or high - low < 1e-10:
        raise ValueError("resolved MuJoCo joint range is invalid")
    return low, high, canonical


def _number(values) -> str:
    return " ".join(f"{float(value):.17g}" for value in values)


def _integers(values) -> str:
    return " ".join(str(int(value)) for value in np.asarray(values).reshape(-1))


def _mesh_bytes(vertices: np.ndarray, faces: np.ndarray, trimesh) -> bytes:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    payload = mesh.export(file_type="stl")
    return payload.encode() if isinstance(payload, str) else bytes(payload)


def _moving_inertia(
    vertices_local: np.ndarray, mass: float
) -> tuple[np.ndarray, float]:
    vertices = np.asarray(vertices_local, dtype=np.float64)
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    size = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-5)
    # Axis-aligned box inertia is robust even when a reconstructed surface is
    # open or inconsistently wound.
    inertia = (
        mass
        / 12.0
        * np.array(
            [
                size[1] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[1] ** 2,
            ]
        )
    )
    inertia = np.maximum(inertia, mass * 1e-8)
    return center, float(max(inertia.max(), mass * 1e-8))


def build_model_bundle(
    prepared: PreparedScene, args, trimesh, offscreen_hw=None
) -> ModelBundle:
    """Create one in-memory MJCF plus STL assets for the interactive replay.

    ``offscreen_hw`` sizes the offscreen framebuffer for ``--export-video``;
    MuJoCo refuses to allocate a renderer larger than this compiled buffer.
    """
    assets: dict[str, bytes] = {}
    mesh_xml = []
    world_xml = []
    deformable_xml = []
    joint_specs: dict[str, JointBuildSpec] = {}
    moving_flex_names = []
    hand_flex_names: dict[str, tuple[str, ...]] = {}
    hand_body_names: dict[str, str] = {}

    static_file = "object_static.stl"
    assets[static_file] = _mesh_bytes(
        prepared.meshes.static.vertices,
        prepared.meshes.static.faces,
        trimesh,
    )
    mesh_xml.append(
        f'<mesh name="object_static_asset" file="{static_file}" '
        f'maxhullvert="{args.max_hull_vertices}"/>'
    )
    world_xml.append(
        '<body name="object_static">\n'
        '  <geom name="object_static_visual" type="mesh" '
        'mesh="object_static_asset" density="0" contype="0" conaffinity="0" '
        f'rgba="{_number(STATIC_RGBA)}"/>\n'
        "</body>"
    )

    for index, (label, record) in enumerate(prepared.joints.items()):
        safe = f"{index}_{_safe_name(label)}"
        mesh = prepared.meshes.moving[label][args.mesh_frame]
        canonical_frame = int(mesh.frame_id)
        low, high, canonical = resolve_joint_range(record, canonical_frame, args)
        axis = np.asarray(record["axis_direction"], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        if record["type"] == "revolute":
            body_position = np.asarray(record["axis_point"], dtype=np.float64)
            vertices_local = np.asarray(mesh.vertices, dtype=np.float64) - body_position
            joint_type = "hinge"
        else:
            body_position = np.zeros(3, dtype=np.float64)
            vertices_local = np.asarray(mesh.vertices, dtype=np.float64)
            joint_type = "slide"

        asset_name = f"moving_asset_{safe}"
        asset_file = f"moving_{safe}.stl"
        geom_name = f"moving_geom_{safe}"
        flex_name = f"moving_flex_{safe}"
        joint_name = f"articulation_{safe}"
        assets[asset_file] = _mesh_bytes(vertices_local, mesh.faces, trimesh)
        mesh_xml.append(
            f'<mesh name="{asset_name}" file="{asset_file}" '
            f'maxhullvert="{args.max_hull_vertices}"/>'
        )

        inertial_position, inertia = _moving_inertia(vertices_local, args.object_mass)
        diagonal = max(float(mesh.diagonal), 1e-4)
        contact_margin = args.contact_margin_scale * diagonal
        contact_radius = args.contact_thickness_scale * diagonal
        raw_center = record.get("position", record.get("axis_point"))
        center = np.asarray(raw_center, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            center = np.asarray(mesh.vertices, dtype=np.float64).mean(axis=0)
        half_length = args.axis_length_scale * diagonal
        radius = max(args.axis_radius_scale * diagonal, 1e-5)
        axis_start = center - half_length * axis
        axis_end = center + half_length * axis
        color = JOINT_RGBA[record["type"]]

        world_xml.append(f"""
<body name="moving_body_{safe}" pos="{_number(body_position)}">
  <joint name="{joint_name}" type="{joint_type}" axis="{_number(axis)}"
         limited="true" range="{_number((low, high))}"
         damping="{args.joint_damping:.17g}"
         frictionloss="{args.joint_friction:.17g}"
         armature="{args.joint_armature:.17g}"/>
  <inertial pos="{_number(inertial_position)}" mass="{args.object_mass:.17g}"
            diaginertia="{_number((inertia, inertia, inertia))}"/>
  <geom name="{geom_name}" type="mesh" mesh="{asset_name}" density="0"
        contype="0" conaffinity="0"
        rgba="{_number(MOVING_RGBA)}"/>
</body>
<body name="joint_visual_{safe}">
  <site name="joint_axis_{safe}" type="capsule"
        fromto="{_number(np.concatenate((axis_start, axis_end)))}"
        size="{radius:.17g}" rgba="{_number(color)}"/>
  <site name="joint_center_{safe}" type="sphere" pos="{_number(center)}"
        size="{(2.0 * radius):.17g}" rgba="{_number(color)}"/>
</body>""".strip())
        deformable_xml.append(f"""
<flex name="{flex_name}" dim="2" radius="{contact_radius:.17g}"
      body="moving_body_{safe}"
      vertex="{_number(vertices_local.reshape(-1))}"
      element="{_integers(mesh.faces)}" rgba="0 0 0 0">
  <contact contype="{MOVING_CONTACT_BIT}"
           conaffinity="{HAND_CONTACT_BIT}" condim="4"
           friction="{_number(args.contact_friction)}"
           margin="{contact_margin:.17g}"
           solref="{args.contact_softness:.17g} 1"/>
</flex>""".strip())
        moving_flex_names.append(flex_name)
        joint_specs[label] = JointBuildSpec(
            label=label,
            safe_label=safe,
            joint_name=joint_name,
            geom_name=geom_name,
            flex_name=flex_name,
            range_min=low,
            range_max=high,
            canonical_state=canonical,
        )

    for side, frames in prepared.hand_frames.items():
        body_name = f"hand_body_{side}"
        hand_body_names[side] = body_name
        names = []
        moving_diagonals = [
            max(float(prepared.meshes.moving[label][args.mesh_frame].diagonal), 1e-4)
            for label in prepared.joints
        ]
        hand_radius = args.contact_thickness_scale * min(moving_diagonals)
        for index, frame in enumerate(frames):
            flex_name = f"hand_flex_{side}_{index:06d}"
            deformable_xml.append(f"""
<flex name="{flex_name}" dim="2" radius="{hand_radius:.17g}"
      body="{body_name}"
      vertex="{_number(frame.vertices_local.reshape(-1))}"
      element="{_integers(frame.faces)}" rgba="0 0 0 0">
  <contact contype="{HAND_CONTACT_BIT}"
           conaffinity="{MOVING_CONTACT_BIT}" condim="4"
           friction="{_number(args.contact_friction)}"
           solref="{args.contact_softness:.17g} 1"/>
</flex>""".strip())
            names.append(flex_name)
        hand_flex_names[side] = tuple(names)
        world_xml.append(f'<body name="{body_name}" mocap="true"/>')

    # A world camera is always present so the offscreen exporter can drive it.
    world_xml.append(
        f'<camera name="{DA3_CAMERA_NAME}" mode="fixed" pos="0 0 0" '
        'quat="1 0 0 0" fovy="45"/>'
    )

    if offscreen_hw is None:
        offscreen = ""
    else:
        offscreen_h, offscreen_w = (int(value) for value in offscreen_hw)
        if offscreen_h <= 0 or offscreen_w <= 0:
            raise ValueError("offscreen framebuffer dimensions must be positive")
        offscreen = f' offwidth="{offscreen_w}" offheight="{offscreen_h}"'

    timestep = 1.0 / args.physics_hz
    xml = f"""<mujoco model="stage63_hawor_contact_retarget">
  <compiler angle="radian" autolimits="true" balanceinertia="true"/>
  <option timestep="{timestep:.17g}" gravity="0 0 0"
          integrator="implicitfast" solver="Newton" cone="elliptic"
          iterations="50" tolerance="1e-10" noslip_iterations="2">
    <flag multiccd="enable"/>
  </option>
  <visual>
    <global azimuth="{args.azimuth:.17g}" elevation="{args.elevation:.17g}"{offscreen}/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8"
               specular="0.35 0.35 0.35"/>
  </visual>
  <asset>
    {chr(10).join(mesh_xml)}
  </asset>
  <worldbody>
    {chr(10).join(world_xml)}
  </worldbody>
  <deformable>
    {chr(10).join(deformable_xml)}
  </deformable>
</mujoco>
"""
    return ModelBundle(
        xml=xml,
        assets=assets,
        joints=joint_specs,
        moving_flex_names=tuple(moving_flex_names),
        hand_flex_names=hand_flex_names,
        hand_body_names=hand_body_names,
    )


def hand_state_at(
    frames: tuple[HandMeshFrame, ...], frame_value: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """Interpolate the palm pose and select the nearest deforming mesh."""
    ids = np.asarray([frame.frame_id for frame in frames], dtype=np.float64)
    position = int(np.searchsorted(ids, frame_value))
    if position <= 0:
        return frames[0].position, frames[0].quaternion_wxyz, 0
    if position >= len(frames):
        index = len(frames) - 1
        return frames[index].position, frames[index].quaternion_wxyz, index
    before = frames[position - 1]
    after = frames[position]
    denominator = max(float(after.frame_id - before.frame_id), 1.0)
    alpha = float(np.clip((frame_value - before.frame_id) / denominator, 0.0, 1.0))
    translation = ((1.0 - alpha) * before.position + alpha * after.position).astype(
        np.float32
    )
    quaternion = STAGE61._slerp_wxyz(
        before.quaternion_wxyz, after.quaternion_wxyz, alpha
    )
    mesh_index = position - 1 if alpha < 0.5 else position
    return translation, quaternion, mesh_index


def _rotation_log(rotation: np.ndarray) -> np.ndarray:
    """Rotation vector (axis times angle) of a proper rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    sine = float(np.sin(angle))
    if angle < 1e-9 or sine < 1e-9:
        # A half-turn inside one physics step is not real hand motion, so the
        # near-pi branch is treated as no rotation rather than guessed at.
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    return axis * (angle / (2.0 * sine))


def hand_motion(previous, current, timestep: float) -> HandMotion:
    """Differentiate one palm pose pair into linear and angular velocity."""
    position = np.asarray(current[0], dtype=np.float64).reshape(3)
    if previous is None or timestep <= 0.0:
        zero = np.zeros(3, dtype=np.float64)
        return HandMotion(position=position, linear=zero, angular=zero)
    before_position = np.asarray(previous[0], dtype=np.float64).reshape(3)
    linear = (position - before_position) / timestep
    relative = (
        _quaternion_to_rotation(current[1]) @ _quaternion_to_rotation(previous[1]).T
    )
    return HandMotion(
        position=position,
        linear=linear,
        angular=_rotation_log(relative) / timestep,
    )


def contacting_hands(
    data, flex_id: int, runtime: RuntimeIds
) -> dict[str, tuple[np.ndarray, int]]:
    """Mean contact point and contact count per hand touching one moving flex.

    Collision masks admit only hand/moving pairs, so any contact naming this
    flex is a hand contact.
    """
    count = int(data.ncon)
    if count <= 0:
        return {}
    pairs = np.asarray(data.contact.flex)[:count]
    positions = np.asarray(data.contact.pos, dtype=np.float64)[:count]
    involved = (pairs[:, 0] == flex_id) | (pairs[:, 1] == flex_id)
    if not involved.any():
        return {}
    partner = np.where(pairs[:, 0] == flex_id, pairs[:, 1], pairs[:, 0])
    grouped: dict[str, list[np.ndarray]] = {}
    for index in np.flatnonzero(involved):
        side = runtime.hand_side_by_flex.get(int(partner[index]))
        if side is not None:
            grouped.setdefault(side, []).append(positions[index])
    return {
        side: (np.mean(points, axis=0), len(points)) for side, points in grouped.items()
    }


def joint_rate_from_velocity(
    record: dict, point: np.ndarray, velocity: np.ndarray, radius_floor: float
) -> float:
    """Project a hand point's velocity onto one estimated joint.

    Prismatic joints take the linear speed along the axis.  Revolute joints
    take the signed orbital angular velocity about the axis, which matches the
    sign convention of Stage 61's ``relative_interval_velocity``.
    """
    axis = np.asarray(record["axis_direction"], dtype=np.float64).reshape(3)
    axis = axis / np.linalg.norm(axis)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    if not np.isfinite(velocity).all():
        return 0.0
    if record["type"] == "prismatic":
        return float(velocity @ axis)
    pivot = np.asarray(record["axis_point"], dtype=np.float64).reshape(3)
    radial = np.asarray(point, dtype=np.float64).reshape(3) - pivot
    radial -= float(radial @ axis) * axis
    radius = float(np.linalg.norm(radial))
    if radius <= radius_floor:
        # On the axis itself no orbital rate is observable.
        return 0.0
    return float(axis @ np.cross(radial, velocity)) / (radius * radius)


def drive_joints(
    data,
    runtime: RuntimeIds,
    prepared: PreparedScene,
    joint_specs: dict[str, JointBuildSpec],
    motions: dict[str, HandMotion],
    state: dict[str, dict[str, float]],
    radius_floors: dict[str, float],
    timestep: float,
    args,
) -> dict[str, bool]:
    """Advance every moving part at its contacting hand's joint-relative rate.

    Contacts must already be current, so call ``mj_forward`` first.  Returns
    whether each label is being driven this step.
    """
    driven: dict[str, bool] = {}
    for label, record in prepared.joints.items():
        spec = joint_specs[label]
        touching = contacting_hands(data, runtime.moving_flex_by_label[label], runtime)
        weight = 0.0
        total = 0.0
        for side, (contact_point, count) in touching.items():
            motion = motions.get(side)
            if motion is None:
                continue
            drive_point = (
                contact_point if args.drive_point == "contact" else motion.position
            )
            rate = joint_rate_from_velocity(
                record,
                drive_point,
                motion.velocity_at(drive_point),
                radius_floors[label],
            )
            # Two hands on one part are averaged by their contact counts.
            total += rate * count
            weight += count
        if weight > 0.0:
            rate = total / weight
            state["rate"][label] = rate
            state["hold"][label] = float(args.contact_hold)
            driven[label] = True
        else:
            remaining = state["hold"][label] - timestep
            state["hold"][label] = max(remaining, 0.0)
            holding = remaining > 0.0
            rate = state["rate"][label] if holding else 0.0
            if not holding:
                state["rate"][label] = 0.0
            driven[label] = holding
        address = runtime.joint_qpos[label]
        target = float(data.qpos[address]) + rate * timestep
        clamped = float(np.clip(target, spec.range_min, spec.range_max))
        data.qpos[address] = clamped
        # A part resting against its limit reports zero rate, not the blocked one.
        data.qvel[runtime.joint_dof[label]] = rate if clamped == target else 0.0
    return driven


def resolve_runtime_ids(mujoco, model, bundle: ModelBundle) -> RuntimeIds:
    joint_qpos = {}
    joint_dof = {}
    moving_geoms = []
    moving_geom_by_label = {}
    moving_flex_by_label = {}
    for label, spec in bundle.joints.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.geom_name)
        flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, spec.flex_name)
        if joint_id < 0 or geom_id < 0 or flex_id < 0:
            raise RuntimeError(f"MuJoCo model omitted joint/geom/flex for {label!r}")
        joint_qpos[label] = int(model.jnt_qposadr[joint_id])
        joint_dof[label] = int(model.jnt_dofadr[joint_id])
        moving_geoms.append(geom_id)
        moving_geom_by_label[label] = int(geom_id)
        moving_flex_by_label[label] = int(flex_id)

    moving_flexes = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
            for name in bundle.moving_flex_names
        ],
        dtype=np.int32,
    )
    if np.any(moving_flexes < 0):
        raise RuntimeError("MuJoCo model omitted moving-part flex colliders")

    hand_flexes = {}
    hand_mocap = {}
    hand_side_by_flex: dict[int, str] = {}
    for side, names in bundle.hand_flex_names.items():
        flex_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
                for name in names
            ],
            dtype=np.int32,
        )
        if np.any(flex_ids < 0):
            raise RuntimeError(f"MuJoCo model omitted {side} hand flexes")
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, bundle.hand_body_names[side]
        )
        mocap_id = int(model.body_mocapid[body_id]) if body_id >= 0 else -1
        if mocap_id < 0:
            raise RuntimeError(f"MuJoCo model omitted {side} mocap body")
        hand_flexes[side] = flex_ids
        hand_mocap[side] = mocap_id
        hand_side_by_flex.update({int(value): side for value in flex_ids})
    return RuntimeIds(
        joint_qpos=joint_qpos,
        joint_dof=joint_dof,
        moving_geoms=np.asarray(moving_geoms, dtype=np.int32),
        moving_geom_by_label=moving_geom_by_label,
        moving_flexes=moving_flexes,
        moving_flex_by_label=moving_flex_by_label,
        hand_flexes=hand_flexes,
        hand_mocap=hand_mocap,
        hand_side_by_flex=hand_side_by_flex,
    )


def set_hand_state(
    model, data, runtime: RuntimeIds, prepared: PreparedScene, frame_value: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Update mocap poses and make exactly one collider active per hand.

    Returns the applied palm poses so the caller can differentiate them.
    """
    applied: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side, frames in prepared.hand_frames.items():
        position, quaternion, active_index = hand_state_at(frames, frame_value)
        mocap_id = runtime.hand_mocap[side]
        data.mocap_pos[mocap_id] = position
        data.mocap_quat[mocap_id] = quaternion
        flex_ids = runtime.hand_flexes[side]
        model.flex_contype[flex_ids] = 0
        model.flex_conaffinity[flex_ids] = 0
        model.flex_rgba[flex_ids, 3] = 0.0
        active_id = int(flex_ids[active_index])
        model.flex_contype[active_id] = HAND_CONTACT_BIT
        model.flex_conaffinity[active_id] = MOVING_CONTACT_BIT
        model.flex_rgba[active_id] = HAND_RGBA[side]
        applied[side] = (
            np.asarray(position, dtype=np.float64),
            np.asarray(quaternion, dtype=np.float64),
        )
    return applied


def _joint_value_text(record: dict, value: float) -> str:
    if record["type"] == "revolute":
        return f"{np.rad2deg(value):+.2f} deg"
    return f"{value:+.5f} scene-unit"


def print_summary(prepared: PreparedScene, args):
    print(f"scene: {prepared.scene_dir}")
    print(
        f"hands: {prepared.hawor_name} / {', '.join(prepared.hand_frames)}; "
        f"frames {prepared.start_frame:06d}..{prepared.end_frame:06d}; "
        f"{prepared.playback_fps:.3f} fps; {prepared.duration:.3f} s"
    )
    for label, record in prepared.joints.items():
        mesh = prepared.meshes.moving[label][args.mesh_frame]
        low, high, canonical = resolve_joint_range(record, int(mesh.frame_id), args)
        print(
            f"joint {label}: {record['type']}; canonical mesh "
            f"{args.mesh_frame}@{mesh.frame_id:06d}; q_abs={canonical:+.6f}; "
            f"dynamic offsets [{_joint_value_text(record, low)}, "
            f"{_joint_value_text(record, high)}]"
        )
    print(
        "physics: no actuator; gravity=0; static mesh collision=off; "
        "only hand<->moving mesh collision is enabled; contact margin="
        f"{100.0 * args.contact_margin_scale:.3f}% and flex radius="
        f"{100.0 * args.contact_thickness_scale:.3f}% of moving-part diagonal"
    )
    if args.drive_mode == "velocity":
        hold = (
            f"held {args.contact_hold:.3f} s after release"
            if args.contact_hold > 0
            else "stops the instant contact ends"
        )
        print(
            f"drive: velocity; each part tracks the {args.drive_point} "
            "velocity of the contacting hand projected onto its joint and "
            f"{hold}; --object-mass, --joint-damping, --joint-friction and "
            "--joint-armature are inert in this mode"
        )
    else:
        print(
            "drive: contact-force; each part is a free 1-DOF inertia "
            f"(mass={args.object_mass:.4g}) accelerated by contact impulses "
            "and coasting until a joint limit stops it"
        )


class ReplayDriver:
    """One hand-contact replay, shared by the viewer and the video exporter.

    Owning the physics stepping in one place keeps interactive playback and
    the exported MP4 showing the same simulation.
    """

    def __init__(self, mujoco, model, data, runtime, bundle, prepared, args):
        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.runtime = runtime
        self.bundle = bundle
        self.prepared = prepared
        self.args = args
        self.timestep = float(model.opt.timestep)
        self.peak = {label: 0.0 for label in prepared.joints}
        self.drive_state = {
            "rate": {label: 0.0 for label in prepared.joints},
            "hold": {label: 0.0 for label in prepared.joints},
        }
        # Mirrors Stage 61's relative radius floor for on-axis contacts.
        self.radius_floors = {
            label: 1e-4
            * max(float(prepared.meshes.moving[label][args.mesh_frame].diagonal), 1e-4)
            for label in prepared.joints
        }
        self.playback_time = 0.0
        self.contact_steps = 0
        self.driven = {label: False for label in prepared.joints}
        self.previous_poses = None
        self.reset()

    @property
    def finished(self) -> bool:
        return self.playback_time >= self.prepared.duration - 1e-12

    def reset(self):
        self.mujoco.mj_resetData(self.model, self.data)
        self.playback_time = 0.0
        self.contact_steps = 0
        for label in self.peak:
            self.peak[label] = 0.0
            self.drive_state["rate"][label] = 0.0
            self.drive_state["hold"][label] = 0.0
            self.driven[label] = False
        # Seeding the pose history here keeps the first driven step from
        # differentiating against a stale palm from the previous lap.
        self.previous_poses = set_hand_state(
            self.model,
            self.data,
            self.runtime,
            self.prepared,
            float(self.prepared.start_frame),
        )
        self.model.geom_rgba[self.runtime.moving_geoms] = MOVING_RGBA
        self.mujoco.mj_forward(self.model, self.data)

    def step(self) -> dict[str, bool]:
        """Advance one physics timestep of the hand replay."""
        prepared, args = self.prepared, self.args
        self.playback_time = min(self.playback_time + self.timestep, prepared.duration)
        frame_value = prepared.start_frame + self.playback_time * prepared.playback_fps
        poses = set_hand_state(
            self.model, self.data, self.runtime, prepared, frame_value
        )
        if args.drive_mode == "velocity":
            motions = {
                side: hand_motion(
                    (
                        None
                        if self.previous_poses is None
                        else self.previous_poses.get(side)
                    ),
                    pose,
                    self.timestep,
                )
                for side, pose in poses.items()
            }
            # Contacts must be current before the parts move.
            self.mujoco.mj_forward(self.model, self.data)
            driven = drive_joints(
                self.data,
                self.runtime,
                prepared,
                self.bundle.joints,
                motions,
                self.drive_state,
                self.radius_floors,
                self.timestep,
                args,
            )
            self.data.time += self.timestep
        else:
            self.mujoco.mj_step(self.model, self.data)
            driven = {label: bool(self.data.ncon) for label in prepared.joints}
        self.previous_poses = poses
        if self.data.ncon:
            self.contact_steps += 1
        for label, address in self.runtime.joint_qpos.items():
            self.peak[label] = max(
                self.peak[label], abs(float(self.data.qpos[address]))
            )
            self.model.geom_rgba[self.runtime.moving_geom_by_label[label]] = (
                MOVING_CONTACT_RGBA if driven[label] else MOVING_RGBA
            )
        self.driven = driven
        return driven

    def advance_to(self, playback_time: float):
        """Step until the replay reaches ``playback_time``."""
        target = min(float(playback_time), self.prepared.duration)
        # The half-step tolerance stops rounding from dropping a whole step.
        while self.playback_time < target - 0.5 * self.timestep:
            before = self.playback_time
            self.step()
            if self.playback_time <= before:
                break

    def report(self, prefix: str):
        details = ", ".join(
            f"{label}="
            f"{_joint_value_text(self.prepared.joints[label], self.peak[label])}"
            for label in self.prepared.joints
        )
        print(
            f"{prefix}: contact physics steps={self.contact_steps}; "
            f"peak |joint offset|: {details}",
            flush=True,
        )


def run_viewer(prepared: PreparedScene, args):
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required") from exc
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "mujoco is required; install the mujoco Python package"
        ) from exc

    print("building per-frame MANO collision assets...", flush=True)
    bundle = build_model_bundle(prepared, args, trimesh)
    try:
        model = mujoco.MjModel.from_xml_string(bundle.xml, assets=bundle.assets)
    except Exception as exc:
        raise RuntimeError(f"could not compile generated MuJoCo scene: {exc}") from exc
    data = mujoco.MjData(model)
    runtime = resolve_runtime_ids(mujoco, model, bundle)

    control_lock = threading.Lock()
    controls = {"playing": bool(args.play), "reset": False}

    def key_callback(keycode):
        with control_lock:
            if keycode == ord(" "):
                controls["playing"] = not controls["playing"]
            elif keycode in (ord("R"), ord("r")):
                controls["reset"] = True

    driver = ReplayDriver(mujoco, model, data, runtime, bundle, prepared, args)
    accumulator = 0.0
    lap = 0
    last_wall = time.perf_counter()
    last_sync = last_wall
    finished_reported = False

    def reset_state():
        nonlocal accumulator, finished_reported
        driver.reset()
        accumulator = 0.0
        finished_reported = False

    center = prepared.scene_bounds.mean(axis=0)
    diagonal = float(
        np.linalg.norm(prepared.scene_bounds[1] - prepared.scene_bounds[0])
    )
    print("viewer controls: Space play/pause, R reset, close window to exit")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        with viewer.lock():
            viewer.cam.azimuth = args.azimuth
            viewer.cam.elevation = args.elevation
            viewer.cam.lookat[:] = args.lookat if args.lookat is not None else center
            viewer.cam.distance = (
                args.distance
                if args.distance is not None
                else max(1.35 * diagonal, 1e-3)
            )
            show_contacts = int(bool(args.show_contacts))
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contacts
        viewer.sync()

        while viewer.is_running():
            now = time.perf_counter()
            wall_delta = min(max(now - last_wall, 0.0), 0.1)
            last_wall = now
            with control_lock:
                do_reset = bool(controls["reset"])
                controls["reset"] = False
                playing = bool(controls["playing"])

            if do_reset:
                with viewer.lock():
                    reset_state()
                last_sync = 0.0

            if playing:
                accumulator += wall_delta * args.speed
                max_steps = max(1, int(np.ceil(0.1 / model.opt.timestep)))
                steps = 0
                with viewer.lock():
                    while accumulator >= model.opt.timestep and steps < max_steps:
                        driver.step()
                        accumulator -= model.opt.timestep
                        steps += 1
                        if driver.finished:
                            if args.loop:
                                lap += 1
                                driver.report(f"lap {lap} complete")
                                reset_state()
                            else:
                                controls["playing"] = False
                            break

            if not finished_reported and driver.finished and not args.loop:
                driver.report("replay complete")
                finished_reported = True

            if now - last_sync >= 1.0 / 60.0:
                viewer.sync()
                last_sync = now
            time.sleep(0.001 if playing else 0.01)


def _da3_camera_pose(sample) -> tuple[np.ndarray, np.ndarray]:
    """Convert one DA3 camera sample into a MuJoCo camera pose.

    DA3 stores OpenCV/RDF axes (X right, Y down, Z forward).  A MuJoCo camera
    looks down its own -Z with +Y up, so the last two axes flip.
    """
    transform = np.asarray(sample.camera_to_reference, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    return transform[:3, 3].copy(), STAGE61._rotation_to_wxyz(rotation)


def _offscreen_plan(cameras, frame_ids, image_hw, max_side):
    """Size one offscreen canvas covering every frame's DA3 intrinsics.

    DA3 refines ``K`` per frame, so each frame needs its own centered render
    size.  The compiled framebuffer cannot be resized, so the largest is used
    for all of them and the per-frame maps are re-centered onto it.
    """
    canvas_h = canvas_w = 0
    output_hw = None
    for frame_id in frame_ids:
        sample = STAGE61.camera_sample_at(cameras, int(frame_id))
        spec = STAGE61.camera_render_spec(sample.intrinsics, image_hw, max_side)
        canvas_h = max(canvas_h, int(spec.render_hw[0]))
        canvas_w = max(canvas_w, int(spec.render_hw[1]))
        current = (int(spec.output_hw[0]), int(spec.output_hw[1]))
        if output_hw is None:
            output_hw = current
        elif current != output_hw:
            raise ValueError(
                "DA3 intrinsics produced inconsistent export sizes "
                f"{current} and {output_hw}"
            )
    if output_hw is None:
        raise ValueError("no frames selected for --export-video")
    return output_hw, (canvas_h, canvas_w)


def _canvas_camera(spec, canvas_hw) -> tuple[float, np.ndarray, np.ndarray]:
    """Retarget one Stage-61 render spec onto a fixed, larger canvas.

    ``camera_render_spec`` centers its remap on its own canvas at one focal
    length.  Holding that focal length while enlarging the canvas changes the
    vertical field of view and shifts the principal point, and nothing else,
    so the maps only need re-centering.
    """
    canvas_h, canvas_w = (int(value) for value in canvas_hw)
    render_h, render_w = (int(value) for value in spec.render_hw)
    focal = render_h / (2.0 * np.tan(float(spec.vertical_fov) / 2.0))
    if not np.isfinite(focal) or focal <= 1e-8:
        raise ValueError("DA3 intrinsics produced an unusable focal length")
    fovy = float(np.rad2deg(2.0 * np.arctan(canvas_h / (2.0 * focal))))
    map_x = np.asarray(spec.map_x, dtype=np.float32) + np.float32(
        (canvas_w - render_w) / 2.0
    )
    map_y = np.asarray(spec.map_y, dtype=np.float32) + np.float32(
        (canvas_h - render_h) / 2.0
    )
    return fovy, map_x, map_y


def copy_to_render_all(scene_dir: Path, output_path: Path, args) -> Path | None:
    """Mirror the exported MP4 as <scene>/render_all/mujoco-retarget.mp4."""
    if args.no_render_all_copy:
        return None
    copy_path = (scene_dir / RENDER_ALL_COPY).resolve()
    if copy_path == output_path:
        return None
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(output_path, copy_path)
    except OSError as exc:
        raise RuntimeError(f"could not write {copy_path}: {exc}") from exc
    return copy_path


def export_video(prepared: PreparedScene, args) -> Path:
    """Render the replay offscreen through the DA3 camera into one MP4."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required") from exc
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError(
            "mujoco is required; install the mujoco Python package"
        ) from exc
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--export-video needs OpenCV; install opencv-python"
        ) from exc

    scene_dir = prepared.scene_dir
    output_path = args.export_video.expanduser()
    if not output_path.is_absolute():
        output_path = scene_dir / output_path
    output_path = output_path.resolve()
    if output_path.exists() and not args.overwrite_video:
        raise FileExistsError(
            f"{output_path} exists; pass --overwrite-video to replace it"
        )

    cameras = STAGE61.load_da3_cameras(scene_dir, prepared.meshes.reference_to_world)
    image_hw = STAGE61._consistent_image_size(STAGE61._numeric_frame_paths(scene_dir))
    frame_ids = list(range(prepared.start_frame, prepared.end_frame + 1))
    output_hw, canvas_hw = _offscreen_plan(
        cameras, frame_ids, image_hw, args.video_max_side
    )
    output_h, output_w = output_hw
    canvas_h, canvas_w = canvas_hw
    fps = float(args.video_fps or prepared.playback_fps)
    print(
        f"video: {len(frame_ids)} frames at {fps:.3f} fps; "
        f"{output_w}x{output_h} from a {canvas_w}x{canvas_h} offscreen canvas; "
        f"source frames {image_hw[1]}x{image_hw[0]}"
    )

    print("building per-frame MANO collision assets...", flush=True)
    bundle = build_model_bundle(prepared, args, trimesh, offscreen_hw=canvas_hw)
    try:
        model = mujoco.MjModel.from_xml_string(bundle.xml, assets=bundle.assets)
    except Exception as exc:
        raise RuntimeError(f"could not compile generated MuJoCo scene: {exc}") from exc
    data = mujoco.MjData(model)
    runtime = resolve_runtime_ids(mujoco, model, bundle)
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, DA3_CAMERA_NAME)
    if camera_id < 0:
        raise RuntimeError("MuJoCo model omitted the DA3 camera")
    driver = ReplayDriver(mujoco, model, data, runtime, bundle, prepared, args)

    scene_option = mujoco.MjvOption()
    show_contacts = int(bool(args.show_contacts))
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = show_contacts
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = show_contacts

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.tmp-", suffix=".mp4", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*args.video_codec),
        fps,
        (output_w, output_h),
    )
    if not writer.isOpened():
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"OpenCV could not open {temporary_path} with codec "
            f"{args.video_codec!r}"
        )

    try:
        renderer = mujoco.Renderer(model, height=canvas_h, width=canvas_w)
    except Exception as exc:
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not create a {canvas_w}x{canvas_h} MuJoCo renderer: {exc}; "
            "set MUJOCO_GL=egl or MUJOCO_GL=osmesa for headless rendering"
        ) from exc

    native_frames: list[int] = []
    inferred_frames: list[int] = []
    next_progress = 0
    try:
        for index, frame_id in enumerate(frame_ids):
            driver.advance_to((frame_id - prepared.start_frame) / prepared.playback_fps)
            sample = STAGE61.camera_sample_at(cameras, frame_id)
            (native_frames if sample.native else inferred_frames).append(frame_id)
            spec = STAGE61.camera_render_spec(
                sample.intrinsics, image_hw, args.video_max_side
            )
            fovy, map_x, map_y = _canvas_camera(spec, canvas_hw)
            position, quaternion = _da3_camera_pose(sample)
            model.cam_pos[camera_id] = position
            model.cam_quat[camera_id] = quaternion
            model.cam_fovy[camera_id] = fovy
            # Refresh cam_xpos/cam_xmat and the driven pose before rendering.
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera_id, scene_option=scene_option)
            rgb = renderer.render()
            frame = cv2.remap(
                rgb,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            if frame.shape[:2] != (output_h, output_w):
                raise RuntimeError(
                    f"frame {frame_id:06d} remapped to "
                    f"{frame.shape[1]}x{frame.shape[0]}, expected "
                    f"{output_w}x{output_h}"
                )
            writer.write(np.ascontiguousarray(frame[..., ::-1]))
            percent = int(100 * (index + 1) / len(frame_ids))
            if percent >= next_progress:
                print(
                    f"  camera video {index + 1}/{len(frame_ids)} ({percent}%)",
                    flush=True,
                )
                next_progress += 10
    except Exception:
        renderer.close()
        writer.release()
        temporary_path.unlink(missing_ok=True)
        raise
    renderer.close()
    writer.release()
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("camera video writer produced no output")
    os.replace(temporary_path, output_path)
    output_path.chmod(0o644)
    copy_path = copy_to_render_all(scene_dir, output_path, args)

    metadata = {
        "version": 1,
        "stage": "63_mujoco_retarget",
        "method": "mujoco_offscreen_da3_full_intrinsics",
        "video": str(output_path),
        "render_all_copy": None if copy_path is None else str(copy_path),
        "frame_count": len(frame_ids),
        "frame_ids": frame_ids,
        "fps": fps,
        "codec": args.video_codec,
        "output_size": [output_w, output_h],
        "offscreen_canvas": [canvas_w, canvas_h],
        "source_image_size": [image_hw[1], image_hw[0]],
        "camera_pose_source": "da3/cameras.npz (camera-to-world, XYZW)",
        "intrinsics_source": "da3/intrinsics.npz",
        "camera_convention": "DA3 RDF converted to MuJoCo -Z forward / +Y up",
        "frame_mapping": "direct numeric frame index (Stage 31/60/60a convention)",
        "native_camera_frames": native_frames,
        "interpolated_or_held_camera_frames": inferred_frames,
        "drive_mode": args.drive_mode,
        "drive_point": args.drive_point,
        "contact_hold": float(args.contact_hold),
        "canonical_moving_mesh": args.mesh_frame,
        "contact_physics_steps": int(driver.contact_steps),
        "peak_joint_offset": {
            label: float(value) for label, value in driver.peak.items()
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    driver.report("export complete")
    print(f"wrote {output_path}")
    print(f"wrote {metadata_path}")
    if copy_path is not None:
        print(f"copied to {copy_path}")
    return output_path


def run(args):
    prepared = prepare_scene(args)
    print_summary(prepared, args)
    if args.dry_run:
        print("--dry-run: MuJoCo was not imported and no viewer was opened")
        return prepared
    if args.export_video is not None:
        export_video(prepared, args)
        return prepared
    run_viewer(prepared, args)
    return prepared


def main(argv=None):
    try:
        args = parse_args(argv)
        run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
