#!/usr/bin/env python
"""Render a Stage-35 joint as a headless MuJoCo animation.

The script uses the same Stage-31 registered scene and named-geometry records
that Stage 35 uses to create ``simple_joint/joint_mesh.glb``.  The
numerically earliest available SegviGen moving-mesh record is selected.  The
registered static geometry is rendered gray; the selected moving geometry is
rendered orange and attached to a MuJoCo hinge or slide joint.

Joint limits may be supplied with ``--range-min`` and ``--range-max``.  Hinge
limits are specified in degrees and slide limits in Stage-31 scene units.  The
limits are offsets from the pose in the selected SegviGen frame.  If omitted,
the observed Stage-35 state range is used, relative to that frame.

The camera reproduces the original video viewpoint by default: the DA3 pose
for the rendered SegviGen frame is read from ``<scene>/da3/cameras.npz``,
mapped into the Stage-31 registered reference frame with the same
reference-to-world similarity Stage 61 uses, and written into the MJCF as a
fixed camera carrying the matching DA3 pinhole intrinsics.  The render
resolution then defaults to the DA3 image resolution.  Pass
``--camera-source free`` for the previous orbit camera driven by
``--azimuth``/``--elevation``/``--roll``/``--distance``/``--lookat``.

This is a pure batch renderer: it creates the MJCF in memory, drives the joint
kinematically with ``mj_forward``, and writes an MP4 without opening a viewer.
The MP4 is also mirrored to ``<scene>/render_all/mujoco-joint.mp4`` alongside
the other Stage-6x renders; pass ``--no-render-all-copy`` to skip that copy.

Examples:
    python scripts/36_render_joint.py --scene-dir data/dryer \\
        --label dryer_door --range-min 0 --range-max 90 --motion sine

    python scripts/36_render_joint.py --scene-dir data/drawer \\
        --range-min 0 --range-max 0.25 --out /tmp/drawer.mp4

    python scripts/36_render_joint.py --scene-dir data/dryer \\
        --camera-source free --azimuth 20 --elevation 60

Runtime dependencies: numpy, trimesh, mujoco, and opencv-python.  On a
headless Linux machine, set ``MUJOCO_GL=egl`` (or ``MUJOCO_GL=osmesa`` for a
software fallback) if an OpenGL backend is not selected automatically.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

STATIC_RGBA = (0.65, 0.65, 0.65, 1.0)
MOVING_RGBA = (1.0, 124.0 / 255.0, 54.0 / 255.0, 1.0)
HEADLIGHT_AMBIENT = (0.35, 0.35, 0.35)
HEADLIGHT_DIFFUSE = (0.8, 0.8, 0.8)
HEADLIGHT_SPECULAR = (0.5, 0.5, 0.5)
DA3_CAMERA_NAME = "da3_camera"
# Any positive sensor size reproduces the DA3 frustum: MuJoCo converts
# focalpixel/principalpixel through sensorsize and resolution, and the
# resulting field of view depends only on the pixel-unit intrinsics.
DA3_SENSOR_HEIGHT = 0.02
# SAM3D GLB axes -> Stage-31 model axes, and PyTorch3D -> right-down-forward.
SAM3D_MODEL_FROM_GLB = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
SAM3D_PYTORCH3D_TO_RDF = np.diag([-1.0, -1.0, 1.0])
# Right-down-forward (DA3/OpenCV) -> right-up-backward (MuJoCo/OpenGL).
RDF_TO_MUJOCO_CAMERA = np.diag([1.0, -1.0, -1.0])
FREE_CAMERA_AZIMUTH = 20.0
FREE_CAMERA_ELEVATION = 60.0
FREE_CAMERA_ROLL = 0.0
FREE_CAMERA_WIDTH = 1280
FREE_CAMERA_HEIGHT = 720
RENDER_ALL_COPY = Path("render_all") / "mujoco-joint.mp4"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        required=True,
        help="Scene directory, for example data/dryer",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Moving label to render (default: auto when Stage 35 has one label)",
    )
    parser.add_argument(
        "--joint-file",
        type=Path,
        default=None,
        help="Stage-35 JSON (default: <scene>/simple_joint/joints.json)",
    )
    parser.add_argument(
        "--range-min",
        type=float,
        default=None,
        help="Lower joint offset: degrees for hinge, scene units for slide",
    )
    parser.add_argument(
        "--range-max",
        type=float,
        default=None,
        help="Upper joint offset: degrees for hinge, scene units for slide",
    )
    parser.add_argument(
        "--motion",
        choices=("triangle", "sine"),
        default="sine",
        help="Joint sweep waveform (default: sine)",
    )
    parser.add_argument(
        "--n-cycles",
        type=float,
        default=3.0,
        help="Number of complete min-to-max-to-min sweeps (default: 3)",
    )
    parser.add_argument(
        "--seconds-per-cycle",
        type=float,
        default=2.0,
        help="Duration of one complete sweep in seconds (default: 2)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Render width (default: DA3 image width, "
            f"or {FREE_CAMERA_WIDTH} for --camera-source free)"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Render height (default: DA3 image height, "
            f"or {FREE_CAMERA_HEIGHT} for --camera-source free)"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MP4 (default: <scene>/joint_render/<label>.mp4)",
    )
    parser.add_argument(
        "--no-render-all-copy",
        action="store_true",
        help=f"Skip the extra copy written to <scene>/{RENDER_ALL_COPY.as_posix()}",
    )
    parser.add_argument(
        "--camera-source",
        choices=("da3", "free"),
        default="da3",
        help="Camera pose source (default: the da3/ pose for the rendered frame)",
    )
    parser.add_argument(
        "--camera-frame",
        type=int,
        default=None,
        help="DA3 frame supplying the camera pose (default: the rendered frame)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=None,
        help=f"Free-camera azimuth in degrees (default: {FREE_CAMERA_AZIMUTH:g})",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=None,
        help=f"Free-camera elevation in degrees (default: {FREE_CAMERA_ELEVATION:g})",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=None,
        help=(
            "Free-camera roll in degrees; positive values rotate clockwise "
            f"(default: {FREE_CAMERA_ROLL:g})"
        ),
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=None,
        help="Free-camera distance (default: fit the full articulated sweep)",
    )
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Free-camera target (default: center of the articulated sweep)",
    )
    return parser.parse_args(argv)


def read_json(path: Path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc


def safe_name(label: str):
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in label)


def choose_joint(payload, requested_label):
    joints = payload.get("joints")
    if not isinstance(joints, dict) or not joints:
        raise RuntimeError("joint JSON contains no Stage-35 joints")
    if requested_label is None:
        if len(joints) != 1:
            raise RuntimeError(
                f"joint JSON has {len(joints)} labels; pass --label to select one: "
                f"{sorted(joints)}"
            )
        label = next(iter(joints))
    else:
        label = requested_label
        if label not in joints:
            raise RuntimeError(
                f"joint label {label!r} not found; available: {sorted(joints)}"
            )
    record = joints[label]
    if not isinstance(record, dict):
        raise RuntimeError(f"joint record {label!r} is not an object")
    if record.get("type") not in ("revolute", "prismatic"):
        raise RuntimeError(
            f"joint {label!r} has unsupported type {record.get('type')!r}"
        )
    return label, record


def resolve_registered_mesh(scene_dir, metadata, metadata_path):
    value = metadata.get("output_mesh", "registered_static/registered_meshes.glb")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    scene_relative = (scene_dir / path).resolve()
    if scene_relative.is_file():
        return scene_relative
    metadata_relative = (metadata_path.parent / path).resolve()
    if metadata_relative.is_file():
        return metadata_relative
    return scene_relative


def collect_label_records(metadata, label):
    records = []
    for frame_record in metadata.get("frames", []):
        try:
            frame = int(frame_record["frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Stage-31 metadata has an invalid frame record") from exc
        for moving in frame_record.get("moving_meshes", []):
            if moving.get("label") != label:
                continue
            geometry = moving.get("geometry")
            if not geometry:
                raise RuntimeError(
                    f"Stage-31 metadata lacks geometry for {label!r} "
                    f"frame {frame:06d}"
                )
            records.append(
                {
                    "frame": frame,
                    "geometry": geometry,
                    "source": moving.get("source"),
                }
            )
            break
    records.sort(key=lambda record: record["frame"])
    if not records:
        raise RuntimeError(f"Stage-31 metadata has no moving meshes for {label!r}")
    duplicates = [
        records[index]["frame"]
        for index in range(1, len(records))
        if records[index]["frame"] == records[index - 1]["frame"]
    ]
    if duplicates:
        raise RuntimeError(
            f"duplicate Stage-31 moving records for {label!r}: {duplicates}"
        )
    return records


def load_named_geometry(scene, geometry_name, trimesh):
    """Copy one named geometry with all Scene instance transforms baked in."""
    instances = []
    for node_name in scene.graph.nodes_geometry:
        transform, node_geometry = scene.graph[node_name]
        if node_geometry != geometry_name:
            continue
        mesh = scene.geometry[node_geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(transform)
        instances.append(mesh)
    if not instances and geometry_name in scene.geometry:
        mesh = scene.geometry[geometry_name].copy()
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            instances.append(mesh)
    if not instances:
        raise RuntimeError(f"registered GLB lacks triangle geometry {geometry_name!r}")
    result = (
        instances[0] if len(instances) == 1 else trimesh.util.concatenate(instances)
    )
    if not np.isfinite(result.vertices).all():
        raise RuntimeError(
            f"registered geometry {geometry_name!r} has non-finite vertices"
        )
    return result


def joint_axis(record):
    axis = np.asarray(record.get("axis_direction"), dtype=np.float64)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise RuntimeError("joint has an invalid axis_direction")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise RuntimeError("joint axis_direction has zero length")
    return axis / norm


def frame_state(record, frame):
    states = []
    selected = None
    for item in record.get("joint_states", []):
        try:
            item_frame = int(item["frame"])
            value = float(item["q"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("joint has a malformed joint_states entry") from exc
        if not np.isfinite(value):
            raise RuntimeError("joint has a non-finite state")
        states.append(value)
        if item_frame == frame:
            selected = value
    if selected is None:
        raise RuntimeError(
            f"joint has no state for selected SegviGen frame {frame:06d}"
        )
    return selected, states


def resolve_range(record, selected_frame, range_min, range_max, moving_diagonal):
    if (range_min is None) != (range_max is None):
        raise RuntimeError("--range-min and --range-max must be supplied together")
    joint_type = record["type"]
    if range_min is not None:
        lo, hi = float(range_min), float(range_max)
        if joint_type == "revolute":
            lo, hi = float(np.deg2rad(lo)), float(np.deg2rad(hi))
        source = "user"
    else:
        selected_state, states = frame_state(record, selected_frame)
        offsets = np.asarray(states, dtype=np.float64) - selected_state
        lo, hi = float(offsets.min()), float(offsets.max())
        source = "observed Stage-35 states"
        if hi - lo <= 1e-8:
            if joint_type == "revolute":
                lo, hi = 0.0, float(np.deg2rad(90.0))
            else:
                lo, hi = 0.0, 0.25 * moving_diagonal
            source = "fallback (observed range was degenerate)"
    if not np.isfinite([lo, hi]).all() or lo >= hi:
        raise RuntimeError("joint range must contain two finite values with min < max")
    return lo, hi, source


def rotation_matrix(axis, angle):
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def articulated_bounds(static_mesh, moving_mesh, record, axis, lo, hi):
    bounds = [np.asarray(static_mesh.bounds, dtype=np.float64)]
    moving_vertices = np.asarray(moving_mesh.vertices, dtype=np.float64)
    # Sampling the range also gives a useful camera fit for wide hinge sweeps.
    for value in np.linspace(lo, hi, 37):
        if record["type"] == "revolute":
            pivot = np.asarray(record.get("axis_point"), dtype=np.float64)
            transformed = (moving_vertices - pivot) @ rotation_matrix(
                axis, value
            ).T + pivot
        else:
            transformed = moving_vertices + value * axis
        bounds.append(np.stack((transformed.min(axis=0), transformed.max(axis=0))))
    low = np.min([item[0] for item in bounds], axis=0)
    high = np.max([item[1] for item in bounds], axis=0)
    return np.stack((low, high))


def format_numbers(values):
    return " ".join(f"{float(value):.17g}" for value in values)


def quat_xyzw_to_rotation(value):
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("DA3 camera quaternion is zero or non-finite")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_rotation(value):
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("SAM3D quaternion is zero or non-finite")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quat_wxyz(value):
    """Convert a proper 3x3 rotation matrix to a normalized WXYZ quaternion."""
    matrix = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(matrix), 1.0, atol=2e-5)
    ):
        raise RuntimeError("camera rotation is not a proper orthonormal matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        root = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * root,
                (matrix[2, 1] - matrix[1, 2]) / root,
                (matrix[0, 2] - matrix[2, 0]) / root,
                (matrix[1, 0] - matrix[0, 1]) / root,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            root = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / root,
                    0.25 * root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                ]
            )
        elif index == 1:
            root = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / root,
                    (matrix[0, 1] + matrix[1, 0]) / root,
                    0.25 * root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                ]
            )
        else:
            root = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / root,
                    (matrix[0, 2] + matrix[2, 0]) / root,
                    (matrix[1, 2] + matrix[2, 1]) / root,
                    0.25 * root,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def resolve_scene_file(scene_dir, value, description):
    path = Path(value).expanduser()
    path = path.resolve() if path.is_absolute() else (scene_dir / path).resolve()
    if not path.is_file():
        raise RuntimeError(f"missing {description}: {path}")
    return path


def load_da3_camera_to_world(scene_dir):
    """Load DA3 camera-to-world poses keyed by their numeric frame index."""
    path = scene_dir / "da3" / "cameras.npz"
    if not path.is_file():
        raise RuntimeError(
            f"missing DA3 cameras: {path} (pass --camera-source free to skip them)"
        )
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
        raise RuntimeError(f"camera array lengths differ in {path}")
    poses = {}
    for frame_id, quaternion, translation in zip(frame_ids, quaternions, translations):
        frame = int(frame_id)
        if frame in poses:
            raise RuntimeError(f"duplicate DA3 camera frame {frame:06d} in {path}")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = quat_xyzw_to_rotation(quaternion)
        transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
        if not np.isfinite(transform).all():
            raise RuntimeError(f"non-finite DA3 pose for frame {frame:06d}")
        poses[frame] = transform
    if not poses:
        raise RuntimeError(f"DA3 camera archive is empty: {path}")
    return poses


def load_da3_intrinsics(scene_dir):
    """Load DA3 pinhole matrices keyed by their numeric frame index."""
    path = scene_dir / "da3" / "intrinsics.npz"
    if not path.is_file():
        raise RuntimeError(f"missing DA3 intrinsics: {path}")
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
        raise RuntimeError(f"intrinsics array lengths differ in {path}")
    result = {}
    for frame_id, value in zip(frame_ids, matrices):
        K = np.asarray(value, dtype=np.float64).reshape(3, 3)
        if (
            not np.isfinite(K).all()
            or K[0, 0] <= 0.0
            or K[1, 1] <= 0.0
            or not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-5)
        ):
            raise RuntimeError(f"invalid DA3 intrinsics for frame {int(frame_id):06d}")
        frame = int(frame_id)
        if frame in result:
            raise RuntimeError(f"duplicate DA3 intrinsics frame {frame:06d}")
        result[frame] = K
    return result


def sam3d_pose_to_rdf_camera(path):
    """Reproduce Stage 31's raw-SAM3D-GLB to RDF-camera transform."""
    pose = read_json(path)
    try:
        rotation = quat_wxyz_to_rotation(pose["rotation_quat_wxyz"])
        translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        scales = np.asarray(pose["scale"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid SAM3D pose in {path}") from exc
    if len(scales) not in (1, 3):
        raise RuntimeError(f"invalid SAM3D scale in {path}: {scales}")
    if len(scales) == 1:
        scales = np.repeat(scales, 3)
    if (
        not np.isfinite(translation).all()
        or not np.isfinite(scales).all()
        or np.any(scales <= 0)
    ):
        raise RuntimeError(f"invalid SAM3D translation/scale in {path}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        SAM3D_PYTORCH3D_TO_RDF @ rotation.T @ np.diag(scales) @ SAM3D_MODEL_FROM_GLB
    )
    transform[:3, 3] = SAM3D_PYTORCH3D_TO_RDF @ translation
    return transform


def stage31_reference_to_world(scene_dir, metadata, poses):
    """Recover the transform Stage 31 used for its reference SAM3D GLB."""
    try:
        reference_frame = int(metadata["reference_frame"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Stage-31 metadata has no valid reference_frame") from exc
    reference_record = next(
        (
            record
            for record in metadata.get("frames", [])
            if isinstance(record, dict)
            and int(record.get("frame", -1)) == reference_frame
        ),
        None,
    )
    if reference_record is None or not reference_record.get("pose"):
        raise RuntimeError(
            "Stage-31 metadata has no SAM3D pose for its reference frame"
        )
    pose_path = resolve_scene_file(
        scene_dir, str(reference_record["pose"]), "reference SAM3D pose"
    )
    reference_to_world = sam3d_pose_to_rdf_camera(pose_path)
    if metadata.get("camera_to_world_used"):
        if reference_frame not in poses:
            raise RuntimeError(
                f"da3/cameras.npz lacks Stage-31 reference frame {reference_frame:06d}"
            )
        reference_to_world = poses[reference_frame] @ reference_to_world
    determinant = float(np.linalg.det(reference_to_world[:3, :3]))
    if (
        not np.isfinite(reference_to_world).all()
        or not np.isfinite(determinant)
        or abs(determinant) < 1e-12
    ):
        raise RuntimeError("Stage-31 reference-to-world transform is singular")
    return reference_to_world


def reference_similarity(reference_to_world):
    """Decompose Stage 31's reference-to-world affine as uniform s*R + t.

    A rigid MuJoCo camera can reproduce the DA3 projection in the registered
    reference frame only when this transform is a similarity, so validate
    instead of silently discarding an anisotropic SAM3D scale component.
    """
    transform = np.asarray(reference_to_world, dtype=np.float64).reshape(4, 4)
    linear = transform[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(scales))
    if (
        not np.isfinite(scale)
        or scale <= 1e-12
        or not np.allclose(scales, scale, rtol=2e-5, atol=1e-8)
    ):
        raise RuntimeError(
            "DA3 camera replay requires the Stage-31 reference transform to have "
            f"uniform scale; column scales are {scales.tolist()}"
        )
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=2e-5
    ):
        raise RuntimeError(
            "DA3 camera replay requires a proper similarity reference transform"
        )
    return scale, rotation, transform[:3, 3].copy()


def da3_camera_pose(scene_dir, metadata, frame):
    """Return one DA3 pose and its intrinsics in Stage-31 reference units."""
    poses = load_da3_camera_to_world(scene_dir)
    intrinsics = load_da3_intrinsics(scene_dir)
    available = sorted(set(poses) & set(intrinsics))
    if not available:
        raise RuntimeError("DA3 cameras and intrinsics have no common frame indices")
    if frame not in poses or frame not in intrinsics:
        raise RuntimeError(
            f"da3/ has no camera for frame {frame:06d}; available frames span "
            f"{available[0]:06d}-{available[-1]:06d}"
        )
    scale, rotation, translation = reference_similarity(
        stage31_reference_to_world(scene_dir, metadata, poses)
    )
    camera_world = poses[frame]
    camera_reference = np.eye(4, dtype=np.float64)
    camera_reference[:3, :3] = rotation.T @ camera_world[:3, :3]
    camera_reference[:3, 3] = (rotation.T @ (camera_world[:3, 3] - translation)) / scale
    # Remove harmless archive round-off so the MJCF quaternion is exact.
    u, _, vt = np.linalg.svd(camera_reference[:3, :3])
    camera_reference[:3, :3] = u @ vt
    if np.linalg.det(camera_reference[:3, :3]) < 0.0:
        u[:, -1] *= -1.0
        camera_reference[:3, :3] = u @ vt
    return camera_reference, intrinsics[frame]


def da3_image_size(scene_dir, intrinsics):
    """Resolution the DA3 intrinsics are expressed in."""
    config_path = scene_dir / "da3" / "config.json"
    if config_path.is_file():
        value = read_json(config_path).get("full_resolution_wh")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                width, height = int(value[0]), int(value[1])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid full_resolution_wh in {config_path}"
                ) from exc
            if width > 0 and height > 0:
                return width, height
    width = int(round(2.0 * float(intrinsics[0, 2])))
    height = int(round(2.0 * float(intrinsics[1, 2])))
    if width < 1 or height < 1:
        raise RuntimeError("could not determine the DA3 image resolution")
    return width, height


def da3_camera_xml(camera_to_reference, intrinsics, image_size, render_size):
    """MJCF fixed camera reproducing one DA3 pose and pinhole projection."""
    image_width, image_height = image_size
    width, height = render_size
    scale_x = width / float(image_width)
    scale_y = height / float(image_height)
    fx = float(intrinsics[0, 0]) * scale_x
    fy = float(intrinsics[1, 1]) * scale_y
    cx = float(intrinsics[0, 2]) * scale_x
    cy = float(intrinsics[1, 2]) * scale_y
    rotation = np.asarray(camera_to_reference, dtype=np.float64)[:3, :3]
    quaternion = rotation_to_quat_wxyz(rotation @ RDF_TO_MUJOCO_CAMERA)
    position = np.asarray(camera_to_reference, dtype=np.float64)[:3, 3]
    sensor_height = DA3_SENSOR_HEIGHT
    sensor_width = sensor_height * width / float(height)
    return f"""    <camera name="{DA3_CAMERA_NAME}" mode="fixed"
            pos="{format_numbers(position)}" quat="{format_numbers(quaternion)}"
            resolution="{int(width)} {int(height)}"
            sensorsize="{format_numbers((sensor_width, sensor_height))}"
            focalpixel="{format_numbers((fx, fy))}"
            principalpixel="{format_numbers((cx - width / 2.0, cy - height / 2.0))}"/>
"""


def da3_camera_summary(camera_to_reference, intrinsics, image_size, render_size, frame):
    position = np.asarray(camera_to_reference, dtype=np.float64)[:3, 3]
    fovy = 2.0 * np.rad2deg(np.arctan(0.5 * image_size[1] / float(intrinsics[1, 1])))
    fovx = 2.0 * np.rad2deg(np.arctan(0.5 * image_size[0] / float(intrinsics[0, 0])))
    return (
        f"camera: da3/cameras.npz frame {frame:06d}, "
        f"pos=({position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}), "
        f"fov=({fovx:.1f}, {fovy:.1f}) deg, "
        f"image={image_size[0]}x{image_size[1]}, "
        f"render={render_size[0]}x{render_size[1]}"
    )


def mesh_stl_bytes(mesh):
    payload = mesh.export(file_type="stl")
    return payload.encode() if isinstance(payload, str) else payload


def make_mjcf(static_mesh, moving_mesh, record, axis, lo, hi, camera_xml=""):
    if record["type"] == "revolute":
        pivot = np.asarray(record.get("axis_point"), dtype=np.float64)
        if pivot.shape != (3,) or not np.isfinite(pivot).all():
            raise RuntimeError("revolute joint has an invalid axis_point")
        body_position = pivot
        exported_moving = moving_mesh.copy()
        exported_moving.apply_translation(-pivot)
        mujoco_type = "hinge"
    else:
        body_position = np.zeros(3)
        exported_moving = moving_mesh
        mujoco_type = "slide"

    number = lambda values: " ".join(f"{float(value):.17g}" for value in values)
    xml = f"""<mujoco model="stage36_joint_render">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.8 0.8 0.8" specular="0.5 0.5 0.5"/>
  </visual>
  <asset>
    <mesh name="static_mesh" file="static.stl"/>
    <mesh name="moving_mesh" file="moving.stl"/>
  </asset>
  <worldbody>
{camera_xml}    <body name="static_link">
      <geom name="static_geom" type="mesh" mesh="static_mesh"
            rgba="{number(STATIC_RGBA)}" contype="0" conaffinity="0"/>
      <body name="moving_link" pos="{number(body_position)}">
        <joint name="articulation" type="{mujoco_type}"
               axis="{number(axis)}" limited="true" range="{number((lo, hi))}"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom name="moving_geom" type="mesh" mesh="moving_mesh"
              rgba="{number(MOVING_RGBA)}" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    assets = {
        "static.stl": mesh_stl_bytes(static_mesh),
        "moving.stl": mesh_stl_bytes(exported_moving),
    }
    return xml, assets


def joint_phase(t, n_cycles, motion):
    cycle_position = n_cycles * t
    fraction = cycle_position - np.floor(cycle_position)
    if t >= 1.0:
        fraction = 0.0
    if motion == "triangle":
        return 1.0 - abs(2.0 * fraction - 1.0)
    if motion == "sine":
        return (1.0 - np.cos(2.0 * np.pi * cycle_position)) / 2.0
    raise ValueError(f"unknown motion: {motion}")


def make_glossy(mujoco, model, scene):
    for scene_geom in scene.geoms[: scene.ngeom]:
        if scene_geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        scene_geom.specular = 0.2
        scene_geom.shininess = 0.1
        scene_geom.reflectance = 0.1


def apply_camera_roll(scene, roll_degrees):
    angle = np.deg2rad(roll_degrees)

    for eye in range(2):
        gl_camera = scene.camera[eye]

        forward = np.asarray(gl_camera.forward, dtype=np.float64)
        forward /= np.linalg.norm(forward)

        up = np.asarray(gl_camera.up, dtype=np.float64)
        up -= np.dot(up, forward) * forward
        up /= np.linalg.norm(up)

        right = np.cross(forward, up)
        rolled_up = np.cos(angle) * up + np.sin(angle) * right
        gl_camera.up[:] = rolled_up


def render_video(mujoco, cv2, xml, assets, out_path, lo, hi, args, render_bounds):
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR
    model.vis.global_.offwidth = max(args.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(args.height, model.vis.global_.offheight)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    if args.camera_source == "da3":
        camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, DA3_CAMERA_NAME
        )
        if camera_id < 0:
            raise RuntimeError("MJCF is missing the DA3 fixed camera")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
    else:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.azimuth = args.azimuth
        camera.elevation = args.elevation
        camera.lookat[:] = (
            args.lookat if args.lookat is not None else render_bounds.mean(axis=0)
        )
        diagonal = float(np.linalg.norm(render_bounds[1] - render_bounds[0]))
        camera.distance = (
            args.distance if args.distance is not None else max(1.5 * diagonal, 1e-3)
        )

    duration = args.n_cycles * args.seconds_per_cycle
    frame_count = max(2, int(round(duration * args.fps)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer for {out_path}")

    renderer = None
    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "articulation")
        qpos_address = int(model.jnt_qposadr[joint_id])
        for index in range(frame_count):
            t = index / (frame_count - 1)
            phase = joint_phase(t, args.n_cycles, args.motion)
            data.qpos[qpos_address] = lo + (hi - lo) * phase
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            if args.camera_source != "da3":
                apply_camera_roll(renderer.scene, args.roll)
            make_glossy(mujoco, model, renderer.scene)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        if renderer is not None:
            renderer.close()
    return frame_count, duration, camera


FREE_CAMERA_FLAGS = (
    ("--azimuth", "azimuth", FREE_CAMERA_AZIMUTH),
    ("--elevation", "elevation", FREE_CAMERA_ELEVATION),
    ("--roll", "roll", FREE_CAMERA_ROLL),
    ("--distance", "distance", None),
    ("--lookat", "lookat", None),
)


def validate_args(args):
    if args.n_cycles <= 0.0 or args.seconds_per_cycle <= 0.0:
        raise RuntimeError("--n-cycles and --seconds-per-cycle must be positive")
    if args.fps <= 0.0:
        raise RuntimeError("--fps must be positive")
    for name, attribute in (("--width", "width"), ("--height", "height")):
        value = getattr(args, attribute)
        if value is not None and value < 1:
            raise RuntimeError(f"{name} must be a positive integer")
    if args.distance is not None and args.distance <= 0.0:
        raise RuntimeError("--distance must be positive")
    if args.camera_source == "da3":
        supplied = [
            flag
            for flag, attribute, _ in FREE_CAMERA_FLAGS
            if getattr(args, attribute) is not None
        ]
        if supplied:
            verb = "applies" if len(supplied) == 1 else "apply"
            raise RuntimeError(
                f"{', '.join(supplied)} only {verb} to --camera-source free"
            )
    else:
        if args.camera_frame is not None:
            raise RuntimeError("--camera-frame only applies to --camera-source da3")
        for _, attribute, default in FREE_CAMERA_FLAGS:
            if default is not None and getattr(args, attribute) is None:
                setattr(args, attribute, default)


def resolve_render_size(args, image_size):
    """Fill in --width/--height, holding the DA3 aspect when only one is given."""
    if image_size is None:
        width = FREE_CAMERA_WIDTH if args.width is None else args.width
        height = FREE_CAMERA_HEIGHT if args.height is None else args.height
        return int(width), int(height)
    image_width, image_height = image_size
    width, height = args.width, args.height
    if width is None and height is None:
        width, height = image_width, image_height
    elif width is None:
        width = max(1, int(round(height * image_width / float(image_height))))
    elif height is None:
        height = max(1, int(round(width * image_height / float(image_width))))
    return int(width), int(height)


def copy_to_render_all(scene_dir, out_path, args):
    """Mirror the finished MP4 as <scene>/render_all/mujoco-joint.mp4."""
    if args.no_render_all_copy:
        return None
    copy_path = (scene_dir / RENDER_ALL_COPY).resolve()
    if copy_path == out_path:
        return None
    copy_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copyfile(out_path, copy_path)
    except OSError as exc:
        raise RuntimeError(f"could not write {copy_path}: {exc}") from exc
    return copy_path


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_args(args)
        try:
            import trimesh
        except ImportError as exc:
            raise RuntimeError("trimesh is required") from exc
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("mujoco is required") from exc
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("opencv-python is required") from exc

        scene_dir = args.scene_dir.expanduser().resolve()
        if not scene_dir.is_dir():
            raise RuntimeError(f"scene directory does not exist: {scene_dir}")
        joint_path = (
            args.joint_file.expanduser().resolve()
            if args.joint_file is not None
            else scene_dir / "simple_joint" / "joints.json"
        )
        registered_path = scene_dir / "registered_static" / "metadata.json"
        joint_payload = read_json(joint_path)
        registered = read_json(registered_path)
        if registered.get("mesh_source") not in (None, "segvigen"):
            raise RuntimeError("Stage-31 metadata is not based on SegviGen pieces")

        label, joint = choose_joint(joint_payload, args.label)
        static_labels = list(dict.fromkeys(registered.get("static_labels", [])))
        if not static_labels:
            raise RuntimeError("Stage-31 metadata contains no static_labels")
        static_geometry = registered.get("static_geometry")
        if not static_geometry:
            raise RuntimeError("Stage-31 metadata contains no static_geometry")
        moving_record = collect_label_records(registered, label)[0]
        frame = moving_record["frame"]
        registered_mesh_path = resolve_registered_mesh(
            scene_dir, registered, registered_path
        )
        if not registered_mesh_path.is_file():
            raise RuntimeError(
                f"missing Stage-31 registered mesh: {registered_mesh_path}"
            )
        try:
            registered_scene = trimesh.load(
                registered_mesh_path, force="scene", process=False
            )
        except Exception as exc:
            raise RuntimeError(f"could not load {registered_mesh_path}: {exc}") from exc
        static_mesh = load_named_geometry(registered_scene, static_geometry, trimesh)
        moving_mesh = load_named_geometry(
            registered_scene, moving_record["geometry"], trimesh
        )

        axis = joint_axis(joint)
        moving_diagonal = float(
            np.linalg.norm(moving_mesh.bounds[1] - moving_mesh.bounds[0])
        )
        if not np.isfinite(moving_diagonal) or moving_diagonal <= 1e-12:
            raise RuntimeError("moving mesh has degenerate bounds")
        lo, hi, range_source = resolve_range(
            joint, frame, args.range_min, args.range_max, moving_diagonal
        )
        bounds = articulated_bounds(static_mesh, moving_mesh, joint, axis, lo, hi)

        camera_xml = ""
        camera_summary = None
        if args.camera_source == "da3":
            camera_frame = (
                frame if args.camera_frame is None else int(args.camera_frame)
            )
            camera_to_reference, intrinsics = da3_camera_pose(
                scene_dir, registered, camera_frame
            )
            image_size = da3_image_size(scene_dir, intrinsics)
            render_size = resolve_render_size(args, image_size)
            args.width, args.height = render_size
            camera_xml = da3_camera_xml(
                camera_to_reference, intrinsics, image_size, render_size
            )
            camera_summary = da3_camera_summary(
                camera_to_reference, intrinsics, image_size, render_size, camera_frame
            )
        else:
            args.width, args.height = resolve_render_size(args, None)

        xml, assets = make_mjcf(
            static_mesh, moving_mesh, joint, axis, lo, hi, camera_xml
        )
        out_path = (
            args.out.expanduser().resolve()
            if args.out is not None
            else scene_dir / "joint_render" / f"{safe_name(label)}.mp4"
        )

        unit = "rad" if joint["type"] == "revolute" else "scene units"
        display_lo, display_hi = lo, hi
        display_unit = unit
        if joint["type"] == "revolute":
            display_lo, display_hi = np.rad2deg([lo, hi])
            display_unit = "deg"
        source = moving_record.get("source") or moving_record["geometry"]
        print(f"frame: {frame:06d} ({source})")
        print(f"registered scene: {registered_mesh_path}")
        print(f"static: {', '.join(static_labels)} = gray")
        print(f"moving: {label} = orange")
        print(
            f"joint: {joint['type']}, range=[{display_lo:.4f}, "
            f"{display_hi:.4f}] {display_unit} ({range_source}, offsets from "
            f"frame {frame:06d})"
        )
        frame_count, duration, camera = render_video(
            mujoco, cv2, xml, assets, out_path, lo, hi, args, bounds
        )
        if camera_summary is not None:
            print(camera_summary)
        else:
            print(
                f"camera: free, azimuth={camera.azimuth:.1f}, "
                f"elevation={camera.elevation:.1f}, distance={camera.distance:.4f}"
            )
        print(
            f"wrote {frame_count} frames ({duration:.2f}s @ {args.fps:g} FPS) "
            f"to {out_path}"
        )
        copy_path = copy_to_render_all(scene_dir, out_path, args)
        if copy_path is not None:
            print(f"copied to {copy_path}")
        return out_path
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
