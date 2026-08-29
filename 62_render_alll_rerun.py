#!/usr/bin/env python
"""Stage 62: replay the complete articulated scene in Rerun.

This is the Rerun counterpart to :mod:`61_render_all`.  It reuses Stage 61's
validated scene preparation and hand-relative joint trajectory, then logs one
synchronized timeline containing:

* the registered static SegViGen mesh;
* a first- or last-frame canonical moving mesh driven at every video frame;
* the HaWoR hand surface and skeleton;
* the ``simple_joint`` axis (an unlabeled shaft by default; pass
  ``--show-labels`` for the floating joint/part names, or ``--joint-style
  arrow`` for Stage 61's arrow glyph); and
* the DA3 camera, RGB frame, and z-depth map.

The 3D view always shows every logged entity. The RGB pane is a filtered
projection: by default it draws only the hand skeleton over the untouched
video, so neither the meshes nor the joint axis hide what the camera saw.
``--rgb-overlays`` changes that set, and ``--media-side`` / ``--scene-share`` /
``--media-shares`` reshape the panes.

The Rerun root is DA3 world space.  Stage-31 geometry stays in its native
registered-reference coordinates below one static similarity transform.  The
camera therefore remains rigid in DA3 world coordinates and raw DA3 depth can
be back-projected beneath its RDF/OpenCV pinhole without an extra scale fix.

Depth and cameras use the same *direct numeric frame IDs* as Stages 31, 60,
60a, and 61.  An exact depth file is used whenever it exists.  A missing depth
at either end of the timeline is endpoint-held by default (matching Stage 61's
camera policy); an internal hole is cleared rather than interpolated.

Examples::

    python scripts/62_render_all_rerun.py --scene-dir data/trashbin
    python scripts/62_render_all_rerun.py --scene-dir data/trashbin \
        --mesh-frame last
    python scripts/62_render_all_rerun.py --scene-dir data/trashbin \
        --save-rrd
    python scripts/62_render_all_rerun.py --scene-dir data/trashbin \
        --show-labels --joint-style arrow
    python scripts/62_render_all_rerun.py --scene-dir data/trashbin \
        --rgb-overlays all --media-side bottom --scene-share 1.4
    python scripts/62_render_all_rerun.py --scene-dir data/trashbin --dry-run

Live replay requires ``rerun-sdk``, numpy, Pillow, and trimesh. ``--dry-run``
keeps the Rerun import lazy, so it can validate the full input bundle without
the SDK.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


WORLD_ROOT = "world"
REFERENCE_ROOT = "world/stage31"
CAMERA_ROOT = "world/camera"
PINHOLE_ROOT = "world/camera/pinhole"
RGB_PATH = "world/camera/pinhole/rgb"
DEPTH_PATH = "world/camera/pinhole/depth"

# Stage 62 overrides Stage 61's palette so the three mesh categories never
# share a hue band: the static part is achromatic gray, every moving part is
# warm (orange, gold, pink, sienna, sand), and both hands are cool (cyan and
# purple). Stage 61 gave the right hand the same orange as its first moving
# part and the left hand the same blue as its second, so a hand resting on a
# door read as part of the door. Every cross-category pair below is at least
# ~27 CIELab dE apart, including against the white viewer background and the
# translucent joint glyphs.
STATIC_COLOR = (150, 158, 172)
MOVING_COLORS = [
    (242, 140, 45),
    (247, 196, 60),
    (232, 100, 150),
    (163, 74, 38),
    (252, 214, 150),
]
HAND_COLORS = {
    "left": (40, 190, 210),
    "right": (178, 92, 240),
}
# In-filled (undetected) HaWoR samples keep their side's hue and only drop in
# lightness, so a tracking gap stays readable as that hand instead of fading
# toward the gray static mesh.
HAND_INFILL_COLORS = {
    "left": (26, 118, 132),
    "right": (110, 60, 168),
}
CAMERA_PATH_COLOR = (220, 75, 75)


def _load_stage61():
    """Load the numeric Stage-61 script without importing Viser or OpenCV."""
    path = SCRIPT_DIR / "61_render_all.py"
    spec = importlib.util.spec_from_file_location(
        "articulate4d_stage61_render_all", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load Stage 61 from {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their module through sys.modules while the module is
    # executing, so registration must happen before exec_module().
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE61 = _load_stage61()


@dataclass(frozen=True)
class MediaSpec:
    """One fixed RGB/depth resolution and its exact pixel scaling."""

    input_hw: tuple[int, int]
    output_hw: tuple[int, int]
    scale_xy: tuple[float, float]


@dataclass(frozen=True)
class DepthTrajectory:
    """Direct-index DA3 depth files with one validated image shape."""

    frame_ids: np.ndarray
    paths: tuple[Path, ...]
    image_hw: tuple[int, int]
    robust_range: tuple[float, float]


@dataclass(frozen=True)
class DepthSample:
    """Depth selected for one animation frame."""

    frame_id: int
    source_frame: int
    path: Path
    native: bool


@dataclass(frozen=True)
class PreparedRerunScene:
    """Stage-61 scene plus Stage-62 depth/media information."""

    scene: Any
    depths: DepthTrajectory | None
    media: MediaSpec


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
        help=("Hand directory. 'auto' prefers hawor_scaled, then hawor"),
    )
    parser.add_argument(
        "--mesh-frame", choices=("first", "last"), default="first",
        help="Canonical registered moving mesh used for every animation frame",
    )
    parser.add_argument(
        "--labels", nargs="*", default=None,
        help="Moving-label subset (default: every simple_joint label)",
    )
    parser.add_argument(
        "--driver-hand", choices=("auto", "left", "right"), default="auto",
        help="HaWoR side whose joint-relative velocity drives moving parts",
    )
    parser.add_argument(
        "--hand-coordinate-source", choices=("auto", "world", "camera"),
        default="auto",
        help="Prefer saved world hands or force DA3 camera conversion",
    )
    parser.add_argument(
        "--detected-only", action="store_true",
        help="Exclude HaWoR in-filled samples when preparing the trajectory",
    )
    parser.add_argument("--start-frame", type=int,
                        help="First logged numeric frame (inclusive)")
    parser.add_argument("--end-frame", type=int,
                        help="Last logged numeric frame (inclusive)")
    parser.add_argument(
        "--velocity-smoothing", type=int, default=5,
        help="Moving-average width for hand-relative joint velocity",
    )
    parser.add_argument(
        "--playback-fps", type=float,
        help="Timeline FPS; defaults to HaWoR overlay FPS, then 24",
    )
    parser.add_argument(
        "--media-max-side", type=int, default=720,
        help=("Maximum RGB/depth side logged to Rerun; both media and the "
              "full K matrix are scaled together"),
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=90,
        help="Rerun JPEG quality for resized RGB frames",
    )
    parser.add_argument(
        "--depth-point-fill-ratio", type=float, default=0.5,
        help=("Rerun point radius/fill ratio for the depth map's automatic "
              "3D point cloud"),
    )
    parser.add_argument(
        "--no-endpoint-depth-hold", action="store_true",
        help="Clear rather than endpoint-hold depth outside native DA3 IDs",
    )
    parser.add_argument(
        "--axis-length-scale", type=float, default=0.60,
        help="Joint-axis half-length in moving-part diagonals",
    )
    parser.add_argument(
        "--axis-radius-scale", type=float, default=0.012,
        help="Joint-axis radius in moving-part diagonals",
    )
    parser.add_argument(
        "--joint-style", choices=("line", "arrow"), default="line",
        help=("Joint-axis glyph: a plain double-ended shaft, or Stage 61's "
              "single arrow"),
    )
    parser.add_argument(
        "--joint-opacity", type=float, default=0.55,
        help="Alpha applied to joint axes, centers, and rotation rings",
    )
    parser.add_argument(
        "--show-labels", action="store_true",
        help=("Draw the floating joint/moving-part name next to each joint "
              "(hidden by default)"),
    )
    parser.add_argument(
        "--no-joint-rings", action="store_true",
        help="Do not draw the revolute rotation ring around each joint",
    )
    parser.add_argument(
        "--rgb-overlays", nargs="*", default=["skeleton"],
        choices=("object", "hands", "skeleton", "joints", "all", "none"),
        metavar="LAYER",
        help=("Geometry projected onto the RGB pane: 'object' meshes, 'hands' "
              "surfaces, hand 'skeleton' lines, 'joints' axes -- or 'all' / "
              "'none'. The 3D view always shows everything"),
    )
    parser.add_argument(
        "--media-side", choices=("right", "left", "bottom", "top"),
        default="right",
        help="Edge of the 3D view that carries the RGB and depth panes",
    )
    parser.add_argument(
        "--scene-share", type=float, default=2.0,
        help="3D view size relative to the media group's share of 1.0",
    )
    parser.add_argument(
        "--media-shares", type=float, nargs=2, default=(1.0, 1.0),
        metavar=("RGB", "DEPTH"),
        help="Relative sizes of the RGB and depth panes within their group",
    )
    parser.add_argument(
        "--hide-panels", action="store_true",
        help=("Hide the viewer's top, blueprint, selection, and time panels "
              "(what Stage 62b uses for video capture)"),
    )
    parser.add_argument(
        "--hide-view-titles", action="store_true",
        help="Blank each view's title text, leaving only its button strip",
    )
    parser.add_argument("--no-rgb", action="store_true",
                        help="Do not log RGB images or create an RGB pane")
    parser.add_argument("--no-depth", action="store_true",
                        help="Do not log depth images or create a depth pane")
    parser.add_argument("--no-hands", action="store_true",
                        help="Do not log HaWoR hand surfaces")
    parser.add_argument("--no-skeleton", action="store_true",
                        help="Do not log HaWoR hand skeletons")
    parser.add_argument("--no-meshes", action="store_true",
                        help="Do not log registered static/moving meshes")
    parser.add_argument("--no-joints", action="store_true",
                        help="Do not log simple_joint axes")
    parser.add_argument(
        "--port", type=int, default=9999,
        help="Port used by the spawned Rerun viewer",
    )
    parser.add_argument(
        "--save-rrd", type=Path, nargs="?",
        const=Path("render_all/stage62.rerun.rrd"), metavar="PATH",
        help=("Save an offline RRD instead of spawning a viewer; without PATH, "
              "write render_all/stage62.rerun.rrd inside the scene"),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing --save-rrd file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and summarize inputs without importing Rerun",
    )
    return parser.parse_args(argv)


def _stage61_args(args):
    """Build Stage 61's complete namespace using its own parser defaults."""
    forwarded = [
        "--scene-dir", str(args.scene_dir),
        "--hawor-name", args.hawor_name,
        "--mesh-frame", args.mesh_frame,
        "--driver-hand", args.driver_hand,
        "--hand-coordinate-source", args.hand_coordinate_source,
        "--velocity-smoothing", str(args.velocity_smoothing),
        "--axis-length-scale", str(args.axis_length_scale),
        "--axis-radius-scale", str(args.axis_radius_scale),
        "--port", str(args.port),
    ]
    if args.labels is not None:
        forwarded.extend(["--labels", *args.labels])
    if args.detected_only:
        forwarded.append("--detected-only")
    if args.start_frame is not None:
        forwarded.extend(["--start-frame", str(args.start_frame)])
    if args.end_frame is not None:
        forwarded.extend(["--end-frame", str(args.end_frame)])
    if args.playback_fps is not None:
        forwarded.extend(["--playback-fps", str(args.playback_fps)])
    return STAGE61.parse_args(forwarded)


ALL_RGB_OVERLAYS = ("object", "hands", "skeleton", "joints")


def _resolve_overlays(requested) -> tuple[str, ...]:
    """Expand the 'all'/'none' shorthands into an explicit layer tuple."""
    layers = list(requested)
    if "none" in layers:
        if len(layers) > 1:
            raise ValueError("--rgb-overlays none cannot list other layers")
        return ()
    if "all" in layers:
        return ALL_RGB_OVERLAYS
    return tuple(layer for layer in ALL_RGB_OVERLAYS if layer in layers)


def make_media_spec(image_hw: tuple[int, int], max_side: int) -> MediaSpec:
    """Choose one no-upscale media size shared by RGB, depth, and K."""
    height, width = map(int, image_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid input media size: {width}x{height}")
    if max_side <= 0:
        raise ValueError("--media-max-side must be positive")
    scale = min(1.0, float(max_side) / max(height, width))
    output_h = max(1, int(round(height * scale)))
    output_w = max(1, int(round(width * scale)))
    return MediaSpec(
        input_hw=(height, width),
        output_hw=(output_h, output_w),
        scale_xy=(output_w / float(width), output_h / float(height)),
    )


def scale_intrinsics(K: np.ndarray, media: MediaSpec) -> np.ndarray:
    """Left-scale a full pinhole K, including skew and principal point."""
    matrix = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    if (not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-5)):
        raise ValueError("DA3 intrinsics must be a finite pinhole matrix")
    sx, sy = media.scale_xy
    matrix[0, :] *= sx
    matrix[1, :] *= sy
    matrix[2] = [0.0, 0.0, 1.0]
    if (abs(float(np.linalg.det(matrix[:2, :2]))) < 1e-12
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("scaled DA3 intrinsics are singular or non-positive")
    return matrix.astype(np.float32)


def _numeric_depth_paths(root: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    if not root.is_dir():
        raise FileNotFoundError(f"missing DA3 depth directory: {root}")
    for path in sorted(root.glob("*.npy")):
        if not path.stem.isdigit():
            continue
        frame_id = int(path.stem)
        if frame_id in paths:
            raise ValueError(
                f"duplicate numeric DA3 depth ID {frame_id}: "
                f"{paths[frame_id].name} and {path.name}")
        paths[frame_id] = path
    if not paths:
        raise FileNotFoundError(f"no numeric *.npy depth maps under {root}")
    return paths


def _estimate_depth_range(paths: tuple[Path, ...]) -> tuple[float, float]:
    """Estimate one robust native-unit range without reading every full map."""
    sample_count = min(12, len(paths))
    indices = np.unique(np.linspace(
        0, len(paths) - 1, sample_count, dtype=np.int64))
    samples = []
    for index in indices:
        depth = np.load(paths[int(index)], mmap_mode="r", allow_pickle=False)
        row_stride = max(1, depth.shape[0] // 96)
        col_stride = max(1, depth.shape[1] // 96)
        subset = np.asarray(depth[::row_stride, ::col_stride], dtype=np.float32)
        valid = subset[np.isfinite(subset) & (subset > 0.0)]
        if len(valid):
            samples.append(valid)
        del depth
    if not samples:
        raise ValueError("sampled DA3 depth maps contain no finite positive depth")
    values = np.concatenate(samples)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite([low, high]).all() or high <= low:
        low, high = float(values.min()), float(values.max())
    if not np.isfinite([low, high]).all() or high <= low:
        high = float(low) + max(abs(float(low)) * 1e-3, 1e-6)
    return float(low), float(high)


def load_depth_trajectory(
        scene_dir: Path, image_hw: tuple[int, int],
        camera_frame_ids: np.ndarray) -> DepthTrajectory:
    """Validate direct-index DA3 depth files against cameras and RGB shape."""
    by_frame = _numeric_depth_paths(scene_dir / "da3" / "depth")
    expected_hw = tuple(map(int, image_hw))
    camera_ids = {int(value) for value in np.asarray(camera_frame_ids).reshape(-1)}
    missing_camera = sorted(set(by_frame) - camera_ids)
    if missing_camera:
        preview = missing_camera[:8]
        suffix = " ..." if len(missing_camera) > len(preview) else ""
        raise ValueError(
            "DA3 depth IDs without an exact camera/intrinsics record: "
            f"{preview}{suffix}")
    for frame_id, path in sorted(by_frame.items()):
        try:
            depth = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not read DA3 depth {path}: {exc}") from exc
        if depth.ndim != 2 or tuple(depth.shape) != expected_hw:
            raise ValueError(
                f"DA3 depth {frame_id:06d} has shape {depth.shape}; "
                f"expected {expected_hw}")
        if not np.issubdtype(depth.dtype, np.number):
            raise ValueError(f"DA3 depth {frame_id:06d} is not numeric")
        row_stride = max(1, depth.shape[0] // 64)
        col_stride = max(1, depth.shape[1] // 64)
        probe = np.asarray(
            depth[::row_stride, ::col_stride], dtype=np.float32)
        if not np.any(np.isfinite(probe) & (probe > 0.0)):
            # A sparse valid map can miss the cheap probe, so scan the full
            # array only on this exceptional path before rejecting it.
            full = np.asarray(depth)
            if not np.any(np.isfinite(full) & (full > 0.0)):
                raise ValueError(
                    f"DA3 depth {frame_id:06d} has no finite positive values")
        del depth
    frame_ids = np.asarray(sorted(by_frame), dtype=np.int64)
    paths = tuple(by_frame[int(frame)] for frame in frame_ids)
    return DepthTrajectory(
        frame_ids=frame_ids,
        paths=paths,
        image_hw=expected_hw,
        robust_range=_estimate_depth_range(paths),
    )


def depth_sample_at(
        trajectory: DepthTrajectory, frame_id: int,
        endpoint_hold: bool = True) -> DepthSample | None:
    """Use direct IDs; optionally hold only outside the native depth range."""
    requested = int(frame_id)
    position = int(np.searchsorted(trajectory.frame_ids, requested))
    if (position < len(trajectory.frame_ids)
            and int(trajectory.frame_ids[position]) == requested):
        return DepthSample(
            frame_id=requested,
            source_frame=requested,
            path=trajectory.paths[position],
            native=True,
        )
    if endpoint_hold and (position == 0 or position == len(trajectory.frame_ids)):
        index = 0 if position == 0 else len(trajectory.frame_ids) - 1
        return DepthSample(
            frame_id=requested,
            source_frame=int(trajectory.frame_ids[index]),
            path=trajectory.paths[index],
            native=False,
        )
    # Pixel-wise depth cannot be meaningfully interpolated across an internal
    # camera/image gap, so clear it rather than displaying stale geometry.
    return None


def camera_reference_to_world(
        camera_to_reference: np.ndarray,
        reference_to_world: np.ndarray) -> np.ndarray:
    """Compose a rigid C2R camera with the Stage-31 R2W similarity.

    Translation is scaled, but camera rotation is not.  This is intentionally
    not a raw 4x4 matrix product, whose upper-left block would contain scale.
    """
    camera = np.asarray(camera_to_reference, dtype=np.float64).reshape(4, 4)
    scale, rotation, translation = STAGE61._reference_similarity(
        reference_to_world)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ camera[:3, :3]
    result[:3, 3] = (
        scale * (rotation @ camera[:3, 3]) + translation)
    if (not np.isfinite(result).all()
            or not np.allclose(result[:3, :3].T @ result[:3, :3],
                               np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=2e-5)):
        raise ValueError("composed DA3 camera-to-world pose is not rigid")
    return result.astype(np.float32)


def moving_transform(record: dict, state_delta: float) -> np.ndarray:
    """Return the child-to-parent transform for a canonical moving mesh."""
    axis = np.asarray(record["axis_direction"], dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("simple_joint axis is invalid")
    axis /= norm
    transform = np.eye(4, dtype=np.float64)
    if record["type"] == "prismatic":
        transform[:3, 3] = float(state_delta) * axis
    elif record["type"] == "revolute":
        pivot = np.asarray(record["axis_point"], dtype=np.float64).reshape(3)
        rotation = STAGE61._axis_angle_rotation(axis, float(state_delta))
        transform[:3, :3] = rotation
        transform[:3, 3] = pivot - rotation @ pivot
    else:
        raise ValueError(f"unsupported simple_joint type: {record['type']!r}")
    return transform.astype(np.float32)


def moving_transform_at(
        prepared, label: str, mesh_frame: str,
        frame_id: int) -> tuple[np.ndarray, float, float]:
    """Evaluate the dense state relative to one canonical mesh's native q."""
    mesh = prepared.meshes.moving[label][mesh_frame]
    if mesh.frame_id is None:
        raise ValueError(f"moving mesh {label!r} has no native frame ID")
    canonical_state = STAGE61.sparse_state_at(
        prepared.joints[label], mesh.frame_id)
    state = prepared.motions[label].state_at(int(frame_id))
    transform = moving_transform(
        prepared.joints[label], state - canonical_state)
    return transform, float(state), float(canonical_state)


def _resize_depth(depth: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    """Validity-aware bilinear depth resize followed by NaN sanitization."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth must be HxW, got {values.shape}")
    valid = np.isfinite(values) & (values > 0.0)
    if not valid.any():
        raise ValueError("DA3 depth map has no finite positive values")
    output_h, output_w = map(int, output_hw)
    if (output_h, output_w) == tuple(values.shape):
        return np.where(valid, values, np.nan).astype(np.float32)
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to resize Stage-62 media") from exc
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    numerator = Image.fromarray(
        np.where(valid, values, 0.0), mode="F").resize(
            (output_w, output_h), resample=resampling)
    denominator = Image.fromarray(valid.astype(np.float32), mode="F").resize(
        (output_w, output_h), resample=resampling)
    numerator_array = np.asarray(numerator, dtype=np.float32)
    denominator_array = np.asarray(denominator, dtype=np.float32)
    resized = np.full((output_h, output_w), np.nan, dtype=np.float32)
    usable = denominator_array > 1e-3
    resized[usable] = (
        numerator_array[usable] / denominator_array[usable])
    resized[~np.isfinite(resized) | (resized <= 0.0)] = np.nan
    return resized


def load_depth_image(sample: DepthSample, media: MediaSpec) -> np.ndarray:
    try:
        depth = np.load(sample.path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not load DA3 depth {sample.path}: {exc}") from exc
    if tuple(depth.shape) != media.input_hw:
        raise ValueError(
            f"DA3 depth changed shape at runtime: {depth.shape} != "
            f"{media.input_hw}")
    return _resize_depth(depth, media.output_hw)


def load_rgb_image(path: Path, output_hw: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read Stage-62 RGB frames") from exc
    try:
        with Image.open(path) as image:
            image = image.convert("RGB")
            output_h, output_w = map(int, output_hw)
            if image.size != (output_w, output_h):
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                image = image.resize(
                    (output_w, output_h), resample=resampling)
            return np.asarray(image, dtype=np.uint8).copy()
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read RGB frame {path}: {exc}") from exc


def prepare_rerun_scene(args) -> PreparedRerunScene:
    if args.jpeg_quality < 1 or args.jpeg_quality > 100:
        raise ValueError("--jpeg-quality must be between 1 and 100")
    if (not np.isfinite(args.depth_point_fill_ratio)
            or args.depth_point_fill_ratio <= 0):
        raise ValueError("--depth-point-fill-ratio must be positive")
    if not 0.0 <= args.joint_opacity <= 1.0:
        raise ValueError("--joint-opacity must be between 0 and 1")
    if args.scene_share <= 0 or min(args.media_shares) <= 0:
        raise ValueError("--scene-share and --media-shares must be positive")
    args.rgb_overlays = _resolve_overlays(args.rgb_overlays)
    stage61_args = _stage61_args(args)
    scene = STAGE61.prepare_scene(stage61_args)
    media = make_media_spec(scene.image_hw, args.media_max_side)
    depths = None
    if not args.no_depth:
        depths = load_depth_trajectory(
            scene.scene_dir, scene.image_hw, scene.cameras.frame_ids)
    return PreparedRerunScene(scene=scene, depths=depths, media=media)


def _rrd_output_path(scene_dir: Path, requested: Path) -> Path:
    path = requested.expanduser()
    path = path.resolve() if path.is_absolute() else (scene_dir / path).resolve()
    if path.suffix.lower() != ".rrd":
        raise ValueError("--save-rrd PATH must end in .rrd")
    return path


def _require_rerun():
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Stage 62 needs the Rerun SDK. Remove the unrelated 'rerun' "
            "package if present, then install it with:\n"
            "  python -m pip uninstall -y rerun\n"
            "  python -m pip install rerun-sdk") from exc
    required = ("DepthImage", "Mesh3D", "Pinhole", "Transform3D", "log")
    missing = [name for name in required if not hasattr(rr, name)]
    if missing or not hasattr(rrb, "Spatial3DView"):
        raise RuntimeError(
            "the imported 'rerun' module is not rerun-sdk or is too old; "
            f"missing APIs: {missing or ['rerun.blueprint.Spatial3DView']}")
    return rr, rrb


def _set_time(rr, frame_id: int, seconds: float):
    """Use current Rerun timelines with a Stage-41-compatible fallback."""
    if hasattr(rr, "set_time"):
        rr.set_time("frame", sequence=int(frame_id))
        rr.set_time("stable_time", duration=float(seconds))
    else:
        rr.set_time_sequence("frame", int(frame_id))
        rr.set_time_seconds("stable_time", float(seconds))


def _mesh_archetype(rr, vertices: np.ndarray, faces: np.ndarray, color):
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.broadcast_to(
        np.asarray(color, dtype=np.uint8), (len(vertices), 3)).copy()
    return rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=np.asarray(faces, dtype=np.uint32),
        vertex_colors=colors,
    )


def _transform_archetype(rr, transform: np.ndarray):
    """Build a parent-from-child pose across Rerun SDK generations.

    Stage 31/61 matrices map entity-local points into their parent space:
    reference -> world, camera -> world, and moving part -> reference. Rerun's
    explicit relation must therefore be ``ParentFromChild``. Using the inverse
    relation mirrors the camera and geometry branches through their poses and
    places otherwise projectable meshes behind the camera.
    """
    matrix = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    kwargs = {
        "translation": matrix[:3, 3],
        "mat3x3": matrix[:3, :3],
    }
    relation_type = getattr(rr, "TransformRelation", None)
    if relation_type is not None and hasattr(relation_type, "ParentFromChild"):
        kwargs["relation"] = relation_type.ParentFromChild
    else:
        # Stage-41-era Rerun: False means the logged space maps to its parent.
        kwargs["from_parent"] = False
    return rr.Transform3D(**kwargs)


def _depth_archetype(rr, depth: np.ndarray,
                     robust_range: tuple[float, float],
                     point_fill_ratio: float):
    """Build DepthImage across SDKs, fixing its color range when supported."""
    common = {
        "meter": 1.0,
        "colormap": "turbo",
        "point_fill_ratio": float(point_fill_ratio),
    }
    try:
        # depth_range was added after the Stage-41-era SDK. Keeping one range
        # across the clip prevents the depth colors from flickering.
        return rr.DepthImage(
            depth, depth_range=list(robust_range), **common)
    except TypeError:
        return rr.DepthImage(depth, **common)


def _fade(color, opacity: float):
    """Return an RGBA color so joint glyphs read as a translucent overlay."""
    alpha = int(round(255 * min(max(float(opacity), 0.0), 1.0)))
    red, green, blue = (int(channel) for channel in color[:3])
    return (red, green, blue, alpha)


def _joint_visual_geometry(prepared, label: str,
                           length_scale: float, radius_scale: float):
    record = prepared.joints[label]
    mesh = prepared.meshes.moving[label]["first"]
    axis = np.asarray(record["axis_direction"], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    center = np.asarray(record.get("position"), dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        center = mesh.vertices.mean(axis=0).astype(np.float64)
    scene_min, scene_max = STAGE61._scene_bounds(prepared)
    scene_diagonal = max(float(np.linalg.norm(scene_max - scene_min)), 1e-6)
    part_diagonal = max(mesh.diagonal, 1e-6)
    # The scene-diagonal floors only keep a joint on a very small part from
    # collapsing to nothing; they are deliberately well below Stage 61's so
    # the axis reads as an annotation rather than as scene geometry.
    half_length = max(
        float(length_scale) * part_diagonal,
        0.12 * scene_diagonal,
    )
    radius = max(
        float(radius_scale) * part_diagonal,
        0.0025 * scene_diagonal,
        1e-5,
    )
    return center, axis, half_length, radius, part_diagonal


def _log_static_scene(rr, prepared, args):
    rr.log(WORLD_ROOT, rr.ViewCoordinates.RDF, static=True)
    reference_to_world = np.asarray(
        prepared.meshes.reference_to_world, dtype=np.float32)
    rr.log(
        REFERENCE_ROOT,
        _transform_archetype(rr, reference_to_world),
        static=True,
    )

    if not args.no_meshes:
        static_mesh = prepared.meshes.static
        rr.log(
            f"{REFERENCE_ROOT}/object/static",
            _mesh_archetype(
                rr, static_mesh.vertices, static_mesh.faces, STATIC_COLOR),
            static=True,
        )
        for index, (label, choices) in enumerate(
                prepared.meshes.moving.items()):
            mesh = choices[args.mesh_frame]
            color = MOVING_COLORS[index % len(MOVING_COLORS)]
            rr.log(
                f"{REFERENCE_ROOT}/object/moving/{label}/mesh",
                _mesh_archetype(rr, mesh.vertices, mesh.faces, color),
                static=True,
            )

    if not args.no_joints:
        for label, record in prepared.joints.items():
            center, axis, half_length, radius, part_diagonal = (
                _joint_visual_geometry(
                    prepared, label,
                    args.axis_length_scale, args.axis_radius_scale))
            color = _fade(
                STAGE61.JOINT_COLORS[record["type"]], args.joint_opacity)
            base = f"{REFERENCE_ROOT}/simple_joint/{label}"
            start = center - half_length * axis
            end = center + half_length * axis
            axis_labels = (
                [f"{label}: {record['type']}"] if args.show_labels else None)
            if args.joint_style == "arrow":
                rr.log(
                    f"{base}/axis",
                    rr.Arrows3D(
                        origins=[start], vectors=[end - start], colors=[color],
                        radii=[radius], labels=axis_labels,
                    ),
                    static=True,
                )
            else:
                # A plain double-ended shaft has no arrow head to catch the
                # eye; the axis is a line, so its two ends are equivalent
                # anyway and the head only added visual weight.
                rr.log(
                    f"{base}/axis",
                    rr.LineStrips3D(
                        [np.asarray([start, end], dtype=np.float32)],
                        colors=[color], radii=[radius], labels=axis_labels,
                    ),
                    static=True,
                )
            rr.log(
                f"{base}/position",
                rr.Points3D(
                    [center], colors=[color], radii=[1.6 * radius],
                    labels=[label] if args.show_labels else None),
                static=True,
            )
            if record["type"] == "revolute" and not args.no_joint_rings:
                first, second = STAGE61._orthogonal_basis(axis)
                angles = np.linspace(0.0, 2.0 * np.pi, 65)
                ring = center + 0.22 * part_diagonal * (
                    np.cos(angles)[:, None] * first
                    + np.sin(angles)[:, None] * second)
                rr.log(
                    f"{base}/rotation",
                    rr.LineStrips3D(
                        [ring.astype(np.float32)], colors=[color],
                        radii=[0.7 * radius]),
                    static=True,
                )

    camera_positions = []
    for frame_id in prepared.frame_ids:
        sample = STAGE61.camera_sample_at(prepared.cameras, int(frame_id))
        camera_world = camera_reference_to_world(
            sample.camera_to_reference,
            prepared.meshes.reference_to_world)
        camera_positions.append(camera_world[:3, 3])
    if len(camera_positions) >= 2:
        rr.log(
            f"{WORLD_ROOT}/camera_path",
            rr.LineStrips3D(
                [np.asarray(camera_positions, dtype=np.float32)],
                colors=[CAMERA_PATH_COLOR]),
            static=True,
        )


def _hidden_panels(rrb):
    """Blueprint parts that hide every piece of viewer chrome we can hide.

    A view still keeps a thin tab bar holding its help/visibility/maximize
    buttons; only its *title text* goes away, by naming the view "". Stage 62b
    crops those bars off using the viewer's own pane rectangles.
    """
    parts = []
    for name in ("BlueprintPanel", "SelectionPanel", "TimePanel", "TopPanel"):
        panel = getattr(rrb, name, None)
        if panel is not None:
            parts.append(panel(state="Hidden"))
    return parts


def _rgb_overlay_contents(prepared, layers) -> list[str]:
    """Entity rules for the RGB pane, one explicit path per enabled layer.

    Rerun's content query only understands a trailing ``/**``; a ``*`` in the
    middle of a path is not supported. The hand layers therefore have to name
    each logged side, which is why this needs the prepared scene.
    """
    contents = [f"/{RGB_PATH}"]
    if "object" in layers:
        contents.append(f"/{REFERENCE_ROOT}/object/**")
    for side in prepared.hands.sides:
        if "hands" in layers:
            contents.append(f"/{REFERENCE_ROOT}/hands/{side}/surface")
        if "skeleton" in layers:
            contents.append(f"/{REFERENCE_ROOT}/hands/{side}/skeleton")
    if "joints" in layers:
        contents.append(f"/{REFERENCE_ROOT}/simple_joint/**")
    return contents


def _compose_layout(rrb, scene_view, media_views, args):
    """Place the media views beside or below the 3D view, honoring the shares.

    ``--media-side`` picks the edge, and the media group stacks along the
    perpendicular axis so each pane keeps a usable aspect ratio.
    """
    if not media_views:
        return scene_view

    horizontal_split = args.media_side in ("left", "right")
    if len(media_views) == 1:
        group = media_views[0]
    elif horizontal_split:
        group = rrb.Vertical(*media_views, row_shares=list(args.media_shares))
    else:
        group = rrb.Horizontal(
            *media_views, column_shares=list(args.media_shares))

    scene_first = args.media_side in ("right", "bottom")
    parts = ((scene_view, group) if scene_first else (group, scene_view))
    shares = ([args.scene_share, 1.0] if scene_first
              else [1.0, args.scene_share])
    if horizontal_split:
        return rrb.Horizontal(*parts, column_shares=shares)
    return rrb.Vertical(*parts, row_shares=shares)


def _log_blueprint(rr, rrb, prepared, args):
    def view_name(name):
        return "" if args.hide_view_titles else name

    scene_view = rrb.Spatial3DView(
        origin=WORLD_ROOT,
        name=view_name("DA3 world + articulated scene"),
        contents=["/world/**"],
        background=[255, 255, 255],
    )
    media_views = []
    if not args.no_rgb:
        media_views.append(rrb.Spatial2DView(
            origin=PINHOLE_ROOT,
            name=view_name("DA3 camera RGB"),
            contents=_rgb_overlay_contents(prepared, args.rgb_overlays),
        ))
    if not args.no_depth:
        media_views.append(rrb.Spatial2DView(
            origin=PINHOLE_ROOT,
            name=view_name("DA3 depth"),
            contents=[f"/{DEPTH_PATH}"],
        ))
    layout = _compose_layout(rrb, scene_view, media_views, args)
    panels = _hidden_panels(rrb) if args.hide_panels else []
    rr.send_blueprint(
        rrb.Blueprint(layout, *panels, collapse_panels=False))


def _log_dynamic_frame(rr, prepared_scene: PreparedRerunScene,
                       args, frame_id: int, _frame_index: int):
    prepared = prepared_scene.scene
    seconds = ((frame_id - int(prepared.frame_ids[0]))
               / float(prepared.playback_fps))
    _set_time(rr, frame_id, seconds)

    camera_sample = STAGE61.camera_sample_at(prepared.cameras, frame_id)
    camera_world = camera_reference_to_world(
        camera_sample.camera_to_reference,
        prepared.meshes.reference_to_world)
    rr.log(
        CAMERA_ROOT,
        _transform_archetype(rr, camera_world),
    )
    K = scale_intrinsics(camera_sample.intrinsics, prepared_scene.media)
    output_h, output_w = prepared_scene.media.output_hw
    rr.log(
        PINHOLE_ROOT,
        rr.Pinhole(
            image_from_camera=K,
            height=output_h,
            width=output_w,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )

    if not args.no_meshes:
        for label in prepared.joints:
            transform, _, _ = moving_transform_at(
                prepared, label, args.mesh_frame, frame_id)
            rr.log(
                f"{REFERENCE_ROOT}/object/moving/{label}",
                _transform_archetype(rr, transform),
            )

    for side in prepared.hands.sides:
        vertices, joints, detected = STAGE61._hand_sample_at(
            prepared.hands.samples[side], frame_id)
        color = (HAND_COLORS[side] if detected
                 else HAND_INFILL_COLORS[side])
        base = f"{REFERENCE_ROOT}/hands/{side}"
        if not args.no_hands:
            rr.log(
                f"{base}/surface",
                _mesh_archetype(
                    rr, vertices, prepared.hands.faces[side], color),
            )
        if not args.no_skeleton:
            segments = STAGE61._hand_skeleton(joints)
            rr.log(
                f"{base}/skeleton",
                rr.LineStrips3D(
                    [segment for segment in segments],
                    colors=[color],
                ),
            )

    if not args.no_rgb:
        rgb_path = prepared.frame_paths.get(int(frame_id))
        if rgb_path is None:
            rr.log(RGB_PATH, rr.Clear(recursive=False))
        else:
            rgb = load_rgb_image(rgb_path, prepared_scene.media.output_hw)
            rr.log(
                RGB_PATH,
                rr.Image(rgb).compress(jpeg_quality=args.jpeg_quality),
            )

    if not args.no_depth:
        if prepared_scene.depths is None:
            raise RuntimeError("depth logging requested but depth was not prepared")
        depth_sample = depth_sample_at(
            prepared_scene.depths, frame_id,
            endpoint_hold=not args.no_endpoint_depth_hold)
        if depth_sample is None:
            rr.log(DEPTH_PATH, rr.Clear(recursive=False))
        else:
            depth = load_depth_image(depth_sample, prepared_scene.media)
            rr.log(
                DEPTH_PATH,
                _depth_archetype(
                    rr, depth, prepared_scene.depths.robust_range,
                    args.depth_point_fill_ratio),
            )


def _flush_rerun(rr, *, disconnect: bool = False):
    """Flush the active recording and optionally close its sinks.

    Recent Rerun SDKs expose these operations on the global RecordingStream,
    while older releases also exposed module-level helpers.
    """
    recording = None
    get_recording = getattr(rr, "get_global_data_recording", None)
    if get_recording is not None:
        recording = get_recording()

    flush_target = (
        recording if recording is not None and hasattr(recording, "flush")
        else rr if hasattr(rr, "flush")
        else None
    )
    if flush_target is not None:
        try:
            flush_target.flush(blocking=True)
        except TypeError:
            flush_target.flush()

    if disconnect:
        disconnect_target = (
            recording
            if recording is not None and hasattr(recording, "disconnect")
            else rr if hasattr(rr, "disconnect")
            else None
        )
        if disconnect_target is not None:
            disconnect_target.disconnect()


def log_rerun_recording(prepared_scene: PreparedRerunScene, args):
    rr, rrb = _require_rerun()
    prepared = prepared_scene.scene
    output_path = None
    rr.init("articulate4d_stage62")
    if args.save_rrd is None:
        rr.spawn(port=args.port)
    else:
        output_path = _rrd_output_path(prepared.scene_dir, args.save_rrd)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"RRD already exists: {output_path}; pass --overwrite")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(output_path))

    _log_blueprint(rr, rrb, prepared, args)
    _log_static_scene(rr, prepared, args)
    for index, frame_id in enumerate(prepared.frame_ids.tolist()):
        _log_dynamic_frame(
            rr, prepared_scene, args, int(frame_id), index)
        print(
            f"\rRerun frame {index + 1}/{len(prepared.frame_ids)} "
            f"({int(frame_id):06d})",
            end="",
            flush=True,
        )
    print()
    _flush_rerun(rr, disconnect=output_path is not None)
    return output_path


def _depth_coverage(prepared_scene: PreparedRerunScene, args):
    if prepared_scene.depths is None:
        return 0, 0, 0
    native = held = missing = 0
    for frame_id in prepared_scene.scene.frame_ids:
        sample = depth_sample_at(
            prepared_scene.depths, int(frame_id),
            endpoint_hold=not args.no_endpoint_depth_hold)
        if sample is None:
            missing += 1
        elif sample.native:
            native += 1
        else:
            held += 1
    return native, held, missing


def print_summary(prepared_scene: PreparedRerunScene, args):
    prepared = prepared_scene.scene
    output_h, output_w = prepared_scene.media.output_hw
    input_h, input_w = prepared_scene.media.input_hw
    print(f"scene: {prepared.scene_dir}")
    print(
        f"frames: {len(prepared.frame_ids)} "
        f"[{int(prepared.frame_ids[0]):06d}, "
        f"{int(prepared.frame_ids[-1]):06d}] at {prepared.playback_fps:g} fps")
    print(
        f"hands: {', '.join(prepared.hands.sides)}; "
        f"driver={prepared.driver_side}; source={prepared.hawor_name}")
    print(
        f"moving labels: {', '.join(prepared.joints)}; "
        f"canonical mesh={args.mesh_frame}")
    media_summary = f"media: {input_w}x{input_h} -> {output_w}x{output_h}"
    if prepared_scene.depths is None:
        print(f"{media_summary}; depth omitted")
    else:
        native, held, missing = _depth_coverage(prepared_scene, args)
        print(
            f"{media_summary}; depth native={native}, "
            f"endpoint-held={held}, cleared={missing}")
        low, high = prepared_scene.depths.robust_range
        print(
            f"DA3 depth IDs: {int(prepared_scene.depths.frame_ids[0]):06d}.."
            f"{int(prepared_scene.depths.frame_ids[-1]):06d}; "
            f"sampled 1%-99% range={low:.5g}..{high:.5g}")
    scale, _, _ = STAGE61._reference_similarity(
        prepared.meshes.reference_to_world)
    print(
        f"coordinates: DA3 world root; Stage-31 reference scale={scale:.6g}")


def main(argv=None):
    args = parse_args(argv)
    try:
        prepared_scene = prepare_rerun_scene(args)
        print_summary(prepared_scene, args)
        if args.save_rrd is not None:
            output = _rrd_output_path(
                prepared_scene.scene.scene_dir, args.save_rrd)
            print(f"RRD output: {output}")
        if args.dry_run:
            print("dry-run complete; Rerun was not imported")
            return 0
        output = log_rerun_recording(prepared_scene, args)
        if output is None:
            print(f"Rerun viewer spawned on port {args.port}")
        else:
            print(f"wrote Rerun recording: {output}")
        return 0
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError,
            ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
