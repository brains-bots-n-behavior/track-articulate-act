#!/usr/bin/env python
"""Stage 06: register frame-specific SegviGen meshes using scaled SAM3D and static parts.

By default, the earliest numeric candidate frame is used as the reference;
``--reference-frame`` can select another available candidate.  Each complete
SegviGen mesh is first similarity-registered to the
scaled SAM3D mesh reconstructed at the same frame.  Each scaled SAM3D mesh is
then placed
with its complete pose, including
the GLB Y-up and PyTorch3D-to-RDF camera-axis conversions.  When DA3 camera
poses are available, these transforms are composed into the world frame before
forming the frame-to-reference initialization.  Static-surface registration
then refines uniform scale and XYZ translation with a fixed rotation selected
from the relative SAM3D pose, identity, or a conservative automatic choice.
The two transforms are composed and applied unchanged to all moving pieces
from that frame.

By default the output GLB contains one copy of the SegviGen reference static
mesh and one moving mesh per label and frame.  With ``--register-sam3d``, the
input switches to the full frame-specific meshes under
``sam3d_scaled/combined`` and the output contains one registered full mesh per
frame instead.  A JSON file records transforms, costs, source paths, and
geometry names.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Scene directory, for example data/trashbin")
    parser.add_argument("--static-labels", nargs="*", default=None,
                        help="Override labels marked static in masks/tracking.json")
    parser.add_argument("--moving-labels", nargs="*", default=None,
                        help="Override labels marked moving in masks/tracking.json")
    parser.add_argument("--register-sam3d", action="store_true",
                        help="Register full meshes from sam3d_scaled/combined "
                             "instead of the default SegviGen static/moving pieces")
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=None,
        metavar="FRAME_ID",
        help=("Candidate frame ID to use as the registration reference "
              "(default: earliest available frame)"),
    )
    parser.add_argument(
        "--static-rotation-source",
        choices=("sam3d", "identity", "auto"),
        default="sam3d",
        help=("Rotation initializer for cross-frame static registration: "
              "'sam3d' preserves the relative SAM3D rotation (default), "
              "'identity' assumes pose-corrected meshes share canonical "
              "orientation, and 'auto' selects identity only for a clear "
              "near-180-degree SAM3D conflict"),
    )
    parser.add_argument("--samples", type=int, default=5000,
                        help="Static-surface samples used for registration (default: 5000)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Fixed-rotation scale/translation registration iterations "
                             "(default: 100)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Surface-sampling random seed (default: 0)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: <scene>/registered_static)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output files")
    return parser.parse_args()


def read_json(path: Path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc


def safe_name(label: str):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def load_mesh(path: Path, trimesh):
    if not path.is_file():
        raise RuntimeError(f"missing mesh piece: {path}")
    try:
        asset = trimesh.load(path, force="scene", process=False)
        meshes = [mesh for mesh in asset.dump(concatenate=False)
                  if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces)]
    except Exception as exc:
        raise RuntimeError(f"could not load mesh {path}: {exc}") from exc
    if not meshes:
        raise RuntimeError(f"mesh has no triangle geometry: {path}")
    return trimesh.util.concatenate(meshes)


def combine_labels(frame_dir: Path, labels, trimesh):
    return trimesh.util.concatenate([
        load_mesh(frame_dir / "pieces" / f"{safe_name(label)}.glb", trimesh)
        for label in labels
    ])


def discover_frames(combined_dir: Path, required_labels):
    frames = []
    skipped = []
    for path in combined_dir.iterdir() if combined_dir.is_dir() else []:
        if path.is_dir() and path.name.isdigit():
            missing = [label for label in required_labels
                       if not (path / "pieces" / f"{safe_name(label)}.glb").is_file()]
            if missing:
                skipped.append((int(path.name), missing))
            else:
                frames.append((int(path.name), path))
    frames.sort(key=lambda item: item[0])
    if not frames:
        raise RuntimeError(f"no complete frame folders found under {combined_dir}")
    return frames, sorted(skipped)


def discover_poses(combined_dir: Path):
    """Map keyframe IDs to stage-03 pose files."""
    poses = {}
    candidates = sorted(combined_dir.glob("cand_*_*"))
    direct = combined_dir / "pose.json"
    if direct.is_file():
        candidates.append(combined_dir)
    for candidate in candidates:
        pose_path = candidate / "pose.json"
        keyframe_path = candidate / "keyframe.txt"
        if not pose_path.is_file() or not keyframe_path.is_file():
            continue
        try:
            frame_id = int(keyframe_path.read_text().strip())
        except ValueError as exc:
            raise RuntimeError(f"invalid keyframe in {keyframe_path}") from exc
        if frame_id in poses:
            raise RuntimeError(f"multiple SAM3D poses found for frame {frame_id:06d}")
        poses[frame_id] = pose_path
    return poses


def mesh_beside_pose(pose_path: Path):
    """Return the frame-specific SAM3D mesh stored beside a discovered pose."""
    mesh_path = pose_path.parent / "mesh.glb"
    if not mesh_path.is_file():
        raise RuntimeError(f"missing SAM3D mesh beside pose: {mesh_path}")
    return mesh_path


def discover_sam3d_frames(pose_paths):
    """Return frame records for every complete SAM3D mesh/pose candidate."""
    frames = []
    for frame_id, pose_path in sorted(pose_paths.items()):
        frames.append((frame_id, mesh_beside_pose(pose_path)))
    if not frames:
        raise RuntimeError("no complete SAM3D mesh/pose candidates found")
    return frames


def quat_wxyz_to_R(q):
    """Convert a WXYZ quaternion to SAM3D/PyTorch3D's rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("SAM3D rotation quaternion has zero/invalid norm")
    w, x, y, z = np.asarray([w, x, y, z], dtype=np.float64) / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def quat_xyzw_to_R(q):
    """Convert a DA3 XYZW camera quaternion to a rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("DA3 camera quaternion has zero/invalid norm")
    x, y, z, w = np.asarray([x, y, z, w], dtype=np.float64) / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def sam3d_pose_to_rdf_camera(path: Path):
    """Load the exact raw-SAM3D-GLB -> RDF-camera transform used by mesh scaling (stage 04)."""
    pose = read_json(path)
    try:
        rotation = quat_wxyz_to_R(pose["rotation_quat_wxyz"])
        translation = np.asarray(pose["translation"], dtype=float).reshape(3)
        scales = np.asarray(pose["scale"], dtype=float).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid rotation, translation, or scale in {path}") from exc
    if len(scales) not in (1, 3) or not np.isfinite(translation).all() \
            or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError(f"invalid translation or scale in {path}")
    if len(scales) == 1:
        scales = np.repeat(scales, 3)

    model_from_glb = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    pytorch3d_to_rdf = np.diag([-1.0, -1.0, 1.0])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (pytorch3d_to_rdf @ rotation.T
                          @ np.diag(scales) @ model_from_glb)
    transform[:3, 3] = pytorch3d_to_rdf @ translation
    return transform


def load_camera_to_world(scene_dir: Path):
    """Return DA3 RDF-camera -> world transforms keyed by frame."""
    path = scene_dir / "da3" / "cameras.npz"
    if not path.is_file():
        return {}
    with np.load(path) as cameras:
        required = {"frame_indices", "cam_quats_xyzw", "cam_trans"}
        if not required.issubset(cameras.files):
            raise RuntimeError(f"invalid DA3 camera archive: {path}")
        transforms = {}
        for index, frame_id in enumerate(cameras["frame_indices"]):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = quat_xyzw_to_R(
                cameras["cam_quats_xyzw"][index])
            transform[:3, 3] = np.asarray(
                cameras["cam_trans"][index], dtype=np.float64).reshape(3)
            transforms[int(frame_id)] = transform
    return transforms


def labels_by_motion(tracking, override, motion):
    if override is not None:
        labels = list(dict.fromkeys(override))
    else:
        labels = [label for label, entry in tracking.items()
                  if entry.get("motion") == motion]
    if not labels:
        raise RuntimeError(f"no {motion} labels selected")
    unknown = [label for label in labels if label not in tracking]
    if unknown:
        raise RuntimeError(f"unknown {motion} labels: {unknown}")
    return labels


def display_path(path: Path, scene_dir: Path):
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def rotation_angle_degrees(transform):
    """Return the proper-rotation angle represented by a similarity transform."""
    linear = np.asarray(transform[:3, :3], dtype=float)
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise RuntimeError("transform has a degenerate scale")
    u, _, vt = np.linalg.svd(linear / scale)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def register_static(source, target_points, initial, args, trimesh):
    """Refine scale/translation while retaining the pose-derived rotation."""
    from scipy.spatial import cKDTree

    source_points, _ = trimesh.sample.sample_surface(
        source, args.samples, seed=args.seed)
    linear = np.asarray(initial[:3, :3], dtype=float)
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise RuntimeError("pose initialization has a degenerate scale")
    rotation = linear / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    rotated_source = source_points @ rotation.T
    translation = np.asarray(initial[:3, 3], dtype=float).copy()
    target_tree = cKDTree(target_points)
    previous_cost = np.inf

    # Bidirectional closest-point correspondences prevent the scale from
    # collapsing toward a small region of the target surface.
    for _ in range(args.iterations):
        transformed = scale * rotated_source + translation
        forward_distance, forward_index = target_tree.query(transformed, k=1)
        source_tree = cKDTree(transformed)
        reverse_distance, reverse_index = source_tree.query(target_points, k=1)

        paired_source = np.vstack((rotated_source, rotated_source[reverse_index]))
        paired_target = np.vstack((target_points[forward_index], target_points))
        source_mean = paired_source.mean(axis=0)
        target_mean = paired_target.mean(axis=0)
        source_zero = paired_source - source_mean
        target_zero = paired_target - target_mean
        denominator = float(np.sum(source_zero * source_zero))
        if denominator <= np.finfo(float).eps:
            raise RuntimeError("static registration correspondences are degenerate")
        new_scale = float(np.sum(source_zero * target_zero) / denominator)
        if new_scale <= np.finfo(float).eps:
            raise RuntimeError("static registration produced a non-positive scale")
        new_translation = target_mean - new_scale * source_mean
        cost = float((np.mean(forward_distance ** 2)
                      + np.mean(reverse_distance ** 2)) / 2.0)
        scale, translation = new_scale, new_translation
        if abs(previous_cost - cost) < 1e-10:
            break
        previous_cost = cost

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    transformed = scale * rotated_source + translation
    forward_distance = target_tree.query(transformed, k=1)[0]
    source_tree = cKDTree(transformed)
    reverse_distance = source_tree.query(target_points, k=1)[0]
    cost = float((np.mean(forward_distance ** 2)
                  + np.mean(reverse_distance ** 2)) / 2.0)
    return transform, cost


def register_static_with_rotation_policy(source, target_points, pose_initial,
                                         args, trimesh):
    """Register static geometry using the requested fixed-rotation policy."""
    policy = args.static_rotation_source
    candidate_costs = {}

    if policy in ("sam3d", "auto"):
        pose_transform, pose_cost = register_static(
            source, target_points, pose_initial, args, trimesh)
        candidate_costs["sam3d"] = float(pose_cost)
    if policy in ("identity", "auto"):
        identity_transform, identity_cost = register_static(
            source, target_points, np.eye(4), args, trimesh)
        candidate_costs["identity"] = float(identity_cost)

    if policy == "sam3d":
        return pose_transform, pose_cost, "sam3d", candidate_costs
    if policy == "identity":
        return identity_transform, identity_cost, "identity", candidate_costs

    pose_angle = rotation_angle_degrees(pose_initial)
    use_identity = (pose_angle >= 120.0
                    and identity_cost < 0.9 * pose_cost)
    if use_identity:
        return identity_transform, identity_cost, "identity", candidate_costs
    return pose_transform, pose_cost, "sam3d", candidate_costs


def _fit_similarity(source, target):
    """Least-squares proper-rotation similarity mapping source to target."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_zero = source - source_mean
    target_zero = target - target_mean
    covariance = target_zero.T @ source_zero
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    denominator = float(np.sum(source_zero * source_zero))
    if denominator <= np.finfo(float).eps:
        raise RuntimeError("pose-registration correspondences are degenerate")
    scale = float(np.sum(singular * np.diag(correction)) / denominator)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise RuntimeError("pose registration produced a non-positive scale")
    translation = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform


def _transform_points(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def _symmetric_cost(source, target, transform, cKDTree):
    transformed = _transform_points(source, transform)
    forward = cKDTree(target).query(transformed, k=1)[0]
    reverse = cKDTree(transformed).query(target, k=1)[0]
    return float((np.mean(forward ** 2) + np.mean(reverse ** 2)) / 2.0)


def _principal_axes(points):
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt.T
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    return axes


def _pose_initializations(source, target):
    """Generate all 24 proper PCA-axis mappings between two point clouds."""
    import itertools

    source_axes = _principal_axes(source)
    target_axes = _principal_axes(target)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_extent = np.linalg.norm(np.ptp(source, axis=0))
    target_extent = np.linalg.norm(np.ptp(target, axis=0))
    if source_extent <= np.finfo(float).eps or target_extent <= np.finfo(float).eps:
        raise RuntimeError("pose-registration mesh has degenerate extent")
    scale = target_extent / source_extent
    transforms = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            mapping = np.zeros((3, 3))
            mapping[range(3), permutation] = signs
            if np.linalg.det(mapping) < 0:
                continue
            rotation = target_axes @ mapping @ source_axes.T
            transform = np.eye(4)
            transform[:3, :3] = scale * rotation
            transform[:3, 3] = target_center - scale * rotation @ source_center
            transforms.append(transform)
    return transforms


def _refine_similarity_icp(source, target, initial, iterations, cKDTree):
    """Refine one pose candidate with bidirectional similarity ICP."""
    transform = initial.copy()
    target_tree = cKDTree(target)
    previous_cost = np.inf
    for _ in range(iterations):
        transformed = _transform_points(source, transform)
        _, forward_index = target_tree.query(transformed, k=1)
        source_tree = cKDTree(transformed)
        _, reverse_index = source_tree.query(target, k=1)
        paired_source = np.vstack((source, source[reverse_index]))
        paired_target = np.vstack((target[forward_index], target))
        transform = _fit_similarity(paired_source, paired_target)
        cost = _symmetric_cost(source, target, transform, cKDTree)
        if abs(previous_cost - cost) < 1e-10:
            break
        previous_cost = cost
    return transform, _symmetric_cost(source, target, transform, cKDTree)


def register_pose_to_sam3d(source, target, args, trimesh):
    """Globally initialize and similarity-ICP SegviGen to same-frame SAM3D."""
    from scipy.spatial import cKDTree

    source_points, _ = trimesh.sample.sample_surface(
        source, args.samples, seed=args.seed)
    target_points, _ = trimesh.sample.sample_surface(
        target, args.samples, seed=args.seed)
    initializations = _pose_initializations(source_points, target_points)
    ranked = sorted(
        initializations,
        key=lambda candidate: _symmetric_cost(
            source_points, target_points, candidate, cKDTree),
    )
    # PCA provides 24 global orientation hypotheses. Refining the best four
    # keeps runtime bounded while avoiding the local-minimum behavior of a
    # single identity-oriented ICP initialization.
    results = [
        _refine_similarity_icp(
            source_points, target_points, candidate, args.iterations, cKDTree)
        for candidate in ranked[:4]
    ]
    return min(results, key=lambda result: result[1])


def main():
    args = parse_args()
    if args.samples < 10 or args.iterations < 1:
        sys.exit("error: --samples must be at least 10 and --iterations positive")

    try:
        import trimesh
    except ImportError:
        sys.exit("error: trimesh is required; run this in the trellis2 environment")

    scene_dir = args.scene_dir.expanduser().resolve()
    tracking_path = scene_dir / "masks" / "tracking.json"
    try:
        if not scene_dir.is_dir():
            raise RuntimeError(f"scene directory does not exist: {scene_dir}")
        pose_paths = discover_poses(scene_dir / "sam3d_scaled" / "combined")
        if args.register_sam3d:
            static_labels = []
            moving_labels = []
            frames = discover_sam3d_frames(pose_paths)
            skipped_frames = []
        else:
            tracking = read_json(tracking_path)
            static_labels = labels_by_motion(
                tracking, args.static_labels, "static")
            moving_labels = labels_by_motion(
                tracking, args.moving_labels, "moving")
            frames, skipped_frames = discover_frames(
                scene_dir / "segvigen" / "combined",
                [*static_labels, *moving_labels],
            )
        camera_to_world = load_camera_to_world(scene_dir)
        missing_poses = [frame_id for frame_id, _ in frames
                         if frame_id not in pose_paths]
        if missing_poses:
            missing_text = ", ".join(f"{frame_id:06d}" for frame_id in missing_poses)
            raise RuntimeError(f"missing SAM3D pose.json for frame(s): {missing_text}")
        if camera_to_world:
            missing_cameras = [frame_id for frame_id, _ in frames
                               if frame_id not in camera_to_world]
            if missing_cameras:
                missing_text = ", ".join(
                    f"{frame_id:06d}" for frame_id in missing_cameras)
                raise RuntimeError(
                    f"DA3 camera archive lacks selected frame(s): {missing_text}"
                )
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    for frame_id, missing in skipped_frames:
        print(f"warning: skipping incomplete frame {frame_id:06d}; "
              f"missing pieces: {', '.join(missing)}")

    frames_by_id = {frame_id: source for frame_id, source in frames}
    reference_id = (frames[0][0] if args.reference_frame is None
                    else args.reference_frame)
    if reference_id not in frames_by_id:
        available = ", ".join(f"{frame_id:06d}" for frame_id in frames_by_id)
        sys.exit(f"error: reference frame {reference_id:06d} is not an "
                 f"available complete candidate; available frame(s): {available}")
    reference_source = frames_by_id[reference_id]

    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir
               else scene_dir / "registered_static")
    output_mesh = out_dir / "registered_meshes.glb"
    metadata_path = out_dir / "metadata.json"
    existing = [path for path in (output_mesh, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        sys.exit("error: output(s) already exist; pass --overwrite: "
                 + ", ".join(str(path) for path in existing))
    for path in existing:
        if not path.is_file():
            sys.exit(f"error: refusing to replace non-file output: {path}")
        path.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"registering {len(frames)} available frame(s); "
          f"reference={reference_id:06d}", flush=True)
    try:
        reference_pose = sam3d_pose_to_rdf_camera(pose_paths[reference_id])
        if reference_id in camera_to_world:
            reference_pose = camera_to_world[reference_id] @ reference_pose
        reference_static = (load_mesh(reference_source, trimesh)
                            if args.register_sam3d
                            else combine_labels(reference_source, static_labels, trimesh))
        reference_pose_correction = np.eye(4)
        reference_pose_cost = 0.0
        if not args.register_sam3d:
            reference_full = combine_labels(
                reference_source, [*static_labels, *moving_labels], trimesh)
            reference_sam3d = load_mesh(
                mesh_beside_pose(pose_paths[reference_id]), trimesh)
            reference_pose_correction, reference_pose_cost = register_pose_to_sam3d(
                reference_full, reference_sam3d, args, trimesh)
            reference_static.apply_transform(reference_pose_correction)
        reference_points, _ = trimesh.sample.sample_surface(
            reference_static, args.samples, seed=args.seed)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    result = trimesh.Scene()
    if not args.register_sam3d:
        result.add_geometry(reference_static.copy(),
                            geom_name=f"static_reference_{reference_id:06d}",
                            node_name=f"static_reference_{reference_id:06d}")
    records = []

    for frame_id, frame_source in frames:
        try:
            frame_static = (load_mesh(frame_source, trimesh)
                            if args.register_sam3d
                            else combine_labels(frame_source, static_labels, trimesh))
            pose_correction = np.eye(4)
            pose_cost = 0.0
            if not args.register_sam3d:
                if frame_id == reference_id:
                    pose_correction = reference_pose_correction
                    pose_cost = reference_pose_cost
                else:
                    frame_full = combine_labels(
                        frame_source, [*static_labels, *moving_labels], trimesh)
                    frame_sam3d = load_mesh(
                        mesh_beside_pose(pose_paths[frame_id]), trimesh)
                    pose_correction, pose_cost = register_pose_to_sam3d(
                        frame_full, frame_sam3d, args, trimesh)
                frame_static.apply_transform(pose_correction)
            if frame_id == reference_id:
                static_transform = np.eye(4)
                initial_transform = np.eye(4)
                cost = 0.0
                static_rotation_source_used = "reference"
                static_candidate_costs = {"reference": 0.0}
            else:
                frame_pose = sam3d_pose_to_rdf_camera(pose_paths[frame_id])
                if frame_id in camera_to_world:
                    frame_pose = camera_to_world[frame_id] @ frame_pose
                initial_transform = np.linalg.inv(reference_pose) @ frame_pose
                (static_transform, cost, static_rotation_source_used,
                 static_candidate_costs) = register_static_with_rotation_policy(
                    frame_static, reference_points, initial_transform,
                    args, trimesh)
            transform = (static_transform if args.register_sam3d
                         else static_transform @ pose_correction)
            if not np.isfinite(transform).all() or not np.isfinite(cost):
                raise RuntimeError(f"registration returned non-finite values for {frame_id}")

            moving_records = []
            sam3d_record = None
            if args.register_sam3d:
                frame_static.apply_transform(transform)
                sam3d_geometry = f"sam3d_scaled_raw_{frame_id:06d}"
                result.add_geometry(
                    frame_static,
                    geom_name=sam3d_geometry,
                    node_name=sam3d_geometry,
                )
                sam3d_record = {
                    "source": display_path(frame_source, scene_dir),
                    "geometry": sam3d_geometry,
                }
            else:
                for label in moving_labels:
                    source = (frame_source / "pieces"
                              / f"{safe_name(label)}.glb")
                    moving = load_mesh(source, trimesh)
                    moving.apply_transform(transform)
                    geometry_name = f"moving_{safe_name(label)}_{frame_id:06d}"
                    result.add_geometry(moving, geom_name=geometry_name,
                                        node_name=geometry_name)
                    moving_records.append({
                        "label": label,
                        "source": display_path(source, scene_dir),
                        "geometry": geometry_name,
                    })
        except (RuntimeError, ValueError) as exc:
            sys.exit(f"error processing frame {frame_id:06d}: {exc}")

        records.append({
            "frame": frame_id,
            "registration_cost": float(cost),
            "pose_registration_cost": float(pose_cost),
            "pose": display_path(pose_paths[frame_id], scene_dir),
            "pose_correction_to_same_frame_sam3d": np.asarray(
                pose_correction).tolist(),
            "initial_transform_from_pose": np.asarray(initial_transform).tolist(),
            "initial_rotation_from_pose_degrees": rotation_angle_degrees(
                initial_transform),
            "static_rotation_source_used": static_rotation_source_used,
            "static_registration_candidate_costs": static_candidate_costs,
            "static_transform_after_pose_correction": np.asarray(
                static_transform).tolist(),
            "transform_to_reference": np.asarray(transform).tolist(),
            "moving_meshes": moving_records,
            "sam3d_mesh": sam3d_record,
        })
        print(f"frame {frame_id:06d}: pose_cost={pose_cost:.8g}, "
              f"static_cost={cost:.8g}, "
              f"rotation_source={static_rotation_source_used}", flush=True)

    try:
        result.export(output_mesh)
    except Exception as exc:
        sys.exit(f"error exporting {output_mesh}: {exc}")

    metadata = {
        "stage": "06_register_static",
        "scene": scene_dir.name,
        "reference_frame": reference_id,
        "mesh_source": "sam3d_scaled" if args.register_sam3d else "segvigen",
        "sam3d_reference_root": "sam3d_scaled/combined",
        "static_labels": static_labels,
        "moving_labels": moving_labels,
        "registration_dof": "selected_fixed_rotation_with_uniform_scale_and_translation_refinement",
        "static_rotation_source_requested": args.static_rotation_source,
        "auto_rotation_policy": {
            "minimum_sam3d_angle_degrees": 120.0,
            "maximum_identity_to_sam3d_cost_ratio": 0.9,
        } if args.static_rotation_source == "auto" else None,
        "registration_stages": [
            "segvigen_full_to_same_frame_sam3d_scaled_similarity",
            "pose_corrected_static_to_reference_fixed_rotation_similarity",
        ] if not args.register_sam3d else ["sam3d_scaled_static_registration"],
        "pose_rotation_used": any(
            record["static_rotation_source_used"] == "sam3d"
            for record in records
        ),
        "pose_convention": "sam3d_glb_to_rdf_camera",
        "camera_to_world_used": bool(camera_to_world),
        "samples": args.samples,
        "static_geometry": (None if args.register_sam3d
                            else f"static_reference_{reference_id:06d}"),
        "output_mesh": display_path(output_mesh, scene_dir),
        "frames": records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"registered mesh: {output_mesh}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
