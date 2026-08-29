#!/usr/bin/env python
"""Replay a stage-40 bundle in Rerun.

Reads the TrackCraft3R products from ``trackcraft/`` and the shared stage-00
geometry from ``da3/``; no model inference is performed. Follows the same
visualization paradigm as
Any4D/scripts/render_exported.py:
    * Module-level helpers (_log_points, _log_camera_and_points, _log_scene_flow)
    * White-background Spatial3DView blueprint, no line grid, panels collapsed
    * stable_time timeline in seconds (frame_time_step per step)
    * View-indexed entity names: pred/pointcloud_view_{i}, pred/image_view_{i},
      pred/scene_flow_{i}
    * Per-frame Clear of the previous transient entities so clouds don't pile up
    * Scene flow as HSV-coloured arrows (hue=direction, sat/val=magnitude)

Also exports an mp4 (by default, next to the trackcraft bundle) of the same
per-label point tracks projected directly onto the original RGB frames, with
a fading trail behind each track's current position — a quick way to eyeball
tracking quality without opening the Rerun viewer.

Run inside any env that has `numpy`, `pillow`, `opencv-python`,
`matplotlib`, and `rerun-sdk`.

Examples:
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01 --no-scene-flow
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01 --no-video
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01 \\
        --video-out /tmp/tracks.mp4 --video-trail-frames 15
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from matplotlib.colors import hsv_to_rgb

# ---------------------------------------------------------------------------
# Shared rendering helpers (adapted from Any4D/scripts/render_exported.py)
# ---------------------------------------------------------------------------


def _log_points(image, pts3d, pts_name, mask=None):
    filtered_pts = pts3d[mask] if mask is not None else pts3d
    filtered_col = image[mask] if mask is not None else image
    rr.log(
        pts_name,
        rr.Points3D(
            positions=filtered_pts.reshape(-1, 3),
            colors=filtered_col.reshape(-1, 3),
        ),
    )


def _log_camera_and_image(image, cam_pose, cam_intrinsics, base_name):
    """Log a Transform3D + Pinhole + RGB image for one view."""
    h, w = image.shape[:2]
    rr.log(
        base_name,
        rr.Transform3D(
            translation=cam_pose[:3, 3],
            mat3x3=cam_pose[:3, :3],
            from_parent=False,
        ),
    )
    rr.log(
        f"{base_name}/pinhole",
        rr.Pinhole(
            image_from_camera=cam_intrinsics,
            height=h,
            width=w,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{base_name}/pinhole/rgb", rr.Image(image))


def _log_camera_and_points(
    image, cam_pose, cam_intrinsics, pts3d, mask, base_name, pts_name
):
    _log_camera_and_image(image, cam_pose, cam_intrinsics, base_name)
    _log_points(image, pts3d, pts_name, mask=mask)


def _log_depth_image(depth, intrinsics, base_name):
    """Attach a DA3 depth map under a per-frame pinhole at full resolution."""
    h, w = depth.shape
    rr.log(
        f"{base_name}/da3/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=h,
            width=w,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{base_name}/da3/pinhole/depth", rr.DepthImage(depth))


def _track_colors(track_positions):
    """Per-track RGB uint8 colors, rainbow-mapped by each track's initial X.

    Args:
        track_positions: (n_tracks, T, 3) float — per-track position at each
                         frame in temporal order.

    Returns: (n_tracks, 3) uint8 RGB.
    """
    from matplotlib import cm

    init_x = track_positions[:, 0, 0]
    x_min, x_max = float(init_x.min()), float(init_x.max())
    if x_max - x_min < 1e-8:
        x_max = x_min + 1e-8
    normalized = (init_x - x_min) / (x_max - x_min)
    try:
        cmap = cm.colormaps["rainbow"]
    except Exception:
        cmap = cm.get_cmap("rainbow")
    return (cmap(normalized)[:, :3] * 255).astype(np.uint8)


def _log_point_tracks(track_positions, base_name):
    """Log per-track 3D polylines, colored by initial X (rainbow).

    Args:
        track_positions: (n_tracks, T, 3) float — per-track position at each
                         frame in temporal order.
        base_name: rerun entity path, e.g. "pred/point_tracks/mug".
    """
    if track_positions.shape[0] == 0 or track_positions.shape[1] < 2:
        return

    colors = _track_colors(track_positions)

    rr.log(
        base_name,
        rr.LineStrips3D(
            strips=[track_positions[i] for i in range(track_positions.shape[0])],
            colors=colors,
        ),
        static=True,
    )


def _log_scene_flow(pts3d, scene_flow_vecs, base_name, mask=None, max_arrows=500):
    """Log arrows colored by HSV (hue = XZ direction, sat/val = magnitude)."""
    pts = pts3d[mask] if mask is not None else pts3d
    vecs = scene_flow_vecs[mask] if mask is not None else scene_flow_vecs
    pts = pts.reshape(-1, 3)
    vecs = vecs.reshape(-1, 3)
    if len(pts) == 0:
        print(f"warning: no valid scene flow points for {base_name}")
        return

    if pts.shape[0] > max_arrows:
        magnitudes_all = np.linalg.norm(vecs, axis=1)
        if magnitudes_all.max() > 1e-6:
            probs = 0.2 + 0.8 * (magnitudes_all / (magnitudes_all.max() + 1e-6))
            probs /= probs.sum()
            indices = np.random.choice(
                len(pts), size=max_arrows, replace=False, p=probs
            )
        else:
            indices = np.random.permutation(len(pts))[:max_arrows]
        pts = pts[indices]
        vecs = vecs[indices]

    magnitudes = np.linalg.norm(vecs, axis=1)
    mag_min, mag_max = magnitudes.min(), magnitudes.max()
    if mag_max == mag_min:
        mag_max = mag_min + 1e-6

    direction = vecs / (magnitudes[:, np.newaxis] + 1e-8)
    hue = (np.arctan2(direction[:, 2], direction[:, 0]) + np.pi) / (2 * np.pi)
    norm_mag = np.clip((magnitudes - mag_min) / (mag_max - mag_min + 1e-8), 0, 1)
    hsv = np.stack([hue, 0.3 + 0.7 * norm_mag, 0.5 + 0.5 * norm_mag], axis=1)
    rgb = hsv_to_rgb(hsv)
    alpha = np.full((len(rgb), 1), 80.0 / 255.0)
    colors = np.concatenate([rgb, alpha], axis=1)

    rr.log(
        f"{base_name}/scene_flow",
        rr.Arrows3D(origins=pts, vectors=vecs, colors=colors),
    )


def project_points_to_frame(points_world, cam_pose_c2w, K):
    """Project (N, 3) world points into one camera's image plane.

    Args:
        points_world: (N, 3) float world-frame points.
        cam_pose_c2w: (4, 4) camera-to-world transform (same convention as
                      ``build_cam_pose`` / ``_log_camera_and_image``).
        K: (3, 3) pinhole intrinsics, full-resolution pixel units, RDF
           camera axes (matches the ``camera_xyz=rr.ViewCoordinates.RDF``
           used when logging the pinhole elsewhere in this file).

    Returns: (uv, valid) — (N, 2) float pixel coords and (N,) bool, True
    where the point is in front of the camera (z > 0).
    """
    R = cam_pose_c2w[:3, :3]
    t = cam_pose_c2w[:3, 3]
    pts_cam = (points_world - t) @ R  # R^T @ (p - t), row-vector form
    valid = pts_cam[:, 2] > 1e-6
    uvw = pts_cam @ K.T
    z_safe = np.where(np.abs(uvw[:, 2:3]) < 1e-8, 1e-8, uvw[:, 2:3])
    uv = uvw[:, :2] / z_safe
    return uv, valid


def render_point_track_video(
    bundle,
    track_payload,
    temporal_order,
    cam_poses,
    frame_intrinsics,
    out_path,
    fps,
    trail_frames,
    point_radius,
    line_thickness,
    trail_alpha,
):
    """Render the per-label point tracks over the original RGB frames.

    For each frame in temporal order, projects every track's recent history
    (``trail_frames`` back) into that frame's camera and draws a
    semi-transparent trail plus a solid dot at the current position, directly
    on top of the full-resolution ``frames/<frame>.jpg`` image. Writes an mp4
    to ``out_path``.
    """
    if not track_payload:
        print("warning: no point tracks available, skipping video export")
        return

    frames_dir = bundle["frames_dir"]
    label_colors = {label: _track_colors(tracks) for label, tracks in track_payload.items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    for t, frame_idx in enumerate(temporal_order):
        rgb_path = frames_dir / f"{frame_idx:06d}.jpg"
        frame_bgr = cv2.imread(str(rgb_path)) if rgb_path.is_file() else None
        if frame_bgr is None:
            print(f"warning: missing/unreadable frame {rgb_path}, skipping in video")
            continue
        h, w = frame_bgr.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        cam_pose = cam_poses.get(frame_idx)
        K = frame_intrinsics.get(frame_idx)
        if cam_pose is None or K is None:
            writer.write(frame_bgr)
            continue

        canvas = frame_bgr
        overlay = canvas.copy()
        t0 = max(0, t - trail_frames)

        for label, tracks in track_payload.items():
            colors = label_colors[label]
            window = tracks[:, t0 : t + 1, :]  # (n_tracks, W, 3)
            n_tracks, W = window.shape[0], window.shape[1]
            uv, valid = project_points_to_frame(window.reshape(-1, 3), cam_pose, K)
            uv = uv.reshape(n_tracks, W, 2)
            in_bounds = (
                (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
            )
            valid = valid.reshape(n_tracks, W) & in_bounds

            for k in range(n_tracks):
                color_bgr = (int(colors[k, 2]), int(colors[k, 1]), int(colors[k, 0]))
                pts_uv = uv[k]
                pts_valid = valid[k]
                for tt in range(1, W):
                    if pts_valid[tt - 1] and pts_valid[tt]:
                        cv2.line(
                            overlay,
                            (int(pts_uv[tt - 1, 0]), int(pts_uv[tt - 1, 1])),
                            (int(pts_uv[tt, 0]), int(pts_uv[tt, 1])),
                            color_bgr,
                            line_thickness,
                            cv2.LINE_AA,
                        )

        canvas = cv2.addWeighted(overlay, trail_alpha, canvas, 1.0 - trail_alpha, 0)

        for label, tracks in track_payload.items():
            colors = label_colors[label]
            cur = tracks[:, t, :]
            uv, valid = project_points_to_frame(cur, cam_pose, K)
            in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            valid = valid & in_bounds
            for k in np.nonzero(valid)[0]:
                color_bgr = (int(colors[k, 2]), int(colors[k, 1]), int(colors[k, 0]))
                cv2.circle(
                    canvas,
                    (int(uv[k, 0]), int(uv[k, 1])),
                    point_radius,
                    color_bgr,
                    -1,
                    cv2.LINE_AA,
                )

        writer.write(canvas)

    if writer is not None:
        writer.release()
        print(f"wrote point-track video to {out_path}")
    else:
        print("warning: no frames written to video (no readable RGB frames)")


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


def quaternion_xyzw_to_R(q):
    """Convert an XYZW quaternion to a 3x3 rotation matrix.

    Stage 00 stores camera-to-world quaternions in XYZW order.
    """
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def build_cam_pose(quat_xyzw, trans):
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion_xyzw_to_R(quat_xyzw)
    T[:3, 3] = trans
    return T


def load_bundle(scene_dir: Path):
    """Load the stage-40 TrackCraft bundle and its stage-00 DA3 geometry."""
    trackcraft_root = scene_dir / "trackcraft"
    da3_root = scene_dir / "da3"
    frames_dir = scene_dir / "frames"
    config_path = trackcraft_root / "config.json"

    for need in (
        config_path,
        trackcraft_root / "pointmap_ref.npy",
        da3_root / "cameras.npz",
        da3_root / "intrinsics.npz",
        da3_root / "depth",
        frames_dir,
    ):
        if not need.exists():
            sys.exit(f"error: missing {need} (did stage 40 finish?)")

    with open(config_path) as f:
        config = json.load(f)

    if "tracked_frames" not in config or "ref_frame" not in config:
        sys.exit(f"error: {config_path} lacks tracked_frames/ref_frame")
    frame_indices = np.asarray(config["tracked_frames"], dtype=np.int64)
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        sys.exit(f"error: invalid tracked_frames in {config_path}")
    if int(config["ref_frame"]) not in frame_indices:
        sys.exit("error: ref_frame is not present in tracked_frames")

    cameras = np.load(da3_root / "cameras.npz")
    intrinsics = np.load(da3_root / "intrinsics.npz")
    pts3d_ref = np.load(trackcraft_root / "pointmap_ref.npy")
    camera_rows = {int(fi): i for i, fi in enumerate(cameras["frame_indices"])}
    intr_rows = {int(fi): i for i, fi in enumerate(intrinsics["frame_indices"])}
    missing = [
        int(fi)
        for fi in frame_indices
        if int(fi) not in camera_rows or int(fi) not in intr_rows
    ]
    if missing:
        sys.exit(
            f"error: tracked frames missing from DA3 cameras/intrinsics: {missing}"
        )

    label_dirs = [
        d
        for d in trackcraft_root.iterdir()
        if d.is_dir() and (d / "pts3d_ref.npy").is_file()
    ]

    bundle = dict(
        trackcraft_root=trackcraft_root,
        da3_root=da3_root,
        frames_dir=frames_dir,
        config=config,
        cameras=cameras,
        intrinsics=intrinsics,
        frame_indices=frame_indices,
        camera_rows=camera_rows,
        intr_rows=intr_rows,
        pts3d_ref=pts3d_ref,
        label_dirs=label_dirs,
    )
    return bundle


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(
    scene_dir,
    show_scene_flow,
    show_trajectories,
    max_tracks,
    frame_time_step,
    max_arrows,
    labels_filter,
    show_depth=True,
    export_video=True,
    video_out=None,
    video_fps=None,
    video_trail_frames=8,
    video_point_radius=4,
    video_line_thickness=2,
    video_trail_alpha=0.6,
):
    bundle = load_bundle(scene_dir)
    config = bundle["config"]
    cameras = bundle["cameras"]
    intrinsics = bundle["intrinsics"]
    pts3d_ref = bundle["pts3d_ref"]
    frame_indices = bundle["frame_indices"]
    if "cam_quats_xyzw" not in cameras.files:
        sys.exit("error: cameras.npz has no cam_quats_xyzw key")
    cam_quats = cameras["cam_quats_xyzw"]
    cam_trans = cameras["cam_trans"]
    H_model, W_model = pts3d_ref.shape[:2]
    ref_frame = int(config["ref_frame"])

    # Filter labels and pre-load shared per-label arrays
    label_dirs = bundle["label_dirs"]
    if labels_filter:
        label_dirs = [d for d in label_dirs if d.name in labels_filter]
    print(f"labels to replay: {[d.name for d in label_dirs]}")

    label_data = {}
    for d in label_dirs:
        pts_ref = np.load(d / "pts3d_ref.npy")
        if pts_ref.ndim != 2 or pts_ref.shape[1] != 3:
            sys.exit(f"error: {d / 'pts3d_ref.npy'} must have shape (N, 3)")
        label_data[d.name] = {
            "pts3d_ref": pts_ref,
            "flow_dir": d / "scene_flow",
        }

    # Pre-build per-label point trajectories in TEMPORAL order so the polylines
    # don't zig-zag when image_indices is non-monotonic (ref first, then targets).
    # Each track is the path of a single masked ref-pixel through time:
    #   position[ref_frame]      = pts3d_ref[i]
    #   position[target_frame f] = pts3d_ref[i] + scene_flow[f][i]
    temporal_order = sorted(set(int(f) for f in frame_indices.tolist()))
    track_payload = {}  # label -> (track_positions, sampled_indices)
    if show_trajectories:
        for label, ld in label_data.items():
            pts_ref = ld["pts3d_ref"]
            N = pts_ref.shape[0]
            if N == 0:
                continue
            rng = np.random.default_rng(0)
            n_tracks = int(min(max_tracks, N))
            sampled = (
                rng.choice(N, size=n_tracks, replace=False)
                if N > n_tracks
                else np.arange(N)
            )

            T = len(temporal_order)
            tracks = np.zeros((n_tracks, T, 3), dtype=np.float32)
            for t, fi in enumerate(temporal_order):
                if fi == ref_frame:
                    tracks[:, t, :] = pts_ref[sampled]
                else:
                    flow_path = ld["flow_dir"] / f"{fi:06d}.npy"
                    if flow_path.is_file():
                        flow = np.load(flow_path)
                        if flow.shape != pts_ref.shape:
                            sys.exit(
                                f"error: {flow_path} has shape {flow.shape}, expected {pts_ref.shape}"
                            )
                        tracks[:, t, :] = pts_ref[sampled] + flow[sampled]
                    else:
                        # No flow file for this frame; fall back to ref pos
                        tracks[:, t, :] = pts_ref[sampled]
            track_payload[label] = tracks
            print(f"  {label}: built {n_tracks} trajectories across {T} frames")

    # Pre-load ref RGB at model resolution for the ref pointcloud coloring
    ref_rgb_path = bundle["frames_dir"] / f"{ref_frame:06d}.jpg"
    if not ref_rgb_path.is_file():
        sys.exit(f"error: missing ref RGB {ref_rgb_path}")
    ref_rgb_full = np.array(Image.open(ref_rgb_path).convert("RGB"))
    ref_rgb_model = np.array(
        Image.fromarray(ref_rgb_full).resize((W_model, H_model), Image.BILINEAR)
    )

    # Stage 40 marks invalid DA3 unprojections as zero.
    ref_pt_mask = np.isfinite(pts3d_ref).all(axis=-1) & np.any(pts3d_ref != 0, axis=-1)

    # Build view_idx -> camera pose lookup (matches order of frame_indices)
    cam_poses = {
        int(fi): build_cam_pose(
            cam_quats[bundle["camera_rows"][int(fi)]],
            cam_trans[bundle["camera_rows"][int(fi)]],
        )
        for fi in frame_indices
    }
    frame_intrinsics = {
        int(fi): intrinsics["intrinsics"][bundle["intr_rows"][int(fi)]]
        for fi in frame_indices
    }

    # --- Export the point tracks over the original RGB video ---
    if export_video:
        video_out = video_out or (bundle["trackcraft_root"] / "point_tracks_video.mp4")
        render_point_track_video(
            bundle,
            track_payload,
            temporal_order,
            cam_poses,
            frame_intrinsics,
            out_path=video_out,
            fps=video_fps if video_fps is not None else 1.0 / frame_time_step,
            trail_frames=video_trail_frames,
            point_radius=video_point_radius,
            line_thickness=video_line_thickness,
            trail_alpha=video_trail_alpha,
        )

    # ---- Rerun init + blueprint ----
    rr.init("articulate4d_replay")
    rr.spawn(port=9999)
    rr.log("pred", rr.ViewCoordinates.RDF, static=True)
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="pred",
            name="3D Scene",
            background=[255, 255, 255],
            line_grid=rrb.archetypes.LineGrid3D(visible=False),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    # Static-log all trajectories once (visible across the whole timeline).
    for label, tracks in track_payload.items():
        _log_point_tracks(tracks, f"pred/point_tracks/{label}")
        print(
            f"  logged trajectories for '{label}' "
            f"({tracks.shape[0]} tracks × {tracks.shape[1]} frames)"
        )

    print(f"Rendering {len(frame_indices)} frames from {scene_dir} ...")

    for v_idx, frame_idx in enumerate(frame_indices.tolist()):
        rr.set_time_seconds("stable_time", frame_time_step * v_idx)

        # Clear previous frame's transient entities
        if v_idx > 0:
            rr.log(f"pred/pointcloud_view_{v_idx - 1}", rr.Clear(recursive=True))
            rr.log(f"pred/image_view_{v_idx - 1}", rr.Clear(recursive=True))
            rr.log(f"pred/scene_flow_{v_idx - 1}", rr.Clear(recursive=True))
            # DA3 depth lives under image_view_, so the Clear above covers it.

        # --- Reference view (logged each step so it stays visible) ---
        ref_base = "pred/image_view_0"
        ref_pts_name = "pred/pointcloud_view_0"
        _log_camera_and_points(
            ref_rgb_model,
            cam_poses[ref_frame],
            frame_intrinsics[ref_frame],
            pts3d_ref,
            ref_pt_mask,
            ref_base,
            ref_pts_name,
        )

        # --- Current view ---
        if frame_idx != ref_frame:
            cur_base = f"pred/image_view_{v_idx}"
            rgb_path = bundle["frames_dir"] / f"{frame_idx:06d}.jpg"
            if rgb_path.is_file():
                rgb = np.array(Image.open(rgb_path).convert("RGB"))
                _log_camera_and_image(
                    rgb,
                    cam_poses[frame_idx],
                    frame_intrinsics[frame_idx],
                    cur_base,
                )

        # --- DA3 depth (full resolution, under the current view's transform) ---
        cur_base = f"pred/image_view_{v_idx}"
        depth_path = bundle["da3_root"] / "depth" / f"{frame_idx:06d}.npy"
        if show_depth and depth_path.is_file():
            depth = np.load(depth_path)
            _log_depth_image(depth, frame_intrinsics[frame_idx], cur_base)

        # --- Per-label scene flow (skip on ref frame) ---
        if show_scene_flow and frame_idx != ref_frame:
            sf_base = f"pred/scene_flow_{v_idx}"
            for label, ld in label_data.items():
                flow_path = ld["flow_dir"] / f"{frame_idx:06d}.npy"
                if not flow_path.is_file():
                    continue
                flow = np.load(flow_path)
                if flow.shape != ld["pts3d_ref"].shape:
                    sys.exit(
                        f"error: {flow_path} has shape {flow.shape}, expected {ld['pts3d_ref'].shape}"
                    )
                _log_scene_flow(
                    ld["pts3d_ref"],
                    flow,
                    f"{sf_base}/{label}",
                    mask=None,
                    max_arrows=max_arrows,
                )

    print("done. Drag the 'stable_time' timeline in the Rerun viewer.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument(
        "--scene-dir", type=Path, required=True, help="Scene folder data/<scene_id>/"
    )
    p.add_argument(
        "--no-scene-flow", action="store_true", help="Skip rendering scene-flow arrows"
    )
    p.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Skip rendering 3D point trajectories (line strips per masked point)",
    )
    p.add_argument(
        "--max-tracks",
        type=int,
        default=200,
        help="Number of trajectories per label to draw (default 200)",
    )
    p.add_argument(
        "--frame-time-step",
        type=float,
        default=0.2,
        help="Seconds per frame on the stable_time timeline (default 0.2)",
    )
    p.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Only replay these labels (default: all in trackcraft/)",
    )
    p.add_argument(
        "--max-arrows",
        type=int,
        default=500,
        help="Subsample arrows per label per frame (default 500)",
    )
    p.add_argument(
        "--no-depth", action="store_true", help="Skip rendering the DA3 depth map"
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip exporting the point-track video over the original RGB frames",
    )
    p.add_argument(
        "--video-out",
        type=Path,
        default=None,
        help="Output mp4 path (default: <trackcraft_root>/point_tracks_video.mp4)",
    )
    p.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Video framerate (default: 1 / --frame-time-step)",
    )
    p.add_argument(
        "--video-trail-frames",
        type=int,
        default=8,
        help="Trailing history length, in frames, drawn behind each track (default 8)",
    )
    p.add_argument(
        "--video-point-radius",
        type=int,
        default=4,
        help="Radius in pixels of each track's current-position dot (default 4)",
    )
    p.add_argument(
        "--video-line-thickness",
        type=int,
        default=2,
        help="Trail line thickness in pixels (default 2)",
    )
    p.add_argument(
        "--video-trail-alpha",
        type=float,
        default=0.6,
        help="Opacity of the trailing lines, 0-1 (default 0.6)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    replay(
        scene_dir=args.scene_dir.resolve(),
        show_scene_flow=not args.no_scene_flow,
        show_trajectories=not args.no_trajectories,
        max_tracks=args.max_tracks,
        frame_time_step=args.frame_time_step,
        max_arrows=args.max_arrows,
        export_video=not args.no_video,
        video_out=args.video_out,
        video_fps=args.video_fps,
        video_trail_frames=args.video_trail_frames,
        video_point_radius=args.video_point_radius,
        video_line_thickness=args.video_line_thickness,
        video_trail_alpha=args.video_trail_alpha,
        labels_filter=args.labels,
        show_depth=not args.no_depth,
    )


if __name__ == "__main__":
    main()
