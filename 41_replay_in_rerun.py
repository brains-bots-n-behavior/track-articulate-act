#!/usr/bin/env python
"""Replay a stage-40 bundle in Rerun.

Reads only files under data/<scene>/ (no Any4D model, no MoGe) and replays
them in Rerun. Follows the same visualization paradigm as
Any4D/scripts/render_exported.py:
    * Module-level helpers (_log_points, _log_camera_and_points, _log_scene_flow)
    * White-background Spatial3DView blueprint, no line grid, panels collapsed
    * stable_time timeline in seconds (frame_time_step per step)
    * View-indexed entity names: pred/pointcloud_view_{i}, pred/image_view_{i},
      pred/scene_flow_{i}
    * Per-frame Clear of the previous transient entities so clouds don't pile up
    * Scene flow as HSV-coloured arrows (hue=direction, sat/val=magnitude)

Run inside any env that has `numpy`, `pillow`, `opencv-python`,
`matplotlib`, and `rerun-sdk`.

Examples:
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01
    python 41_replay_in_rerun.py --scene-dir data/kitchen_pour_01 --no-scene-flow
"""

import argparse
import json
import sys
from pathlib import Path

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
    """Attach a MoGe depth map under a per-frame pinhole at full resolution."""
    h, w = depth.shape
    rr.log(
        f"{base_name}/moge/pinhole",
        rr.Pinhole(
            image_from_camera=intrinsics,
            height=h,
            width=w,
            camera_xyz=rr.ViewCoordinates.RDF,
        ),
    )
    rr.log(f"{base_name}/moge/pinhole/depth", rr.DepthImage(depth))


def _log_point_tracks(track_positions, base_name):
    """Log per-track 3D polylines, colored by initial X (rainbow).

    Args:
        track_positions: (n_tracks, T, 3) float — per-track position at each
                         frame in temporal order.
        base_name: rerun entity path, e.g. "pred/point_tracks/mug".
    """
    from matplotlib import cm

    if track_positions.shape[0] == 0 or track_positions.shape[1] < 2:
        return

    init_x = track_positions[:, 0, 0]
    x_min, x_max = float(init_x.min()), float(init_x.max())
    if x_max - x_min < 1e-8:
        x_max = x_min + 1e-8
    normalized = (init_x - x_min) / (x_max - x_min)
    try:
        cmap = cm.colormaps["rainbow"]
    except Exception:
        cmap = cm.get_cmap("rainbow")
    colors = (cmap(normalized)[:, :3] * 255).astype(np.uint8)

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


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------


def quaternion_xyzw_to_R(q):
    """Convert an XYZW quaternion to a 3x3 rotation matrix.

    Any4D stores quaternions in XYZW order (see
    Any4D/any4d/utils/geometry.py:601). Do NOT call this with a wxyz tuple.
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
    """Read every file produced by stage 40 (lazily for per-frame data)."""
    any4d_root = scene_dir / "any4d"
    frames_dir = scene_dir / "frames"
    config_path = any4d_root / "config.json"

    for need in (
        config_path,
        any4d_root / "cameras.npz",
        any4d_root / "frame_indices.npy",
        any4d_root / "pointmap_ref.npy",
    ):
        if not need.exists():
            sys.exit(f"error: missing {need} (did stage 40 finish?)")

    with open(config_path) as f:
        config = json.load(f)

    cameras = np.load(any4d_root / "cameras.npz")
    pts3d_ref = np.load(any4d_root / "pointmap_ref.npy")  # (H, W, 3)

    moge_root = any4d_root / "moge"
    if moge_root.is_dir():
        moge_intr = np.load(moge_root / "intrinsics.npz")
        moge_K_map = {
            int(fi): K
            for fi, K in zip(moge_intr["frame_indices"], moge_intr["intrinsics"])
        }
    else:
        moge_root = None
        moge_K_map = {}

    label_dirs = [
        d
        for d in any4d_root.iterdir()
        if d.is_dir() and d.name != "moge" and (d / "pts3d_ref.npy").is_file()
    ]

    bundle = dict(
        any4d_root=any4d_root,
        frames_dir=frames_dir,
        moge_root=moge_root,
        config=config,
        cameras=cameras,
        pts3d_ref=pts3d_ref,
        moge_K_map=moge_K_map,
        label_dirs=label_dirs,
    )
    return bundle


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay(scene_dir, show_scene_flow, show_trajectories, max_tracks,
           frame_time_step, max_arrows, labels_filter, show_depth=True):
    bundle = load_bundle(scene_dir)
    config = bundle["config"]
    cameras = bundle["cameras"]
    pts3d_ref = bundle["pts3d_ref"]
    frame_indices = cameras["frame_indices"]
    # Any4D stores quaternions as XYZW. Older bundles used the misleading key
    # name `cam_quats_wxyz` but stored the same XYZW bytes; accept both.
    if "cam_quats_xyzw" in cameras.files:
        cam_quats = cameras["cam_quats_xyzw"]
    elif "cam_quats_wxyz" in cameras.files:
        print("note: cameras.npz uses legacy key 'cam_quats_wxyz' but bytes are "
              "actually XYZW (Any4D convention). Reading as XYZW.")
        cam_quats = cameras["cam_quats_wxyz"]
    else:
        sys.exit("error: cameras.npz has no cam_quats_xyzw key")
    cam_trans = cameras["cam_trans"]
    intr_model = cameras["intrinsics"]
    H_model, W_model = pts3d_ref.shape[:2]
    ref_frame = config["ref_frame"]

    # Filter labels and pre-load shared per-label arrays
    label_dirs = bundle["label_dirs"]
    if labels_filter:
        label_dirs = [d for d in label_dirs if d.name in labels_filter]
    print(f"labels to replay: {[d.name for d in label_dirs]}")

    label_data = {}
    for d in label_dirs:
        label_data[d.name] = {
            "pts3d_ref": np.load(d / "pts3d_ref.npy"),  # (N, 3)
            "flow_dir": d / "scene_flow",
        }

    # Pre-build per-label point trajectories in TEMPORAL order so the polylines
    # don't zig-zag when image_indices is non-monotonic (ref first, then targets).
    # Each track is the path of a single masked ref-pixel through time:
    #   position[ref_frame]      = pts3d_ref[i]
    #   position[target_frame f] = pts3d_ref[i] + scene_flow[f][i]
    track_payload = {}  # label -> (track_positions, sampled_indices)
    if show_trajectories:
        temporal_order = sorted(set(int(f) for f in frame_indices.tolist()))
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

    # MoGe mask for ref frame, resized to model resolution (filter the ref cloud)
    ref_pt_mask = None
    if bundle["moge_root"] is not None:
        mp = bundle["moge_root"] / "mask" / f"{ref_frame:06d}.png"
        if mp.is_file():
            m_full = np.array(Image.open(mp))
            if m_full.ndim == 3:
                m_full = m_full[..., -1]
            ref_pt_mask = (
                np.array(
                    Image.fromarray((m_full > 0).astype(np.uint8) * 255).resize(
                        (W_model, H_model), Image.NEAREST
                    )
                )
                > 0
            )

    # Build view_idx -> camera pose lookup (matches order of frame_indices)
    cam_poses = {
        int(fi): build_cam_pose(q, t)
        for fi, q, t in zip(frame_indices.tolist(), cam_quats, cam_trans)
    }

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
        print(f"  logged trajectories for '{label}' "
              f"({tracks.shape[0]} tracks × {tracks.shape[1]} frames)")

    print(f"Rendering {len(frame_indices)} frames from {scene_dir} ...")

    for v_idx, frame_idx in enumerate(frame_indices.tolist()):
        rr.set_time_seconds("stable_time", frame_time_step * v_idx)

        # Clear previous frame's transient entities
        if v_idx > 0:
            rr.log(f"pred/pointcloud_view_{v_idx - 1}", rr.Clear(recursive=True))
            rr.log(f"pred/image_view_{v_idx - 1}", rr.Clear(recursive=True))
            rr.log(f"pred/scene_flow_{v_idx - 1}", rr.Clear(recursive=True))
            # MoGe depth/mask live under image_view_, so the Clear above covers them.

        # --- Reference view (logged each step so it stays visible) ---
        ref_base = "pred/image_view_0"
        ref_pts_name = "pred/pointcloud_view_0"
        _log_camera_and_points(
            ref_rgb_model,
            cam_poses[ref_frame],
            intr_model[list(frame_indices).index(ref_frame)],
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
                    intr_model[v_idx],
                    cur_base,
                )

        # --- MoGe depth + mask (full resolution, under the current view's transform) ---
        cur_base = f"pred/image_view_{v_idx}"
        if bundle["moge_root"] is not None and frame_idx in bundle["moge_K_map"]:
            depth_path = bundle["moge_root"] / "depth" / f"{frame_idx:06d}.npy"
            mask_path = bundle["moge_root"] / "mask" / f"{frame_idx:06d}.png"
            if show_depth and depth_path.is_file():
                depth = np.load(depth_path)
                _log_depth_image(depth, bundle["moge_K_map"][frame_idx], cur_base)
            if mask_path.is_file():
                m = np.array(Image.open(mask_path))
                rr.log(
                    f"{cur_base}/moge/mask",
                    rr.SegmentationImage((m > 0).astype(np.uint8)),
                )

        # --- Per-label scene flow (skip on ref frame) ---
        if show_scene_flow and frame_idx != ref_frame:
            sf_base = f"pred/scene_flow_{v_idx}"
            for label, ld in label_data.items():
                flow_path = ld["flow_dir"] / f"{frame_idx:06d}.npy"
                if not flow_path.is_file():
                    continue
                flow = np.load(flow_path)
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
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument(
        "--no-scene-flow", action="store_true", help="Skip rendering scene-flow arrows"
    )
    p.add_argument(
        "--no-trajectories", action="store_true",
        help="Skip rendering 3D point trajectories (line strips per masked point)"
    )
    p.add_argument(
        "--max-tracks", type=int, default=200,
        help="Number of trajectories per label to draw (default 200)"
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
        help="Only replay these labels (default: all in any4d/)",
    )
    p.add_argument(
        "--max-arrows",
        type=int,
        default=500,
        help="Subsample arrows per label per frame (default 500)",
    )
    p.add_argument(
        "--no-depth", action="store_true",
        help="Skip rendering the MoGe depth map (mask overlay is unaffected)"
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
        labels_filter=args.labels,
        show_depth=not args.no_depth,
    )


if __name__ == "__main__":
    main()
