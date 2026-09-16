#!/usr/bin/env python
"""Stage 08: TrackCraft3R point tracking -> per-label scene flow.

Runs TrackCraft3R once on a fixed window of frames and converts the model's
dense frame-0-anchored 3D track field into the per-label schema consumed by
point-track joint estimation and legacy replay utilities.

Depth + camera/world come from stage 00 (`00_da3_depth_cameras.py`, DA3): this
stage reads `da3/depth`, `da3/intrinsics.npz`, and `da3/cameras.npz`, then
writes its own products under `trackcraft/`.

Adapted from TrackCraft3r/scripts/build_user_npz.py (assemble RGB + depth +
w2c camera + intrinsics, frame-0-normalize the extrinsics) and
inference_user_video.py (run the WanSceneFlowPredictor, read its dense cache),
but wired to the Articulate4D scene layout.

Reads (produced by stages 00 + 02):
    data/<scene>/frames/*.jpg
    data/<scene>/masks/tracking.json
    data/<scene>/masks/<label>/<ref>.png
    data/<scene>/da3/depth/<frame>.npy               # z-depth, full res (stage 00)
    data/<scene>/da3/intrinsics.npz                  # per-frame K, full res (stage 00)
    data/<scene>/da3/cameras.npz                     # cam->world quats+trans (stage 00)

Writes:
    data/<scene>/trackcraft/
        config.json                       # updated: ref_frame, tracker, tracked_frames
        pointmap_ref.npy                  # recomputed dense ref pointmap (world coords)
        point_tracks_video.mp4            # tracked 2D points/trails on source RGB
        <label>/
            ref_mask.png                  # mask at TRACKER resolution
            pixel_ij.npy                  # (N, 2) int16, ref pixel (row, col)
            pts3d_ref.npy                 # (N, 3) float32, ref 3D position (WORLD)
            scene_flow/<frame>.npy        # (N, 3) float32, ref->target flow (WORLD)

Frame selection (TrackCraft3R runs on a fixed-length window):
    tracked_frames = [start_idx + k * frame_stride for k in range(num_frames)]
    ref_frame      = tracked_frames[0]
When ``--num-frames`` is omitted, every strided frame from ``start_idx``
through the end of the clip is used. ``--frame-stride`` defaults to 5.
Every window frame must have a stage-00 depth map and camera pose. Scene flow
is written only for these frames. Point-track joint estimation (stage 10)
uses the available observations; choose a window spanning real motion to
make the joint identifiable.

Coordinate frame:
    TrackCraft3R returns tracks in the reference frame's CAMERA space. Point-track joint estimation
    localizes the joint axis in whatever frame pts_ref lives in. To keep the
    joint in the pipeline WORLD frame, we lift tracks to world using the
    ref frame's camera-to-world (from stage-00 cameras.npz):
        p_world = R_c2w[ref] @ p_cam + t_c2w[ref]
    Scene flow is a difference of two world points, so only the rotation applies:
        flow_world = R_c2w[ref] @ (p_cam[t] - p_cam[ref])

Run inside the TrackCraft3R conda env.

Example:
    python scripts/08_trackcraft_flow.py \\
        --scene-dir        data/kitchen_pour_01 \\
        --trackcraft-repo  /home/jeremy/research/Articulate4D/TrackCraft3r \\
        --checkpoint       /path/to/trackcraft3r/model.safetensors \\
        --start-idx 0 --num-frames 12 --frame-stride 5 \\
        --overwrite
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Geometry helpers (same conventions as the other stages)
# ---------------------------------------------------------------------------


def quat_xyzw_to_R(q):
    """Camera-to-world rotation from an XYZW quaternion (matches stages 11/12)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_binary_mask_to(path: Path, target_hw) -> np.ndarray:
    """Read an 8-bit mask PNG, threshold > 0, stretch (nearest) to (H, W) bool.

    A plain resize is correct here: TrackCraft3R runs in 'stretch' mode, so its
    output grid is a non-aspect-preserving resize of the full image — matching a
    plain PIL resize of the full-res mask pixel-for-pixel."""
    with Image.open(path) as im:
        m = np.array(im)
    if m.ndim == 3:
        m = m[..., -1]
    m_bool = m > 0
    H, W = target_hw
    if m_bool.shape != (H, W):
        m_u8 = (m_bool.astype(np.uint8) * 255)
        m_u8 = np.array(Image.fromarray(m_u8).resize((W, H), Image.NEAREST))
        m_bool = m_u8 > 0
    return m_bool


def unproject_to_world(depth, K, R_c2w, t_c2w):
    """Dense back-projection of a z-depth map into world coords -> (H, W, 3)."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_world = np.stack([x, y, z], axis=-1) @ R_c2w.T + t_c2w
    pts_world[~valid] = 0.0
    return pts_world.astype(np.float32)


def project_world_to_image(points_world, R_c2w, t_c2w, K):
    """Project (N, 3) world points to full-resolution pixels."""
    pts_cam = (points_world.astype(np.float64) - t_c2w) @ R_c2w
    valid = np.isfinite(pts_cam).all(axis=1) & (pts_cam[:, 2] > 1e-6)
    uvw = pts_cam @ K.T
    uv = uvw[:, :2] / np.where(np.abs(uvw[:, 2:3]) > 1e-8,
                               uvw[:, 2:3], 1e-8)
    return uv, valid


def track_colors_bgr(tracks):
    """Stable rainbow BGR colors based on each track's initial world X."""
    x = tracks[:, 0, 0]
    span = max(float(np.ptp(x)), 1e-8)
    hue = np.round(179.0 * (x - float(x.min())) / span).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 220), np.full_like(hue, 255)], axis=1)
    return cv2.cvtColor(hsv[:, None, :], cv2.COLOR_HSV2BGR)[:, 0, :]


def write_point_track_video(path, frame_paths, frame_indices, track_payload,
                            cam_c2w, K_for_frame, fps, trail_frames,
                            point_radius):
    """Write sampled 3D tracks projected over their original RGB frames."""
    if not track_payload:
        print("warning: no tracks available; skipping point-track video")
        return False

    first = cv2.imread(str(frame_paths[frame_indices[0]]))
    if first is None:
        print("warning: could not read first RGB frame; skipping point-track video")
        return False
    H, W = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             float(fps), (W, H))
    if not writer.isOpened():
        print(f"warning: OpenCV could not open video writer for {path}")
        return False

    colors = {label: track_colors_bgr(tracks)
              for label, tracks in track_payload.items()}
    for k, fi in enumerate(frame_indices):
        canvas = cv2.imread(str(frame_paths[fi]))
        if canvas is None or canvas.shape[:2] != (H, W):
            writer.release()
            raise RuntimeError(f"unreadable or differently sized RGB frame: {frame_paths[fi]}")
        R_c2w, t_c2w = cam_c2w(fi)
        K = K_for_frame(fi)
        trail_layer = canvas.copy()
        k0 = max(0, k - trail_frames)

        for label, tracks in track_payload.items():
            label_colors = colors[label]
            history = tracks[:, k0:k + 1]
            n_tracks, n_steps = history.shape[:2]
            uv, valid = project_world_to_image(
                history.reshape(-1, 3), R_c2w, t_c2w, K)
            uv = uv.reshape(n_tracks, n_steps, 2)
            valid = valid.reshape(n_tracks, n_steps)
            valid &= ((uv[..., 0] >= 0) & (uv[..., 0] < W) &
                      (uv[..., 1] >= 0) & (uv[..., 1] < H))
            for i in range(n_tracks):
                color = tuple(int(v) for v in label_colors[i])
                for j in range(1, n_steps):
                    if valid[i, j - 1] and valid[i, j]:
                        cv2.line(trail_layer, tuple(uv[i, j - 1].astype(int)),
                                 tuple(uv[i, j].astype(int)), color, 3, cv2.LINE_AA)

            cur_uv, cur_valid = project_world_to_image(
                tracks[:, k], R_c2w, t_c2w, K)
            cur_valid &= ((cur_uv[:, 0] >= 0) & (cur_uv[:, 0] < W) &
                          (cur_uv[:, 1] >= 0) & (cur_uv[:, 1] < H))
            for i in np.flatnonzero(cur_valid):
                cv2.circle(canvas, tuple(cur_uv[i].astype(int)), point_radius,
                           tuple(int(v) for v in label_colors[i]), -1, cv2.LINE_AA)

        canvas = cv2.addWeighted(trail_layer, 0.8, canvas, 0.2, 0.0)
        writer.write(canvas)

    writer.release()
    print(f"wrote point-track overlay: {path}")
    return True


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/ (needs stages 00 + 02 done)")
    p.add_argument("--trackcraft-repo", type=Path, required=True,
                   help="Path to the TrackCraft3r checkout (added to sys.path)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="TrackCraft3R model checkpoint (.safetensors)")
    p.add_argument("--model-cache", type=Path, default=None,
                   help="Wan base-model cache (default: <trackcraft-repo>/checkpoints/wan_models)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only process these labels (default: all in tracking.json)")

    # Window selection
    p.add_argument("--start-idx", type=int, default=0,
                   help="Scene frame index of the window's first (reference) frame")
    p.add_argument("--num-frames", type=int, default=None,
                   help="Frames per model run (default: all remaining strided frames)")
    p.add_argument("--frame-stride", type=int, default=5,
                   help="Sample every Nth frame (default 5)")

    # Predictor params (mirror inference_user_video.py defaults)
    p.add_argument("--model-id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    p.add_argument("--lora-rank", type=int, default=1024)
    p.add_argument("--lora-target-modules", type=str, default="q,k,v,o,ffn.0,ffn.2")
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--regression-timestep", type=int, default=-1)
    p.add_argument("--track-latent-length", type=int, default=12)
    p.add_argument("--resize-mode", type=str, default="stretch",
                   choices=["pad", "stretch"],
                   help="'stretch' (default) keeps the mask<->track pixel mapping trivial")
    p.add_argument("--diag-max-depth", type=float, default=80.0)
    p.add_argument("--pj-norm-percentile-lo", type=float, default=2.0)
    p.add_argument("--pj-norm-percentile-hi", type=float, default=98.0)
    p.add_argument("--device", type=str, default="cuda")

    p.add_argument("--no-track-video", action="store_true",
                   help="Do not write trackcraft/point_tracks_video.mp4")
    p.add_argument("--track-video-fps", type=float, default=5.0,
                   help="Point-track overlay video FPS (default: 5)")
    p.add_argument("--track-video-max-tracks", type=int, default=200,
                   help="Maximum displayed tracks per label (default: 200)")
    p.add_argument("--track-video-trail-frames", type=int, default=8,
                   help="Number of prior tracked frames shown as trails (default: 8)")
    p.add_argument("--track-video-point-radius", type=int, default=5,
                   help="Current-point radius in pixels (default: 5)")

    p.add_argument("--overwrite", action="store_true",
                   help="Replace any existing per-label tracking outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.track_video_fps <= 0:
        sys.exit("error: --track-video-fps must be positive")
    if args.track_video_max_tracks < 0 or args.track_video_trail_frames < 0:
        sys.exit("error: video track/trail counts must be non-negative")
    if args.track_video_point_radius < 1:
        sys.exit("error: --track-video-point-radius must be at least 1")
    if args.start_idx < 0:
        sys.exit("error: --start-idx must be non-negative")
    if args.frame_stride < 1:
        sys.exit("error: --frame-stride must be at least 1")
    if args.num_frames is not None and args.num_frames < 1:
        sys.exit("error: --num-frames must be at least 1 when provided")

    scene_dir = args.scene_dir.resolve()
    tc_repo = args.trackcraft_repo.resolve()
    model_cache = (args.model_cache.resolve() if args.model_cache is not None
                   else tc_repo / "checkpoints" / "wan_models")
    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"
    tracking_path = masks_root / "tracking.json"
    da3_root = scene_dir / "da3"
    depth_dir = da3_root / "depth"
    intrinsics_path = da3_root / "intrinsics.npz"
    cameras_path = da3_root / "cameras.npz"
    out_root = scene_dir / "trackcraft"

    for need in (frames_dir, tracking_path, da3_root, depth_dir,
                 intrinsics_path, cameras_path):
        if not Path(need).exists():
            sys.exit(f"error: missing {need} (run stage 00 (DA3) and stage 02 first)")

    # Labels
    with open(tracking_path) as f:
        tracking = json.load(f)
    labels = sorted(tracking.keys())
    if args.labels:
        unknown = [l for l in args.labels if l not in tracking]
        if unknown:
            sys.exit(f"error: --labels not in tracking.json: {unknown}")
        labels = [l for l in labels if l in args.labels]
    if not labels:
        sys.exit("error: nothing to process")

    # Frames
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"error: no *.jpg in {frames_dir}")
    n_frames = len(frame_paths)

    if args.start_idx >= n_frames:
        sys.exit(f"error: --start-idx {args.start_idx} is outside clip length {n_frames}")
    if args.num_frames is None:
        win = list(range(args.start_idx, n_frames, args.frame_stride))
        num_frames = len(win)
        print(f"auto-selected {num_frames} frames from {args.start_idx} to "
              f"{n_frames - 1} with stride {args.frame_stride}")
    else:
        num_frames = args.num_frames
        win = [args.start_idx + k * args.frame_stride for k in range(num_frames)]
        if win[-1] >= n_frames:
            sys.exit(f"error: window {win[0]}..{win[-1]} (num_frames={num_frames} x "
                     f"frame_stride={args.frame_stride}) exceeds clip length {n_frames}. "
                     f"Lower --num-frames / --frame-stride / --start-idx.")
    ref_frame = win[0]
    print(f"scene:          {scene_dir.name}")
    print(f"tracked frames: {win}")
    print(f"ref_frame:      {ref_frame:06d}")
    print(f"labels:         {labels}\n")

    # --- Load stage-00 depth / intrinsics / cameras for the window ---
    intr_npz = np.load(intrinsics_path)
    intr_frames = intr_npz["frame_indices"].tolist()
    intr_arr = intr_npz["intrinsics"]
    cams = np.load(cameras_path)
    cam_frames = cams["frame_indices"].tolist()
    cam_quats = cams["cam_quats_xyzw"]
    cam_trans = cams["cam_trans"]

    def cam_c2w(frame_idx):
        i = cam_frames.index(frame_idx)
        return quat_xyzw_to_R(cam_quats[i]), cam_trans[i].astype(np.float64)

    def intrinsics_for(frame_idx):
        return intr_arr[intr_frames.index(frame_idx)].astype(np.float64)

    for fi in win:
        if not (depth_dir / f"{fi:06d}.npy").is_file():
            sys.exit(f"error: no stage-00 depth for frame {fi:06d}")
        if fi not in intr_frames or fi not in cam_frames:
            sys.exit(f"error: frame {fi:06d} missing from stage-00 intrinsics/cameras")

    # Depth stack (z-depth, full res) + intrinsics 4-vec at the ref frame.
    depth_stack = np.stack(
        [np.load(depth_dir / f"{fi:06d}.npy").astype(np.float32) for fi in win])
    K_ref = intrinsics_for(ref_frame)
    fx_fy_cx_cy = np.array(
        [K_ref[0, 0], K_ref[1, 1], K_ref[0, 2], K_ref[1, 2]], dtype=np.float64)

    # W2C extrinsics for the window, then frame-0 normalize (build_user_npz).
    extr_w2c = np.zeros((len(win), 4, 4), dtype=np.float64)
    for k, fi in enumerate(win):
        R_c2w, t_c2w = cam_c2w(fi)
        M = np.eye(4)
        M[:3, :3] = R_c2w
        M[:3, 3] = t_c2w
        extr_w2c[k] = np.linalg.inv(M)          # world-to-camera
    inv0 = np.linalg.inv(extr_w2c[0])
    extr_w2c_norm = np.stack([extr_w2c[k] @ inv0 for k in range(len(win))]).astype(np.float32)

    # RGB PIL frames at original resolution.
    video_list = [Image.open(frame_paths[fi]).convert("RGB") for fi in win]

    # --- Build TrackCraft3R predictor + run once ---
    if not (tc_repo / "evaluation").is_dir():
        sys.exit(f"error: {tc_repo} does not look like the TrackCraft3r repo "
                 f"(no evaluation/ package inside)")

    # DiffSynth resolves the Wan model ID through ModelScope. Point it at the
    # repository-local download and force offline mode, otherwise it may use a
    # different global cache and download the same multi-GB weights again.
    wan_model_dir = model_cache / args.model_id
    required_wan_files = (
        wan_model_dir / "diffusion_pytorch_model.safetensors",
        wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        wan_model_dir / "Wan2.1_VAE.pth",
    )
    missing_wan_files = [p for p in required_wan_files if not p.is_file()]
    if missing_wan_files:
        missing_text = "\n  ".join(str(p) for p in missing_wan_files)
        sys.exit(
            f"error: local Wan base model is incomplete; missing:\n  {missing_text}\n"
            f"download it with:\n  MODELSCOPE_CACHE={model_cache} "
            f"python {tc_repo / 'scripts' / 'download_wan_1.3B.py'}"
        )
    os.environ["MODELSCOPE_CACHE"] = str(model_cache)
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    print(f"Wan model cache: {wan_model_dir} (offline)\n")

    sys.path.insert(0, str(tc_repo))
    try:
        from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor
    except ImportError as e:
        sys.exit(f"error: failed to import TrackCraft3R predictor: {e}")

    print("building TrackCraft3R predictor ...")
    predictor = WanSceneFlowPredictor(
        checkpoint_path=str(args.checkpoint),
        model_id=args.model_id,
        lora_rank=args.lora_rank,
        lora_target_modules=args.lora_target_modules,
        height=args.height, width=args.width, device=args.device,
        regression_timestep=args.regression_timestep,
        track_latent_length=args.track_latent_length,
        resize_mode=args.resize_mode,
        diag_max_depth=args.diag_max_depth,
        pj_norm_percentile_lo=args.pj_norm_percentile_lo,
        pj_norm_percentile_hi=args.pj_norm_percentile_hi,
    )

    # Dense query grid (the outputs we use come from the predictor's dense cache).
    H_img, W_img = video_list[0].height, video_list[0].width
    stride = max(1, min(H_img, W_img) // 16)
    vv, uu = np.meshgrid(np.arange(0, H_img, stride), np.arange(0, W_img, stride),
                         indexing="ij")
    query_uv = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=-1).astype(np.float32)
    vis_dummy = np.ones((len(win), query_uv.shape[0]), dtype=bool)

    print("running TrackCraft3R inference ...")
    with torch.no_grad():
        predictor.predict(
            video_list, query_uv, vis_dummy, fx_fy_cx_cy,
            depth_map=depth_stack, extrinsics_w2c=extr_w2c_norm,
        )

    track_map = predictor._last_row_dense.astype(np.float32)  # (T, Hm, Wm, 3) cam-0 space
    T_out, Hm, Wm, _ = track_map.shape
    if T_out != len(win):
        print(f"warning: model returned {T_out} frames for a {len(win)}-frame window; "
              f"aligning to the first {min(T_out, len(win))}")
    n_use = min(T_out, len(win))
    print(f"tracker output: T={T_out}  resolution {Wm}x{Hm} (cam-0 space)\n")

    # --- Lift to world frame via the ref camera-to-world ---
    R_ref, t_ref = cam_c2w(ref_frame)
    pts_ref_cam = track_map[0]                                  # (Hm, Wm, 3)

    # --- Recompute the dense ref pointmap in world coords (keeps 41/51 consistent
    #     with the new ref_frame) ---
    depth_ref_full = np.load(depth_dir / f"{ref_frame:06d}.npy").astype(np.float32)
    pointmap_ref = unproject_to_world(depth_ref_full, K_ref, R_ref, t_ref)
    out_root.mkdir(parents=True, exist_ok=True)
    np.save(out_root / "pointmap_ref.npy", pointmap_ref)

    # --- Per-label outputs ---
    ref_stem = f"{ref_frame:06d}"
    written = []
    track_payload = {}
    for label in labels:
        mask_path = masks_root / label / f"{ref_stem}.png"
        if not mask_path.is_file():
            print(f"[{label}] SKIP: no mask at {mask_path}")
            continue
        mask = load_binary_mask_to(mask_path, (Hm, Wm))
        if not mask.any():
            print(f"[{label}] SKIP: empty mask at ref frame {ref_stem}")
            continue

        label_out = out_root / label
        if label_out.exists():
            if args.overwrite:
                shutil.rmtree(label_out)
            else:
                sys.exit(f"error: {label_out} exists; pass --overwrite to replace")
        flow_dir = label_out / "scene_flow"
        flow_dir.mkdir(parents=True)

        cv2.imwrite(str(label_out / "ref_mask.png"), (mask.astype(np.uint8) * 255))

        ys, xs = np.where(mask)
        pixel_ij = np.stack([ys, xs], axis=1).astype(np.int16)
        np.save(label_out / "pixel_ij.npy", pixel_ij)

        # Reference 3D positions -> world.
        pcam_ref = pts_ref_cam[mask].astype(np.float64)                 # (N, 3)
        pts3d_ref_world = (pcam_ref @ R_ref.T + t_ref).astype(np.float32)
        np.save(label_out / "pts3d_ref.npy", pts3d_ref_world)

        # Sample complete world-space tracks for the RGB overlay video.
        n_overlay = min(args.track_video_max_tracks, pts3d_ref_world.shape[0])
        if n_overlay > 0:
            rng = np.random.default_rng(0)
            sampled = (rng.choice(pts3d_ref_world.shape[0], n_overlay, replace=False)
                       if pts3d_ref_world.shape[0] > n_overlay
                       else np.arange(n_overlay))
            tracks_cam = track_map[:n_use, mask][:, sampled, :]  # (T, M, 3)
            tracks_world = tracks_cam.astype(np.float64) @ R_ref.T + t_ref
            track_payload[label] = tracks_world.transpose(1, 0, 2).astype(np.float32)

        # Scene flow per target frame: rotation-only (translation cancels).
        for k in range(1, n_use):
            fi = win[k]
            delta_cam = (track_map[k][mask].astype(np.float64) - pcam_ref)  # (N, 3)
            flow_world = (delta_cam @ R_ref.T).astype(np.float32)
            np.save(flow_dir / f"{fi:06d}.npy", flow_world)

        written.append(label)
        print(f"[{label}] {pixel_ij.shape[0]} masked pixels, {n_use - 1} target frames "
              f"-> {label_out}/")

    video_written = False
    video_path = out_root / "point_tracks_video.mp4"
    if not args.no_track_video:
        video_written = write_point_track_video(
            video_path,
            frame_paths,
            win[:n_use],
            track_payload,
            cam_c2w,
            intrinsics_for,
            args.track_video_fps,
            args.track_video_trail_frames,
            args.track_video_point_radius,
        )

    # --- Update config.json ---
    cfg_path = out_root / "config.json"
    config = {}
    da3_cfg_path = da3_root / "config.json"
    if da3_cfg_path.is_file():
        with open(da3_cfg_path) as f:
            config = json.load(f)
    config.update({
        "scene_id": scene_dir.name,
        "geometry_source": "da3",
        "tracker": "trackcraft3r",
        "checkpoint": str(args.checkpoint),
        "ref_frame": int(ref_frame),
        "start_idx": int(args.start_idx),
        "num_frames": int(num_frames),
        "num_frames_auto": args.num_frames is None,
        "frame_stride": int(args.frame_stride),
        "tracked_frames": [int(f) for f in win[:n_use]],
        "n_target_frames": int(n_use - 1),
        "tracker_resolution_wh": [int(Wm), int(Hm)],
        "labels_processed": written,
        "has_scene_flow": True,
        "point_tracks_video": (video_path.name if video_written else None),
        "point_tracks_video_fps": (float(args.track_video_fps)
                                    if video_written else None),
        "point_tracks_video_max_tracks_per_label": int(args.track_video_max_tracks),
    })
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\ndone. tracked {len(written)} label(s); ref_frame={ref_frame:06d}.")
    if not written:
        print("WARNING: no labels tracked (all masks empty at the ref frame). "
              "Pick a --start-idx where the objects are visible.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
