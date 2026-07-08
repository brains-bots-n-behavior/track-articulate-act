#!/usr/bin/env python
"""Stage 40: TrackCraft3R point tracking -> per-label scene flow (Any4D substitute).

Drop-in replacement for the old Any4D stage 40. Instead of running Any4D, it
runs TrackCraft3R once on a fixed window of frames and converts the model's
dense frame-0-anchored 3D track field into the *exact* per-label schema the
downstream stages already read (50 joints, 51/41 replay).

Depth + camera/world come from stage 00 (`00_da3_depth_cameras.py`, DA3): this
stage READS that bundle (any4d/moge/depth, any4d/cameras.npz, any4d/moge/
intrinsics) and only ADDS the tracking outputs. Together, DA3 (stage 00) +
TrackCraft3R (this stage) fully replace Any4D.

Adapted from TrackCraft3r/scripts/build_user_npz.py (assemble RGB + depth +
w2c camera + intrinsics, frame-0-normalize the extrinsics) and
inference_user_video.py (run the WanSceneFlowPredictor, read its dense cache),
but wired to the Articulate4D scene layout and emitting Any4D's file schema.

Reads (produced by stages 00 + 10):
    data/<scene>/frames/*.jpg
    data/<scene>/masks/tracking.json
    data/<scene>/masks/<label>/<ref>.png
    data/<scene>/any4d/moge/depth/<frame>.npy        # z-depth, full res (stage 00)
    data/<scene>/any4d/moge/intrinsics.npz           # per-frame K, full res (stage 00)
    data/<scene>/any4d/cameras.npz                   # cam->world quats+trans (stage 00)

Writes (same layout as the old Any4D stage 40):
    data/<scene>/any4d/
        config.json                       # updated: ref_frame, tracker, tracked_frames
        pointmap_ref.npy                  # recomputed dense ref pointmap (world coords)
        <label>/
            ref_mask.png                  # mask at TRACKER resolution
            pixel_ij.npy                  # (N, 2) int16, ref pixel (row, col)
            pts3d_ref.npy                 # (N, 3) float32, ref 3D position (WORLD)
            scene_flow/<frame>.npy        # (N, 3) float32, ref->target flow (WORLD)

Frame selection (TrackCraft3R runs on a fixed-length window, unlike Any4D's
stride-1 full range):
    tracked_frames = [start_idx + k * frame_stride for k in range(num_frames)]
    ref_frame      = tracked_frames[0]
Every window frame must have a stage-00 depth map and camera pose. Scene flow
is written only for these frames; 50/51 guard on file presence, so a sparse set
is fine. (Stage 50 still needs *some* rotated frames — pick a window that spans
real motion.)

Coordinate frame:
    TrackCraft3R returns tracks in the reference frame's CAMERA space. Stage 50
    localizes the joint axis in whatever frame pts_ref lives in, and stage 51
    renders it in the pipeline WORLD frame, so we lift tracks to world using the
    ref frame's camera-to-world (from stage-00 cameras.npz):
        p_world = R_c2w[ref] @ p_cam + t_c2w[ref]
    Scene flow is a difference of two world points, so only the rotation applies:
        flow_world = R_c2w[ref] @ (p_cam[t] - p_cam[ref])

Run inside the TrackCraft3R conda env.

Example:
    python scripts/40_trackcraft_flow.py \\
        --scene-dir        data/kitchen_pour_01 \\
        --trackcraft-repo  /home/jeremy/research/Articulate4D/TrackCraft3r \\
        --checkpoint       /path/to/trackcraft3r/model.safetensors \\
        --start-idx 0 --num-frames 12 --frame-stride 5 \\
        --overwrite
"""

import argparse
import json
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
    """Camera-to-world rotation from an XYZW quaternion (matches stages 52/60)."""
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


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/ (needs stages 00 + 10 done)")
    p.add_argument("--trackcraft-repo", type=Path, required=True,
                   help="Path to the TrackCraft3r checkout (added to sys.path)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="TrackCraft3R model checkpoint (.safetensors)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only process these labels (default: all in tracking.json)")

    # Window selection
    p.add_argument("--start-idx", type=int, default=0,
                   help="Scene frame index of the window's first (reference) frame")
    p.add_argument("--num-frames", type=int, default=12,
                   help="Frames per model run (default 12 = training length)")
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

    p.add_argument("--overwrite", action="store_true",
                   help="Replace any existing per-label tracking outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    scene_dir = args.scene_dir.resolve()
    tc_repo = args.trackcraft_repo.resolve()
    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"
    tracking_path = masks_root / "tracking.json"
    any4d_root = scene_dir / "any4d"
    moge_depth_dir = any4d_root / "moge" / "depth"
    moge_intr_path = any4d_root / "moge" / "intrinsics.npz"
    cameras_path = any4d_root / "cameras.npz"

    for need in (frames_dir, tracking_path, any4d_root, moge_depth_dir,
                 moge_intr_path, cameras_path):
        if not Path(need).exists():
            sys.exit(f"error: missing {need} (run stages 00 (DA3) and 10 first)")

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

    win = [args.start_idx + k * args.frame_stride for k in range(args.num_frames)]
    if win[-1] >= n_frames:
        sys.exit(f"error: window {win[0]}..{win[-1]} (num_frames={args.num_frames} x "
                 f"frame_stride={args.frame_stride}) exceeds clip length {n_frames}. "
                 f"Lower --num-frames / --frame-stride / --start-idx.")
    ref_frame = win[0]
    print(f"scene:          {scene_dir.name}")
    print(f"tracked frames: {win}")
    print(f"ref_frame:      {ref_frame:06d}")
    print(f"labels:         {labels}\n")

    # --- Load stage-00 depth / intrinsics / cameras for the window ---
    intr_npz = np.load(moge_intr_path)
    intr_frames = intr_npz["frame_indices"].tolist()
    intr_arr = intr_npz["intrinsics"]
    cams = np.load(cameras_path)
    cam_frames = cams["frame_indices"].tolist()
    cam_quats = cams["cam_quats_xyzw"]
    cam_trans = cams["cam_trans"]

    def cam_c2w(frame_idx):
        i = cam_frames.index(frame_idx)
        return quat_xyzw_to_R(cam_quats[i]), cam_trans[i].astype(np.float64)

    for fi in win:
        if not (moge_depth_dir / f"{fi:06d}.npy").is_file():
            sys.exit(f"error: no stage-00 depth for frame {fi:06d}")
        if fi not in intr_frames or fi not in cam_frames:
            sys.exit(f"error: frame {fi:06d} missing from stage-00 intrinsics/cameras")

    # Depth stack (z-depth, full res) + intrinsics 4-vec at the ref frame.
    depth_stack = np.stack(
        [np.load(moge_depth_dir / f"{fi:06d}.npy").astype(np.float32) for fi in win])
    K_ref = intr_arr[intr_frames.index(ref_frame)].astype(np.float64)
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
    depth_ref_full = np.load(moge_depth_dir / f"{ref_frame:06d}.npy").astype(np.float32)
    pointmap_ref = unproject_to_world(depth_ref_full, K_ref, R_ref, t_ref)
    np.save(any4d_root / "pointmap_ref.npy", pointmap_ref)

    # --- Per-label outputs ---
    ref_stem = f"{ref_frame:06d}"
    written = []
    for label in labels:
        mask_path = masks_root / label / f"{ref_stem}.png"
        if not mask_path.is_file():
            print(f"[{label}] SKIP: no mask at {mask_path}")
            continue
        mask = load_binary_mask_to(mask_path, (Hm, Wm))
        if not mask.any():
            print(f"[{label}] SKIP: empty mask at ref frame {ref_stem}")
            continue

        label_out = any4d_root / label
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

        # Scene flow per target frame: rotation-only (translation cancels).
        for k in range(1, n_use):
            fi = win[k]
            delta_cam = (track_map[k][mask].astype(np.float64) - pcam_ref)  # (N, 3)
            flow_world = (delta_cam @ R_ref.T).astype(np.float32)
            np.save(flow_dir / f"{fi:06d}.npy", flow_world)

        written.append(label)
        print(f"[{label}] {pixel_ij.shape[0]} masked pixels, {n_use - 1} target frames "
              f"-> {label_out}/")

    # --- Update config.json ---
    cfg_path = any4d_root / "config.json"
    config = {}
    if cfg_path.is_file():
        with open(cfg_path) as f:
            config = json.load(f)
    config.update({
        "scene_id": scene_dir.name,
        "tracker": "trackcraft3r",
        "checkpoint": str(args.checkpoint),
        "ref_frame": int(ref_frame),
        "start_idx": int(args.start_idx),
        "num_frames": int(args.num_frames),
        "frame_stride": int(args.frame_stride),
        "tracked_frames": [int(f) for f in win[:n_use]],
        "n_target_frames": int(n_use - 1),
        "tracker_resolution_wh": [int(Wm), int(Hm)],
        "labels_processed": written,
        "has_scene_flow": True,
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
