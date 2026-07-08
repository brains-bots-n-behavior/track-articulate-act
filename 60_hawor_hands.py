#!/usr/bin/env python
"""Stage 60: video-temporal HaWoR hand tracking (MANO mesh + 3D keypoints).

HaWoR runs a full video pipeline — detect/track -> per-frame motion estimation
-> masked DROID-SLAM (camera trajectory + metric scale) -> a transformer
in-filler that completes both hands across the whole clip. The result is a
temporally smooth, two-hand (left=0, right=1) trajectory in a SLAM world frame.

This stage runs that pipeline on data/<scene>/frames/ and re-projects the hands
into the **camera frame** of each image (RDF / OpenCV, matching the stage-00
depth + cameras), then writes a per-frame npz to `data/<scene>/hawor/` that the
scene-authoring track (52c / 52d / 52e) consumes. If any4d/cameras.npz exists,
the cam->world transform is baked in to produce `verts_world` / `joints_world`
in the shared world frame.

Reads:
    data/<scene>/frames/*.jpg
    data/<scene>/any4d/cameras.npz            (optional, enables world-frame output)
    data/<scene>/any4d/moge/intrinsics.npz    (optional, supplies a metric focal)
    <hawor-repo>/weights/hawor/checkpoints/{hawor.ckpt, infiller.pt}
    <hawor-repo>/_DATA/...                     (MANO models)

Writes:
    data/<scene>/hawor/
        config.json                       # run params + summary stats
        faces.npy                         # (Nf, 3) MANO faces, shared (right-hand winding)
        _work/                            # HaWoR seq intermediates (tracks, SLAM, params); cached
        per_frame/<frame_idx>.npz         # only frames that have at least one hand
            verts          (n_hands, 778, 3)   float32  — camera frame (RDF)
            joints         (n_hands, 21, 3)    float32  — camera frame
            is_right       (n_hands,)          bool
            valid          (n_hands,)          bool     — True=detected, False=infilled  [HaWoR-specific]
            cam_t          (n_hands, 3)        float32  — camera-frame wrist (joint 0)
            bbox           (n_hands, 4)        float32  — image-space xyxy (from projected verts)
            focal_length   ()                  float32
            img_size_wh    (2,)                int32
            frame_idx      ()                  int32
            (verts_world)  (n_hands, 778, 3)   float32  — only if any4d/cameras.npz present
            (joints_world) (n_hands, 21, 3)    float32

Left-hand meshes share `faces.npy` but with reversed winding (a convenience
`faces_left.npy` with corrected winding is also written). The `valid` field
marks detected (True) vs. in-filled (False) hands.

Run inside the `hawor` conda env (see project notes: torch 2.0.1+cu118 on
dsailogin). Needs a GPU.

Examples:
    python scripts/60_hawor_hands.py \\
        --scene-dir data/oven \\
        --hawor-repo /home/jeremy/research/Articulate4D/HaWoR

    # Save only frames where the hand was actually detected (drop in-filled), and
    # force a focal length instead of using the MoGe one.
    python scripts/60_hawor_hands.py \\
        --scene-dir data/oven \\
        --hawor-repo /home/jeremy/research/Articulate4D/HaWoR \\
        --detected-only --img-focal 1500 --overwrite
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry / IO helpers
# ---------------------------------------------------------------------------


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_cam_poses(scene_dir: Path):
    """Return dict {frame_idx -> (R_c2w, t_c2w)} from any4d/cameras.npz, or {}."""
    path = scene_dir / "any4d" / "cameras.npz"
    if not path.is_file():
        return {}
    cams = np.load(path)
    out = {}
    for i, fi in enumerate(cams["frame_indices"].tolist()):
        out[int(fi)] = (
            quat_xyzw_to_R(cams["cam_quats_xyzw"][i]),
            cams["cam_trans"][i].astype(np.float64),
        )
    return out


def load_moge_focal_map(scene_dir: Path):
    """Return dict {frame_idx -> focal_px} from any4d/moge/intrinsics.npz, or {}."""
    p = scene_dir / "any4d" / "moge" / "intrinsics.npz"
    if not p.is_file():
        return {}
    d = np.load(p)
    return {int(fi): float(d["intrinsics"][i][0, 0])
            for i, fi in enumerate(d["frame_indices"].tolist())}


def prepare_seq(work_dir: Path, frame_paths):
    """Symlink the scene frames into the extracted_images layout HaWoR expects
    and return a synthetic video_path whose dir/stem resolve to the seq folder.

    HaWoR's stage functions all derive
        img_folder = f"{dirname(video_path)}/{stem(video_path)}/extracted_images"
    so we just make {dirname}/{stem} a real seq folder full of jpgs.
    """
    seq_folder = work_dir / "seq"
    extracted = seq_folder / "extracted_images"
    extracted.mkdir(parents=True, exist_ok=True)
    for src in frame_paths:
        dst = extracted / src.name
        if not dst.exists():
            os.symlink(src.resolve(), dst)
    return str(seq_folder) + ".mp4", seq_folder


def project_bbox(verts_cam, focal, W, H):
    """Tight image-space xyxy bbox from camera-frame verts (N,3)."""
    z = np.clip(verts_cam[:, 2], 1e-4, None)
    xs = focal * verts_cam[:, 0] / z + W / 2.0
    ys = focal * verts_cam[:, 1] / z + H / 2.0
    x0 = float(np.clip(xs.min(), 0, W)); x1 = float(np.clip(xs.max(), 0, W))
    y0 = float(np.clip(ys.min(), 0, H)); y1 = float(np.clip(ys.max(), 0, H))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--hawor-repo", type=Path, required=True,
                   help="Path to the HaWoR/ checkout (needs weights/ and _DATA/)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="HaWoR ckpt (default: <repo>/weights/hawor/checkpoints/hawor.ckpt)")
    p.add_argument("--infiller-weight", type=str, default=None,
                   help="Infiller weights (default: <repo>/weights/hawor/checkpoints/infiller.pt)")
    p.add_argument("--img-focal", type=float, default=None,
                   help="Pinhole focal in px. Default: median MoGe focal if "
                        "any4d/moge/intrinsics.npz is present, else HaWoR's own "
                        "estimate (~600 fallback).")
    p.add_argument("--ignore-moge-focal", action="store_true",
                   help="Do not auto-load the MoGe focal; let HaWoR estimate it.")
    p.add_argument("--start-idx", type=int, default=None,
                   help="Only SAVE frames with index >= this (inference still runs "
                        "on the whole clip — SLAM needs continuity).")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Only SAVE frames with index < this.")
    p.add_argument("--detected-only", action="store_true",
                   help="Save only hands actually detected at a frame; drop "
                        "in-filled poses. Default keeps the full trajectory and "
                        "marks in-filled hands with valid=False.")
    p.add_argument("--no-world", action="store_true",
                   help="Skip applying Any4D cam2world; save camera frame only.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite per_frame/faces/config (keeps the _work SLAM "
                        "cache for fast re-derivation; add --recompute to clear it).")
    p.add_argument("--recompute", action="store_true",
                   help="Also clear the _work cache, forcing tracking/SLAM to rerun.")
    return p.parse_args()


def run_hawor_pipeline(args_ns, work_dir, frame_paths):
    """Run detect/track -> motion -> SLAM -> infiller. Returns
    (imgfiles, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
     R_w2c, t_w2c, img_focal)."""
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_video import (
        hawor_motion_estimation, hawor_infiller)
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from lib.eval_utils.custom_utils import load_slam_cam

    video_path, seq_folder = prepare_seq(work_dir, frame_paths)
    args_ns.video_path = video_path

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args_ns)
    frame_chunks_all, img_focal = hawor_motion_estimation(
        args_ns, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args_ns, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args_ns, start_idx, end_idx, frame_chunks_all)

    return (imgfiles, pred_trans, pred_rot, pred_hand_pose, pred_betas,
            pred_valid, R_w2c, t_w2c, img_focal)


def hand_camera_frame(idx, pred_trans, pred_rot, pred_hand_pose, pred_betas,
                      R_w2c_np, t_w2c_np):
    """World-space MANO for one hand (idx 0=left, 1=right), then transform every
    frame into the camera frame with the SLAM extrinsics. Returns
    (verts_cam (T,778,3), joints_cam (T,21,3)) as float32 numpy."""
    from hawor.utils.process import run_mano, run_mano_left

    sl = slice(idx, idx + 1)
    runner = run_mano if idx == 1 else run_mano_left
    out = runner(pred_trans[sl], pred_rot[sl], pred_hand_pose[sl], betas=pred_betas[sl])
    vw = out["vertices"][0].detach().cpu().numpy().astype(np.float64)   # (T,778,3) world
    jw = out["joints"][0].detach().cpu().numpy().astype(np.float64)     # (T,21,3) world

    # x_cam = R_w2c @ x_world + t_w2c, per frame
    v_cam = np.einsum('tij,tnj->tni', R_w2c_np, vw) + t_w2c_np[:, None, :]
    j_cam = np.einsum('tij,tnj->tni', R_w2c_np, jw) + t_w2c_np[:, None, :]
    return v_cam.astype(np.float32), j_cam.astype(np.float32)


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    hawor_repo = args.hawor_repo.resolve()

    frames_dir = scene_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    ckpt = Path(args.checkpoint) if args.checkpoint else \
        hawor_repo / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
    infiller = Path(args.infiller_weight) if args.infiller_weight else \
        hawor_repo / "weights" / "hawor" / "checkpoints" / "infiller.pt"
    for p in (ckpt, infiller):
        if not p.is_file():
            sys.exit(f"error: missing {p}")

    out_root = scene_dir / "hawor"
    per_frame_dir = out_root / "per_frame"
    work_dir = out_root / "_work"

    if per_frame_dir.exists() and not args.overwrite:
        sys.exit(f"error: {per_frame_dir} exists; pass --overwrite")
    if args.overwrite:
        for sub in (per_frame_dir, out_root / "faces.npy",
                    out_root / "faces_left.npy", out_root / "config.json"):
            if sub.is_dir():
                shutil.rmtree(sub)
            elif sub.exists():
                sub.unlink()
        if args.recompute and work_dir.exists():
            shutil.rmtree(work_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"error: no .jpg frames in {frames_dir}")
    img0 = cv2.imread(str(frame_paths[0]))
    H, W = img0.shape[:2]
    img_size_wh = [int(W), int(H)]

    # Focal resolution: explicit > MoGe median > HaWoR's own estimate.
    moge_focal_map = {} if args.ignore_moge_focal else load_moge_focal_map(scene_dir)
    if args.img_focal is not None:
        img_focal_in, focal_source = float(args.img_focal), "provided"
    elif moge_focal_map:
        img_focal_in = float(np.median(list(moge_focal_map.values())))
        focal_source = "moge_median"
    else:
        img_focal_in, focal_source = None, "hawor_estimate"

    print(f"scene:        {scene_dir.name}")
    print(f"hawor_repo:   {hawor_repo}")
    print(f"frames:       {len(frame_paths)} ({W}x{H})")
    print(f"focal:        {img_focal_in} ({focal_source})")

    # HaWoR stage functions use repo-relative paths (_DATA, thirdparty, weights),
    # so run them from the repo root. Scene paths were already resolved absolute.
    sys.path.insert(0, str(hawor_repo))
    os.chdir(hawor_repo)

    args_ns = SimpleNamespace(
        img_focal=img_focal_in,
        checkpoint=str(ckpt),
        infiller_weight=str(infiller),
        input_type="file",
        video_path=None,
    )

    t_start = time.time()
    try:
        (imgfiles, pred_trans, pred_rot, pred_hand_pose, pred_betas,
         pred_valid, R_w2c, t_w2c, img_focal) = run_hawor_pipeline(
            args_ns, work_dir, frame_paths)
    except ImportError as e:
        sys.exit(f"error: failed to import HaWoR dependencies ({e}). "
                 f"Activate the hawor conda env and retry.")

    from hawor.utils.process import get_mano_faces

    T = pred_trans.shape[1]
    pred_valid = np.asarray(pred_valid).astype(bool)        # (2, T)
    R_w2c_np = np.asarray(R_w2c.detach().cpu() if torch.is_tensor(R_w2c) else R_w2c,
                          dtype=np.float64)                  # (T, 3, 3)
    t_w2c_np = np.asarray(t_w2c.detach().cpu() if torch.is_tensor(t_w2c) else t_w2c,
                          dtype=np.float64)                  # (T, 3)
    img_focal = float(img_focal)

    # frame index for each pipeline timestep, from the (natsorted) image names
    frame_indices = [int(Path(f).stem) for f in imgfiles]

    # Shared MANO topology (standard right-hand winding).
    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    np.save(out_root / "faces.npy", faces)
    np.save(out_root / "faces_left.npy", faces[:, [0, 2, 1]])

    cam_pose_map = {} if args.no_world else load_cam_poses(scene_dir)
    print(f"camera poses available: {len(cam_pose_map)} frame(s) "
          f"{'(world-frame output enabled)' if cam_pose_map else '(camera-frame only)'}")

    # Which hands ever appear? (a hand with no detected frame is pure infill.)
    active = [idx for idx in (0, 1) if pred_valid[idx].any()]
    side_name = {0: "left", 1: "right"}
    print(f"active hands: {[side_name[i] for i in active]}")

    # Pre-compute camera-frame verts/joints for each active hand (T, *, 3).
    cam_cache = {idx: hand_camera_frame(
        idx, pred_trans, pred_rot, pred_hand_pose, pred_betas, R_w2c_np, t_w2c_np)
        for idx in active}

    save_start = args.start_idx if args.start_idx is not None else -np.inf
    save_end = args.end_idx if args.end_idx is not None else np.inf

    n_with_hands, total_hands = 0, 0
    for t in range(T):
        frame_idx = frame_indices[t]
        if not (save_start <= frame_idx < save_end):
            continue

        verts, joints, is_right, valid, cam_t, bbox = [], [], [], [], [], []
        for idx in active:
            if args.detected_only and not pred_valid[idx, t]:
                continue
            v_cam = cam_cache[idx][0][t]      # (778,3)
            j_cam = cam_cache[idx][1][t]      # (21,3)
            verts.append(v_cam)
            joints.append(j_cam)
            is_right.append(idx == 1)
            valid.append(bool(pred_valid[idx, t]))
            cam_t.append(j_cam[0].astype(np.float32))            # wrist position
            bbox.append(project_bbox(v_cam, img_focal, W, H))

        if not verts:
            continue

        verts = np.stack(verts).astype(np.float32)
        joints = np.stack(joints).astype(np.float32)
        save_dict = dict(
            verts=verts,
            joints=joints,
            is_right=np.array(is_right, dtype=bool),
            valid=np.array(valid, dtype=bool),
            cam_t=np.stack(cam_t).astype(np.float32),
            bbox=np.stack(bbox).astype(np.float32),
            focal_length=np.float32(img_focal),
            img_size_wh=np.asarray(img_size_wh, dtype=np.int32),
            frame_idx=np.int32(frame_idx),
        )

        if frame_idx in cam_pose_map:
            R_cw, t_cw = cam_pose_map[frame_idx]
            R_cw = R_cw.T.astype(np.float32)
            t_cw = t_cw.astype(np.float32)
            save_dict["verts_world"] = (verts @ R_cw + t_cw).astype(np.float32)
            save_dict["joints_world"] = (joints @ R_cw + t_cw).astype(np.float32)

        np.savez_compressed(per_frame_dir / f"{frame_idx:06d}.npz", **save_dict)
        n_with_hands += 1
        total_hands += len(verts)
        sides = ",".join("R" if r else "L" for r in is_right)
        inf = "" if all(valid) else " (some infilled)"
        print(f"  [{frame_idx:06d}] {len(verts)} hand(s) [{sides}]{inf}")

    elapsed = time.time() - t_start

    config = {
        "scene_id": scene_dir.name,
        "stage": "60_hawor_hands",
        "hawor_repo": str(hawor_repo),
        "checkpoint": str(ckpt),
        "infiller_weight": str(infiller),
        "img_focal": img_focal,
        "focal_source": focal_source,
        "img_size_wh": img_size_wh,
        "n_frames_total": int(T),
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "detected_only": bool(args.detected_only),
        "active_hands": [side_name[i] for i in active],
        "world_frame_baked": bool(cam_pose_map),
        "n_frames_with_hands": int(n_with_hands),
        "total_hands": int(total_hands),
        "elapsed_s": float(elapsed),
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"done in {elapsed:.1f}s. {n_with_hands}/{T} frames had hands; "
          f"{total_hands} total hand instances.")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
