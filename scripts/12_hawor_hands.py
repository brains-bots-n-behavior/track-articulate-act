#!/usr/bin/env python
"""Stage 12: video-temporal HaWoR hand tracking (MANO mesh + 3D keypoints).

HaWoR runs a full video pipeline — detect/track -> per-frame motion estimation
-> masked DROID-SLAM (camera trajectory + metric scale) -> a transformer
in-filler that completes both hands across the whole clip. The result is a
temporally smooth, two-hand (left=0, right=1) trajectory in a SLAM world frame.

This stage runs that pipeline on data/<scene>/frames/ and re-projects the hands
into the **camera frame** of each image (RDF / OpenCV, matching DA3). If
da3/cameras.npz exists, its cam->world transform is baked in to produce
`verts_world` / `joints_world` in the shared DA3 world frame.

Reads:
    data/<scene>/frames/*.jpg
    data/<scene>/da3/cameras.npz              (optional, enables world-frame output)
    data/<scene>/da3/intrinsics.npz           (optional, supplies a focal length)
    <hawor-repo>/weights/hawor/checkpoints/{hawor.ckpt, infiller.pt}
    <hawor-repo>/_DATA/...                     (MANO models)

Writes:
    data/<scene>/hawor/
        config.json                       # run params + summary stats
        overlay.mp4                       # selected hand over original RGB frames
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
            (verts_world)  (n_hands, 778, 3)   float32  — only if da3/cameras.npz present
            (joints_world) (n_hands, 21, 3)    float32

Only one hand is saved. `--hand left` or `--hand right` selects it explicitly;
the default `--hand auto` chooses the side detected in the most distinct frames
across the clip, with mean YOLO detector confidence as a tie-breaker. `valid`
records whether each saved pose was detected or in-filled. Left-hand meshes
share `faces.npy` but with reversed winding (a convenience `faces_left.npy`
with corrected winding is also written).

Run inside the `hawor` conda env (see project notes: torch 2.0.1+cu118 on
dsailogin). Needs a GPU.

Examples:
    python scripts/12_hawor_hands.py \\
        --scene-dir data/oven \\
        --hawor-repo /home/jeremy/research/Articulate4D/HaWoR

    # Save only frames where the hand was actually detected (drop in-filled), and
    # force a focal length instead of using the DA3 one.
    python scripts/12_hawor_hands.py \\
        --scene-dir data/oven \\
        --hawor-repo /home/jeremy/research/Articulate4D/HaWoR \\
        --hand right --detected-only --img-focal 1500 --overwrite
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


# HaWoR's MANO stack imports the unmaintained ``chumpy`` package, which still
# expects aliases removed in NumPy 1.24. Define only missing legacy names before
# any HaWoR module has a chance to import chumpy. Keeping this local avoids
# requiring users to downgrade NumPy in the otherwise working hawor environment.
for _name, _type in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _type)


# ---------------------------------------------------------------------------
# Geometry / IO helpers — shared RDF/OpenCV pipeline conventions
# ---------------------------------------------------------------------------


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_cam_poses(scene_dir: Path):
    """Return dict {frame_idx -> (R_c2w, t_c2w)} from da3/cameras.npz, or {}."""
    path = scene_dir / "da3" / "cameras.npz"
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


def load_da3_focal_map(scene_dir: Path):
    """Return dict {frame_idx -> focal_px} from da3/intrinsics.npz, or {}."""
    p = scene_dir / "da3" / "intrinsics.npz"
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


def project_points(points_cam, focal, W, H):
    """Project camera-frame 3D points to image pixels; return (xy, valid_z)."""
    points_cam = np.asarray(points_cam)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-4)
    z = np.maximum(points_cam[:, 2], 1e-4)
    xy = np.column_stack([
        focal * points_cam[:, 0] / z + W / 2.0,
        focal * points_cam[:, 1] / z + H / 2.0,
    ])
    return np.rint(xy).astype(np.int32), valid


def draw_hand_overlay(image, verts_cam, joints_cam, focal, is_right):
    """Draw a translucent MANO silhouette and 21-joint skeleton in-place."""
    H, W = image.shape[:2]
    verts_2d, verts_valid = project_points(verts_cam, focal, W, H)
    joints_2d, joints_valid = project_points(joints_cam, focal, W, H)
    color = (70, 190, 255) if is_right else (255, 145, 70)  # BGR

    visible = verts_2d[verts_valid]
    if len(visible) >= 3:
        visible[:, 0] = np.clip(visible[:, 0], 0, W - 1)
        visible[:, 1] = np.clip(visible[:, 1], 0, H - 1)
        hull = cv2.convexHull(visible)
        tint = image.copy()
        cv2.fillConvexPoly(tint, hull, color, lineType=cv2.LINE_AA)
        cv2.addWeighted(tint, 0.32, image, 0.68, 0, dst=image)
        cv2.polylines(image, [hull], True, color, 2, cv2.LINE_AA)

    # MANO joint order: wrist, then four joints for each of five fingers.
    bones = [(0, 1), (1, 2), (2, 3), (3, 4),
             (0, 5), (5, 6), (6, 7), (7, 8),
             (0, 9), (9, 10), (10, 11), (11, 12),
             (0, 13), (13, 14), (14, 15), (15, 16),
             (0, 17), (17, 18), (18, 19), (19, 20)]
    for a, b in bones:
        if joints_valid[a] and joints_valid[b]:
            cv2.line(image, tuple(joints_2d[a]), tuple(joints_2d[b]),
                     (255, 255, 255), 2, cv2.LINE_AA)
    for xy, valid in zip(joints_2d, joints_valid):
        if valid:
            cv2.circle(image, tuple(xy), 3, color, -1, cv2.LINE_AA)
    return image


def write_overlay_video(path, frame_paths, frame_to_t, verts_cam, joints_cam,
                        pred_valid, selected, focal, fps, save_start, save_end,
                        detected_only, W, H):
    """Write original RGB frames with the selected hand overlaid."""
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    written = 0
    try:
        for frame_path in frame_paths:
            frame_idx = int(frame_path.stem)
            if not (save_start <= frame_idx < save_end):
                continue
            image = cv2.imread(str(frame_path))
            if image is None:
                raise RuntimeError(f"could not read overlay frame {frame_path}")
            if image.shape[:2] != (H, W):
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            t = frame_to_t.get(frame_idx)
            if t is not None and (not detected_only or pred_valid[selected, t]):
                draw_hand_overlay(image, verts_cam[t], joints_cam[t], focal,
                                  is_right=(selected == 1))
            writer.write(image)
            written += 1
    finally:
        writer.release()
    if written == 0:
        path.unlink(missing_ok=True)
        raise RuntimeError("overlay video range contains no frames")
    return written


def summarize_hand_detections(seq_folder: Path, start_idx: int, end_idx: int):
    """Return per-side appearance count and mean YOLO confidence.

    HaWoR assigns each tracklet to left/right by its majority handedness. For
    duplicate detections of one side in a frame, keep only the highest score so
    appearance means distinct video frames rather than raw detector boxes.
    """
    seq_folder = Path(seq_folder)
    tracks_path = seq_folder / f"tracks_{start_idx}_{end_idx}" / "model_tracks.npy"
    tracks = np.load(tracks_path, allow_pickle=True).item()
    scores_by_frame = {0: {}, 1: {}}

    for tracklet in tracks.values():
        detected = [entry for entry in tracklet if bool(entry.get("det", False))]
        if not detected:
            continue
        handedness = np.concatenate([entry["det_handedness"] for entry in detected])
        side = int(float(handedness.mean()) >= 0.5)  # 0=left, 1=right
        for entry in detected:
            frame = int(entry["frame"])
            box = np.asarray(entry["det_box"]).reshape(-1)
            confidence = float(box[4]) if box.size >= 5 else 0.0
            previous = scores_by_frame[side].get(frame, -np.inf)
            scores_by_frame[side][frame] = max(previous, confidence)

    stats = {}
    for side in (0, 1):
        scores = list(scores_by_frame[side].values())
        stats[side] = {
            "detected_frames": len(scores),
            "mean_confidence": float(np.mean(scores)) if scores else 0.0,
        }
    return stats


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
                   help="Pinhole focal in px. Default: median DA3 focal if "
                        "da3/intrinsics.npz is present, else HaWoR's own "
                        "estimate (~600 fallback).")
    p.add_argument("--ignore-da3-focal", "--ignore-moge-focal",
                   dest="ignore_da3_focal", action="store_true",
                   help="Do not auto-load the DA3 focal; let HaWoR estimate it. "
                        "(--ignore-moge-focal is retained as a compatibility alias.)")
    p.add_argument("--start-idx", type=int, default=None,
                   help="Only SAVE frames with index >= this (inference still runs "
                        "on the whole clip — SLAM needs continuity).")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Only SAVE frames with index < this.")
    p.add_argument("--detected-only", action="store_true",
                   help="Save only hands actually detected at a frame; drop "
                        "in-filled poses. Default keeps the full trajectory and "
                        "marks in-filled hands with valid=False.")
    p.add_argument("--hand", choices=["auto", "left", "right"], default="auto",
                   help="Hand to save and show in the overlay. Default 'auto' "
                        "chooses the side seen in the most frames, breaking a "
                        "tie by mean detector confidence.")
    p.add_argument("--no-world", action="store_true",
                   help="Skip applying DA3 cam2world; save camera frame only.")
    p.add_argument("--overlay-fps", type=float, default=30.0,
                   help="Frame rate for hawor/overlay.mp4 (default: 30).")
    p.add_argument("--no-overlay", action="store_true",
                   help="Do not write hawor/overlay.mp4.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite per_frame/faces/config (keeps the _work SLAM "
                        "cache for fast re-derivation; add --recompute to clear it).")
    p.add_argument("--recompute", action="store_true",
                   help="Also clear the _work cache, forcing tracking/SLAM to rerun.")
    return p.parse_args()


def run_hawor_pipeline(args_ns, work_dir, frame_paths):
    """Run detect/track -> motion -> SLAM -> infiller. Returns
    (imgfiles, pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid,
     R_w2c, t_w2c, img_focal, detection_stats)."""
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
    detection_stats = summarize_hand_detections(seq_folder, start_idx, end_idx)
    direct_valid = np.zeros((2, len(imgfiles)), dtype=bool)
    for idx in (0, 1):
        for chunk in frame_chunks_all[idx]:
            direct_valid[idx, np.asarray(chunk, dtype=np.int64)] = True

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args_ns, start_idx, end_idx)
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, _infilled_valid = hawor_infiller(
        args_ns, start_idx, end_idx, frame_chunks_all)

    return (imgfiles, pred_trans, pred_rot, pred_hand_pose, pred_betas,
            direct_valid, R_w2c, t_w2c, img_focal, detection_stats)


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

    ckpt = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else \
        hawor_repo / "weights" / "hawor" / "checkpoints" / "hawor.ckpt"
    infiller = Path(args.infiller_weight).expanduser().resolve() if args.infiller_weight else \
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
                    out_root / "faces_left.npy", out_root / "config.json",
                    out_root / "overlay.mp4"):
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
    if args.overlay_fps <= 0:
        sys.exit("error: --overlay-fps must be greater than zero")
    img0 = cv2.imread(str(frame_paths[0]))
    H, W = img0.shape[:2]
    img_size_wh = [int(W), int(H)]

    # Focal resolution: explicit > DA3 median > HaWoR's own estimate.
    da3_focal_map = {} if args.ignore_da3_focal else load_da3_focal_map(scene_dir)
    if args.img_focal is not None:
        img_focal_in, focal_source = float(args.img_focal), "provided"
    elif da3_focal_map:
        img_focal_in = float(np.median(list(da3_focal_map.values())))
        focal_source = "da3_median"
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
         pred_valid, R_w2c, t_w2c, img_focal, detection_stats) = run_hawor_pipeline(
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

    # Shared MANO topology (standard right-hand winding -> matches WiLoR).
    faces = np.asarray(get_mano_faces(), dtype=np.int32)
    np.save(out_root / "faces.npy", faces)
    np.save(out_root / "faces_left.npy", faces[:, [0, 2, 1]])

    cam_pose_map = {} if args.no_world else load_cam_poses(scene_dir)
    print(f"camera poses available: {len(cam_pose_map)} frame(s) "
          f"{'(world-frame output enabled)' if cam_pose_map else '(camera-frame only)'}")

    side_name = {0: "left", 1: "right"}
    candidates = [idx for idx in (0, 1)
                  if detection_stats[idx]["detected_frames"] > 0]
    if not candidates:
        sys.exit("error: HaWoR did not detect either hand in any frame")
    if args.hand == "auto":
        selected = max(candidates, key=lambda idx: (
            detection_stats[idx]["detected_frames"],
            detection_stats[idx]["mean_confidence"],
        ))
        selection_method = "most_detected_frames_then_mean_confidence"
    else:
        selected = 0 if args.hand == "left" else 1
        if selected not in candidates:
            sys.exit(f"error: requested --hand {args.hand}, but HaWoR did not "
                     "detect that hand in any frame")
        selection_method = "user_selected"
    active = [selected]
    print("hand detections: " + ", ".join(
        f"{side_name[idx]}={detection_stats[idx]['detected_frames']} frames, "
        f"mean confidence {detection_stats[idx]['mean_confidence']:.3f}"
        for idx in (0, 1)))
    print(f"selected hand: {side_name[selected]}")

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

    overlay_path = out_root / "overlay.mp4"
    overlay_frames = 0
    if not args.no_overlay:
        frame_to_t = {frame_idx: t for t, frame_idx in enumerate(frame_indices)}
        overlay_frames = write_overlay_video(
            overlay_path, frame_paths, frame_to_t,
            cam_cache[selected][0], cam_cache[selected][1], pred_valid,
            selected, img_focal, args.overlay_fps, save_start, save_end,
            args.detected_only, W, H)
        print(f"overlay video: {overlay_path} ({overlay_frames} frames at "
              f"{args.overlay_fps:g} fps)")

    elapsed = time.time() - t_start

    config = {
        "scene_id": scene_dir.name,
        "stage": "12_hawor_hands",
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
        "requested_hand": args.hand,
        "selected_hand": side_name[selected],
        "active_hands": [side_name[selected]],
        "hand_selection": selection_method,
        "hand_detection_stats": {
            side_name[idx]: detection_stats[idx] for idx in (0, 1)
        },
        "world_frame_baked": bool(cam_pose_map),
        "n_frames_with_hands": int(n_with_hands),
        "total_hands": int(total_hands),
        "overlay_video": None if args.no_overlay else str(overlay_path),
        "overlay_fps": None if args.no_overlay else float(args.overlay_fps),
        "overlay_frames": int(overlay_frames),
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
