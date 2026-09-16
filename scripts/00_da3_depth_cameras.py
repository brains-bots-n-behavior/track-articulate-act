#!/usr/bin/env python
"""Stage 00: Depth-Anything-3 depth + camera/world frame.

Runs DA3 once over the clip and writes the depth/camera products consumed by
downstream stages. Pair it with the TrackCraft3R stage 08,
`08_trackcraft_flow.py`, for point tracking. The output schema lets
mesh alignment, HaWoR world-baking, and legacy replay/Any6D consumers
read the shared geometry products.

Adapted from TrackCraft3r/scripts/preprocess_da3.py, but instead of dumping
raw depth/extrinsics/intrinsics NPYs it emits the pipeline's shared schema.

Produces (under data/<scene>/da3/ by default):
    config.json                       # ref_frame, resolutions, source=da3
    cameras.npz                       # cam_quats_xyzw, cam_trans, intrinsics
                                      #   quats are CAMERA-TO-WORLD, XYZW order
                                      #   (matches quat_xyzw_to_R in stages 11/12)
    frame_indices.npy                 # which frame each row of cameras.npz is
    pointmap_ref.npy                  # dense (H, W, 3) ref pointmap, world coords
    intrinsics.npz                    # per-frame K (FULL resolution, pixel units)
    depth/<frame>.npy                 # (H, W) float32 z-depth, full resolution

NOT produced (DA3 has no temporal correspondence):
    <label>/scene_flow, <label>/pts3d_ref, <label>/pixel_ij
Point-track joint estimation needs scene flow from the tracking stage. Stages
that only need depth + camera/world use these DA3 products directly.

Conventions (verified against consumers):
  * Depth is z-depth in the RDF camera frame, same as MoGe (11_align_meshes.py
    back_project treats it as z). DA3 emits z-depth directly.
  * DA3's extrinsics are WORLD-TO-CAMERA (see preprocess_da3.py header); the
    pipeline stores CAMERA-TO-WORLD (12_hawor_hands.py builds R_c2w/t_c2w from
    cam_quats_xyzw/cam_trans and does p_world = R_c2w @ p_cam + t_c2w). We
    invert here so the stored pose is C2W.
  * The world frame is DA3's own (frame-0 not re-normalized). It is arbitrary
    but self-consistent: every stage bakes into this same frame.

Scale caveat: DA3 depth/pose are not guaranteed to match MoGe's metric scale.
Consumers that reason about a consistent scene scale are fine. Legacy
Any6D pose estimation expects *metric* depth — sanity-check the recovered translations if you
feed it a DA3 bundle.

Setup (one-time):
    git clone https://github.com/ByteDance-Seed/depth-anything-3
    cd depth-anything-3 && pip install -e .

Example:
    python scripts/00_da3_depth_cameras.py \\
        --scene-dir data/kitchen_pour_01 \\
        --da3-root  /home/jeremy/research/Articulate4D/depth-anything-3 \\
        --ref-frame 42 \\
        --overwrite
"""

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/ (reads frames/*.jpg)")
    p.add_argument("--da3-root", type=Path, default=None,
                   help="Path to the depth-anything-3 checkout (added to sys.path). "
                        "Skip if depth_anything_3 is pip-installed.")
    p.add_argument("--model-name", type=str,
                   default="depth-anything/DA3NESTED-GIANT-LARGE",
                   help="DA3 hub model id. Smaller: da3-large / da3-base.")
    p.add_argument("--process-res", type=int, default=504,
                   help="DA3 processing resolution (upper-bound resize).")
    p.add_argument("--ref-frame", type=int, default=0,
                   help="Frame index used as the reference for pointmap_ref / "
                        "config.ref_frame (default: first processed frame).")
    p.add_argument("--start-idx", type=int, default=None,
                   help="Inclusive start frame index (default: 0).")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Exclusive end frame index (default: end of clip).")
    p.add_argument("--out-name", type=str, default="da3",
                   help="Output subfolder under the scene dir (default: da3).")
    p.add_argument("--no-pointmap-ref", action="store_true",
                   help="Skip writing the dense pointmap_ref.npy (saves disk; "
                        "legacy replay utilities need it; alignment and HaWoR do not).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace the existing output folder.")
    return p.parse_args()


def R_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w), numerically stable.

    Inverse of quat_xyzw_to_R used in stages 11/12."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float32)
    return q / np.linalg.norm(q)


def unproject_to_world(depth, K, R_c2w, t_c2w):
    """Dense back-projection of a z-depth map into world coords.

    Returns (H, W, 3) float32; pixels with invalid depth are set to 0."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us, vs = np.meshgrid(np.arange(W), np.arange(H))          # (H, W)
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)                    # (H, W, 3)
    pts_world = pts_cam @ R_c2w.T + t_c2w                     # (H, W, 3)
    pts_world[~valid] = 0.0
    return pts_world.astype(np.float32)


def main():
    args = parse_args()

    scene_dir = args.scene_dir.resolve()
    frames_dir = scene_dir / "frames"
    out_root = scene_dir / args.out_name

    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        sys.exit(f"error: no *.jpg in {frames_dir}")
    n_frames = len(frame_paths)

    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else n_frames
    if not (0 <= start_idx < end_idx <= n_frames):
        sys.exit(f"error: bad range [{start_idx}, {end_idx}) for n_frames={n_frames}")
    frame_indices = list(range(start_idx, end_idx))
    if args.ref_frame not in frame_indices:
        sys.exit(f"error: --ref-frame {args.ref_frame} not in processed range "
                 f"[{start_idx}, {end_idx})")

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            sys.exit(f"error: {out_root} already exists; pass --overwrite to replace")
    out_root.mkdir(parents=True)

    # --- Load DA3 ---
    if args.da3_root:
        sys.path.insert(0, str(Path(args.da3_root).resolve() / "src"))
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        sys.exit("Failed to import depth_anything_3. Pass --da3-root <repo> "
                 "or `pip install -e .` from the depth-anything-3 repo.")

    pil_images = [Image.open(frame_paths[i]).convert("RGB") for i in frame_indices]
    T = len(pil_images)
    full_W, full_H = pil_images[0].size
    print(f"scene: {scene_dir.name}")
    print(f"frames: {T} at {full_W}x{full_H}  range [{start_idx}, {end_idx})")
    print(f"ref_frame: {args.ref_frame:06d}")

    print(f"loading {args.model_name} on {args.device} ...")
    model = DepthAnything3.from_pretrained(args.model_name).to(args.device).eval()

    print(f"running DA3 inference (process_res={args.process_res}) ...")
    with torch.no_grad():
        pred = model.inference(
            image=pil_images,
            process_res=args.process_res,
            process_res_method="upper_bound_resize",
            export_dir=None,
        )

    depth_proc = np.asarray(pred.depth)          # (T, H_proc, W_proc) z-depth
    extr_3x4 = np.asarray(pred.extrinsics)       # (T, 3, 4) W2C
    intr_proc = np.asarray(pred.intrinsics)      # (T, 3, 3) at proc res
    H_proc, W_proc = depth_proc.shape[1], depth_proc.shape[2]

    # --- Depth -> full resolution ---
    if (H_proc, W_proc) != (full_H, full_W):
        sh, sw = full_H / H_proc, full_W / W_proc
        depth = np.stack([zoom(depth_proc[t], (sh, sw), order=1) for t in range(T)]
                         ).astype(np.float32)
    else:
        depth = depth_proc.astype(np.float32)

    # --- Intrinsics -> full resolution, pixel units ---
    sx, sy = full_W / W_proc, full_H / H_proc
    intr = intr_proc.copy().astype(np.float32)
    intr[:, 0, 0] *= sx
    intr[:, 1, 1] *= sy
    intr[:, 0, 2] *= sx
    intr[:, 1, 2] *= sy

    # --- Extrinsics W2C -> camera-to-world quats (XYZW) + translation ---
    cam_quats, cam_trans = [], []
    R_c2w_list, t_c2w_list = [], []
    for t in range(T):
        R_w2c = extr_3x4[t, :3, :3].astype(np.float64)
        t_w2c = extr_3x4[t, :3, 3].astype(np.float64)
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        cam_quats.append(R_to_quat_xyzw(R_c2w))
        cam_trans.append(t_c2w.astype(np.float32))
        R_c2w_list.append(R_c2w)
        t_c2w_list.append(t_c2w)

    # --- Write per-frame depth and intrinsics directly under da3/ ---
    depth_dir = out_root / "depth"
    depth_dir.mkdir(parents=True)
    for t, fi in enumerate(frame_indices):
        np.save(depth_dir / f"{fi:06d}.npy", depth[t])
    np.savez(
        out_root / "intrinsics.npz",
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        intrinsics=intr,
    )

    # --- Write cameras.npz + frame_indices.npy ---
    np.savez(
        out_root / "cameras.npz",
        frame_indices=np.asarray(frame_indices, dtype=np.int32),
        cam_quats_xyzw=np.stack(cam_quats).astype(np.float32),
        cam_trans=np.stack(cam_trans).astype(np.float32),
        intrinsics=intr,   # FULL resolution (model res == full res for DA3)
    )
    np.save(out_root / "frame_indices.npy", np.asarray(frame_indices, dtype=np.int32))

    # --- Dense ref pointmap in world coords ---
    if not args.no_pointmap_ref:
        r = frame_indices.index(args.ref_frame)
        pointmap_ref = unproject_to_world(depth[r], intr[r], R_c2w_list[r], t_c2w_list[r])
        np.save(out_root / "pointmap_ref.npy", pointmap_ref)

    # --- config.json (model res == full res for DA3) ---
    config = {
        "scene_id": scene_dir.name,
        "source": "da3",
        "da3_model": args.model_name,
        "process_res": int(args.process_res),
        "ref_frame": int(args.ref_frame),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "stride": 1,
        "n_target_frames": int(T - 1),
        "model_resolution_wh": [int(full_W), int(full_H)],
        "full_resolution_wh": [int(full_W), int(full_H)],
        "labels_processed": [],
        "has_scene_flow": False,
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nwrote {out_root}/")
    print(f"  depth:      {depth.shape}  (z-depth, full res)")
    print(f"  cameras:    {len(cam_quats)} poses (cam->world, XYZW quats)")
    print(f"  intrinsics: full res, fx_fy_cx_cy[ref]="
          f"[{intr[r if not args.no_pointmap_ref else 0, 0, 0]:.1f}, "
          f"{intr[0, 1, 1]:.1f}, {intr[0, 0, 2]:.1f}, {intr[0, 1, 2]:.1f}]")
    print("  NOTE: DA3 does not produce scene flow; run the tracking stage for it.")
    print("done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
