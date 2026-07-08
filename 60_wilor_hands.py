#!/usr/bin/env python
"""Stage 60: per-frame WiLoR hand tracking (MANO mesh + 3D keypoints).

For each frame in data/<scene>/frames/, runs the YOLO hand detector + WiLoR
fit, saves a compact per-frame npz with the mesh and 3D keypoints already in
camera frame (RDF, matching MoGe and Any4D). If any4d/cameras.npz exists,
also bakes the camera-to-world transform so a `verts_world` / `joints_world`
field is ready for the stage-51 viewer.

Reads:
    data/<scene>/frames/*.jpg
    data/<scene>/any4d/cameras.npz   (optional, enables world-frame output)
    <wilor-repo>/pretrained_models/{wilor_final.ckpt, model_config.yaml, detector.pt}

Writes:
    data/<scene>/wilor/
        config.json                       # run params + summary stats
        faces.npy                         # (Nf, 3) MANO faces, shared
        per_frame/<frame_idx>.npz         # only frames where YOLO found a hand
            verts          (n_hands, 778, 3)   float32  — camera frame, cam_t applied
            joints         (n_hands, 21, 3)    float32  — camera frame
            is_right       (n_hands,) bool
            cam_t          (n_hands, 3)        float32
            bbox           (n_hands, 4)        float32  — image-space (xyxy)
            yolo_conf      (n_hands,)          float32
            focal_length   ()                  float32
            img_size_wh    (2,)                int32
            frame_idx      ()                  int32
            (verts_world)  (n_hands, 778, 3)   float32  — only if cam pose known
            (joints_world) (n_hands, 21, 3)    float32

Run inside the `wilor` conda env (torch 2.12+cu130 on dsailogin per the
project notes).

Examples:
    python scripts/60_wilor_hands.py \\
        --scene-dir data/macbook-all \\
        --wilor-repo /home/jeremy/research/Articulate4D/WiLoR

    # Subrange + faster inference
    python scripts/60_wilor_hands.py \\
        --scene-dir data/macbook-all \\
        --wilor-repo /home/jeremy/research/Articulate4D/WiLoR \\
        --start-idx 20 --end-idx 80 --fast --overwrite

    # Cap to top-2 hands per frame, also dump .obj for ad-hoc viewing
    python scripts/60_wilor_hands.py \\
        --scene-dir data/macbook-all \\
        --wilor-repo /home/jeremy/research/Articulate4D/WiLoR \\
        --max-hands 2 --save-obj --overwrite
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry helper (same convention as the other stages — XYZW quaternion)
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
    """Return dict {frame_idx -> focal_px} from any4d/moge/intrinsics.npz, or {}.

    MoGe writes full-image-resolution intrinsics (same resolution WiLoR runs
    on), so K[0, 0] is the real focal length in pixels for that frame. This
    is what we want to feed into cam_crop_to_full — using WiLoR's nominal
    `EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * img_max` is a training-time
    constant unrelated to the actual camera, and produces depths off by
    roughly focal_wilor / focal_real (~30× for a typical 1920-px capture)."""
    p = scene_dir / "any4d" / "moge" / "intrinsics.npz"
    if not p.is_file():
        return {}
    d = np.load(p)
    return {int(fi): float(d["intrinsics"][i][0, 0])
            for i, fi in enumerate(d["frame_indices"].tolist())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--wilor-repo", type=Path, required=True,
                   help="Path to the WiLoR/ checkout (needs pretrained_models/)")
    p.add_argument("--checkpoint", type=str, default="wilor_final.ckpt",
                   help="WiLoR checkpoint filename under pretrained_models/")
    p.add_argument("--model-config", type=str, default="model_config.yaml",
                   help="Model config filename under pretrained_models/")
    p.add_argument("--detector", type=str, default="detector.pt",
                   help="YOLO hand-detector filename under pretrained_models/")
    p.add_argument("--start-idx", type=int, default=None,
                   help="Inclusive start frame index (default: 0 = beginning)")
    p.add_argument("--end-idx", type=int, default=None,
                   help="Exclusive end frame index (default: end of clip)")
    p.add_argument("--detector-conf", type=float, default=0.3,
                   help="YOLO confidence threshold (default 0.3)")
    p.add_argument("--rescale-factor", type=float, default=2.0,
                   help="Bbox padding factor for the WiLoR crop (default 2.0)")
    p.add_argument("--max-hands", type=int, default=None,
                   help="Cap to top-N detections per frame (by YOLO conf)")
    p.add_argument("--batch-size", type=int, default=16,
                   help="WiLoR fit batch size within a single frame (default 16)")
    p.add_argument("--fast", action="store_true",
                   help="FP16 + torch.compile + layer dropping for speed")
    p.add_argument("--no-world", action="store_true",
                   help="Skip applying Any4D cam2world; save camera frame only")
    p.add_argument("--ignore-moge-focal", action="store_true",
                   help="Use WiLoR's nominal focal "
                        "(EXTRA.FOCAL_LENGTH / MODEL.IMAGE_SIZE * img_max) "
                        "even when any4d/moge/intrinsics.npz is available. "
                        "WiLoR's nominal focal is a training-time constant, "
                        "not the actual camera focal, and over-estimates "
                        "depth by ~30x on typical 1920-px input. Default is "
                        "to use the MoGe per-frame focal whenever present.")
    p.add_argument("--save-obj", action="store_true",
                   help="Also write per-hand .obj files (compatibility with "
                        "WiLoR/demo.py's output layout)")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing wilor/ outputs")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    wilor_repo = args.wilor_repo.resolve()

    frames_dir = scene_dir / "frames"
    out_root = scene_dir / "wilor"

    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    ckpt_path = wilor_repo / "pretrained_models" / args.checkpoint
    cfg_path = wilor_repo / "pretrained_models" / args.model_config
    det_path = wilor_repo / "pretrained_models" / args.detector
    for p in (ckpt_path, cfg_path, det_path):
        if not p.is_file():
            sys.exit(f"error: missing {p}")

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            sys.exit(f"error: {out_root} exists; pass --overwrite")
    out_root.mkdir(parents=True)
    per_frame_dir = out_root / "per_frame"
    per_frame_dir.mkdir()
    obj_dir = out_root / "obj"
    if args.save_obj:
        obj_dir.mkdir()

    # Defer WiLoR imports until after path setup — they need the repo on sys.path
    sys.path.insert(0, str(wilor_repo))
    try:
        from wilor.models import load_wilor                     # noqa: E402
        from wilor.utils import recursive_to                    # noqa: E402
        from wilor.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
        from wilor.utils.renderer import cam_crop_to_full        # noqa: E402
        if args.save_obj:
            from wilor.utils.renderer import Renderer            # noqa: E402
        from ultralytics import YOLO                             # noqa: E402
    except ImportError as e:
        sys.exit(f"error: failed to import WiLoR dependencies ({e}). "
                 f"Activate the wilor conda env and retry.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"scene:        {scene_dir.name}")
    print(f"device:       {device}")
    print(f"wilor_repo:   {wilor_repo}")

    print("loading WiLoR + YOLO ...")
    model, model_cfg = load_wilor(checkpoint_path=str(ckpt_path),
                                  cfg_path=str(cfg_path))
    if args.fast:
        torch.set_float32_matmul_precision("high")
        model = model.half()
        model.backbone = torch.compile(model.backbone)
        model.backbone.skip_blocks = True

    detector = YOLO(str(det_path))
    model = model.to(device).eval()
    detector = detector.to(device)

    # Save the shared MANO face topology once
    faces = np.asarray(model.mano.faces, dtype=np.int32)
    np.save(out_root / "faces.npy", faces)
    print(f"saved faces.npy ({faces.shape})")

    cam_pose_map = {} if args.no_world else load_cam_poses(scene_dir)
    print(f"camera poses available: {len(cam_pose_map)} frame(s) "
          f"{'(world-frame output enabled)' if cam_pose_map else '(camera-frame only)'}")

    moge_focal_map = {} if args.ignore_moge_focal else load_moge_focal_map(scene_dir)
    if moge_focal_map:
        sample_focal = next(iter(moge_focal_map.values()))
        print(f"MoGe focals available: {len(moge_focal_map)} frame(s); "
              f"example fx={sample_focal:.1f} px — using these instead of "
              f"WiLoR's nominal focal so cam_t depth is metric-correct")
    else:
        print("MoGe focals NOT available — falling back to WiLoR's nominal "
              "focal. Expect depths to be off by roughly focal_wilor / "
              "focal_real (~30x for typical 1920-px capture).")

    if args.save_obj:
        renderer = Renderer(model_cfg, faces=model.mano.faces)
        LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

    # Enumerate frames
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    n_total = len(frame_paths)
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else n_total
    sub_paths = frame_paths[start:end]
    print(f"processing {len(sub_paths)} frames [{start}, {end}) "
          f"of {n_total} total\n")

    summary = {"n_frames_processed": 0, "n_frames_with_hands": 0,
               "total_hands": 0, "frames_failed": []}
    img_size_wh = None
    focal_used = None

    t_start = time.time()
    for img_path in sub_paths:
        try:
            frame_idx = int(img_path.stem)
        except ValueError:
            print(f"  skip: non-integer name {img_path.name}")
            continue
        summary["n_frames_processed"] += 1

        try:
            img_cv2 = cv2.imread(str(img_path))
            if img_cv2 is None:
                print(f"  [{frame_idx:06d}] SKIP: failed to read")
                continue
            if img_size_wh is None:
                img_size_wh = [int(img_cv2.shape[1]), int(img_cv2.shape[0])]

            detections = detector(img_cv2, conf=args.detector_conf,
                                   verbose=False)[0]
            bboxes, is_right_list, confs = [], [], []
            for det in detections:
                arr = det.boxes.data.cpu().detach().squeeze().numpy()
                bboxes.append(arr[:4].astype(np.float32).tolist())
                is_right_list.append(
                    float(det.boxes.cls.cpu().detach().squeeze().item())
                )
                confs.append(
                    float(det.boxes.conf.cpu().detach().squeeze().item())
                )

            if not bboxes:
                continue

            if args.max_hands is not None and len(bboxes) > args.max_hands:
                order = np.argsort(confs)[::-1][:args.max_hands].tolist()
                bboxes = [bboxes[i] for i in order]
                is_right_list = [is_right_list[i] for i in order]
                confs = [confs[i] for i in order]

            boxes = np.stack(bboxes).astype(np.float32)
            right_arr = np.stack(is_right_list).astype(np.float32)
            dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right_arr,
                                     rescale_factor=args.rescale_factor,
                                     fp16=args.fast)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=0
            )

            all_verts, all_joints, all_cam_t, all_right_int = [], [], [], []
            scaled_focal_length = None
            for batch in loader:
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    out = model(batch)

                multiplier = (2 * batch["right"] - 1)
                pred_cam = out["pred_cam"]
                pred_cam[:, 1] = multiplier * pred_cam[:, 1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size_t = batch["img_size"].float()
                if frame_idx in moge_focal_map:
                    scaled_focal_length = torch.tensor(
                        moge_focal_map[frame_idx],
                        device=img_size_t.device,
                        dtype=img_size_t.dtype,
                    )
                else:
                    scaled_focal_length = (model_cfg.EXTRA.FOCAL_LENGTH
                                            / model_cfg.MODEL.IMAGE_SIZE
                                            * img_size_t.max())
                pred_cam_t_full = cam_crop_to_full(
                    pred_cam, box_center, box_size,
                    img_size_t, scaled_focal_length
                ).detach().cpu().numpy()

                B = batch["img"].shape[0]
                for n in range(B):
                    v = out["pred_vertices"][n].detach().cpu().numpy()
                    j = out["pred_keypoints_3d"][n].detach().cpu().numpy()
                    ir = int(batch["right"][n].cpu().numpy())
                    # The MANO output is for the right hand; reflect across the
                    # YZ plane for left hands (same convention as demo.py).
                    v[:, 0] = (2 * ir - 1) * v[:, 0]
                    j[:, 0] = (2 * ir - 1) * j[:, 0]
                    cam_t = pred_cam_t_full[n].astype(np.float32)
                    # Apply cam_t to place mesh + joints in camera frame
                    all_verts.append((v + cam_t).astype(np.float32))
                    all_joints.append((j + cam_t).astype(np.float32))
                    all_cam_t.append(cam_t)
                    all_right_int.append(ir)

            if not all_verts:
                continue

            verts_cam = np.stack(all_verts)
            joints_cam = np.stack(all_joints)
            cam_t_arr = np.stack(all_cam_t)
            is_right_arr = np.array(all_right_int, dtype=bool)
            bbox_arr = np.array(bboxes, dtype=np.float32)
            conf_arr = np.array(confs, dtype=np.float32)
            focal_used = float(scaled_focal_length)

            save_dict = dict(
                verts=verts_cam,
                joints=joints_cam,
                is_right=is_right_arr,
                cam_t=cam_t_arr,
                bbox=bbox_arr,
                yolo_conf=conf_arr,
                focal_length=np.float32(focal_used),
                img_size_wh=np.asarray(img_size_wh, dtype=np.int32),
                frame_idx=np.int32(frame_idx),
            )

            # World-frame (Any4D) outputs
            if frame_idx in cam_pose_map:
                R_cw, t_cw = cam_pose_map[frame_idx]
                verts_w = verts_cam @ R_cw.T.astype(np.float32) + t_cw.astype(np.float32)
                joints_w = joints_cam @ R_cw.T.astype(np.float32) + t_cw.astype(np.float32)
                save_dict["verts_world"] = verts_w.astype(np.float32)
                save_dict["joints_world"] = joints_w.astype(np.float32)

            np.savez_compressed(
                per_frame_dir / f"{frame_idx:06d}.npz", **save_dict
            )

            if args.save_obj:
                # vertices_to_trimesh takes canonical verts (pre cam_t) + cam_t
                for n in range(len(all_verts)):
                    v_canon = verts_cam[n] - cam_t_arr[n]
                    tmesh = renderer.vertices_to_trimesh(
                        v_canon, cam_t_arr[n], LIGHT_PURPLE,
                        is_right=int(is_right_arr[n]),
                    )
                    tmesh.export(str(obj_dir / f"{frame_idx:06d}_{n}.obj"))

            summary["n_frames_with_hands"] += 1
            summary["total_hands"] += len(all_verts)
            sides = ("L" if not ir else "R" for ir in all_right_int)
            print(f"  [{frame_idx:06d}] {len(all_verts)} hand(s) "
                  f"[{','.join(sides)}]  conf={conf_arr.round(2).tolist()}")
        except Exception as e:
            summary["frames_failed"].append(int(frame_idx))
            print(f"  [{frame_idx:06d}] FAIL ({type(e).__name__}: {e})")
            traceback.print_exc()
            continue

    elapsed = time.time() - t_start

    config = {
        "scene_id": scene_dir.name,
        "wilor_repo": str(wilor_repo),
        "checkpoint": str(ckpt_path.relative_to(wilor_repo)),
        "start_idx": int(start),
        "end_idx": int(end),
        "detector_conf": float(args.detector_conf),
        "rescale_factor": float(args.rescale_factor),
        "max_hands": args.max_hands,
        "batch_size": int(args.batch_size),
        "fast": bool(args.fast),
        "img_size_wh": img_size_wh,
        "focal_length": focal_used,
        "world_frame_baked": bool(cam_pose_map),
        "n_frames_processed": int(summary["n_frames_processed"]),
        "n_frames_with_hands": int(summary["n_frames_with_hands"]),
        "total_hands": int(summary["total_hands"]),
        "frames_failed": summary["frames_failed"],
        "elapsed_s": float(elapsed),
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"done in {elapsed:.1f}s. "
          f"{summary['n_frames_with_hands']}/{summary['n_frames_processed']} "
          f"frames had hands; {summary['total_hands']} total detections.")
    print(f"wrote {out_root}")


if __name__ == "__main__":
    main()
