#!/usr/bin/env python
"""Stage 32: estimate per-object 6D pose with Any6D.

Adapts Any6D/run_pose.py (the SAM2/InstantMesh-free driver) to the
Articulate4D scene layout. For each label with a stage-30 mesh.glb, Any6D
registers the mesh to the masked metric pointcloud at the label's keyframe and
returns the object->camera pose (and rescales the mesh to metric size via its
oriented-bounding-box ratio fit).

Per-label inputs, all already produced by earlier stages:
    sam3d/<label>/[cand_NN_<kf>/]{mesh.glb, keyframe.txt}   (stage 30)
    frames/<kf>.jpg                                          (stage 00)
    masks/<label>/<kf>.png                                   (stage 10)
    any4d/moge/depth/<kf>.npy                                (stage 40; METRIC z-depth, meters)
    any4d/moge/intrinsics.npz                                (stage 40; full-res pixel K)
    any4d/moge/mask/<kf>.png                                 (stage 40; MoGe valid-depth mask, optional)
    any4d/cameras.npz                                        (stage 40; optional, enables world-frame pose)

Unlike Any6D's demo, the MoGe depth is ALREADY in meters, so there is no
`depth_scale` divisor — the .npy is fed straight to register_any6d.

Outputs (data/<scene>/any6d/<label>/):
    pose.txt            4x4 object->camera transform (Any6D's raw output)
    pose.json           structured pose: object->camera, object->world (if a
                        camera pose is known), keyframe, K, provenance
    K.txt               the 3x3 intrinsics used
    final_mesh.glb      Any6D's metric-rescaled mesh, in its own object frame
    mesh_world.glb      that mesh placed by the estimated pose into the Any4D
                        world frame (if any4d/cameras.npz exists), else
                        mesh_cam.glb in the camera frame — handy for stage-51
                        style overlays

Run inside the Any6D conda env (needs a GPU: nvdiffrast / open3d / FoundationPose).

Headless: this script opens no GUI and renders nothing to a display. Any6D's
refiner uses an offscreen CUDA rasterizer (nvdiffrast RasterizeCudaContext, not
a GL/display context) and its Open3D draw_geometries calls are disabled, so it
runs on a render-less server. All results are written as plain data files under
data/<scene>/any6d/<label>/ (poses + .glb meshes) for transfer to a workstation
for rendering/visualization. Leave --debug at its default of 0.

Examples:
    python scripts/32_any6d_pose.py \\
        --scene-dir data/dryer \\
        --any6d-repo /home/jeremy/research/Articulate4D/Any6D

    # One label, more refiner iterations, a specific sam3d candidate
    python scripts/32_any6d_pose.py \\
        --scene-dir data/dryer \\
        --any6d-repo /home/jeremy/research/Articulate4D/Any6D \\
        --labels dryer_door --candidate 1 --iteration 8 --overwrite
"""

import argparse
import json
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image


# ---------------------------------------------------------------------------
# Geometry helper (same XYZW-quaternion convention as stages 50/52/60)
# ---------------------------------------------------------------------------


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def cam_pose_at(cams_npz, frame_idx: int):
    """Return camera->world (R, t) for one frame, or None. Matches stage 52."""
    fis = list(cams_npz["frame_indices"].tolist())
    if frame_idx not in fis:
        return None
    i = fis.index(frame_idx)
    R = quat_xyzw_to_R(cams_npz["cam_quats_xyzw"][i])
    t = cams_npz["cam_trans"][i].astype(np.float64)
    return R, t


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Mesh / mask / depth IO (same conventions as stages 52 / 31)
# ---------------------------------------------------------------------------


def find_mesh_dir(label_dir: Path, candidate_idx: int):
    """Return (mesh_path, keyframe) for one label.

    Tries label_dir/mesh.glb (single-candidate stage-30 layout), then
    label_dir/cand_<candidate_idx:02d>_*/mesh.glb, then the first cand_*.
    """
    direct = label_dir / "mesh.glb"
    if direct.is_file():
        kf_txt = label_dir / "keyframe.txt"
        kf = int(kf_txt.read_text().strip()) if kf_txt.is_file() else None
        return direct, kf

    matches = sorted(label_dir.glob(f"cand_{candidate_idx:02d}_*"))
    if not matches:
        matches = sorted(label_dir.glob("cand_*"))
    for c in matches:
        mp = c / "mesh.glb"
        if mp.is_file():
            kf_txt = c / "keyframe.txt"
            kf = int(kf_txt.read_text().strip()) if kf_txt.is_file() else None
            if kf is None:
                try:
                    kf = int(c.name.split("_")[-1])
                except ValueError:
                    kf = None
            return mp, kf
    return None, None


def load_mask_bool(path: Path, target_hw):
    """Read an 8-bit mask, threshold > 0, resize (nearest) to (H, W)."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[..., -1]
    m = m > 0
    H, W = target_hw
    if m.shape != (H, W):
        m_u8 = (m.astype(np.uint8) * 255)
        m_u8 = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        m = m_u8 > 0
    return m


def sanitize_depth(depth: np.ndarray) -> np.ndarray:
    """MoGe depth can carry NaN/inf outside the valid region. Any6D runs an
    erode + bilateral filter on the FULL frame before masking, so replace
    non-finite / negative values with 0 (Any6D's convention for 'no depth')."""
    d = depth.astype(np.float32).copy()
    d[~np.isfinite(d)] = 0.0
    d[d < 0] = 0.0
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--any6d-repo", type=Path, required=True,
                   help="Path to the Any6D/ checkout (provides estimater.py + weights)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only estimate these labels (default: all sam3d labels with a mesh)")
    p.add_argument("--candidate", type=int, default=0,
                   help="sam3d candidate index when stage 30 ran --all-candidates (default 0)")
    p.add_argument("--iteration", type=int, default=5,
                   help="Any6D refiner iterations (default 5)")
    p.add_argument("--no-refinement", action="store_true",
                   help="Skip Any6D's render-and-compare refinement (coarse OBB fit only)")
    p.add_argument("--no-axis-align", action="store_true",
                   help="Disable the coarse OBB axis alignment inside register_any6d")
    p.add_argument("--no-coarse", action="store_true",
                   help="Disable the coarse OBB scale/pose initialization")
    p.add_argument("--debug", type=int, default=0,
                   help="Any6D debug level. Default 0 = headless: only the pose + "
                        "mesh data files (below) are written, no intermediate "
                        "render-compare artifacts. Any6D never opens a GUI window "
                        "(its draw_geometries calls are disabled), so >0 is still "
                        "disk-only — leave it at 0 on a render-less server to keep "
                        "the output clean.")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing any6d/<label>/ outputs")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    any6d_repo = args.any6d_repo.resolve()

    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"
    sam3d_root = scene_dir / "sam3d"
    moge_root = scene_dir / "any4d" / "moge"
    out_root = scene_dir / "any6d"

    for need in (frames_dir, masks_root, sam3d_root,
                 moge_root / "depth", moge_root / "intrinsics.npz"):
        if not need.exists():
            sys.exit(f"error: missing {need} (run stages 10/30/40 first)")
    if not (any6d_repo / "estimater.py").is_file():
        sys.exit(f"error: {any6d_repo}/estimater.py not found — is --any6d-repo correct?")

    # Discover labels (any sam3d label that has a mesh; skip stage-31 previews)
    label_dirs = sorted(d for d in sam3d_root.iterdir()
                        if d.is_dir() and d.name != "_previews")
    labels = [d.name for d in label_dirs]
    if args.labels:
        unknown = [l for l in args.labels if l not in labels]
        if unknown:
            sys.exit(f"error: --labels not present under sam3d/: {unknown}")
        labels = [l for l in labels if l in args.labels]
    if not labels:
        sys.exit("error: nothing to process")

    out_root.mkdir(parents=True, exist_ok=True)

    # Intrinsics + (optional) camera poses, loaded once
    intr_npz = np.load(moge_root / "intrinsics.npz")
    intr_frames = list(intr_npz["frame_indices"].tolist())
    intr_arr = intr_npz["intrinsics"]
    cams_path = scene_dir / "any4d" / "cameras.npz"
    cams_npz = np.load(cams_path) if cams_path.is_file() else None

    # Any6D imports are heavy (torch + nvdiffrast + open3d + FoundationPose);
    # defer until after argument validation. The repo on sys.path is enough —
    # FoundationPose resolves its weights relative to its own __file__.
    set_seed(0)
    sys.path.insert(0, str(any6d_repo))
    try:
        import nvdiffrast.torch as dr
        from estimater import Any6D
    except ImportError as e:
        missing = getattr(e, "name", None)
        hint = (f"missing module '{missing}' — `python -m pip install {missing}` "
                f"(keep numpy pinned: append \"numpy==1.26.4\"). FoundationPose "
                f"pulls in a few deps not in requirements-pose.txt, e.g. psutil "
                f"and warp (pip name `warp-lang`)." if missing else
                "check that the Any6D conda env is active and fully installed.")
        sys.exit(f"error: failed to import Any6D dependencies ({e}).\n       {hint}")

    # Offscreen CUDA rasterizer used by Any6D's render-and-compare refiner.
    # This is a pure-CUDA context (NOT RasterizeGLContext) — no X / EGL display
    # is required, so it runs on a headless server.
    glctx = dr.RasterizeCudaContext()

    print(f"scene:  {scene_dir.name}")
    print(f"labels: {labels}")
    print(f"world-frame baking: "
          f"{'on (any4d/cameras.npz found)' if cams_npz is not None else 'off (no cameras.npz)'}\n")

    summary = {}
    for label in labels:
        try:
            print(f"[{label}]")
            mesh_path, kf = find_mesh_dir(sam3d_root / label, args.candidate)
            if mesh_path is None:
                print(f"  SKIP: no mesh.glb")
                continue
            if kf is None:
                print(f"  SKIP: no keyframe recorded for {mesh_path}")
                continue
            print(f"  mesh = {mesh_path.relative_to(scene_dir)}  kf = {kf:06d}")

            out_label = out_root / label
            if out_label.exists():
                if args.overwrite:
                    shutil.rmtree(out_label)
                else:
                    print(f"  SKIP: {out_label} exists (pass --overwrite)")
                    continue
            out_label.mkdir(parents=True)

            # --- inputs ---
            rgb_path = frames_dir / f"{kf:06d}.jpg"
            mask_path = masks_root / label / f"{kf:06d}.png"
            depth_path = moge_root / "depth" / f"{kf:06d}.npy"
            if not rgb_path.is_file():
                print(f"  SKIP: missing {rgb_path}")
                continue
            if not mask_path.is_file():
                print(f"  SKIP: missing {mask_path}")
                continue
            if not depth_path.is_file():
                print(f"  SKIP: missing {depth_path}")
                continue
            if kf not in intr_frames:
                print(f"  SKIP: keyframe {kf} not in MoGe intrinsics")
                continue

            color = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
            H, W = color.shape[:2]
            depth = sanitize_depth(np.load(depth_path))      # METRIC z-depth (m)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            K = intr_arr[intr_frames.index(kf)].astype(np.float64).copy()

            mask = load_mask_bool(mask_path, (H, W))
            # Only trust pixels with valid metric depth: drop holes/zeros and,
            # if present, intersect with MoGe's own valid-depth mask. Feeding
            # depth==0 pixels inside ob_mask would inject (0,0,0) points.
            valid = mask & np.isfinite(depth) & (depth > 0)
            moge_mask_path = moge_root / "mask" / f"{kf:06d}.png"
            if moge_mask_path.is_file():
                moge_valid = load_mask_bool(moge_mask_path, (H, W))
                valid &= moge_valid
            n_valid = int(valid.sum())
            if n_valid < 50:
                print(f"  SKIP: only {n_valid} valid masked depth pixels "
                      f"(need >= 50) — check mask / MoGe depth at this frame")
                continue
            print(f"  valid masked depth pixels: {n_valid}")

            # --- mesh ---
            mesh = trimesh.load(str(mesh_path), force="mesh")

            # --- Any6D registration ---
            est = Any6D(symmetry_tfs=None, mesh=mesh, glctx=glctx,
                        debug_dir=str(out_label), debug=args.debug)
            pred_pose = est.register_any6d(
                K=K, rgb=color, depth=depth, ob_mask=valid,
                iteration=args.iteration, name=label,
                refinement=not args.no_refinement,
                axis_align=not args.no_axis_align,
                coarse_est=not args.no_coarse,
            )
            pred_pose = np.asarray(pred_pose, dtype=np.float64).reshape(4, 4)

            # --- save: raw pose + intrinsics + metric-rescaled mesh ---
            np.savetxt(out_label / "pose.txt", pred_pose)
            np.savetxt(out_label / "K.txt", K)
            est.mesh.export(str(out_label / "final_mesh.glb"))

            # --- optional: bake camera->world so the pose lands in Any4D world ---
            pose_record = {
                "label": label,
                "keyframe": int(kf),
                "mesh_source": str(mesh_path.relative_to(scene_dir)),
                "iteration": int(args.iteration),
                "refinement": not args.no_refinement,
                "axis_align": not args.no_axis_align,
                "coarse_est": not args.no_coarse,
                "K": K.tolist(),
                "image_wh": [int(W), int(H)],
                "n_valid_pixels": n_valid,
                "pose_object_to_camera": pred_pose.tolist(),
            }

            posed_world_mesh = None
            cw = cam_pose_at(cams_npz, kf) if cams_npz is not None else None
            if cw is not None:
                R_cw, t_cw = cw
                T_cw = np.eye(4)
                T_cw[:3, :3] = R_cw
                T_cw[:3, 3] = t_cw
                pose_world = T_cw @ pred_pose          # object -> world
                pose_record["cam_to_world"] = {
                    "rotation_R": R_cw.tolist(),
                    "translation": t_cw.tolist(),
                }
                pose_record["pose_object_to_world"] = pose_world.tolist()

                v = np.asarray(est.mesh.vertices, dtype=np.float64)
                v_world = v @ pose_world[:3, :3].T + pose_world[:3, 3]
                posed_world_mesh = est.mesh.copy()
                posed_world_mesh.vertices = v_world
                posed_world_mesh.export(str(out_label / "mesh_world.glb"))
            else:
                # No camera pose — at least dump the mesh placed in camera frame.
                v = np.asarray(est.mesh.vertices, dtype=np.float64)
                v_cam = v @ pred_pose[:3, :3].T + pred_pose[:3, 3]
                cam_mesh = est.mesh.copy()
                cam_mesh.vertices = v_cam
                cam_mesh.export(str(out_label / "mesh_cam.glb"))

            with open(out_label / "pose.json", "w") as f:
                json.dump(pose_record, f, indent=2)

            t = pred_pose[:3, 3]
            print(f"  pose t(cam) = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] m"
                  + ("  + world-frame pose baked" if cw is not None else "")
                  + f"\n  -> wrote {out_label.relative_to(scene_dir)}/\n")
            summary[label] = {
                "keyframe": int(kf),
                "t_cam": t.tolist(),
                "world_baked": cw is not None,
            }
        except Exception as e:
            print(f"  FAIL ({type(e).__name__}: {e})")
            traceback.print_exc()
            continue

    print("done.")
    if summary:
        print("summary:")
        for lbl, s in summary.items():
            print(f"  {lbl:16s} kf={s['keyframe']:06d}  "
                  f"t_cam={[round(x, 3) for x in s['t_cam']]}  "
                  f"world={'yes' if s['world_baked'] else 'no'}")


if __name__ == "__main__":
    main()
