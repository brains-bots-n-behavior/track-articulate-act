#!/usr/bin/env python
"""Stage 52: align sam3d meshes to each label's keyframe image + mask.

Pipeline per label, using the keyframe recorded in keyframe.txt:
  1. Load mesh.glb (sam3d output, canonical frame).
  2. Apply pose.json as initial camera-frame transform:
         v_cam = scale * R(quat_wxyz) * v_canonical + translation
     pose.json is sam3d's own predicted alignment; ignoring it would throw
     away the rotation and start from canonical orientation.
  3. mesh_alignment.py-style coarse refinement against the MoGe pointmap:
     re-fit scale (Y-height ratio) + translation (centroid offset) using only
     MoGe points inside the label's mask at the keyframe.
  4. Reprojection refinement: optimize (delta_R, delta_t, delta_s) via
     Nelder-Mead with cost = (1 - IoU(rendered_silhouette, mask))
                            + lambda_icp * mean(dist to MoGe pointmap).
  5. Bake camera-to-world transform for the keyframe (from any4d/cameras.npz)
     so the saved mesh is directly in Any4D's world frame for stage 51 to
     render alongside the pointcloud / scene-flow / joint axes.

Inputs:
    data/<scene>/sam3d/<label>/[cand_NN_<kf>/]{mesh.glb, pose.json, keyframe.txt}
    data/<scene>/masks/<label>/<kf>.png
    data/<scene>/any4d/moge/depth/<kf>.npy
    data/<scene>/any4d/moge/intrinsics.npz
    data/<scene>/any4d/cameras.npz

Outputs:
    data/<scene>/aligned/<label>/
        mesh.glb            # aligned mesh in Any4D world frame
        align.json          # init / coarse / refined params + diagnostics

Run inside any env with numpy / opencv / trimesh / scipy.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

try:
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw_to_R(q):
    return quat_wxyz_to_R([q[3], q[0], q[1], q[2]])


def axis_angle_to_R(a):
    angle = float(np.linalg.norm(a))
    if angle < 1e-12:
        return np.eye(3)
    axis = a / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def apply_rts(verts: np.ndarray, R: np.ndarray, t: np.ndarray, s: float):
    return s * (verts @ R.T) + t


# ---------------------------------------------------------------------------
# Mesh / mask / depth IO
# ---------------------------------------------------------------------------


def find_mesh_dir(label_dir: Path, candidate_idx: int):
    """Return (mesh_path, pose_path, keyframe) for a label.

    Order of attempts:
        1. label_dir/mesh.glb   (single candidate written by stage 30)
        2. label_dir/cand_<candidate_idx:02d>_*/mesh.glb
        3. first cand_*/mesh.glb
    """
    direct = label_dir / "mesh.glb"
    if direct.is_file():
        pose = label_dir / "pose.json"
        kf_txt = label_dir / "keyframe.txt"
        kf = int(kf_txt.read_text().strip()) if kf_txt.is_file() else None
        return direct, pose if pose.is_file() else None, kf

    matches = sorted(label_dir.glob(f"cand_{candidate_idx:02d}_*"))
    if not matches:
        matches = sorted(label_dir.glob("cand_*"))
    for c in matches:
        mp = c / "mesh.glb"
        if mp.is_file():
            pose = c / "pose.json"
            kf_txt = c / "keyframe.txt"
            kf = int(kf_txt.read_text().strip()) if kf_txt.is_file() else None
            if kf is None:
                # Parse from folder name `cand_NN_<6-digit-frame>`
                parts = c.name.split("_")
                try:
                    kf = int(parts[-1])
                except ValueError:
                    kf = None
            return mp, pose if pose.is_file() else None, kf
    return None, None, None


def load_mesh(mesh_path: Path):
    """Return (vertices, faces) as float64 / int64 numpy arrays in the mesh's
    own canonical frame (no convention flip applied — pose.json handles that)."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    v = np.asarray(mesh.vertices, dtype=np.float64).copy()
    f = np.asarray(mesh.faces, dtype=np.int64).copy()
    return v, f


def load_pose(pose_path: Path):
    if pose_path is None:
        return None
    with open(pose_path) as f:
        d = json.load(f)
    return {
        "R": quat_wxyz_to_R(d["rotation_quat_wxyz"]),
        "t": np.asarray(d["translation"], dtype=np.float64).reshape(3),
        # sam3d's scale is always isotropic — take the first component
        "s": float(np.asarray(d["scale"]).reshape(-1)[0]),
    }


def load_mask(mask_path: Path, target_hw):
    if not mask_path.is_file():
        return None
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., -1]
    m_bool = m > 0
    H, W = target_hw
    if m_bool.shape != (H, W):
        m_u8 = (m_bool.astype(np.uint8) * 255)
        m_u8 = cv2.resize(m_u8, (W, H), interpolation=cv2.INTER_NEAREST)
        m_bool = m_u8 > 0
    return m_bool


def back_project(depth: np.ndarray, K: np.ndarray, mask: np.ndarray = None):
    """Back-project depth into 3D camera points (RDF). Returns (N, 3)."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= mask
    ys, xs = np.where(valid)
    z = depth[ys, xs].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def filter_flying_points(pts: np.ndarray) -> np.ndarray:
    """Drop the far-tail z outliers (per mesh_alignment.py heuristic)."""
    if pts.shape[0] == 0:
        return pts
    z_range = float(pts[:, 2].max() - pts[:, 2].min())
    if z_range > 6.0:
        q = 0.90
    elif z_range > 2.0:
        q = 0.93
    else:
        q = 0.95
    thr = float(np.quantile(pts[:, 2], q))
    pts = pts[pts[:, 2] <= thr]
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts


# ---------------------------------------------------------------------------
# Coarse refinement (mesh_alignment.py-style scale + translate)
# ---------------------------------------------------------------------------


def coarse_scale_translate(verts_cam: np.ndarray, target_points: np.ndarray):
    """Re-fit isotropic scale (Y-height ratio) + translation (centroid offset).
    Returns (verts_aligned, scale_factor, translation_delta)."""
    if target_points.shape[0] == 0:
        return verts_cam.copy(), 1.0, np.zeros(3)
    # Use the mesh's CURRENT extent (after pose.json applied)
    h_src = float(verts_cam[:, 1].max() - verts_cam[:, 1].min())
    h_tgt = float(target_points[:, 1].max() - target_points[:, 1].min())
    if h_src < 1e-9:
        return verts_cam.copy(), 1.0, np.zeros(3)
    s = h_tgt / h_src
    v_scaled = verts_cam * s
    c_src = v_scaled.mean(axis=0)
    c_tgt = target_points.mean(axis=0)
    t = c_tgt - c_src
    return v_scaled + t, s, t


# ---------------------------------------------------------------------------
# Silhouette rasterization (one cv2.fillPoly call for all visible faces)
# ---------------------------------------------------------------------------


def project_verts(verts_cam: np.ndarray, K: np.ndarray):
    Z = verts_cam[:, 2]
    valid = Z > 1e-3
    u = np.full(Z.shape, -1.0)
    v = np.full(Z.shape, -1.0)
    u[valid] = K[0, 0] * verts_cam[valid, 0] / Z[valid] + K[0, 2]
    v[valid] = K[1, 1] * verts_cam[valid, 1] / Z[valid] + K[1, 2]
    return np.stack([u, v], axis=1), valid


def render_silhouette(verts_cam: np.ndarray, faces: np.ndarray,
                      K: np.ndarray, H: int, W: int) -> np.ndarray:
    pts2d, valid = project_verts(verts_cam, K)
    # Triangle is rendered only if all three verts are in front of the camera
    face_valid = valid[faces].all(axis=1)
    if not face_valid.any():
        return np.zeros((H, W), dtype=np.uint8)
    tri_pts = pts2d[faces[face_valid]]  # (Nf, 3, 2)
    polys = [t.astype(np.int32) for t in tri_pts]
    sil = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(sil, polys, 255)
    return sil


def silhouette_iou(sil: np.ndarray, mask: np.ndarray) -> float:
    a = sil > 0
    b = mask > 0
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Reprojection refinement (Nelder-Mead)
# ---------------------------------------------------------------------------


def downsample_for_render(depth, mask, K, factor):
    if factor <= 1:
        return depth, mask, K
    H, W = depth.shape
    nH, nW = H // factor, W // factor
    d_ds = cv2.resize(depth, (nW, nH), interpolation=cv2.INTER_NEAREST)
    m_ds = cv2.resize((mask.astype(np.uint8) * 255), (nW, nH),
                      interpolation=cv2.INTER_NEAREST) > 0
    K_ds = K.copy()
    K_ds[0, :] /= factor
    K_ds[1, :] /= factor
    return d_ds, m_ds, K_ds


def silhouette_chamfer(sil: np.ndarray, mask: np.ndarray,
                       mask_dt: np.ndarray = None, sil_dt: np.ndarray = None):
    """Symmetric chamfer-like distance between mask and silhouette boundaries,
    using distance transforms. Smoother than IoU under small pixel changes."""
    if mask_dt is None:
        mask_dt = cv2.distanceTransform((~(mask > 0)).astype(np.uint8) * 255,
                                        cv2.DIST_L2, 3)
    sil_b = (sil > 0).astype(np.uint8)
    if sil_b.sum() == 0:
        # Penalize a fully-empty silhouette by the mask-area mean distance
        return float(mask_dt[mask > 0].mean()) if (mask > 0).any() else 0.0
    if sil_dt is None:
        sil_dt = cv2.distanceTransform((~(sil > 0)).astype(np.uint8) * 255,
                                       cv2.DIST_L2, 3)
    # Mean distance from silhouette pixels to nearest mask pixel + vice versa
    d1 = float(mask_dt[sil_b > 0].mean())
    mask_b = (mask > 0)
    d2 = float(sil_dt[mask_b].mean()) if mask_b.any() else 0.0
    return 0.5 * (d1 + d2)


def refine_reprojection(verts_init: np.ndarray, faces: np.ndarray,
                        K: np.ndarray, mask: np.ndarray,
                        target_points: np.ndarray,
                        max_face_samples: int, max_iters: int,
                        iou_weight: float, icp_weight: float,
                        chamfer_weight: float,
                        rng: np.random.Generator):
    """Optimize (delta_R, delta_t, delta_log_scale) around verts_init via
    Nelder-Mead. Cost = iou_weight*(1 - IoU)
                     + chamfer_weight * silhouette<->mask chamfer
                     + icp_weight * mean 3D ICP distance.

    Initial simplex uses physically-meaningful per-axis scales so the optimizer
    actually explores (rotation ~10 deg, translation ~5 cm, scale ~15%)."""
    if not _HAS_SCIPY:
        return verts_init, None, {"refined": False,
                                  "reason": "scipy.optimize not available"}

    H, W = mask.shape
    Nf = faces.shape[0]
    if Nf > max_face_samples:
        face_idx = rng.choice(Nf, size=max_face_samples, replace=False)
        faces_render = faces[face_idx]
    else:
        faces_render = faces

    centroid = verts_init.mean(axis=0)
    if target_points.shape[0] > 0:
        tree = cKDTree(target_points)
        n_v = verts_init.shape[0]
        v_idx = rng.choice(n_v, size=min(2000, n_v), replace=False)
    else:
        tree, v_idx = None, None

    # Precompute mask distance transform (constant across iterations)
    mask_dt = cv2.distanceTransform((~(mask > 0)).astype(np.uint8) * 255,
                                    cv2.DIST_L2, 3)
    diag_px = float(np.hypot(H, W))

    def evaluate(params):
        rvec, tvec, lscale = params[:3], params[3:6], params[6]
        R = axis_angle_to_R(rvec)
        s = float(np.exp(lscale))
        v = s * (verts_init - centroid) @ R.T + centroid + tvec
        sil = render_silhouette(v, faces_render, K, H, W)
        iou = silhouette_iou(sil, mask)
        ch = silhouette_chamfer(sil, mask, mask_dt=mask_dt) / diag_px
        icp = 0.0
        if tree is not None and v_idx is not None:
            dists, _ = tree.query(v[v_idx], k=1)
            icp = float(np.mean(dists ** 2))
        cost = (iou_weight * (1.0 - iou)
                + chamfer_weight * ch
                + icp_weight * icp)
        return cost, iou, ch, icp, v

    def cost(params):
        return evaluate(params)[0]

    # Physically-meaningful initial simplex so Nelder-Mead actually moves
    x0 = np.zeros(7)
    step = np.array([
        np.deg2rad(8.0), np.deg2rad(8.0), np.deg2rad(8.0),   # rotation
        0.03, 0.03, 0.05,                                     # translation (m)
        0.10,                                                 # log scale (~10%)
    ])
    simplex = np.vstack([x0, x0 + np.diag(step)])  # (8, 7)

    cost0, iou0, ch0, icp0, _ = evaluate(x0)
    t0 = time.time()
    result = minimize(cost, x0, method="Nelder-Mead",
                      options={"maxiter": max_iters,
                               "xatol": 1e-4, "fatol": 1e-5,
                               "adaptive": True,
                               "initial_simplex": simplex})
    cost1, iou1, ch1, icp1, v_refined = evaluate(result.x)

    diag = {
        "refined": True,
        "iou_init": float(iou0), "iou_final": float(iou1),
        "chamfer_norm_init": float(ch0), "chamfer_norm_final": float(ch1),
        "icp_init": float(icp0), "icp_final": float(icp1),
        "cost_init": float(cost0), "cost_final": float(cost1),
        "nfev": int(result.nfev), "niter": int(result.nit),
        "elapsed_s": float(time.time() - t0),
        "delta_axis_angle": result.x[:3].tolist(),
        "delta_translation": result.x[3:6].tolist(),
        "delta_log_scale": float(result.x[6]),
        "delta_scale_factor": float(np.exp(result.x[6])),
    }
    return v_refined, result.x, diag


# ---------------------------------------------------------------------------
# Camera-to-world transform for one frame
# ---------------------------------------------------------------------------


def cam_pose_at(cams_npz, frame_idx: int):
    fis = list(cams_npz["frame_indices"].tolist())
    if frame_idx not in fis:
        return None
    i = fis.index(frame_idx)
    R = quat_xyzw_to_R(cams_npz["cam_quats_xyzw"][i])
    t = cams_npz["cam_trans"][i].astype(np.float64)
    return R, t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only align these labels (default: all)")
    p.add_argument("--candidate", type=int, default=0,
                   help="When sam3d ran with --all-candidates, which candidate "
                        "subfolder (default 0 = top).")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing aligned/<label>/ outputs")
    p.add_argument("--coarse", dest="coarse", action="store_true", default=False,
                   help="Re-fit scale (Y-height ratio) + translation against MoGe "
                        "target points (mesh_alignment.py-style). DEFAULT off: "
                        "sam3d's pose.json is already a strong prior, and the "
                        "height-ratio heuristic over-shrinks meshes whose back is "
                        "occluded in the keyframe. Useful for sam3d_body-style "
                        "canonical meshes without a usable pose.json.")
    p.add_argument("--no-coarse", dest="coarse", action="store_false",
                   help="(default) skip the mesh_alignment.py-style step")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip the reprojection refinement step")
    p.add_argument("--render-factor", type=int, default=4,
                   help="Image/mask/K downscale factor used during refinement "
                        "(default 4 -> renders at 270x480 for a 1080p clip)")
    p.add_argument("--max-faces", type=int, default=15000,
                   help="Random face subsample for silhouette rendering during "
                        "refinement (default 15000)")
    p.add_argument("--max-iters", type=int, default=300,
                   help="Nelder-Mead max iterations (default 300)")
    p.add_argument("--iou-weight", type=float, default=1.0,
                   help="Silhouette IoU cost weight (default 1.0)")
    p.add_argument("--icp-weight", type=float, default=1.0,
                   help="3D ICP (mesh verts -> MoGe pointmap) cost weight (default 1.0)")
    p.add_argument("--chamfer-weight", type=float, default=2.0,
                   help="Weight on silhouette-mask chamfer-on-distance-transform "
                        "term (smoother than IoU; helps the optimizer move "
                        "even when IoU is locally piecewise-constant)")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for face subsampling")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    sam3d_root = scene_dir / "sam3d"
    masks_root = scene_dir / "masks"
    any4d_root = scene_dir / "any4d"
    moge_root = any4d_root / "moge"
    out_root = scene_dir / "aligned"

    for need in (sam3d_root, masks_root, moge_root,
                 any4d_root / "cameras.npz", moge_root / "intrinsics.npz"):
        if not need.exists():
            sys.exit(f"error: missing {need}")

    out_root.mkdir(parents=True, exist_ok=True)

    # Discover labels
    label_dirs = sorted(d for d in sam3d_root.iterdir() if d.is_dir())
    labels = [d.name for d in label_dirs]
    if args.labels:
        unknown = [l for l in args.labels if l not in labels]
        if unknown:
            sys.exit(f"error: --labels not present: {unknown}")
        labels = [l for l in labels if l in args.labels]
    if not labels:
        sys.exit("error: nothing to process")

    intr_npz = np.load(moge_root / "intrinsics.npz")
    intr_frames = list(intr_npz["frame_indices"].tolist())
    intr_arr = intr_npz["intrinsics"]
    cams_npz = np.load(any4d_root / "cameras.npz")

    rng = np.random.default_rng(args.seed)
    summary = {}

    print(f"scene:  {scene_dir.name}")
    print(f"labels: {labels}\n")

    for label in labels:
        label_dir = sam3d_root / label
        try:
            print(f"[{label}]")
            mesh_path, pose_path, kf = find_mesh_dir(label_dir, args.candidate)
            if mesh_path is None:
                print(f"  SKIP: no mesh.glb")
                continue
            if kf is None:
                print(f"  SKIP: no keyframe known for {mesh_path}")
                continue
            print(f"  mesh   = {mesh_path.relative_to(scene_dir)}")
            print(f"  kf     = {kf:06d}")

            out_label = out_root / label
            if out_label.exists():
                if args.overwrite:
                    import shutil
                    shutil.rmtree(out_label)
                else:
                    print(f"  SKIP: {out_label} exists (pass --overwrite)")
                    continue
            out_label.mkdir(parents=True)

            # Load inputs
            verts_canon, faces = load_mesh(mesh_path)
            pose = load_pose(pose_path)
            if pose is None:
                print(f"  SKIP: no pose.json")
                continue

            depth_path = moge_root / "depth" / f"{kf:06d}.npy"
            if not depth_path.is_file():
                print(f"  SKIP: missing {depth_path}")
                continue
            depth = np.load(depth_path)
            H, W = depth.shape

            if kf not in intr_frames:
                print(f"  SKIP: keyframe {kf} not in moge intrinsics")
                continue
            K = intr_arr[intr_frames.index(kf)].astype(np.float64).copy()

            mask = load_mask(masks_root / label / f"{kf:06d}.png", (H, W))
            if mask is None or not mask.any():
                print(f"  SKIP: missing or empty mask at kf {kf:06d}")
                continue

            # Build MoGe target points within the mask
            target_full = back_project(depth, K, mask=mask)
            target = filter_flying_points(target_full)
            if target.shape[0] == 0:
                print(f"  WARN: no target points after flying-point filter; "
                      f"falling back to full set")
                target = target_full

            # ---- Step A: pose.json applied (camera frame) ----
            # sam3d's pose puts the mesh in a PyTorch3D-style camera frame
            # (X-left, Y-up). Convert to RDF (X-right, Y-down) by rotating 180
            # degrees about Z (== flipping X and Y). This is a proper rotation
            # so face winding stays valid.
            v_cam_init = apply_rts(verts_canon, pose["R"], pose["t"], pose["s"])
            v_cam_init[:, 0] *= -1
            v_cam_init[:, 1] *= -1
            sil_init = render_silhouette(v_cam_init, faces, K, H, W)
            iou_init = silhouette_iou(sil_init, mask)
            print(f"  step A (pose.json):  scale={pose['s']:.4f}  "
                  f"t={pose['t'].round(3).tolist()}  iou={iou_init:.3f}")

            # ---- Step B: mesh_alignment.py-style scale+translate ----
            if args.coarse:
                v_cam_coarse, s_coarse, t_coarse = coarse_scale_translate(
                    v_cam_init, target
                )
                iou_coarse = silhouette_iou(
                    render_silhouette(v_cam_coarse, faces, K, H, W), mask)
                print(f"  step B (coarse):     scale*={s_coarse:.4f}  "
                      f"dt={t_coarse.round(3).tolist()}  iou={iou_coarse:.3f}")
            else:
                v_cam_coarse = v_cam_init
                s_coarse, t_coarse, iou_coarse = 1.0, np.zeros(3), float(iou_init)

            # ---- Step C: reprojection refinement ----
            refine_diag = {"refined": False}
            v_cam_refined = v_cam_coarse
            if not args.no_refine:
                depth_ds, mask_ds, K_ds = downsample_for_render(
                    depth, mask, K, args.render_factor
                )
                target_ds = filter_flying_points(
                    back_project(depth_ds, K_ds, mask=mask_ds)
                )
                v_cam_refined, _params, refine_diag = refine_reprojection(
                    v_cam_coarse, faces, K_ds, mask_ds, target_ds,
                    max_face_samples=args.max_faces,
                    max_iters=args.max_iters,
                    iou_weight=args.iou_weight, icp_weight=args.icp_weight,
                    chamfer_weight=args.chamfer_weight,
                    rng=rng,
                )
                # Re-score IoU at full resolution for honest comparison
                iou_refined_full = silhouette_iou(
                    render_silhouette(v_cam_refined, faces, K, H, W), mask)
                refine_diag["iou_full_after"] = float(iou_refined_full)
                print(f"  step C (reproject):  iou_ds {refine_diag['iou_init']:.3f} -> "
                      f"{refine_diag['iou_final']:.3f}  icp {refine_diag['icp_init']:.4f} -> "
                      f"{refine_diag['icp_final']:.4f}  "
                      f"iou_full={iou_refined_full:.3f}  "
                      f"nfev={refine_diag['nfev']}  t={refine_diag['elapsed_s']:.1f}s")

            # ---- Step D: bake camera-to-world (keyframe cam pose) ----
            cw = cam_pose_at(cams_npz, kf)
            if cw is None:
                print(f"  WARN: no camera pose for kf {kf}; saving in camera frame")
                R_cw, t_cw = np.eye(3), np.zeros(3)
            else:
                R_cw, t_cw = cw
            v_world = v_cam_refined @ R_cw.T + t_cw

            # ---- Save outputs ----
            # NOTE: faces unchanged across all transforms; export trimesh in world frame.
            out_mesh = trimesh.Trimesh(vertices=v_world, faces=faces, process=False)
            # Preserve vertex colors if present in the source mesh
            try:
                src = trimesh.load(str(mesh_path), force="mesh")
                if hasattr(src.visual, "vertex_colors"):
                    out_mesh.visual.vertex_colors = np.asarray(src.visual.vertex_colors)
            except Exception:
                pass
            out_glb = out_label / "mesh.glb"
            out_mesh.export(str(out_glb))

            align = {
                "label": label,
                "keyframe": int(kf),
                "mesh_source": str(mesh_path.relative_to(scene_dir)),
                "render_resolution_full_wh": [int(W), int(H)],
                "step_A_pose_json": {
                    "rotation_R": pose["R"].tolist(),
                    "translation": pose["t"].tolist(),
                    "scale": float(pose["s"]),
                    "iou_full": float(iou_init),
                },
                "step_B_coarse": {
                    "scale_factor": float(s_coarse),
                    "translation_delta": t_coarse.tolist(),
                    "iou_full": float(iou_coarse),
                },
                "step_C_refine": refine_diag,
                "step_D_cam_to_world": {
                    "rotation_R": R_cw.tolist(),
                    "translation": t_cw.tolist(),
                },
                "n_target_points": int(target.shape[0]),
                "n_mesh_verts": int(verts_canon.shape[0]),
                "n_mesh_faces": int(faces.shape[0]),
            }
            with open(out_label / "align.json", "w") as fp:
                json.dump(align, fp, indent=2)

            summary[label] = {
                "iou_init": float(iou_init),
                "iou_coarse": float(iou_coarse),
                "iou_refined_full": float(refine_diag.get("iou_full_after", iou_coarse)),
                "keyframe": int(kf),
            }
            print(f"  -> wrote {out_glb.relative_to(scene_dir)}\n")
        except Exception as e:
            print(f"  FAIL ({type(e).__name__}: {e})")
            traceback.print_exc()
            continue

    print("done.")
    if summary:
        print("summary:")
        for lbl, s in summary.items():
            print(f"  {lbl:12s}  kf={s['keyframe']:06d}  "
                  f"iou: pose={s['iou_init']:.3f}  "
                  f"coarse={s['iou_coarse']:.3f}  "
                  f"refined={s['iou_refined_full']:.3f}")


if __name__ == "__main__":
    main()
