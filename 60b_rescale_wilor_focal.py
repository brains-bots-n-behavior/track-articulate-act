#!/usr/bin/env python
"""Post-fix WiLoR per-frame outputs to use the real MoGe focal length.

Stage 60 (prior to the MoGe-focal patch) wrote `cam_t`, `verts`, `joints`
computed with WiLoR's *nominal* focal — typically EXTRA.FOCAL_LENGTH /
MODEL.IMAGE_SIZE * img_max (5000 / 256 * 1920 = 37500 for a 1920-px-wide
capture). The real camera focal is much smaller (~1000–1500 px), so depths
came out too large by `focal_wilor / focal_real` (~30x).

This script rescales every existing `data/<scene>/wilor/per_frame/<f>.npz`
so the hand sits at the correct camera-frame depth, without re-running
the WiLoR fit. The math:
    cam_t' = cam_t with z scaled by focal_real / focal_wilor
    verts' = verts shifted by (cam_t'.z - cam_t.z) in camera z
    joints' similarly
Projection onto the image is preserved (focal·x/z stays the same), so the
hand still aligns with the image — but it's now placed at a metric-real
depth, and verts_world / joints_world (if present) are re-baked through
any4d/cameras.npz so the world-frame trajectory is correct too.

Reads:
    data/<scene>/wilor/per_frame/*.npz   (in place; we re-write)
    data/<scene>/any4d/moge/intrinsics.npz   (real focals)
    data/<scene>/any4d/cameras.npz       (optional; needed only to re-bake
                                          verts_world / joints_world)

Writes:
    Overwrites each per_frame npz in place. Adds `focal_length_orig` so the
    correction is auditable; updates `focal_length` to the real focal.

Run inside any env with numpy.

Example:
    python scripts/60b_rescale_wilor_focal.py --scene-dir data-final/dryer
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def load_focal_map(scene_dir: Path):
    p = scene_dir / "any4d" / "moge" / "intrinsics.npz"
    if not p.is_file():
        sys.exit(f"error: {p} not found — need MoGe intrinsics to rescale")
    d = np.load(p)
    return {int(fi): float(d["intrinsics"][i][0, 0])
            for i, fi in enumerate(d["frame_indices"].tolist())}


def load_cam_poses(scene_dir: Path):
    p = scene_dir / "any4d" / "cameras.npz"
    if not p.is_file():
        return {}
    cams = np.load(p)
    return {int(fi): (quat_xyzw_to_R(cams["cam_quats_xyzw"][i]),
                       cams["cam_trans"][i].astype(np.float64))
            for i, fi in enumerate(cams["frame_indices"].tolist())}


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change, don't write")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    per_frame_dir = scene_dir / "wilor" / "per_frame"
    if not per_frame_dir.is_dir():
        sys.exit(f"error: {per_frame_dir} does not exist")

    focal_map = load_focal_map(scene_dir)
    cam_pose_map = load_cam_poses(scene_dir)
    print(f"MoGe focals: {len(focal_map)} frames")
    print(f"Cam poses:   {len(cam_pose_map)} frames "
          f"{'(verts_world will be re-baked)' if cam_pose_map else '(camera-frame only)'}")

    npz_paths = sorted(per_frame_dir.glob("*.npz"))
    print(f"per_frame: {len(npz_paths)} files\n")

    n_done, n_skip = 0, 0
    sample_log = []
    for npz_p in npz_paths:
        d = dict(np.load(npz_p))
        fi = int(d["frame_idx"])
        if fi not in focal_map:
            n_skip += 1
            continue
        if "focal_length_orig" in d:
            # Already rescaled — skip to keep idempotent
            n_skip += 1
            continue

        wilor_focal = float(d["focal_length"])
        real_focal = focal_map[fi]
        scale = real_focal / wilor_focal

        cam_t = d["cam_t"].astype(np.float64).copy()
        dz = cam_t[:, 2] * (scale - 1.0)   # per-hand z shift
        cam_t[:, 2] *= scale
        verts = d["verts"].astype(np.float64).copy()
        joints = d["joints"].astype(np.float64).copy()
        verts[..., 2] += dz[:, None]
        joints[..., 2] += dz[:, None]

        new = dict(d)
        new["verts"] = verts.astype(np.float32)
        new["joints"] = joints.astype(np.float32)
        new["cam_t"] = cam_t.astype(np.float32)
        new["focal_length"] = np.float32(real_focal)
        new["focal_length_orig"] = np.float32(wilor_focal)

        # Re-bake verts_world / joints_world if cam pose available
        if "verts_world" in d and fi in cam_pose_map:
            R_cw, t_cw = cam_pose_map[fi]
            new["verts_world"] = (
                verts @ R_cw.T + t_cw
            ).astype(np.float32)
            new["joints_world"] = (
                joints @ R_cw.T + t_cw
            ).astype(np.float32)

        if len(sample_log) < 3:
            sample_log.append(
                f"  [{fi:06d}] wilor_focal={wilor_focal:.1f} -> "
                f"real_focal={real_focal:.1f} (scale={scale:.4f}); "
                f"hand0 z {d['cam_t'][0, 2]:.2f} -> {cam_t[0, 2]:.2f}"
            )

        if not args.dry_run:
            np.savez_compressed(npz_p, **new)
        n_done += 1

    for line in sample_log:
        print(line)
    print()
    print(f"rescaled: {n_done} files; skipped: {n_skip} "
          f"(no MoGe focal or already rescaled)")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    main()
