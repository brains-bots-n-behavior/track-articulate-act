#!/usr/bin/env python
"""Stage 33: visualize Any6D object poses against the scene point cloud.

An alignment check for stage 32 (Any6D). For every label under
data/<scene>/any6d/ that has a `mesh_world.glb`, this logs the posed mesh into
Rerun on top of a scene point cloud in the *same* Any4D world frame, so you can
eyeball whether each object sits where the geometry says it should.

Two point-cloud sources (choose with --point-source):

    any4d  (DEFAULT)
        The Any4D reference pointmap (any4d/pointmap_ref.npy), one global cloud
        for the whole scene, coloured by the reference-frame RGB and masked to
        MoGe's valid pixels. This is the same cloud stages 41/51 draw. Best for
        a quick "does everything sit together" look; note it is a single frame
        (config.ref_frame), so a part that MOVED between the ref frame and its
        own keyframe will not line up with it (use `moge` for those).

    moge
        The MoGe *metric* depth at EACH object's own keyframe, unprojected to
        that frame's camera and baked into the Any4D world frame. This is the
        exact geometry Any6D registered to, per object, so a good pose should
        overlay tightly even for a part that articulated between frames. One
        cloud per distinct keyframe.

    both
        Draw both of the above (any4d cloud dimmed under the moge clouds).

The posed meshes come straight from stage 32's `mesh_world.glb` (Any6D's
metric-rescaled mesh already placed by pose_object_to_world) — no extra
transform is applied here, exactly as stage 51 renders aligned meshes.

Reads:
    data/<scene>/any6d/<label>/mesh_world.glb     (stage 32; required per label)
    data/<scene>/any6d/<label>/pose.json          (stage 32; for keyframe + sanity)
    data/<scene>/any4d/config.json                (stage 40)
    data/<scene>/any4d/cameras.npz                (stage 40)
    data/<scene>/any4d/pointmap_ref.npy           (stage 40; --point-source any4d/both)
    data/<scene>/any4d/moge/{depth,mask,intrinsics.npz}  (stage 40; --point-source moge/both)
    data/<scene>/frames/*.jpg                      (stage 00; point-cloud colour)

Writes: nothing under data/ — only a Rerun recording (viewer, .rrd, or web).

Run inside any env with numpy + pillow + rerun-sdk (the stage 41/51 env).
Any6D itself is headless; run this wherever you can open (or forward) a Rerun
viewer, or pass --save-rrd to produce a file you scp to a workstation.

Examples:
    # Spawn the Rerun viewer with the global Any4D cloud + all posed meshes
    python scripts/33_visualize_any6d.py --scene-dir data/oven

    # Check one object against the MoGe depth at its own keyframe
    python scripts/33_visualize_any6d.py --scene-dir data/oven \\
        --labels oven_body --point-source moge

    # Headless server: write a recording to open on a workstation
    python scripts/33_visualize_any6d.py --scene-dir data/oven \\
        --save-rrd data/oven/any6d/preview.rrd
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

# Reuse 41's camera helpers (same folder) so the world frame matches exactly.
_THIS_DIR = Path(__file__).resolve().parent
_REPLAY41_PATH = _THIS_DIR / "41_replay_in_rerun.py"
if not _REPLAY41_PATH.is_file():
    sys.exit(f"error: {_REPLAY41_PATH} not found; stage 33 reuses 41's camera helpers")
_spec = importlib.util.spec_from_file_location("_replay41", _REPLAY41_PATH)
_m41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m41)
build_cam_pose = _m41.build_cam_pose  # (quat_xyzw, trans) -> 4x4 cam->world


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config_and_cameras(any4d_root: Path):
    for need in (any4d_root / "config.json", any4d_root / "cameras.npz"):
        if not need.is_file():
            sys.exit(f"error: missing {need} (did stage 40 finish?)")
    with open(any4d_root / "config.json") as f:
        config = json.load(f)
    cameras = np.load(any4d_root / "cameras.npz")

    # Any4D stores quaternions as XYZW; older bundles used a misleading key.
    if "cam_quats_xyzw" in cameras.files:
        cam_quats = cameras["cam_quats_xyzw"]
    elif "cam_quats_wxyz" in cameras.files:
        print("note: cameras.npz uses legacy key 'cam_quats_wxyz' but bytes are "
              "XYZW (Any4D convention). Reading as XYZW.")
        cam_quats = cameras["cam_quats_wxyz"]
    else:
        sys.exit("error: cameras.npz has no cam_quats_xyzw key")

    cam_pose = {
        int(fi): build_cam_pose(q, t)
        for fi, q, t in zip(cameras["frame_indices"].tolist(),
                            cam_quats, cameras["cam_trans"])
    }
    return config, cam_pose


def find_any6d_labels(any6d_root: Path, labels_filter):
    """Return [(label, mesh_world_path, pose_dict|None)] for labels with a
    world-frame mesh. Skips empty folders (e.g. labels Any6D couldn't pose)."""
    if not any6d_root.is_dir():
        sys.exit(f"error: {any6d_root} does not exist (run stage 32 first)")
    out = []
    for d in sorted(p for p in any6d_root.iterdir() if p.is_dir()):
        if labels_filter and d.name not in labels_filter:
            continue
        mesh_world = d / "mesh_world.glb"
        if not mesh_world.is_file():
            # mesh_cam.glb (no cameras.npz at stage 32) can't be world-compared.
            reason = "mesh_cam.glb only (no world frame)" \
                if (d / "mesh_cam.glb").is_file() else "no mesh_world.glb"
            print(f"  [{d.name}] SKIP: {reason}")
            continue
        pose = None
        pj = d / "pose.json"
        if pj.is_file():
            with open(pj) as f:
                pose = json.load(f)
        out.append((d.name, mesh_world, pose))
    if labels_filter:
        unknown = set(labels_filter) - {p.name for p in any6d_root.iterdir() if p.is_dir()}
        if unknown:
            print(f"  warning: --labels not found under {any6d_root}: {sorted(unknown)}")
    return out


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------

def _subsample(positions, colors, max_points, rng):
    n = positions.shape[0]
    if max_points and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        return positions[idx], colors[idx]
    return positions, colors


def log_any4d_cloud(scene_dir, any4d_root, config, entity, max_points, rng,
                    use_mask=True, dim=False):
    """Log the Any4D reference pointmap (world frame) coloured by ref RGB."""
    pm_path = any4d_root / "pointmap_ref.npy"
    if not pm_path.is_file():
        print(f"  warning: {pm_path} missing — skipping any4d cloud")
        return 0
    pts = np.load(pm_path)                      # (H, W, 3), Any4D world frame
    H, W = pts.shape[:2]
    ref_frame = int(config["ref_frame"])

    rgb_path = scene_dir / "frames" / f"{ref_frame:06d}.jpg"
    if rgb_path.is_file():
        rgb_full = np.array(Image.open(rgb_path).convert("RGB"))
        rgb = np.array(Image.fromarray(rgb_full).resize((W, H), Image.BILINEAR))
    else:
        rgb = np.full((H, W, 3), 200, np.uint8)

    mask = np.isfinite(pts).all(axis=2)
    if use_mask:
        mp = any4d_root / "moge" / "mask" / f"{ref_frame:06d}.png"
        if mp.is_file():
            m = np.array(Image.open(mp))
            if m.ndim == 3:
                m = m[..., -1]
            m = np.array(Image.fromarray((m > 0).astype(np.uint8) * 255)
                         .resize((W, H), Image.NEAREST)) > 0
            mask &= m

    positions = pts[mask].reshape(-1, 3).astype(np.float32)
    colors = rgb[mask].reshape(-1, 3).astype(np.uint8)
    if dim:                                      # push toward gray so meshes/moge pop
        colors = (colors * 0.55 + 110 * 0.45).astype(np.uint8)
    positions, colors = _subsample(positions, colors, max_points, rng)
    rr.log(entity, rr.Points3D(positions=positions, colors=colors), static=True)
    print(f"  any4d cloud: {positions.shape[0]:>7d} pts "
          f"(ref frame {ref_frame}) -> {entity}")
    return positions.shape[0]


def _unproject_zdepth(depth, K):
    """(H,W) z-depth + 3x3 K -> (H,W,3) camera-frame points (RDF)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.arange(W, dtype=np.float32)[None, :]
    v = np.arange(H, dtype=np.float32)[:, None]
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=2)


def log_moge_cloud_for_keyframe(scene_dir, any4d_root, cam_pose, kf, entity,
                                max_points, rng):
    """Unproject MoGe metric depth at frame `kf` and bake it into world frame."""
    moge = any4d_root / "moge"
    depth_path = moge / "depth" / f"{kf:06d}.npy"
    intr_path = moge / "intrinsics.npz"
    if not depth_path.is_file() or not intr_path.is_file():
        print(f"  warning: MoGe depth/intrinsics missing for frame {kf} — skipping")
        return 0
    if kf not in cam_pose:
        print(f"  warning: no camera pose for frame {kf} in cameras.npz — skipping")
        return 0

    intr = np.load(intr_path)
    K_map = {int(fi): K for fi, K in zip(intr["frame_indices"], intr["intrinsics"])}
    if kf not in K_map:
        print(f"  warning: no MoGe intrinsics for frame {kf} — skipping")
        return 0

    depth = np.load(depth_path)                  # (H, W) metric z-depth
    pts_cam = _unproject_zdepth(depth, K_map[kf])
    valid = np.isfinite(depth) & (depth > 0)
    mp = moge / "mask" / f"{kf:06d}.png"
    if mp.is_file():
        m = np.array(Image.open(mp))
        if m.ndim == 3:
            m = m[..., -1]
        valid &= (m > 0)

    rgb_path = scene_dir / "frames" / f"{kf:06d}.jpg"
    H, W = depth.shape
    if rgb_path.is_file():
        rgb = np.array(Image.open(rgb_path).convert("RGB").resize((W, H)))
    else:
        rgb = np.full((H, W, 3), 200, np.uint8)

    pc = pts_cam[valid].reshape(-1, 3)
    colors = rgb[valid].reshape(-1, 3).astype(np.uint8)
    T = cam_pose[kf]                             # cam -> world
    pw = (T[:3, :3] @ pc.T + T[:3, 3:4]).T.astype(np.float32)
    pw, colors = _subsample(pw, colors, max_points, rng)
    rr.log(entity, rr.Points3D(positions=pw, colors=colors), static=True)
    print(f"  moge cloud (frame {kf}): {pw.shape[0]:>7d} pts -> {entity}")
    return pw.shape[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only visualize these labels (default: all in any6d/)")
    p.add_argument("--point-source", choices=["any4d", "moge", "both"],
                   default="any4d",
                   help="Point cloud to check against: 'any4d' global ref cloud "
                        "(default), 'moge' per-keyframe metric cloud, or 'both'")
    p.add_argument("--no-mask", action="store_true",
                   help="Don't mask the any4d cloud by MoGe's valid-pixel mask "
                        "(shows the full reference pointmap)")
    p.add_argument("--max-points", type=int, default=250_000,
                   help="Subsample each point cloud to at most this many points "
                        "(default 250000; 0 = keep all)")
    p.add_argument("--save-rrd", type=Path, default=None,
                   help="Write the Rerun recording to this .rrd file instead of "
                        "spawning a viewer (for headless servers)")
    p.add_argument("--serve", action="store_true",
                   help="Serve the recording over the web instead of spawning a "
                        "local viewer")
    p.add_argument("--port", type=int, default=9999,
                   help="Port for the spawned viewer (default 9999)")
    p.add_argument("--seed", type=int, default=0, help="Subsampling RNG seed")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    any4d_root = scene_dir / "any4d"
    any6d_root = scene_dir / "any6d"

    config, cam_pose = load_config_and_cameras(any4d_root)

    print(f"any6d labels under {any6d_root}:")
    labels = find_any6d_labels(any6d_root, args.labels)
    if not labels:
        sys.exit("error: no posed meshes (mesh_world.glb) to visualize — "
                 "run stage 32 first (and ensure any4d/cameras.npz existed then)")

    # ---- Rerun init + output sink ----
    rr.init("articulate4d_any6d")
    if args.save_rrd is not None:
        args.save_rrd.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(args.save_rrd))
        print(f"recording -> {args.save_rrd}")
    elif args.serve:
        rr.serve_web()
        print("serving Rerun over the web (see printed URL)")
    else:
        rr.spawn(port=args.port)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Spatial3DView(origin="world", name="Any6D alignment",
                          background=[255, 255, 255],
                          line_grid=rrb.archetypes.LineGrid3D(visible=False)),
        collapse_panels=True,
    ))

    rng = np.random.default_rng(args.seed)
    maxp = args.max_points if args.max_points > 0 else None

    # ---- Point clouds ----
    print("\npoint cloud(s):")
    if args.point_source in ("any4d", "both"):
        log_any4d_cloud(scene_dir, any4d_root, config, "world/cloud/any4d",
                        maxp, rng, use_mask=not args.no_mask,
                        dim=(args.point_source == "both"))
    if args.point_source in ("moge", "both"):
        keyframes = {}
        for name, _, pose in labels:
            kf = int(pose["keyframe"]) if pose and "keyframe" in pose else None
            if kf is None:
                print(f"  warning: {name} has no keyframe in pose.json — "
                      "can't build its MoGe cloud")
                continue
            keyframes.setdefault(kf, []).append(name)
        for kf, names in sorted(keyframes.items()):
            log_moge_cloud_for_keyframe(
                scene_dir, any4d_root, cam_pose, kf,
                f"world/cloud/moge/frame_{kf:06d}", maxp, rng)
            print(f"    (frame {kf} keyframe for: {', '.join(names)})")

    # ---- Posed meshes ----
    print("\nposed meshes (stage 32 mesh_world.glb, Any4D world frame):")
    for name, mesh_world, pose in labels:
        rr.log(f"world/any6d/{name}", rr.Asset3D(path=str(mesh_world)), static=True)
        if pose and "pose_object_to_world" in pose:
            t = [round(pose["pose_object_to_world"][i][3], 3) for i in range(3)]
            print(f"  [{name}] kf={pose.get('keyframe')}  t_world={t}  <- "
                  f"{mesh_world.relative_to(scene_dir)}")
        else:
            print(f"  [{name}]  <- {mesh_world.relative_to(scene_dir)}")

    if args.save_rrd is not None:
        print(f"\ndone. Open it with:  rerun {args.save_rrd}")
    else:
        print("\ndone. Objects should sit inside/on the point cloud where the "
              "real objects are.")


if __name__ == "__main__":
    main()
