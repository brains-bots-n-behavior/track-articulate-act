#!/usr/bin/env python
"""Stage 51: Rerun replay (scene flow + trajectories) + estimated joint axes.

Same visualization paradigm as 41_replay_in_rerun.py — same blueprint, same
helpers, same dynamic timeline — but also overlays the joint axes produced by
stage 50.

For each label with a joint.json (or an entry in any4d/joints.json):
  - revolute: draws an infinite axis line through axis_point along axis_direction
              (length scaled to the part's bbox diagonal), plus a small marker
              at axis_point.
  - prismatic: draws a single arrow at the part centroid along axis_direction.

Joint logs are STATIC (visible across the whole timeline) so they sit on top
of the moving scene-flow arrows and point trajectories.

Run inside any env that has the same deps as stage 41 (numpy, pillow,
opencv-python, matplotlib, rerun-sdk).

Example:
    python scripts/51_replay_with_joints.py --scene-dir macbook-all
    python scripts/51_replay_with_joints.py --scene-dir macbook-all --labels laptop_up
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr


# ---------------------------------------------------------------------------
# Reuse helpers from 41 (which lives in the same folder)
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_REPLAY41_PATH = _THIS_DIR / "41_replay_in_rerun.py"
if not _REPLAY41_PATH.is_file():
    sys.exit(f"error: {_REPLAY41_PATH} not found; 51 reuses 41's replay")

_spec = importlib.util.spec_from_file_location("_replay41", _REPLAY41_PATH)
_m41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m41)


# ---------------------------------------------------------------------------
# Joint loading + rendering
# ---------------------------------------------------------------------------


# Colors (matplotlib-tab10 inspired): green=revolute, blue=prismatic
JOINT_COLORS = {
    "revolute": (35, 180, 75),
    "prismatic": (50, 110, 230),
}


def load_joints(any4d_root: Path, labels_filter):
    """Prefer any4d/joints.json; fall back to per-label any4d/<label>/joint.json."""
    joints = {}
    summary_path = any4d_root / "joints.json"
    if summary_path.is_file():
        with open(summary_path) as f:
            joints = json.load(f)
    else:
        for d in any4d_root.iterdir():
            if not d.is_dir() or d.name == "moge":
                continue
            jp = d / "joint.json"
            if jp.is_file():
                with open(jp) as f:
                    joints[d.name] = json.load(f)

    if not joints:
        return joints

    if labels_filter:
        joints = {k: v for k, v in joints.items() if k in labels_filter}
    return joints


def _bbox_diag(points: np.ndarray) -> float:
    if points.shape[0] == 0:
        return 1.0
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(np.linalg.norm(hi - lo))


def log_joint(joint: dict, pts_ref: np.ndarray, base_name: str,
              line_radius_frac: float = 0.005, axis_extend_frac: float = 0.75):
    """Log one joint's axis as a static line + marker (revolute) or arrow (prismatic).

    line_radius_frac, axis_extend_frac are fractions of the part's bbox
    diagonal; the line extends ±extend·diag past axis_point.
    """
    axis_dir = joint.get("axis_direction")
    jtype = joint.get("type")
    if axis_dir is None or jtype is None:
        print(f"  skipped: missing axis_direction/type")
        return False

    a = np.asarray(axis_dir, dtype=np.float32)
    a_norm = float(np.linalg.norm(a))
    if a_norm < 1e-8:
        print(f"  skipped: zero-length axis_direction")
        return False
    a /= a_norm

    diag = _bbox_diag(pts_ref)
    extend = max(axis_extend_frac * diag, 1e-3)
    radius = max(line_radius_frac * diag, 1e-4)
    color = JOINT_COLORS.get(jtype, (200, 200, 200))

    if jtype == "revolute":
        axis_pt = joint.get("axis_point")
        if axis_pt is None:
            print(f"  skipped: revolute without axis_point")
            return False
        # axis_point returned by stage 50 is the closest point on the line to
        # the world origin — which can be meters away from the part. Re-anchor
        # the rendered segment at the projection of the part centroid onto the
        # axis so the line visibly passes near the object.
        c_orig = np.asarray(axis_pt, dtype=np.float32)
        centroid = pts_ref.mean(axis=0)
        lam = float((centroid - c_orig) @ a)
        c_render = c_orig + lam * a
        offset = float(np.linalg.norm(centroid - c_render))
        # Extend the line at least far enough to span the offset; keeps the
        # segment visible even when the fit places the axis well off the part.
        extend = max(extend, 1.5 * offset)
        p0 = c_render - a * extend
        p1 = c_render + a * extend
        rr.log(
            f"{base_name}/axis_line",
            rr.LineStrips3D(
                strips=[np.stack([p0, p1])],
                colors=[color],
                radii=[radius],
            ),
            static=True,
        )
        rr.log(
            f"{base_name}/axis_point",
            rr.Points3D(
                positions=[c_render],
                colors=[color],
                radii=[radius * 4.0],
            ),
            static=True,
        )
        rr.log(
            f"{base_name}/axis_arrow",
            rr.Arrows3D(
                origins=[c_render],
                vectors=[a * (extend * 0.5)],
                colors=[color],
                radii=[radius],
            ),
            static=True,
        )
        if offset > 0.5 * _bbox_diag(pts_ref):
            print(f"  note: axis line is {offset:.3f} m from part centroid "
                  f"(bbox diag {_bbox_diag(pts_ref):.3f} m) — fit likely "
                  f"biased by finite-rotation/noise; line re-anchored for display")
    else:  # prismatic
        centroid = pts_ref.mean(axis=0)
        rr.log(
            f"{base_name}/axis_arrow",
            rr.Arrows3D(
                origins=[centroid - a * (extend * 0.5)],
                vectors=[a * extend],
                colors=[color],
                radii=[radius],
            ),
            static=True,
        )

    label = jtype
    rt = joint.get("residual_revolute")
    rp = joint.get("residual_prismatic")
    if rt is not None and rp is not None:
        label = f"{jtype}  E_rev={rt:.3g}  E_pris={rp:.3g}"
    elif "votes" in joint:
        label = f"{jtype}  votes={joint['votes']}"
    print(f"  {jtype}: {label}")
    return True


def overlay_joints(scene_dir: Path, labels_filter):
    """Load joints.json, load each label's pts3d_ref, static-log axes."""
    any4d_root = scene_dir / "any4d"
    joints = load_joints(any4d_root, labels_filter)
    if not joints:
        print("no joints found "
              f"(expected {any4d_root}/joints.json or "
              f"{any4d_root}/<label>/joint.json — run stage 50 first)")
        return

    print(f"overlaying joints for: {list(joints.keys())}")
    for label, joint in joints.items():
        pts_path = any4d_root / label / "pts3d_ref.npy"
        if not pts_path.is_file():
            print(f"[{label}] SKIP: missing {pts_path}")
            continue
        pts = np.load(pts_path).astype(np.float32)
        print(f"[{label}]")
        log_joint(joint, pts, base_name=f"pred/joints/{label}")


# ---------------------------------------------------------------------------
# SAM 3D mesh overlay (no pose applied — "forget alignment first")
# ---------------------------------------------------------------------------


def find_mesh_path(label_dir: Path, candidate_idx: int):
    """Return the GLB to render for one label, or None if none found.

    Looks first at <label>/mesh.glb (aligned/ layout, or sam3d/ single-candidate
    layout), then at <label>/cand_<NN>_<kf>/mesh.glb (--all-candidates layout).
    """
    direct = label_dir / "mesh.glb"
    if direct.is_file():
        return direct
    cands = sorted(label_dir.glob(f"cand_{candidate_idx:02d}_*"))
    for c in cands:
        mp = c / "mesh.glb"
        if mp.is_file():
            return mp
    # Fallback: first available candidate
    cands = sorted(label_dir.glob("cand_*"))
    for c in cands:
        mp = c / "mesh.glb"
        if mp.is_file():
            print(f"  note: cand_{candidate_idx:02d}_* not found, using {c.name}")
            return mp
    return None


def overlay_meshes(scene_dir: Path, labels_filter, candidate_idx: int,
                   source: str):
    """Log each label's GLB under pred/meshes/<label>.

    source='aligned' -> aligned/<label>/mesh.glb (output of stage 52,
                        in Any4D world frame; rendered with no extra transform)
    source='sam3d'   -> sam3d/<label>/[cand_NN_<kf>/]mesh.glb (canonical,
                        before alignment — overlaps at world origin)
    source='auto'    -> aligned/ if it exists, else sam3d/
    """
    aligned_root = scene_dir / "aligned"
    sam3d_root = scene_dir / "sam3d"

    if source == "auto":
        source = "aligned" if aligned_root.is_dir() else "sam3d"
    root = aligned_root if source == "aligned" else sam3d_root

    if not root.is_dir():
        print(f"no {root.name}/ folder at {root} — skipping mesh overlay")
        return

    label_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if labels_filter:
        label_dirs = [d for d in label_dirs if d.name in labels_filter]
    if not label_dirs:
        print(f"no labels under {root.name}/ to render")
        return

    print(f"loading meshes from {root.relative_to(scene_dir)}/ (source={source}):")
    n_logged = 0
    for d in label_dirs:
        mp = find_mesh_path(d, candidate_idx)
        if mp is None:
            print(f"  [{d.name}] SKIP: no mesh.glb")
            continue
        rr.log(f"pred/meshes/{d.name}", rr.Asset3D(path=str(mp)), static=True)
        print(f"  [{d.name}] -> {mp.relative_to(scene_dir)}")
        n_logged += 1
    if n_logged == 0:
        print("  (nothing rendered)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--no-scene-flow", action="store_true",
                   help="Skip rendering scene-flow arrows")
    p.add_argument("--no-trajectories", action="store_true",
                   help="Skip rendering 3D point trajectories")
    p.add_argument("--no-joints", action="store_true",
                   help="Skip joint-axis overlay (then this script == stage 41)")
    p.add_argument("--no-meshes", action="store_true",
                   help="Skip the sam3d GLB mesh overlay")
    p.add_argument("--mesh-candidate", type=int, default=0,
                   help="When sam3d was run with --all-candidates, which candidate "
                        "to render per label (default 0 = top). Falls back to the "
                        "first available candidate if the requested index is missing.")
    p.add_argument("--mesh-source", choices=["auto", "aligned", "sam3d"],
                   default="auto",
                   help="Which mesh folder to render. 'aligned' = stage-52 output "
                        "(in world frame, properly placed). 'sam3d' = pre-alignment "
                        "(overlaps near origin). 'auto' = aligned if present, else "
                        "sam3d.")
    p.add_argument("--max-tracks", type=int, default=200,
                   help="Trajectories per label (default 200)")
    p.add_argument("--frame-time-step", type=float, default=0.2,
                   help="Seconds per frame on the stable_time timeline (default 0.2)")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only replay these labels (default: all)")
    p.add_argument("--max-arrows", type=int, default=500,
                   help="Subsample scene-flow arrows per label per frame (default 500)")
    p.add_argument("--no-depth", action="store_true",
                   help="Skip rendering the MoGe depth map (mask overlay is unaffected)")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()

    # Run the same replay 41 does (this opens the rerun recording and spawns the viewer).
    _m41.replay(
        scene_dir=scene_dir,
        show_scene_flow=not args.no_scene_flow,
        show_trajectories=not args.no_trajectories,
        max_tracks=args.max_tracks,
        frame_time_step=args.frame_time_step,
        max_arrows=args.max_arrows,
        labels_filter=args.labels,
        show_depth=not args.no_depth,
    )

    if not args.no_joints:
        overlay_joints(scene_dir, args.labels)

    if not args.no_meshes:
        overlay_meshes(scene_dir, args.labels, args.mesh_candidate,
                       source=args.mesh_source)

    print("done. Drag the 'stable_time' timeline in the Rerun viewer.")


if __name__ == "__main__":
    main()
