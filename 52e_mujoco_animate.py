#!/usr/bin/env python
"""Stage 52e: animate a saved 52d scene in MuJoCo.

Reads the `rerun/transforms.json` produced by stage 52d and rebuilds the
articulation as a real MuJoCo scene — fixed body welded to ground, moving
body attached via a hinge or slide joint at the user-set pose, hand mocap
bodies driven through the full WiLoR trajectory. Joint motion is generated
on the fly (linear / triangle / sine / static) since the saved JSON only
records the user's static drive snapshot.

Reads:
    data/<scene>/rerun/transforms.json      (or --transforms PATH)
    data/<scene>/sam3d/<label>/[cand_NN_<kf>/]mesh.glb   (per provenance)
    data/<scene>/wilor/per_frame/*.npz
    data/<scene>/wilor/faces.npy
    data/<scene>/frames/*.jpg

Writes (under data/<scene>/mujoco_anim/):
    scene.xml             Compileable MJCF — re-runnable with
                          `python -m mujoco.viewer --mjcf=scene.xml`
    object_fixed.stl      Decimated, centred fixed body geometry
    object_moving.stl     Decimated, centred moving body geometry
    hand_left.stl         Canonical (first-frame) MANO mesh, centred
    hand_right.stl        (only emitted for lateralities the trajectory has)

Run inside any env with mujoco + numpy + trimesh.

Examples:
    # Default: linear joint sweep 0 -> drive_at_save over the trajectory.
    python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer

    # Two open/close cycles, 30 FPS playback:
    python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer \\
        --joint-motion triangle --n-cycles 2 --fps 30

    # Hold the joint static at 1.5x the saved drive while the hand plays:
    python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer \\
        --joint-motion static --joint-amplitude 1.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("error: mujoco not installed. `pip install mujoco`")


# ---------------------------------------------------------------------------
# Mesh / data loading (shared style with 52c)
# ---------------------------------------------------------------------------


def find_label_mesh(scene_dir: Path, label: str, candidate: int) -> Path:
    root = scene_dir / "sam3d" / label
    if not root.is_dir():
        sys.exit(f"error: {root} does not exist")
    direct = root / "mesh.glb"
    if direct.is_file():
        return direct
    matches = sorted(root.glob(f"cand_{candidate:02d}_*"))
    if not matches:
        matches = sorted(root.glob("cand_*"))
    for c in matches:
        if (c / "mesh.glb").is_file():
            return c / "mesh.glb"
    sys.exit(f"error: no mesh.glb under {root}")


def load_glb(path: Path):
    mesh = trimesh.load(str(path), force="mesh")
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64))


def center_and_export_mesh(verts: np.ndarray, faces: np.ndarray,
                           out: Path, max_faces: int = 150_000):
    """Centre at origin, decimate if over max_faces, write binary STL.
    Same order as 52c/52d so body-local frames stay consistent."""
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        sys.exit(f"error: empty mesh export to {out}")
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    if max_faces > 0 and faces.shape[0] > max_faces:
        try:
            full = trimesh.Trimesh(vertices=centered, faces=faces,
                                   process=False)
            small = full.simplify_quadric_decimation(face_count=max_faces)
            centered = np.asarray(small.vertices, dtype=np.float64)
            faces = np.asarray(small.faces, dtype=np.int64)
            print(f"  decimated {out.name}: "
                  f"{centered.shape[0]} verts, {faces.shape[0]} faces")
        except Exception as e:
            rng = np.random.default_rng(0)
            keep = rng.choice(faces.shape[0], max_faces, replace=False)
            faces = faces[keep]
            print(f"  random-subsampled {out.name} "
                  f"({type(e).__name__}: {e})")
    out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=centered, faces=faces, process=False).export(
        str(out), file_type="stl")
    return centroid


def list_scene_frames(scene_dir: Path):
    frames_dir = scene_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")
    frames = sorted(int(p.stem) for p in frames_dir.glob("*.jpg")
                    if p.stem.isdigit())
    if not frames:
        sys.exit(f"error: no JPEG frames in {frames_dir}")
    return frames


def load_hand_trajectory(scene_dir: Path):
    """Same as 52c/52d. Returns (traj, faces, canon, traj_centroids)."""
    per_frame_dir = scene_dir / "wilor" / "per_frame"
    faces_p = scene_dir / "wilor" / "faces.npy"
    if not per_frame_dir.is_dir() or not faces_p.is_file():
        sys.exit(f"error: WiLoR output missing under {scene_dir / 'wilor'}")
    faces = np.load(faces_p).astype(np.int64)
    traj = {}
    first = {False: None, True: None}
    centroids = {False: [], True: []}
    for npz_p in sorted(per_frame_dir.glob("*.npz")):
        fi = int(npz_p.stem)
        d = np.load(npz_p)
        verts = d["verts"]
        is_right_arr = d["is_right"].astype(bool)
        per_frame = []
        for i in range(verts.shape[0]):
            v = verts[i].astype(np.float64)
            key = bool(is_right_arr[i])
            per_frame.append({"is_right": key, "verts": v})
            centroids[key].append(v.mean(axis=0))
            if first[key] is None:
                first[key] = v.copy()
        traj[fi] = per_frame
    canon = {}
    traj_c = {}
    for k in (False, True):
        if first[k] is not None:
            canon[k] = first[k] - first[k].mean(axis=0)
            traj_c[k] = np.mean(centroids[k], axis=0)
    return traj, faces, canon, traj_c


# ---------------------------------------------------------------------------
# Math helpers (shared with 52c/52d)
# ---------------------------------------------------------------------------


def euler_xyz_to_R(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_quat_wxyz(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        return np.array([0.25 * s,
                         (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s,
                         0.25 * s,
                         (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s])
    if R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 2] - R[2, 0]) / s,
                         (R[0, 1] + R[1, 0]) / s,
                         0.25 * s,
                         (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[1, 0] - R[0, 1]) / s,
                     (R[0, 2] + R[2, 0]) / s,
                     (R[1, 2] + R[2, 1]) / s,
                     0.25 * s])


def euler_xyz_to_quat_wxyz(rx, ry, rz):
    return R_to_quat_wxyz(euler_xyz_to_R(rx, ry, rz))


def fit_rigid(P_canon, P_curr):
    c1 = P_canon.mean(axis=0)
    c2 = P_curr.mean(axis=0)
    H = (P_canon - c1).T @ (P_curr - c2)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2] *= -1.0
        R = Vt.T @ U.T
    t = c2 - R @ c1
    return R, t


def interpolate_keyframes(slot: int, keyframes, default_value: float) -> float:
    """Piecewise-linear interpolation through (slot, value) keyframes
    (sorted ascending by slot). Clamps to endpoints outside the range.

    Mirrors 52d's helper so the animation in 52e matches the rerun
    preview exactly. Values are in whatever native units the caller
    passed in — degrees → radians conversion happens at the call site."""
    if not keyframes:
        return float(default_value)
    if len(keyframes) == 1:
        return float(keyframes[0][1])
    if slot <= keyframes[0][0]:
        return float(keyframes[0][1])
    if slot >= keyframes[-1][0]:
        return float(keyframes[-1][1])
    for i in range(len(keyframes) - 1):
        s0, v0 = keyframes[i]
        s1, v1 = keyframes[i + 1]
        if s0 <= slot <= s1:
            span = max(1, s1 - s0)
            t = (slot - s0) / span
            return float(v0 + (v1 - v0) * t)
    return float(default_value)


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


def build_mjcf(fixed: dict, fixed_stl: str,
               moving: dict, moving_stl: str,
               joint_type: str,
               joint_pos_local: np.ndarray,
               joint_axis_local: np.ndarray,
               hand_stls: dict,
               hand_initial_pos: np.ndarray,
               stat_center: np.ndarray,
               stat_extent: float) -> str:
    """MJCF for the animated scene.

    - object_fixed: welded to ground (no joint), placed at the saved pose.
    - object_moving: child of worldbody with a real <joint type=hinge|slide
      pos=joint_pos_local axis=joint_axis_local>. Mesh placed at the saved
      baseline pose.
    - hand_left / hand_right: mocap bodies updated per frame via mocap_pos /
      mocap_quat. **Initialised at hand_initial_pos** (close to the scene)
      so the model's bounding box stays tight — otherwise MuJoCo's auto
      camera placement sees the (100, 100, 100) sentinel and zooms the
      viewer all the way out, leaving the actual scene a black dot.
    - <statistic center/extent> pins the auto-camera target to the scene
      centre regardless of bbox quirks (e.g. the missing-detection sentinel
      we use at run time).
    """
    fixed_quat = euler_xyz_to_quat_wxyz(*fixed["euler_deg_xyz"])
    moving_quat = euler_xyz_to_quat_wxyz(*moving["euler_deg_xyz"])

    def fmt(v):
        return " ".join(f"{x:.6f}" for x in np.asarray(v).ravel())

    fs = float(fixed["scale"])
    ms = float(moving["scale"])

    # Body colours: pick mid-saturation hues that stay clearly distinct
    # against a white background while still showing the directional-light
    # shading. Avoid near-white / pastel tints (they wash out on white)
    # and avoid near-black (kills the shading). Slate / orange / coral /
    # cobalt are all far apart on the hue wheel so the moving body and
    # hand_right don't visually collide as they did with two blues.
    RGBA_FIXED = "0.48 0.55 0.62 1"     # slate grey  (cool, matt)
    RGBA_MOVING = "0.95 0.58 0.22 1"    # warm orange (vivid, the door)
    RGBA_HAND_L = "0.92 0.40 0.42 1"    # coral red   (warm hand)
    RGBA_HAND_R = "0.30 0.45 0.85 1"    # cobalt blue (cool hand)

    asset_lines = [
        # Flat white skybox makes the viewer background render as pure
        # white. MuJoCo auto-uses skybox textures as the GL clear/skybox.
        '<texture name="skybox_white" type="skybox" builtin="flat" '
        'rgb1="1 1 1" rgb2="1 1 1" width="512" height="512" mark="none"/>',
        f'<mesh name="fixed_mesh" file="{fixed_stl}" scale="{fs} {fs} {fs}"/>',
        f'<mesh name="moving_mesh" file="{moving_stl}" scale="{ms} {ms} {ms}"/>',
    ]
    for name, stl in hand_stls.items():
        asset_lines.append(
            f'<mesh name="{name}_mesh" file="{stl.name}" scale="1 1 1"/>'
        )

    bodies = []
    bodies.append(
        f'<body name="object_fixed" '
        f'pos="{fmt(fixed["pos"])}" quat="{fmt(fixed_quat)}">'
        f'<geom type="mesh" mesh="fixed_mesh" rgba="{RGBA_FIXED}"/>'
        f'</body>'
    )
    bodies.append(
        f'<body name="object_moving" '
        f'pos="{fmt(moving["pos"])}" quat="{fmt(moving_quat)}">'
        f'<joint name="art" type="{joint_type}" '
        f'axis="{fmt(joint_axis_local)}" pos="{fmt(joint_pos_local)}"/>'
        f'<geom type="mesh" mesh="moving_mesh" rgba="{RGBA_MOVING}"/>'
        f'</body>'
    )
    for name in hand_stls:
        color = RGBA_HAND_L if "left" in name else RGBA_HAND_R
        bodies.append(
            f'<body name="{name}" mocap="true" '
            f'pos="{fmt(hand_initial_pos)}" quat="1 0 0 0">'
            f'<geom type="mesh" mesh="{name}_mesh" rgba="{color}"/>'
            f'</body>'
        )

    assets_xml = "\n    ".join(asset_lines)
    bodies_xml = "\n    ".join(bodies)

    return f"""<mujoco model="articulate4d_anim">
  <compiler angle="radian" autolimits="true" meshdir="."/>
  <option timestep="0.005"/>
  <statistic center="{fmt(stat_center)}" extent="{stat_extent:.4f}"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.45 0.45 0.45"/>
    <rgba haze="1 1 1 1"/>
  </visual>
  <asset>
    {assets_xml}
  </asset>
  <worldbody>
    <light directional="true" diffuse="0.6 0.6 0.6" dir="0 0 -1"/>
    <light directional="true" diffuse="0.3 0.3 0.3" dir="0 1 -0.5"/>
    {bodies_xml}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def save_view_to_file(cam, path: Path):
    """Snapshot the passive viewer's orbit camera (azimuth/elevation/
    distance/lookat) to JSON so the next run can restore it."""
    payload = {
        "azimuth": float(cam.azimuth),
        "elevation": float(cam.elevation),
        "distance": float(cam.distance),
        "lookat": [float(x) for x in cam.lookat],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  saved view -> {path}")


def load_view_from_file(cam, path: Path):
    """Restore camera state previously written by save_view_to_file."""
    view = json.loads(path.read_text())
    cam.azimuth = float(view["azimuth"])
    cam.elevation = float(view["elevation"])
    cam.distance = float(view["distance"])
    cam.lookat[:] = [float(x) for x in view["lookat"]]
    print(f"  loaded view <- {path}")


def set_mocap(model, data, name, pos, quat):
    try:
        bid = model.body(name).id
    except KeyError:
        return
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        return
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat


def joint_value(slot: int, n_total: int, drive_max: float,
                motion: str, n_cycles: float, keyframes=None) -> float:
    """Joint qpos at frame slot. drive_max + keyframe values are already in
    radians (hinge) or metres (slide).

    motion="keyframes" uses the per-slot piecewise-linear interpolation
    saved in transforms.json. Other modes synthesize a motion bounded by
    drive_max for scenes without keyframes."""
    if motion == "keyframes":
        return interpolate_keyframes(slot, keyframes or [], 0.0)
    t = slot / max(1, n_total - 1) if n_total > 1 else 0.0
    if motion == "static":
        return drive_max
    if motion == "linear":
        return drive_max * t
    if motion == "triangle":
        cycle = n_cycles * t * 2.0
        phase = cycle - np.floor(cycle)
        v = phase if (int(np.floor(cycle)) % 2 == 0) else (1.0 - phase)
        return drive_max * v
    if motion == "sine":
        return drive_max * np.sin(np.pi * n_cycles * t)
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--transforms", type=Path, default=None,
                   help="Path to transforms.json "
                        "(default: <scene>/rerun/transforms.json)")
    p.add_argument("--fps", type=float, default=15.0,
                   help="Playback rate in Hz (default 15)")
    p.add_argument("--joint-motion",
                   choices=["keyframes", "static", "linear", "triangle", "sine"],
                   default=None,
                   help="How the joint drives over the trajectory. Default: "
                        "`keyframes` if transforms.json has any saved "
                        "joint_keyframes, otherwise `linear` (0 -> "
                        "drive_at_save). Pass an explicit value to "
                        "override.")
    p.add_argument("--joint-amplitude", type=float, default=1.0,
                   help="Scale factor on drive_at_save (default 1.0).")
    p.add_argument("--n-cycles", type=float, default=1.0,
                   help="Number of cycles for triangle / sine motion "
                        "(default 1).")
    p.add_argument("--max-faces", type=int, default=150_000,
                   help="Decimate object meshes to this face count "
                        "(MuJoCo STL caps at 200000; default 150000).")
    p.add_argument("--once", action="store_true",
                   help="Play through once then exit. Default: loop.")
    p.add_argument("--view-file", type=Path, default=None,
                   help="JSON file holding the saved orbit-camera state "
                        "(default: <scene>/mujoco_anim/view.json). Auto-"
                        "loaded at startup if it exists. Press V in the "
                        "viewer to overwrite it with the current view.")
    p.add_argument("--reset-view", action="store_true",
                   help="Skip auto-loading the saved view this run.")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    transforms_path = (args.transforms.resolve() if args.transforms
                       else scene_dir / "rerun" / "transforms.json")
    if not transforms_path.is_file():
        sys.exit(f"error: {transforms_path} not found. "
                 f"Run 52d and click Save first.")
    print(f"transforms: {transforms_path.relative_to(scene_dir)}")
    transforms = json.loads(transforms_path.read_text())
    provenance = transforms["provenance"]

    save_dir = scene_dir / "mujoco_anim"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Object meshes ----
    fixed_label = provenance["fixed_label"]
    moving_label = provenance["moving_label"]
    fixed_glb = find_label_mesh(scene_dir, fixed_label, 0)
    moving_glb = find_label_mesh(scene_dir, moving_label, 0)
    print(f"fixed:  {fixed_glb.relative_to(scene_dir)} ({fixed_label})")
    print(f"moving: {moving_glb.relative_to(scene_dir)} ({moving_label})")

    fixed_v, fixed_f = load_glb(fixed_glb)
    moving_v, moving_f = load_glb(moving_glb)
    fixed_stl = save_dir / "object_fixed.stl"
    moving_stl = save_dir / "object_moving.stl"
    center_and_export_mesh(fixed_v, fixed_f, fixed_stl, args.max_faces)
    center_and_export_mesh(moving_v, moving_f, moving_stl, args.max_faces)

    # ---- Hand trajectory ----
    traj, hand_faces, canon, traj_centroids = load_hand_trajectory(scene_dir)
    frame_indices = list_scene_frames(scene_dir)
    n_with_hands = sum(1 for fi in frame_indices if fi in traj)
    print(f"trajectory: {len(frame_indices)} scene frames, "
          f"{n_with_hands} with WiLoR hands")

    hand_stls = {}
    for is_right, name in [(False, "hand_left"), (True, "hand_right")]:
        if is_right not in canon:
            continue
        stl = save_dir / f"{name}.stl"
        trimesh.Trimesh(vertices=canon[is_right], faces=hand_faces,
                        process=False).export(str(stl), file_type="stl")
        hand_stls[name] = stl
    print(f"hand bodies: {list(hand_stls.keys())}")

    # ---- Joint in moving body's local frame ----
    saved_joint = transforms["joint"]
    saved_moving = transforms["moving_baseline"]
    baseline_pos = np.array(saved_moving["pos"], dtype=np.float64)
    baseline_euler = np.array(saved_moving["euler_deg_xyz"], dtype=np.float64)
    R_baseline = euler_xyz_to_R(*baseline_euler)
    R_joint_world = euler_xyz_to_R(*saved_joint["euler_world_deg_xyz"])
    joint_axis_world = R_joint_world @ np.array([0., 0., 1.])
    joint_pos_world = np.array(saved_joint["pos_world"], dtype=np.float64)
    joint_pos_local = R_baseline.T @ (joint_pos_world - baseline_pos)
    joint_axis_local = R_baseline.T @ joint_axis_world

    # ---- Hand baseline (also used as the model's compile-time hand pos) ----
    saved_hand_baseline = transforms["hand_baseline"]
    hand_baseline_pos = np.array(saved_hand_baseline["pos"],
                                  dtype=np.float64)
    R_hand_user = euler_xyz_to_R(*saved_hand_baseline["euler_deg_xyz"])

    # ---- Auto-camera anchor ----
    # MuJoCo's viewer uses <statistic> to pick the initial camera distance.
    # Centre on the midpoint of the two objects, with an extent that spans
    # them generously. Avoids the all-black-viewer trap where the camera
    # lands far enough away that the headlight can't reach the geometry.
    fixed_pos_np = np.array(transforms["fixed"]["pos"], dtype=np.float64)
    moving_pos_np = np.array(saved_moving["pos"], dtype=np.float64)
    stat_center = (fixed_pos_np + moving_pos_np) / 2.0
    body_separation = float(np.linalg.norm(fixed_pos_np - moving_pos_np))
    stat_extent = max(0.5, body_separation * 2.0 + 0.5)

    # ---- Build + compile MJCF ----
    xml = build_mjcf(
        transforms["fixed"], fixed_stl.name,
        saved_moving, moving_stl.name,
        saved_joint["type"], joint_pos_local, joint_axis_local,
        hand_stls,
        hand_initial_pos=hand_baseline_pos,
        stat_center=stat_center,
        stat_extent=stat_extent,
    )
    xml_path = save_dir / "scene.xml"
    xml_path.write_text(xml)
    print(f"wrote {xml_path.relative_to(scene_dir)}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    art_jid = model.joint("art").id
    art_qadr = int(model.jnt_qposadr[art_jid])

    # ---- Drive parameters ----
    if saved_joint["type"] == "hinge":
        drive_max = np.deg2rad(
            float(transforms["drive_at_save"]["hinge_deg"])
        )
        drive_units = "deg"
    else:
        drive_max = float(transforms["drive_at_save"]["slide_m"])
        drive_units = "m"
    drive_max *= args.joint_amplitude

    # ---- Keyframes (parsed into joint-native units: rad for hinge, m for slide).
    # joint_amplitude scales keyframe values too so previewing a bigger /
    # smaller motion stays consistent with the synthetic modes.
    raw_keyframes = transforms.get("joint_keyframes", []) or []
    if saved_joint["type"] == "hinge":
        keyframes = [
            (int(kf["frame_slot"]),
             float(np.deg2rad(float(kf["value"]))) * args.joint_amplitude)
            for kf in raw_keyframes if "frame_slot" in kf and "value" in kf
        ]
    else:
        keyframes = [
            (int(kf["frame_slot"]),
             float(kf["value"]) * args.joint_amplitude)
            for kf in raw_keyframes if "frame_slot" in kf and "value" in kf
        ]
    keyframes.sort(key=lambda x: x[0])

    # Smart default: keyframes if present, linear otherwise.
    if args.joint_motion is None:
        args.joint_motion = "keyframes" if keyframes else "linear"
    if args.joint_motion == "keyframes" and not keyframes:
        print("warning: --joint-motion=keyframes but transforms.json has no "
              "joint_keyframes; falling back to linear.")
        args.joint_motion = "linear"

    drive_display = (np.rad2deg(drive_max) if saved_joint["type"] == "hinge"
                     else drive_max)
    print(f"\nanimating {len(frame_indices)} frames at {args.fps:.1f} FPS, "
          f"joint motion: {args.joint_motion}")
    if args.joint_motion == "keyframes":
        unit_display = "deg" if saved_joint["type"] == "hinge" else "m"
        kf_preview = ", ".join(
            f"({s}, {np.rad2deg(v) if saved_joint['type']=='hinge' else v:+.2f} {unit_display})"
            for s, v in keyframes[:4]
        )
        more = " …" if len(keyframes) > 4 else ""
        print(f"  {len(keyframes)} keyframe(s): {kf_preview}{more}"
              + (f"  ×amplitude {args.joint_amplitude}"
                 if args.joint_amplitude != 1.0 else ""))
    else:
        print(f"  amplitude {drive_display:+.2f} {drive_units}, "
              f"{args.n_cycles:g} cycle"
              f"{'s' if args.n_cycles != 1 else ''}")
    print("close the viewer window to exit "
          f"{'(--once: single pass)' if args.once else '(loops)'}.")

    dt = 1.0 / args.fps
    n_total = len(frame_indices)

    # Camera view persistence. View JSON lives next to scene.xml by default
    # so it follows the scene across re-runs. Press V in the viewer to save.
    view_path = (args.view_file.resolve() if args.view_file
                 else save_dir / "view.json")
    view_ref = {"viewer": None}

    def key_callback(key):
        # GLFW letter codes are the uppercase ASCII value, so both 'v' and
        # Shift+V land here regardless of caps-lock state.
        if key == ord('V') and view_ref["viewer"] is not None:
            save_view_to_file(view_ref["viewer"].cam, view_path)

    print(f"viewer keys: press V to save current camera view to "
          f"{view_path.name}")
    if view_path.is_file() and not args.reset_view:
        print(f"  will restore view from {view_path.name} once viewer opens")
    elif args.reset_view and view_path.is_file():
        print(f"  --reset-view: ignoring saved {view_path.name} this run")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback,
    ) as viewer:
        view_ref["viewer"] = viewer
        if view_path.is_file() and not args.reset_view:
            try:
                load_view_from_file(viewer.cam, view_path)
            except Exception as e:
                print(f"  warning: failed to load view "
                      f"({type(e).__name__}: {e}); using default")
        loop_count = 0
        while viewer.is_running():
            for slot, fi in enumerate(frame_indices):
                if not viewer.is_running():
                    break
                # Joint drive (keyframe-interpolated, or synthetic per --joint-motion)
                data.qpos[art_qadr] = joint_value(
                    slot, n_total, drive_max,
                    args.joint_motion, args.n_cycles,
                    keyframes=keyframes,
                )

                # Hand mocap per laterality
                hands_now = traj.get(fi, [])
                for is_right, body_name in [(False, "hand_left"),
                                            (True, "hand_right")]:
                    if is_right not in canon:
                        continue
                    found = next(
                        (h for h in hands_now if h["is_right"] == is_right),
                        None,
                    )
                    if found is None:
                        set_mocap(model, data, body_name,
                                  np.array([100.0, 100.0, 100.0]),
                                  np.array([1.0, 0.0, 0.0, 0.0]))
                    else:
                        R_fit, t_fit = fit_rigid(canon[is_right],
                                                 found["verts"])
                        t_centered = t_fit - traj_centroids.get(
                            is_right, np.zeros(3)
                        )
                        final_t = R_hand_user @ t_centered + hand_baseline_pos
                        final_R = R_hand_user @ R_fit
                        set_mocap(model, data, body_name, final_t,
                                  R_to_quat_wxyz(final_R))

                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)

            loop_count += 1
            if args.once:
                break


if __name__ == "__main__":
    main()
