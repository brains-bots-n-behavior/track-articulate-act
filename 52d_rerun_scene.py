#!/usr/bin/env python
"""Stage 52d (Rerun variant of 52c): two sam3d objects + WiLoR hand
trajectory + any4d scene flow, visualised in Rerun, edited via the same
gradio slider UI as 52c.

The arrangement of bodies (fixed body, moving body articulated about a
joint, hand trajectory placed via a user-set baseline) is identical to 52c.
The difference is the visualisation: instead of a MuJoCo passive viewer,
this script logs to a Rerun recording and uses Rerun's entity Transform3D
hierarchy so that slider edits only re-log cheap Transform3Ds (meshes are
uploaded once as static archetypes). No real joint constraint is enforced
here — the "drive" slider applies the rotation/translation manually as in
52c's original mocap path.

Reads:
    data/<scene>/sam3d/<label-fixed>/[cand_NN_<kf>/]mesh.glb
    data/<scene>/sam3d/<label-moving>/[cand_NN_<kf>/]mesh.glb
    data/<scene>/wilor/per_frame/<frame>.npz   (full trajectory)
    data/<scene>/wilor/faces.npy               (shared MANO topology)
    data/<scene>/any4d/{config.json, pointmap_ref.npy, moge/mask/*}
    data/<scene>/any4d/<label>/{pts3d_ref.npy, scene_flow/<frame>.npy}
    data/<scene>/frames/<frame>.jpg            (frame range + ref RGB)

Editor:
    Same slider grid as 52c (fixed body, moving body baseline, joint pose,
    hand baseline, drive, frame slot, play/pause). Slider changes re-log
    Transform3D entities — viewer updates immediately. Scene-flow arrows
    and the per-frame hand pose refresh whenever the frame slider moves.

Save (data/<scene>/rerun/):
    transforms.json   Slider state + provenance (same schema as 52c minus
                      the MJCF-specific fields).

Run inside any env with rerun-sdk + gradio + numpy + trimesh + pillow.

Examples:
    python scripts/52d_rerun_scene.py \\
        --scene-dir data/macbook-all \\
        --label-fixed laptop_base --label-moving laptop_up \\
        --joint hinge --fps 15
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from matplotlib.colors import hsv_to_rgb

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    sys.exit("error: rerun-sdk not installed. `pip install rerun-sdk`")

try:
    import gradio as gr
except ImportError:
    sys.exit("error: gradio not installed. `pip install gradio`")


# ---------------------------------------------------------------------------
# Mesh / data loading (mostly shared with 52c)
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
    return (np.asarray(mesh.vertices, dtype=np.float32),
            np.asarray(mesh.faces, dtype=np.int32))


def center_and_decimate(verts: np.ndarray, faces: np.ndarray, max_faces: int):
    """Centre verts at origin and (optionally) quadric-decimate to
    `max_faces`. Returns (centered_verts, faces, source_centroid)."""
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    if max_faces > 0 and faces.shape[0] > max_faces:
        print(f"  decimating {faces.shape[0]} -> ~{max_faces} faces ...")
        try:
            full = trimesh.Trimesh(vertices=centered, faces=faces,
                                   process=False)
            small = full.simplify_quadric_decimation(face_count=max_faces)
            centered = np.asarray(small.vertices, dtype=np.float32)
            faces = np.asarray(small.faces, dtype=np.int32)
            print(f"    -> {centered.shape[0]} verts, {faces.shape[0]} faces "
                  f"(quadric)")
        except Exception as e:
            rng = np.random.default_rng(0)
            keep = rng.choice(faces.shape[0], max_faces, replace=False)
            faces = faces[keep]
            print(f"    quadric decimation failed ({type(e).__name__}: {e}); "
                  f"random-subsampled to {max_faces} faces")
    return centered, faces, centroid


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
    """Same as 52c. Returns (traj, faces, canon, traj_centroids)."""
    per_frame_dir = scene_dir / "wilor" / "per_frame"
    faces_p = scene_dir / "wilor" / "faces.npy"
    if not per_frame_dir.is_dir() or not faces_p.is_file():
        sys.exit(f"error: WiLoR output missing under {scene_dir / 'wilor'}")
    faces = np.load(faces_p).astype(np.int32)
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
            canon[k] = (first[k] - first[k].mean(axis=0)).astype(np.float32)
            traj_c[k] = np.mean(centroids[k], axis=0)
    return traj, faces, canon, traj_c


def load_any4d_label_data(scene_dir: Path, labels):
    """Returns (any4d_root, label_data dict) for the scene-flow display.
    label_data[label] = {'pts3d_ref': (N,3) float, 'flow_dir': Path}.
    Missing labels are silently dropped."""
    any4d_root = scene_dir / "any4d"
    if not any4d_root.is_dir():
        return None, {}
    label_data = {}
    for label in labels:
        ldir = any4d_root / label
        pts_p = ldir / "pts3d_ref.npy"
        flow_dir = ldir / "scene_flow"
        if pts_p.is_file() and flow_dir.is_dir():
            label_data[label] = {
                "pts3d_ref": np.load(pts_p).astype(np.float32),
                "flow_dir": flow_dir,
            }
    return any4d_root, label_data


# ---------------------------------------------------------------------------
# Math helpers (shared with 52c)
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


def axis_angle_to_R(axis, theta_rad):
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta_rad) * K + (1.0 - np.cos(theta_rad)) * (K @ K)


def fit_rigid(P_canon, P_curr):
    P_canon = np.asarray(P_canon, dtype=np.float64)
    P_curr = np.asarray(P_curr, dtype=np.float64)
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


def load_existing_transforms(path: Path):
    """Read a previously-saved transforms.json. Returns the parsed dict or
    None if the file is missing / malformed. Tolerant of older saves that
    predate joint_keyframes / frame_time_step."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"warning: could not parse {path}: {type(e).__name__}: {e}")
        return None


def apply_loaded_transforms(state, loaded: dict):
    """Push every field from a saved transforms.json into SceneState.
    Each top-level key is `.get`-style optional so a partial / old file
    still gets us the fields it does contain."""
    if "fixed" in loaded:
        f = loaded["fixed"]
        state.fixed_pos = np.array(f.get("pos", state.fixed_pos), dtype=np.float64)
        state.fixed_euler = np.array(f.get("euler_deg_xyz", state.fixed_euler),
                                     dtype=np.float64)
        state.fixed_scale = float(f.get("scale", state.fixed_scale))
    if "moving_baseline" in loaded:
        m = loaded["moving_baseline"]
        state.moving_baseline_pos = np.array(
            m.get("pos", state.moving_baseline_pos), dtype=np.float64)
        state.moving_baseline_euler = np.array(
            m.get("euler_deg_xyz", state.moving_baseline_euler),
            dtype=np.float64)
        state.moving_scale = float(m.get("scale", state.moving_scale))
    if "joint" in loaded:
        j = loaded["joint"]
        state.joint_type = j.get("type", state.joint_type)
        state.joint_pos = np.array(j.get("pos_world", state.joint_pos),
                                   dtype=np.float64)
        state.joint_euler = np.array(
            j.get("euler_world_deg_xyz", state.joint_euler),
            dtype=np.float64)
    if "drive_at_save" in loaded:
        d = loaded["drive_at_save"]
        state.drive_hinge_deg = float(d.get("hinge_deg", state.drive_hinge_deg))
        state.drive_slide_m = float(d.get("slide_m", state.drive_slide_m))
    if "hand_baseline" in loaded:
        h = loaded["hand_baseline"]
        state.hand_baseline_pos = np.array(
            h.get("pos", state.hand_baseline_pos), dtype=np.float64)
        state.hand_baseline_euler = np.array(
            h.get("euler_deg_xyz", state.hand_baseline_euler),
            dtype=np.float64)
    if "joint_keyframes" in loaded:
        state.joint_keyframes = sorted(
            ((int(kf["frame_slot"]), float(kf["value"]))
             for kf in loaded["joint_keyframes"]
             if "frame_slot" in kf and "value" in kf),
            key=lambda x: x[0],
        )
    if "frame_time_step" in loaded:
        # Honour the saved timeline spacing; pre_log_frame_timeline will
        # overwrite this with the CLI value, so the loaded value is mainly
        # useful for the velocity-profile units math.
        state.frame_time_step = float(loaded["frame_time_step"])


def interpolate_keyframes(slot: int, keyframes, default_value: float) -> float:
    """Piecewise-linear interpolation through (slot, value) keyframes
    (sorted ascending by slot). Clamps to endpoints outside the range.

    Returns default_value when keyframes is empty so the legacy "single
    drive slider" path keeps working."""
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
# Scene state (plain numpy fields; no model/data, no thread lock needed —
# gradio serialises callbacks for us)
# ---------------------------------------------------------------------------


class SceneState:
    def __init__(self, hand_traj, canon_hands, traj_centroids,
                 frame_indices, joint_type,
                 any4d_label_data, max_arrows,
                 pos_center=None, frame_time_step=0.2):
        self.hand_traj = hand_traj
        self.canon_hands = canon_hands
        self.traj_centroids = traj_centroids
        self.frame_indices = frame_indices
        self.joint_type = joint_type
        self.label_data = any4d_label_data
        self.max_arrows = max_arrows
        # pos_center anchors all body defaults inside any4d's world frame so
        # the user-controlled meshes appear near the pointcloud / scene flow
        # instead of stranded at world origin (which is where any4d puts
        # the *camera*, not the scene).
        pc = (np.asarray(pos_center, dtype=np.float64)
              if pos_center is not None else np.zeros(3))
        self.pos_center = pc

        self.fixed_pos = pc.copy()
        self.fixed_euler = np.zeros(3)
        self.fixed_scale = 1.0
        self.moving_baseline_pos = pc + np.array([0.25, 0.0, 0.0])
        self.moving_baseline_euler = np.zeros(3)
        self.moving_scale = 1.0
        self.joint_pos = pc + np.array([0.125, 0.0, 0.0])
        self.joint_euler = np.zeros(3)
        self.drive_hinge_deg = 0.0
        self.drive_slide_m = 0.0
        self.hand_baseline_pos = pc.copy()
        self.hand_baseline_euler = np.zeros(3)

        # Joint keyframes: sorted list of (slot, value) tuples. When empty,
        # the drive slider value is broadcast across all frames. When
        # populated, joint value at frame slot k is the piecewise-linear
        # interpolation through the keyframes (clamped at endpoints).
        # Value units track joint_type — deg for hinge, m for slide.
        self.joint_keyframes: list = []
        self.frame_time_step = float(frame_time_step)


# ---------------------------------------------------------------------------
# Rerun setup + static logs
# ---------------------------------------------------------------------------


def _uniform_colors(n: int, rgb_u8):
    return np.tile(np.array(rgb_u8, dtype=np.uint8), (n, 1))


def init_rerun(rerun_port: int):
    rr.init("articulate4d_52d")
    rr.spawn(port=rerun_port)
    rr.log("pred", rr.ViewCoordinates.RDF, static=True)
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(
            origin="pred",
            name="3D Scene",
            background=[255, 255, 255],
            line_grid=rrb.archetypes.LineGrid3D(visible=False),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)


def log_static_meshes(fixed_verts, fixed_faces,
                      moving_verts, moving_faces,
                      canon_hands, hand_faces):
    """Upload every mesh once (static). Per-slider updates only re-log
    Transform3Ds and never touch these archetypes again."""
    rr.log(
        "pred/objects/fixed/mesh",
        rr.Mesh3D(
            vertex_positions=fixed_verts,
            triangle_indices=fixed_faces,
            vertex_colors=_uniform_colors(len(fixed_verts), [191, 191, 199]),
        ),
        static=True,
    )
    rr.log(
        "pred/objects/moving/at_pivot/spin/back/baseline/mesh",
        rr.Mesh3D(
            vertex_positions=moving_verts,
            triangle_indices=moving_faces,
            vertex_colors=_uniform_colors(len(moving_verts), [140, 165, 217]),
        ),
        static=True,
    )
    for is_right, name, color in [
        (False, "left",  [230, 107, 107]),
        (True,  "right", [107, 107, 230]),
    ]:
        if is_right not in canon_hands:
            continue
        rr.log(
            f"pred/hands/{name}/per_frame/mesh",
            rr.Mesh3D(
                vertex_positions=canon_hands[is_right],
                triangle_indices=hand_faces,
                vertex_colors=_uniform_colors(
                    len(canon_hands[is_right]), color
                ),
            ),
            static=True,
        )
    # Joint marker: a small sphere at the origin and an arrow along +Z.
    # The marker entity's Transform3D moves the whole assembly to joint
    # pos and orients it so the arrow lines up with the user's axis.
    rr.log(
        "pred/joint/marker/sphere",
        rr.Points3D(positions=[[0.0, 0.0, 0.0]],
                    radii=[0.012],
                    colors=[[50, 220, 60]]),
        static=True,
    )
    rr.log(
        "pred/joint/marker/axis",
        rr.Arrows3D(origins=[[0.0, 0.0, 0.0]],
                    vectors=[[0.0, 0.0, 0.10]],
                    colors=[[50, 220, 60]]),
        static=True,
    )


def load_ref_pointcloud(any4d_root: Path, scene_dir: Path, frame_indices):
    """Load the masked reference pointcloud + colors. Returns
    (pts, colors, ref_fi) or None if unavailable. Separating load from log
    so we can compute the scene's spatial anchor (median centroid) for the
    slider defaults even when the user passes --no-pointcloud."""
    if any4d_root is None:
        return None
    ptmap_path = any4d_root / "pointmap_ref.npy"
    cfg_path = any4d_root / "config.json"
    if not ptmap_path.is_file():
        return None
    ptmap = np.load(ptmap_path)
    try:
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    except Exception:
        cfg = {}
    ref_fi = int(cfg.get("ref_frame", frame_indices[0]))
    ref_rgb_path = scene_dir / "frames" / f"{ref_fi:06d}.jpg"
    if not ref_rgb_path.is_file():
        return None
    ref_rgb_full = np.array(Image.open(ref_rgb_path).convert("RGB"))
    Hf, Wf = ptmap.shape[:2]
    rgb_resized = np.array(Image.fromarray(ref_rgb_full).resize(
        (Wf, Hf), Image.BILINEAR))
    mask = None
    moge_mask = any4d_root / "moge" / "mask" / f"{ref_fi:06d}.png"
    if moge_mask.is_file():
        mfull = np.array(Image.open(moge_mask))
        if mfull.ndim == 3:
            mfull = mfull[..., -1]
        mask = np.array(Image.fromarray(
            (mfull > 0).astype(np.uint8) * 255
        ).resize((Wf, Hf), Image.NEAREST)) > 0
    pts = ptmap.reshape(-1, 3)
    cols = rgb_resized.reshape(-1, 3)
    if mask is not None:
        m = mask.reshape(-1)
        pts = pts[m]
        cols = cols[m]
    return pts, cols, ref_fi


def log_ref_pointcloud(pts: np.ndarray, cols: np.ndarray, ref_fi: int):
    rr.log("pred/ref_pointcloud",
           rr.Points3D(positions=pts, colors=cols),
           static=True)
    print(f"ref pointcloud at frame {ref_fi}: {pts.shape[0]} points")


# ---------------------------------------------------------------------------
# Per-slider Transform3D re-logs (cheap; meshes are not touched)
# ---------------------------------------------------------------------------


def log_object_transforms(state: SceneState):
    """All slider-driven Transform3Ds are logged STATIC — they apply at
    every point on the stable_time timeline, so slider edits never advance
    the frame cursor.

    Drive composition (with keyframes) uses a 4-level hierarchy under the
    moving body so that joint params and keyframes are independent:

        pred/objects/moving/at_pivot              T_pivot      (STATIC)
            T_pivot = translate(joint_pos) @ rotate(R_joint)
        pred/objects/moving/at_pivot/spin         T_spin       (PER-FRAME)
            For hinge: rotation about local Z by theta(slot)
            For slide: translation along local Z by d(slot)
        pred/objects/moving/at_pivot/spin/back    T_pivot_inv  (STATIC)
            T_pivot_inv = rotate(R_joint^T) @ translate(-R_joint^T·joint_pos)
        pred/objects/moving/at_pivot/spin/back/baseline T_baseline (STATIC)
            Moving body's baseline pose + scale
        pred/.../baseline/mesh                    Mesh3D       (STATIC)

    Composition (parent ⨉ child):
        world = T_pivot · T_spin · T_pivot_inv · T_baseline · scale · v
              = T_drive · T_baseline · scale · v   (T_drive about pivot)

    With this, joint-pos / joint-euler slider edits re-log only at_pivot +
    back (2 static logs); keyframe / joint-type edits re-log only the
    per-frame spin chain. Body baseline + scale stays in /baseline.
    """
    R_fixed = euler_xyz_to_R(*state.fixed_euler)
    rr.log("pred/objects/fixed",
           rr.Transform3D(translation=state.fixed_pos,
                          mat3x3=R_fixed,
                          scale=float(state.fixed_scale)),
           static=True)

    R_baseline = euler_xyz_to_R(*state.moving_baseline_euler)
    R_joint = euler_xyz_to_R(*state.joint_euler)

    # at_pivot: translate to joint_pos and rotate by R_joint.
    # Order in Transform3D is T·R·S, so translation=joint_pos, rotation=R_joint
    # already gives the right composition.
    rr.log("pred/objects/moving/at_pivot",
           rr.Transform3D(translation=state.joint_pos, mat3x3=R_joint),
           static=True)

    # back: T_pivot_inv = R_joint^T · translate(-joint_pos).
    # Apply to point x: R_joint^T · (x - joint_pos) = R_joint^T · x - R_joint^T·joint_pos.
    # In Transform3D T·R·S form that's translation = -R_joint^T·joint_pos,
    # rotation = R_joint^T.
    R_joint_inv = R_joint.T
    rr.log("pred/objects/moving/at_pivot/spin/back",
           rr.Transform3D(translation=-(R_joint_inv @ state.joint_pos),
                          mat3x3=R_joint_inv),
           static=True)

    # baseline: moving body's slider-driven baseline pose + scale.
    rr.log("pred/objects/moving/at_pivot/spin/back/baseline",
           rr.Transform3D(translation=state.moving_baseline_pos,
                          mat3x3=R_baseline,
                          scale=float(state.moving_scale)),
           static=True)

    # Joint marker (sphere + axis capsule) — independent visualization.
    rr.log("pred/joint/marker",
           rr.Transform3D(translation=state.joint_pos, mat3x3=R_joint),
           static=True)


def log_hand_baseline(state: SceneState):
    """Hand baseline pose is slider-controlled and applies to every frame,
    so log STATIC. The per-frame fit lives at a child entity and gets
    pre-logged with a stable_time stamp."""
    R_user = euler_xyz_to_R(*state.hand_baseline_euler)
    for is_right, name in [(False, "left"), (True, "right")]:
        if is_right not in state.canon_hands:
            continue
        rr.log(f"pred/hands/{name}",
               rr.Transform3D(translation=state.hand_baseline_pos,
                              mat3x3=R_user),
               static=True)


def log_scene_flow_at_frame(state: SceneState, fi: int):
    """Log scene-flow arrows for ONE frame at the current time context.
    Caller is responsible for setting stable_time before calling."""
    if not state.label_data:
        return
    for label, ld in state.label_data.items():
        flow_path = ld["flow_dir"] / f"{fi:06d}.npy"
        path = f"pred/scene_flow/{label}"
        if not flow_path.is_file():
            rr.log(path, rr.Clear(recursive=False))
            continue
        flow = np.load(flow_path).astype(np.float32)
        pts = ld["pts3d_ref"].reshape(-1, 3)
        vecs = flow.reshape(-1, 3)
        if pts.shape[0] != vecs.shape[0]:
            continue

        if pts.shape[0] > state.max_arrows:
            mags = np.linalg.norm(vecs, axis=1)
            if mags.max() > 1e-6:
                probs = 0.2 + 0.8 * (mags / (mags.max() + 1e-6))
                probs /= probs.sum()
                idx = np.random.choice(len(pts), size=state.max_arrows,
                                       replace=False, p=probs)
            else:
                idx = np.random.permutation(len(pts))[:state.max_arrows]
            pts = pts[idx]
            vecs = vecs[idx]

        mags = np.linalg.norm(vecs, axis=1)
        mn, mx = float(mags.min()), float(mags.max())
        if mx == mn:
            mx = mn + 1e-6
        directions = vecs / (mags[:, None] + 1e-8)
        hue = (np.arctan2(directions[:, 2], directions[:, 0]) + np.pi) / (2 * np.pi)
        norm_mag = np.clip((mags - mn) / (mx - mn + 1e-8), 0.0, 1.0)
        hsv = np.stack([hue, 0.3 + 0.7 * norm_mag, 0.5 + 0.5 * norm_mag], axis=1)
        rgb = hsv_to_rgb(hsv)
        alpha = np.full((len(rgb), 1), 0.7)
        colors = np.concatenate([rgb, alpha], axis=1)
        rr.log(path, rr.Arrows3D(origins=pts, vectors=vecs, colors=colors))


def pre_log_drive_per_frame(state: SceneState):
    """Walk every scene frame and log T_spin (at_pivot/spin) per frame on
    the stable_time timeline. The spin Transform3D in the pivot's LOCAL
    frame is a rotation about local Z by theta (hinge) or a translation
    along local Z by d (slide). Called at startup AND on every keyframe /
    joint-type edit.

    theta / d at each slot come from interpolate_keyframes(state.joint_keyframes);
    when keyframes is empty we fall back to the static drive slider value."""
    if not state.frame_indices:
        return
    default_value = (state.drive_hinge_deg if state.joint_type == "hinge"
                     else state.drive_slide_m)
    is_hinge = (state.joint_type == "hinge")
    path = "pred/objects/moving/at_pivot/spin"
    for slot in range(len(state.frame_indices)):
        rr.set_time_seconds("stable_time", state.frame_time_step * slot)
        value = interpolate_keyframes(slot, state.joint_keyframes, default_value)
        if is_hinge:
            theta = np.deg2rad(value)
            c, s = np.cos(theta), np.sin(theta)
            R_spin = np.array([[c, -s, 0.0],
                               [s,  c, 0.0],
                               [0.0, 0.0, 1.0]])
            rr.log(path,
                   rr.Transform3D(translation=np.zeros(3), mat3x3=R_spin))
        else:
            rr.log(path,
                   rr.Transform3D(translation=[0.0, 0.0, float(value)],
                                  mat3x3=np.eye(3)))
    rr.set_time_seconds("stable_time", 0.0)


def pre_log_frame_timeline(state: SceneState, frame_time_step: float):
    """Walk every scene frame once at startup, stamping each per-frame log
    with stable_time = frame_time_step * slot. After this runs, scrubbing
    rerun's stable_time timeline drives the hand animation + scene-flow
    display + joint drive. Slider edits (logged STATIC elsewhere) never
    advance the cursor."""
    if not state.frame_indices:
        return
    state.frame_time_step = float(frame_time_step)
    for slot, fi in enumerate(state.frame_indices):
        rr.set_time_seconds("stable_time", frame_time_step * slot)

        for is_right, name in [(False, "left"), (True, "right")]:
            if is_right not in state.canon_hands:
                continue
            found = next(
                (h for h in state.hand_traj.get(fi, [])
                 if h["is_right"] == is_right),
                None,
            )
            if found is None:
                # No detection at this frame: push the body off-screen for
                # this timestamp. The Mesh3D under per_frame stays static
                # so the hand can re-appear at the next frame's timestamp.
                rr.log(f"pred/hands/{name}/per_frame",
                       rr.Transform3D(translation=[1000.0, 1000.0, 1000.0],
                                      mat3x3=np.eye(3)))
            else:
                R_fit, t_fit = fit_rigid(state.canon_hands[is_right],
                                         found["verts"])
                t_centered = t_fit - state.traj_centroids.get(
                    is_right, np.zeros(3)
                )
                rr.log(f"pred/hands/{name}/per_frame",
                       rr.Transform3D(translation=t_centered, mat3x3=R_fit))

        log_scene_flow_at_frame(state, fi)

    # Joint drive (interpolated through keyframes) is logged per-frame too,
    # under at_pivot/spin. Done as a separate pass so keyframe edits can
    # re-run just this without touching hand/scene_flow logs.
    pre_log_drive_per_frame(state)

    # Park the time cursor at the first frame so subsequent static logs
    # don't bias toward the last timestamp.
    rr.set_time_seconds("stable_time", 0.0)


def log_static_state(state: SceneState):
    """Re-log every slider-driven Transform3D as STATIC. Cheap; called on
    each gradio slider change."""
    log_object_transforms(state)
    log_hand_baseline(state)


# ---------------------------------------------------------------------------
# Gradio UI (same slider grid as 52c)
# ---------------------------------------------------------------------------


def make_ui(state: SceneState, save_dir: Path, provenance: dict,
            pos_range: float = 1.5):
    POS_RANGE = float(pos_range)
    n_frames = len(state.frame_indices)
    # Per-axis slider centre = pointcloud anchor; range = ±POS_RANGE around it.
    cx, cy, cz = (float(state.pos_center[0]),
                  float(state.pos_center[1]),
                  float(state.pos_center[2]))

    n_left_total = sum(1 for hs in state.hand_traj.values()
                       for h in hs if not h["is_right"])
    n_right_total = sum(1 for hs in state.hand_traj.values()
                        for h in hs if h["is_right"])

    def info_text():
        axis_w = euler_xyz_to_R(*state.joint_euler) @ np.array([0., 0., 1.])
        kf_unit = "deg" if state.joint_type == "hinge" else "m"
        if state.joint_keyframes:
            kf_str = (
                f"{len(state.joint_keyframes)} keyframe(s): "
                + ", ".join(f"({s}, {v:+.2f} {kf_unit})"
                            for s, v in state.joint_keyframes[:4])
                + (" …" if len(state.joint_keyframes) > 4 else "")
            )
        else:
            kf_str = (f"none — drive slider held constant at "
                      f"{state.drive_hinge_deg:+.1f}° / "
                      f"{state.drive_slide_m:+.3f} m")
        return (
            f"**Trajectory** {n_frames} scene frames "
            f"(L: {n_left_total}, R: {n_right_total} detections) — "
            f"scrub `stable_time` in the rerun viewer to play.  \n"
            f"**Fixed** pos=({state.fixed_pos[0]:+.3f}, "
            f"{state.fixed_pos[1]:+.3f}, {state.fixed_pos[2]:+.3f}) "
            f"scale={state.fixed_scale:.3f}  \n"
            f"**Moving** baseline=({state.moving_baseline_pos[0]:+.3f}, "
            f"{state.moving_baseline_pos[1]:+.3f}, "
            f"{state.moving_baseline_pos[2]:+.3f}) "
            f"scale={state.moving_scale:.3f}  \n"
            f"**Joint** type=`{state.joint_type}` "
            f"axis_world=({axis_w[0]:+.3f}, {axis_w[1]:+.3f}, "
            f"{axis_w[2]:+.3f})  \n"
            f"**Keyframes** {kf_str}"
        )

    def on_change(fx, fy, fz, frx, fry, frz, fs,
                  mx, my, mz, mrx, mry, mrz, ms,
                  jpx, jpy, jpz, jrx, jry, jrz,
                  hpx, hpy, hpz, hrx, hry, hrz,
                  dh, ds, jt):
        # Snapshot fields the per-frame drive prelog depends on so we can
        # skip the expensive re-walk when none of them changed.
        old_joint_type = state.joint_type
        old_dh = state.drive_hinge_deg
        old_ds = state.drive_slide_m

        state.fixed_pos = np.array([fx, fy, fz], dtype=np.float64)
        state.fixed_euler = np.array([frx, fry, frz], dtype=np.float64)
        state.fixed_scale = float(fs)
        state.moving_baseline_pos = np.array([mx, my, mz], dtype=np.float64)
        state.moving_baseline_euler = np.array([mrx, mry, mrz], dtype=np.float64)
        state.moving_scale = float(ms)
        state.joint_pos = np.array([jpx, jpy, jpz], dtype=np.float64)
        state.joint_euler = np.array([jrx, jry, jrz], dtype=np.float64)
        state.hand_baseline_pos = np.array([hpx, hpy, hpz], dtype=np.float64)
        state.hand_baseline_euler = np.array([hrx, hry, hrz], dtype=np.float64)
        state.drive_hinge_deg = float(dh)
        state.drive_slide_m = float(ds)
        state.joint_type = jt
        # Slider-driven Transform3Ds are STATIC — re-logging them does NOT
        # advance rerun's stable_time cursor.
        log_static_state(state)

        # Re-walk the per-frame T_spin only when something it depends on
        # actually changed: joint type (formula switches), or the drive
        # slider (only relevant when there are no keyframes overriding).
        drive_changed = (state.joint_type == "hinge" and dh != old_dh) or \
                        (state.joint_type == "slide" and ds != old_ds)
        if state.joint_type != old_joint_type or \
                (drive_changed and not state.joint_keyframes):
            pre_log_drive_per_frame(state)
        return info_text()

    def keyframes_md():
        """Markdown summary of the current keyframe list, ordered by slot."""
        if not state.joint_keyframes:
            return "**Keyframes** — (none; drive slider applies at every frame)"
        unit = "°" if state.joint_type == "hinge" else "m"
        rows = "\n".join(
            f"- slot **{s:4d}** → {v:+.3f}{unit}"
            for s, v in state.joint_keyframes
        )
        return (f"**Keyframes** — {len(state.joint_keyframes)} saved:\n"
                + rows)

    def on_add_keyframe(frame_slot, dh, ds):
        """Capture (frame_slot, current_drive_value) into the keyframe list.
        Replaces any existing keyframe at the same slot. Drive slider value
        is read from whichever knob matches the current joint_type."""
        slot = int(round(float(frame_slot)))
        if slot < 0 or slot >= n_frames:
            return keyframes_md(), info_text()
        value = float(dh) if state.joint_type == "hinge" else float(ds)
        seen = {s: v for s, v in state.joint_keyframes}
        seen[slot] = value
        state.joint_keyframes = sorted(seen.items())
        pre_log_drive_per_frame(state)
        return keyframes_md(), info_text()

    def on_remove_last_keyframe():
        if state.joint_keyframes:
            # Drop the last-added (highest-slot) keyframe rather than the
            # last-edited, since the list is sorted by slot. If the user
            # wanted a different one, Clear-and-readd is the escape hatch.
            state.joint_keyframes = state.joint_keyframes[:-1]
            pre_log_drive_per_frame(state)
        return keyframes_md(), info_text()

    def on_clear_keyframes():
        state.joint_keyframes = []
        pre_log_drive_per_frame(state)
        return keyframes_md(), info_text()

    def on_frame_slider_change(slot):
        """Tiny helper so the slider label shows the corresponding
        stable_time in seconds — match this against the rerun viewer's
        time-panel cursor."""
        s = int(round(float(slot)))
        return (f"Frame slot **{s}** ↔ rerun `stable_time = "
                f"{s * state.frame_time_step:.3f} s`")

    def on_save(overwrite):
        try:
            # Velocity profile: piecewise-constant velocity between adjacent
            # keyframes. Velocity = Δvalue / (Δslot · frame_time_step), so
            # units are deg/s for hinge or m/s for slide.
            kf_units = "deg" if state.joint_type == "hinge" else "m"
            vel_units = f"{kf_units}/s"
            velocity_profile = []
            for i in range(len(state.joint_keyframes) - 1):
                s0, v0 = state.joint_keyframes[i]
                s1, v1 = state.joint_keyframes[i + 1]
                dt = max(1, s1 - s0) * state.frame_time_step
                velocity_profile.append({
                    "from_slot": int(s0),
                    "to_slot": int(s1),
                    "from_value": float(v0),
                    "to_value": float(v1),
                    "velocity": float((v1 - v0) / dt),
                    "units": vel_units,
                })
            payload = {
                "fixed": {
                    "pos": state.fixed_pos.tolist(),
                    "euler_deg_xyz": state.fixed_euler.tolist(),
                    "scale": float(state.fixed_scale),
                },
                "moving_baseline": {
                    "pos": state.moving_baseline_pos.tolist(),
                    "euler_deg_xyz": state.moving_baseline_euler.tolist(),
                    "scale": float(state.moving_scale),
                },
                "joint": {
                    "type": state.joint_type,
                    "pos_world": state.joint_pos.tolist(),
                    "euler_world_deg_xyz": state.joint_euler.tolist(),
                },
                "drive_at_save": {
                    "hinge_deg": float(state.drive_hinge_deg),
                    "slide_m": float(state.drive_slide_m),
                },
                "joint_keyframes": [
                    {"frame_slot": int(s), "value": float(v),
                     "units": kf_units}
                    for s, v in state.joint_keyframes
                ],
                "joint_velocity_profile": velocity_profile,
                "frame_time_step": float(state.frame_time_step),
                "hand_baseline": {
                    "pos": state.hand_baseline_pos.tolist(),
                    "euler_deg_xyz": state.hand_baseline_euler.tolist(),
                },
                "provenance": provenance,
            }
            save_dir.mkdir(parents=True, exist_ok=True)
            json_path = save_dir / "transforms.json"
            if json_path.exists() and not overwrite:
                return (f"**SKIP**: {json_path.name} exists. "
                        f"Tick `overwrite` to replace.")
            json_path.write_text(json.dumps(payload, indent=2))
            return (f"**Saved** `{json_path.name}` to `{save_dir}`. "
                    f"(Meshes live in `sam3d/`; see provenance.)")
        except Exception as e:
            return f"**FAIL**: {type(e).__name__}: {e}"

    with gr.Blocks(title="Articulate4D Rerun scene") as demo:
        gr.Markdown("# Two-body articulation + hand trajectory (Rerun)")
        gr.Markdown(
            "Slider edits re-log slider-driven Transform3Ds as **static** — "
            "they apply at every timestamp and never advance the cursor. "
            "Per-frame hand pose + scene-flow arrows were pre-logged with "
            "`stable_time` stamps at startup. **Scrub the `stable_time` "
            "timeline in the rerun viewer to play the trajectory.**"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Fixed body")
                fx = gr.Slider(cx - POS_RANGE, cx + POS_RANGE,
                                float(state.fixed_pos[0]),
                                step=0.005, label="pos X")
                fy = gr.Slider(cy - POS_RANGE, cy + POS_RANGE,
                                float(state.fixed_pos[1]),
                                step=0.005, label="pos Y")
                fz = gr.Slider(cz - POS_RANGE, cz + POS_RANGE,
                                float(state.fixed_pos[2]),
                                step=0.005, label="pos Z")
                frx = gr.Slider(-180, 180, float(state.fixed_euler[0]),
                                 step=1.0, label="rot X (deg)")
                fry = gr.Slider(-180, 180, float(state.fixed_euler[1]),
                                 step=1.0, label="rot Y (deg)")
                frz = gr.Slider(-180, 180, float(state.fixed_euler[2]),
                                 step=1.0, label="rot Z (deg)")
                fs_  = gr.Slider(0.05, 3.0, float(state.fixed_scale),
                                  step=0.01, label="scale")
            with gr.Column():
                gr.Markdown("### Moving body (baseline)")
                mx = gr.Slider(cx - POS_RANGE, cx + POS_RANGE,
                                float(state.moving_baseline_pos[0]),
                                step=0.005, label="pos X")
                my = gr.Slider(cy - POS_RANGE, cy + POS_RANGE,
                                float(state.moving_baseline_pos[1]),
                                step=0.005, label="pos Y")
                mz = gr.Slider(cz - POS_RANGE, cz + POS_RANGE,
                                float(state.moving_baseline_pos[2]),
                                step=0.005, label="pos Z")
                mrx = gr.Slider(-180, 180, float(state.moving_baseline_euler[0]),
                                 step=1.0, label="rot X (deg)")
                mry = gr.Slider(-180, 180, float(state.moving_baseline_euler[1]),
                                 step=1.0, label="rot Y (deg)")
                mrz = gr.Slider(-180, 180, float(state.moving_baseline_euler[2]),
                                 step=1.0, label="rot Z (deg)")
                ms_  = gr.Slider(0.05, 3.0, float(state.moving_scale),
                                  step=0.01, label="scale")
            with gr.Column():
                gr.Markdown("### Joint (axis = R(rot)·[0,0,1])")
                jpx = gr.Slider(cx - POS_RANGE, cx + POS_RANGE,
                                 float(state.joint_pos[0]),
                                 step=0.005, label="pos X")
                jpy = gr.Slider(cy - POS_RANGE, cy + POS_RANGE,
                                 float(state.joint_pos[1]),
                                 step=0.005, label="pos Y")
                jpz = gr.Slider(cz - POS_RANGE, cz + POS_RANGE,
                                 float(state.joint_pos[2]),
                                 step=0.005, label="pos Z")
                jrx = gr.Slider(-180, 180, float(state.joint_euler[0]),
                                 step=1.0, label="rot X (deg)")
                jry = gr.Slider(-180, 180, float(state.joint_euler[1]),
                                 step=1.0, label="rot Y (deg)")
                jrz = gr.Slider(-180, 180, float(state.joint_euler[2]),
                                 step=1.0, label="rot Z (deg)")
                jt = gr.Dropdown(["hinge", "slide"], value=state.joint_type,
                                  label="joint type")
            with gr.Column():
                gr.Markdown("### Hand baseline")
                hpx = gr.Slider(cx - POS_RANGE, cx + POS_RANGE,
                                 float(state.hand_baseline_pos[0]),
                                 step=0.005, label="pos X")
                hpy = gr.Slider(cy - POS_RANGE, cy + POS_RANGE,
                                 float(state.hand_baseline_pos[1]),
                                 step=0.005, label="pos Y")
                hpz = gr.Slider(cz - POS_RANGE, cz + POS_RANGE,
                                 float(state.hand_baseline_pos[2]),
                                 step=0.005, label="pos Z")
                hrx = gr.Slider(-180, 180, float(state.hand_baseline_euler[0]),
                                 step=1.0, label="rot X (deg)")
                hry = gr.Slider(-180, 180, float(state.hand_baseline_euler[1]),
                                 step=1.0, label="rot Y (deg)")
                hrz = gr.Slider(-180, 180, float(state.hand_baseline_euler[2]),
                                 step=1.0, label="rot Z (deg)")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Joint keyframes")
                gr.Markdown(
                    "Workflow: scrub the rerun viewer's `stable_time` to "
                    "the frame you want. Read the time off rerun, set "
                    "`frame slot` below to match "
                    f"(`slot · {state.frame_time_step:.3f}s = stable_time`). "
                    "Dial **drive** on the right, then **Add keyframe** — "
                    "it records `(slot, current drive)`. Re-adding at an "
                    "existing slot overwrites that keyframe. Between "
                    "adjacent keyframes the joint moves at constant velocity."
                )
                kf_frame_slider = gr.Slider(
                    0, max(0, n_frames - 1), 0, step=1,
                    label=f"frame slot for new keyframe (0..{n_frames - 1})"
                )
                kf_time_hint = gr.Markdown(
                    f"Frame slot **0** ↔ rerun `stable_time = "
                    f"{0.0:.3f} s`"
                )
                with gr.Row():
                    kf_add_btn = gr.Button("Add keyframe", variant="primary")
                    kf_remove_btn = gr.Button("Remove last",
                                               variant="secondary")
                    kf_clear_btn = gr.Button("Clear all",
                                              variant="secondary")
                kf_display = gr.Markdown(keyframes_md())
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Drive (captured by Add keyframe; "
                    "applied to all frames when keyframes is empty)"
                )
                drive_h = gr.Slider(-180, 180, float(state.drive_hinge_deg),
                                     step=0.5, label="hinge drive (deg)")
                drive_s = gr.Slider(-0.5, 0.5, float(state.drive_slide_m),
                                     step=0.005, label="slide drive (m)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Save")
                overwrite_cb = gr.Checkbox(
                    label="Overwrite rerun/transforms.json if exists",
                    value=True,
                )
                save_btn = gr.Button("Save transforms.json", variant="primary")
                status = gr.Markdown("")

        info = gr.Markdown(info_text())

        live_inputs = [fx, fy, fz, frx, fry, frz, fs_,
                       mx, my, mz, mrx, mry, mrz, ms_,
                       jpx, jpy, jpz, jrx, jry, jrz,
                       hpx, hpy, hpz, hrx, hry, hrz,
                       drive_h, drive_s, jt]
        for sld in live_inputs:
            sld.change(on_change, inputs=live_inputs, outputs=info)

        save_btn.click(on_save, inputs=[overwrite_cb], outputs=status)
        kf_add_btn.click(on_add_keyframe,
                         inputs=[kf_frame_slider, drive_h, drive_s],
                         outputs=[kf_display, info])
        kf_remove_btn.click(on_remove_last_keyframe,
                             outputs=[kf_display, info])
        kf_clear_btn.click(on_clear_keyframes,
                            outputs=[kf_display, info])
        kf_frame_slider.change(on_frame_slider_change,
                                inputs=[kf_frame_slider],
                                outputs=[kf_time_hint])
        demo.load(on_change, inputs=live_inputs, outputs=info)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--label-fixed", type=str, required=True,
                   help="sam3d label welded to ground in the saved scene")
    p.add_argument("--label-fixed-candidate", type=int, default=0,
                   help="sam3d candidate index for the fixed body (default 0)")
    p.add_argument("--label-moving", type=str, required=True,
                   help="sam3d label attached via the joint")
    p.add_argument("--label-moving-candidate", type=int, default=0,
                   help="sam3d candidate index for the moving body (default 0)")
    p.add_argument("--joint", choices=["hinge", "slide"], default="hinge",
                   help="Joint connecting the moving body to ground (default hinge)")
    p.add_argument("--frame-time-step", type=float, default=0.2,
                   help="Seconds per frame on the stable_time timeline "
                        "(default 0.2 — matches stage 41).")
    p.add_argument("--port", type=int, default=7860, help="Gradio port")
    p.add_argument("--rerun-port", type=int, default=9999,
                   help="Rerun viewer port (default 9999)")
    p.add_argument("--share", action="store_true",
                   help="Expose a public gradio URL")
    p.add_argument("--max-faces", type=int, default=150_000,
                   help="Quadric-decimate each object mesh down to this face "
                        "count (0 = no decimation; rerun handles large meshes "
                        "but the viewer can get slow above ~200k faces).")
    p.add_argument("--no-scene-flow", action="store_true",
                   help="Don't render any4d scene flow arrows")
    p.add_argument("--no-pointcloud", action="store_true",
                   help="Don't render the any4d reference pointcloud")
    p.add_argument("--max-arrows", type=int, default=500,
                   help="Subsample scene-flow arrows per label per frame "
                        "(default 500)")
    p.add_argument("--load-transforms", type=Path, default=None,
                   help="Path to a previously-saved transforms.json to "
                        "restore at startup (default: "
                        "<scene>/rerun/transforms.json if it exists).")
    p.add_argument("--no-load-transforms", action="store_true",
                   help="Skip auto-loading the existing transforms.json. "
                        "Sliders + keyframes start from the pointcloud-"
                        "anchored defaults.")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    save_dir = scene_dir / "rerun"

    # Object meshes
    fixed_glb = find_label_mesh(scene_dir, args.label_fixed,
                                args.label_fixed_candidate)
    fixed_v_raw, fixed_f_raw = load_glb(fixed_glb)
    print(f"  loaded {fixed_glb.name}: {len(fixed_v_raw)} verts, "
          f"{len(fixed_f_raw)} faces")
    fixed_v, fixed_f, fixed_c = center_and_decimate(
        fixed_v_raw, fixed_f_raw, args.max_faces)
    print(f"fixed: {fixed_glb.relative_to(scene_dir)} -> "
          f"{len(fixed_v)} verts, {len(fixed_f)} faces, source centroid "
          f"{fixed_c.round(3).tolist()}")

    moving_glb = find_label_mesh(scene_dir, args.label_moving,
                                  args.label_moving_candidate)
    moving_v_raw, moving_f_raw = load_glb(moving_glb)
    print(f"  loaded {moving_glb.name}: {len(moving_v_raw)} verts, "
          f"{len(moving_f_raw)} faces")
    moving_v, moving_f, moving_c = center_and_decimate(
        moving_v_raw, moving_f_raw, args.max_faces)
    print(f"moving: {moving_glb.relative_to(scene_dir)} -> "
          f"{len(moving_v)} verts, {len(moving_f)} faces, source centroid "
          f"{moving_c.round(3).tolist()}")

    # Hand trajectory
    traj, hand_faces, canon, traj_centroids = load_hand_trajectory(scene_dir)
    for k, c in traj_centroids.items():
        side = "right" if k else "left"
        print(f"hand_{side}: trajectory centroid {c.round(3).tolist()}")

    frame_indices = list_scene_frames(scene_dir)
    n_with_hands = sum(1 for fi in frame_indices if fi in traj)
    print(f"trajectory: {len(frame_indices)} scene frames, "
          f"{n_with_hands} with WiLoR hands")

    # Any4D label data (for scene-flow display). any4d_root is needed for
    # the pointcloud too, so always resolve it; only label_data is gated by
    # --no-scene-flow.
    any4d_root, label_data = load_any4d_label_data(
        scene_dir, [args.label_fixed, args.label_moving]
    )
    if args.no_scene_flow:
        label_data = {}
    else:
        for lab in (args.label_fixed, args.label_moving):
            if lab not in label_data:
                print(f"  scene flow not found for label '{lab}', skipping")

    provenance = {
        "scene_dir": str(scene_dir),
        "fixed_label": args.label_fixed,
        "fixed_mesh_src": str(fixed_glb.relative_to(scene_dir)),
        "fixed_source_centroid": fixed_c.tolist(),
        "moving_label": args.label_moving,
        "moving_mesh_src": str(moving_glb.relative_to(scene_dir)),
        "moving_source_centroid": moving_c.tolist(),
        "joint_type": args.joint,
        "n_frames": len(frame_indices),
        "n_frames_with_hands": n_with_hands,
    }

    # Load the reference pointcloud upfront — even if --no-pointcloud is set,
    # we still need its centroid to anchor the slider defaults inside
    # any4d's world frame (otherwise user-controlled meshes land at world
    # origin = the camera, not where the scene actually sits).
    pc_data = load_ref_pointcloud(any4d_root, scene_dir, frame_indices)
    if pc_data is not None:
        pc_pts, pc_cols, ref_fi = pc_data
        # Median is robust to a few stray far-depth pixels that any4d's
        # MoGe mask sometimes leaks in; mean would yank the centre out.
        pos_center = np.median(pc_pts, axis=0)
        pc_q05 = np.percentile(pc_pts, 5, axis=0)
        pc_q95 = np.percentile(pc_pts, 95, axis=0)
        print(f"scene anchor (pointcloud median): "
              f"{pos_center.round(3).tolist()}")
        print(f"pointcloud 5–95% bounds: "
              f"min={pc_q05.round(3).tolist()}, "
              f"max={pc_q95.round(3).tolist()}")
    else:
        pos_center = np.zeros(3)
        print("no pointcloud available — slider defaults anchored at origin")

    # Load existing transforms.json if present. The file controls slider
    # defaults, keyframes, and the per-axis slider centre — so we read it
    # BEFORE constructing SceneState and use it to override the pointcloud-
    # anchored defaults.
    transforms_path = (args.load_transforms.resolve()
                       if args.load_transforms is not None
                       else save_dir / "transforms.json")
    loaded = (None if args.no_load_transforms
              else load_existing_transforms(transforms_path))
    if loaded is not None:
        print(f"loading existing transforms from "
              f"{transforms_path.relative_to(scene_dir)}")
        # Anchor sliders on the saved fixed body so all loaded positions
        # land near the centre of the slider grid. Expand POS_RANGE so the
        # furthest loaded position is well inside ±range of the new anchor.
        pos_center = np.array(loaded["fixed"]["pos"], dtype=np.float64)
        loaded_positions = [
            np.asarray(loaded.get("fixed",          {}).get("pos", pos_center)),
            np.asarray(loaded.get("moving_baseline",{}).get("pos", pos_center)),
            np.asarray(loaded.get("joint",          {}).get("pos_world", pos_center)),
            np.asarray(loaded.get("hand_baseline",  {}).get("pos", pos_center)),
        ]
        max_offset = max(
            float(np.max(np.abs(p - pos_center))) for p in loaded_positions
        )
        pos_range = max(1.5, max_offset + 0.5)
    else:
        pos_range = 1.5

    state = SceneState(
        hand_traj=traj,
        canon_hands=canon,
        traj_centroids=traj_centroids,
        frame_indices=frame_indices,
        joint_type=args.joint,
        any4d_label_data=label_data,
        max_arrows=args.max_arrows,
        pos_center=pos_center,
        frame_time_step=args.frame_time_step,
    )
    if loaded is not None:
        apply_loaded_transforms(state, loaded)
        kf_n = len(state.joint_keyframes)
        print(f"  applied loaded state: joint={state.joint_type}, "
              f"{kf_n} keyframe(s), pos_range={pos_range:.2f} m")

    # Init Rerun + static log of meshes and pointcloud
    init_rerun(args.rerun_port)
    log_static_meshes(fixed_v, fixed_f, moving_v, moving_f, canon, hand_faces)
    if not args.no_pointcloud and pc_data is not None:
        log_ref_pointcloud(pc_pts, pc_cols, ref_fi)

    # Pre-log every frame's hand pose + scene flow onto rerun's stable_time
    # timeline. This is the one-time cost that turns the scene into a real
    # frame-indexed animation — scrubbing stable_time in the viewer now
    # plays the trajectory, and subsequent slider edits (logged STATIC)
    # never advance the cursor.
    print(f"pre-logging {len(frame_indices)} frames onto stable_time "
          f"(step={args.frame_time_step}s) ...")
    pre_log_frame_timeline(state, args.frame_time_step)
    print("  done.")

    # Initial static log of all slider-driven Transform3Ds
    log_static_state(state)

    # Gradio (blocks main thread)
    demo = make_ui(state, save_dir, provenance, pos_range=pos_range)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
