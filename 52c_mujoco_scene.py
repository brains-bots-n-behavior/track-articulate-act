#!/usr/bin/env python
"""Stage 52 (MuJoCo variant): two-body articulation scene + HaWoR hand replay.

Loads two sam3d objects by label — one welded to ground, one attached via a
hinge or slide joint — plus the full HaWoR hand trajectory played back as
an animation loop over all scene frames.

Reads:
    data/<scene>/sam3d/<label-fixed>/[cand_NN_<kf>/]mesh.glb
    data/<scene>/sam3d/<label-moving>/[cand_NN_<kf>/]mesh.glb
    data/<scene>/hawor/per_frame/<frame>.npz   (one per frame with detections)
    data/<scene>/hawor/faces.npy               (shared MANO topology)
    data/<scene>/frames/*.jpg                  (frame range for the loop)

Editor (gradio sliders + MuJoCo passive viewer side-by-side):
    - Fixed body: pos (m), euler XYZ (deg), scale.
    - Moving body baseline: pos (m), euler XYZ (deg), scale. The "baseline"
      is the body's rest pose at joint angle 0.
    - Joint: pos (m), orientation euler XYZ (deg). The joint axis in world
      frame is R_joint @ [0, 0, 1]; the orientation sliders rotate that axis.
      A small green sphere + capsule marks the joint position and axis.
    - Drive: separate sliders for hinge (deg) and slide (m). The slider that
      matches the current joint type drives the moving body about the joint
      axis live — purely visual; the saved MJCF carries the real joint.
    - Animation: frame index slider iterating all data/<scene>/frames/*.jpg.
      Play/Pause runs a gr.Timer at --fps. Each frame, hand_left and hand_right
      are placed via best-fit rigid pose from canonical mesh -> per-frame
      HaWoR verts. Lateralities with no detection at the current frame are
      moved off-screen.
    - Apply Scale: rebuilds the model so mesh scale takes effect (MuJoCo bakes
      scale at compile time).

Save (data/<scene>/mujoco/):
    scene.xml        Fixed body welded to ground (no joint).
                     Moving body with a real <joint type="hinge|slide" ...>
                     in its local frame. Hands omitted (they're a trajectory).
    transforms.json  All slider values + provenance.

Run inside any env with mujoco + gradio + numpy + trimesh.

Examples:
    # Laptop lid hinge about user-adjusted axis, full hand trajectory:
    python scripts/52c_mujoco_scene.py \\
        --scene-dir data/macbook-all \\
        --label-fixed laptop_base \\
        --label-moving laptop_up \\
        --joint hinge

    # Drawer (slide), 30 FPS playback:
    python scripts/52c_mujoco_scene.py \\
        --scene-dir data/<scene> \\
        --label-fixed cabinet \\
        --label-moving drawer \\
        --joint slide --fps 30
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import trimesh

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("error: mujoco not installed. `pip install mujoco`")

try:
    import gradio as gr
except ImportError:
    sys.exit("error: gradio not installed. `pip install gradio`")


# ---------------------------------------------------------------------------
# Mesh / data loading
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


def center_and_export_mesh(verts: np.ndarray, faces: np.ndarray, out: Path,
                           max_faces: int = 150_000):
    """Centre verts at origin, decimate if over `max_faces`, write binary STL.

    MuJoCo's STL decoder caps faces at 200 000. sam3d meshes can be 1 M+,
    so we quadric-decimate down to `max_faces` (default 150 000 to leave
    margin). Falls back to random face subsampling if quadric decimation is
    unavailable (no `fast-simplification` / `open3d`)."""
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        sys.exit(f"error: trying to export an empty mesh to {out} "
                 f"({verts.shape[0]} verts, {faces.shape[0]} faces)")
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    if faces.shape[0] > max_faces:
        print(f"  decimating {faces.shape[0]} -> ~{max_faces} faces ...")
        try:
            full = trimesh.Trimesh(vertices=centered, faces=faces,
                                   process=False)
            small = full.simplify_quadric_decimation(face_count=max_faces)
            centered = np.asarray(small.vertices, dtype=np.float64)
            faces = np.asarray(small.faces, dtype=np.int64)
            print(f"    -> {centered.shape[0]} verts, {faces.shape[0]} faces "
                  f"(quadric)")
        except Exception as e:
            rng = np.random.default_rng(0)
            keep = rng.choice(faces.shape[0], max_faces, replace=False)
            faces = faces[keep]
            print(f"    quadric decimation failed ({type(e).__name__}: {e}); "
                  f"random-subsampled to {max_faces} faces "
                  f"(geometry will have holes)")

    out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=centered, faces=faces, process=False).export(
        str(out), file_type="stl",
    )
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
    """Returns:
        traj    dict[frame_idx] -> list of {'is_right': bool, 'verts': (778, 3)}
        faces   (Nf, 3) int32, shared MANO topology
        canon   dict[bool] -> (778, 3) canonical mesh per laterality (centered)
        traj_c  dict[bool] -> (3,) mean centroid across all frames for that
                laterality. Subtracting this from each frame's fit
                translation centers the trajectory near origin so the user
                baseline slider can drop it next to the object.
    Canonical mesh = the first per-frame mesh seen for that laterality.
    """
    per_frame_dir = scene_dir / "hawor" / "per_frame"
    faces_p = scene_dir / "hawor" / "faces.npy"
    if not per_frame_dir.is_dir() or not faces_p.is_file():
        sys.exit(f"error: HaWoR output missing under {scene_dir / 'hawor'}")
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
# Math helpers
# ---------------------------------------------------------------------------


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)


def euler_xyz_to_R(rx_deg, ry_deg, rz_deg):
    """Intrinsic Rz @ Ry @ Rx."""
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
    """SVD best-fit SE(3): P_curr ≈ R @ P_canon + t."""
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


# ---------------------------------------------------------------------------
# MJCF
# ---------------------------------------------------------------------------


OFFSCREEN = np.array([100.0, 100.0, 100.0])     # used to "hide" mocap hands
IDENT_Q = np.array([1.0, 0.0, 0.0, 0.0])


def build_mjcf(meshes: dict, scales: dict, joint_type: str,
               body_poses: dict, joint_pose: dict,
               for_save: bool, save_joint_local: dict = None) -> str:
    """Build an MJCF string.

    Editor mode (for_save=False): object_fixed, object_moving, joint_marker
    are mocap; hand_left / hand_right are mocap when meshes for them exist.

    Save mode (for_save=True): no mocap. object_fixed is welded to world (no
    joint). object_moving has a real <joint> in its local frame, taken from
    save_joint_local. Hands are omitted (they're a trajectory, not state).
    """
    def fmt(v):
        return " ".join(f"{x:.6f}" for x in np.asarray(v).ravel())

    asset_lines = []
    for asset_name, mesh_file in meshes.items():
        s = float(scales.get(asset_name, 1.0))
        asset_lines.append(
            f'<mesh name="{asset_name}" file="{mesh_file}" '
            f'scale="{s} {s} {s}"/>'
        )
    assets_xml = "\n    ".join(asset_lines)

    bodies = []
    if for_save:
        fpos, fquat = body_poses["object_fixed"]
        mpos, mquat = body_poses["object_moving"]
        bodies.append(
            f'<body name="object_fixed" pos="{fmt(fpos)}" quat="{fmt(fquat)}">'
            f'<geom type="mesh" mesh="fixed_mesh" rgba="0.75 0.75 0.78 1"/>'
            f'</body>'
        )
        jp_loc = save_joint_local["pos"]
        ja_loc = save_joint_local["axis_str"]
        bodies.append(
            f'<body name="object_moving" pos="{fmt(mpos)}" quat="{fmt(mquat)}">'
            f'<joint name="articulation" type="{joint_type}" '
            f'axis="{ja_loc}" pos="{fmt(jp_loc)}"/>'
            f'<geom type="mesh" mesh="moving_mesh" rgba="0.55 0.65 0.85 1"/>'
            f'</body>'
        )
    else:
        fpos, fquat = body_poses["object_fixed"]
        mpos, mquat = body_poses["object_moving"]
        bodies.append(
            f'<body name="object_fixed" mocap="true" '
            f'pos="{fmt(fpos)}" quat="{fmt(fquat)}">'
            f'<geom type="mesh" mesh="fixed_mesh" rgba="0.75 0.75 0.78 1"/>'
            f'</body>'
        )
        # Moving body is NOT mocap in the editor: it carries a real <joint>
        # so MuJoCo enforces the articulation when we drive qpos. Baseline
        # pose, joint pos and joint axis are all updated live by writing
        # to model.body_pos / body_quat / jnt_pos / jnt_axis in sync_mocap.
        bodies.append(
            f'<body name="object_moving" '
            f'pos="{fmt(mpos)}" quat="{fmt(mquat)}">'
            f'<joint name="art" type="{joint_type}" axis="0 0 1" pos="0 0 0"/>'
            f'<geom type="mesh" mesh="moving_mesh" rgba="0.55 0.65 0.85 1"/>'
            f'</body>'
        )
        jp = joint_pose["pos"]
        jq = joint_pose["quat_wxyz"]
        bodies.append(
            f'<body name="joint_marker" mocap="true" '
            f'pos="{fmt(jp)}" quat="{fmt(jq)}">'
            f'<geom type="sphere" size="0.012" rgba="0.2 0.9 0.2 0.9"/>'
            f'<geom type="capsule" size="0.004" '
            f'fromto="0 0 -0.06 0 0 0.06" rgba="0.2 0.9 0.2 0.7"/>'
            f'</body>'
        )
        if "hand_left_mesh" in meshes:
            bodies.append(
                f'<body name="hand_left" mocap="true" '
                f'pos="{fmt(OFFSCREEN)}" quat="{fmt(IDENT_Q)}">'
                f'<geom type="mesh" mesh="hand_left_mesh" rgba="0.90 0.42 0.42 1"/>'
                f'</body>'
            )
        if "hand_right_mesh" in meshes:
            bodies.append(
                f'<body name="hand_right" mocap="true" '
                f'pos="{fmt(OFFSCREEN)}" quat="{fmt(IDENT_Q)}">'
                f'<geom type="mesh" mesh="hand_right_mesh" rgba="0.42 0.42 0.90 1"/>'
                f'</body>'
            )

    bodies_xml = "\n    ".join(bodies)

    return f"""<mujoco model="articulate4d_two_body">
  <compiler angle="radian" autolimits="true" meshdir="."/>
  <option timestep="0.005"/>
  <asset>
    {assets_xml}
  </asset>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
    <rgba haze="0.95 0.95 0.95 1"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    {bodies_xml}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class SceneState:
    def __init__(self, mesh_dir: Path,
                 obj_meshes: dict, hand_meshes: dict,
                 hand_traj: dict, canon_hands: dict, traj_centroids: dict,
                 frame_indices: list, joint_type: str):
        self.lock = threading.Lock()
        self.mesh_dir = mesh_dir
        self.obj_meshes = obj_meshes
        self.hand_meshes = hand_meshes
        self.hand_traj = hand_traj
        self.canon_hands = canon_hands
        self.traj_centroids = traj_centroids   # dict[bool] -> (3,) mean centroid
        self.frame_indices = frame_indices
        self.joint_type = joint_type

        self.fixed_pos = np.zeros(3)
        self.fixed_euler = np.zeros(3)
        self.fixed_scale = 1.0
        self.moving_baseline_pos = np.array([0.25, 0.0, 0.0])
        self.moving_baseline_euler = np.zeros(3)
        self.moving_scale = 1.0
        # Joint orientation given by euler (deg). Joint axis in world =
        # R(joint_euler) @ [0, 0, 1].
        self.joint_pos = np.array([0.125, 0.0, 0.0])
        self.joint_euler = np.zeros(3)
        self.drive_hinge_deg = 0.0
        self.drive_slide_m = 0.0
        # Hand baseline pose: applied AFTER trajectory centering. Default at
        # origin so the centered trajectory sits where the fixed body starts.
        self.hand_baseline_pos = np.zeros(3)
        self.hand_baseline_euler = np.zeros(3)
        self.current_frame_slot = 0   # index into self.frame_indices

        self.rebuild_pending = False
        self.shutdown = False
        self.model = None
        self.data = None
        self.art_jid = -1     # joint id of the articulation joint
        self.art_qadr = -1    # qpos index for that joint
        self.compile()

    # ---- pose math ----

    def compute_poses(self):
        """Returns body_poses, joint_pose, joint_axis_world.

        Now that the moving body has a real MuJoCo joint, body_poses
        carries only baseline poses — the joint kinematics take it from
        there when we set qpos in sync_mocap.
        """
        fixed_quat = euler_xyz_to_quat_wxyz(*self.fixed_euler)
        baseline_pos = self.moving_baseline_pos.copy()
        baseline_quat = euler_xyz_to_quat_wxyz(*self.moving_baseline_euler)
        R_joint = euler_xyz_to_R(*self.joint_euler)
        joint_axis_world = R_joint @ np.array([0.0, 0.0, 1.0])
        joint_quat = R_to_quat_wxyz(R_joint)
        return (
            {
                "object_fixed":  (self.fixed_pos, fixed_quat),
                "object_moving": (baseline_pos, baseline_quat),
            },
            {"pos": self.joint_pos, "quat_wxyz": joint_quat},
            joint_axis_world,
        )

    def compute_baseline_moving_pose(self):
        return (self.moving_baseline_pos.copy(),
                euler_xyz_to_quat_wxyz(*self.moving_baseline_euler))

    # ---- mjcf compile + sync ----

    def _all_meshes(self):
        return {**self.obj_meshes, **self.hand_meshes}

    def _scales(self):
        s = {"fixed_mesh": self.fixed_scale, "moving_mesh": self.moving_scale}
        for k in self.hand_meshes:
            s[k] = 1.0
        return s

    def compile(self):
        body_poses, joint_pose, _ = self.compute_poses()
        xml = build_mjcf(
            self._all_meshes(), self._scales(), self.joint_type,
            body_poses, joint_pose, for_save=False,
        )
        tmp = self.mesh_dir / "_live.xml"
        tmp.write_text(xml)
        self.model = mujoco.MjModel.from_xml_path(str(tmp))
        self.data = mujoco.MjData(self.model)
        for i in range(self.model.nmesh):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, i)
            nv = int(self.model.mesh_vertnum[i])
            nf = int(self.model.mesh_facenum[i])
            print(f"  mesh[{i}] {name}: {nv} verts, {nf} faces")
        try:
            self.art_jid = self.model.joint("art").id
            self.art_qadr = int(self.model.jnt_qposadr[self.art_jid])
        except KeyError:
            self.art_jid = -1
            self.art_qadr = -1
        self.sync_mocap()

    def _set_mocap(self, name, pos, quat):
        try:
            bid = self.model.body(name).id
        except KeyError:
            return
        mid = int(self.model.body_mocapid[bid])
        if mid < 0:
            return
        self.data.mocap_pos[mid] = pos
        self.data.mocap_quat[mid] = quat

    def sync_mocap(self):
        body_poses, joint_pose, axis_world = self.compute_poses()

        # Mocap bodies: fixed object, joint marker, hands
        self._set_mocap("object_fixed", *body_poses["object_fixed"])
        self._set_mocap("joint_marker",
                        joint_pose["pos"], joint_pose["quat_wxyz"])

        # Moving body: NOT mocap. Its baseline pose lives in model.body_pos /
        # body_quat; the joint anchor + axis live in model.jnt_pos / jnt_axis
        # (both in the body's local frame); the drive value lives in
        # data.qpos. Write all four live so slider changes propagate
        # through MuJoCo's real joint kinematics on the next mj_forward.
        try:
            mov_bid = self.model.body("object_moving").id
        except KeyError:
            mov_bid = -1
        if mov_bid >= 0:
            mpos, mquat = body_poses["object_moving"]
            self.model.body_pos[mov_bid] = mpos
            self.model.body_quat[mov_bid] = mquat
        if self.art_jid >= 0:
            R_baseline = euler_xyz_to_R(*self.moving_baseline_euler)
            self.model.jnt_pos[self.art_jid] = R_baseline.T @ (
                self.joint_pos - self.moving_baseline_pos
            )
            self.model.jnt_axis[self.art_jid] = R_baseline.T @ axis_world
            if self.art_qadr >= 0:
                if self.joint_type == "hinge":
                    self.data.qpos[self.art_qadr] = np.deg2rad(
                        self.drive_hinge_deg
                    )
                else:
                    self.data.qpos[self.art_qadr] = self.drive_slide_m

        # Hands: rigid fit per frame, then centre the trajectory and apply
        # the user's hand baseline pose so the hand drops next to the object.
        if self.frame_indices:
            fi = self.frame_indices[self.current_frame_slot]
            hands_now = self.hand_traj.get(fi, [])
            R_user = euler_xyz_to_R(*self.hand_baseline_euler)
            for is_right, body_name in [(False, "hand_left"),
                                        (True,  "hand_right")]:
                if is_right not in self.canon_hands:
                    continue
                found = next(
                    (h for h in hands_now if h["is_right"] == is_right),
                    None,
                )
                if found is None:
                    self._set_mocap(body_name, OFFSCREEN, IDENT_Q)
                    continue
                R_fit, t_fit = fit_rigid(self.canon_hands[is_right],
                                         found["verts"])
                t_centered = t_fit - self.traj_centroids.get(
                    is_right, np.zeros(3)
                )
                final_t = R_user @ t_centered + self.hand_baseline_pos
                final_R = R_user @ R_fit
                self._set_mocap(body_name, final_t, R_to_quat_wxyz(final_R))

        # Propagate everything (mocap_pos/quat, body_pos/quat, jnt_pos/axis,
        # qpos) into data.xpos / xquat for the renderer. The passive viewer's
        # sync() does NOT re-run kinematics.
        mujoco.mj_forward(self.model, self.data)

    # ---- save ----

    def build_save_xml(self):
        """Compose the saved MJCF. Joint pos/axis converted to moving body's
        local frame using the baseline orientation."""
        body_poses, joint_pose, axis_world = self.compute_poses()
        baseline_pos, baseline_quat = self.compute_baseline_moving_pose()
        R_baseline = euler_xyz_to_R(*self.moving_baseline_euler)
        joint_pos_local = R_baseline.T @ (self.joint_pos - baseline_pos)
        axis_local = R_baseline.T @ axis_world
        axis_str = " ".join(f"{a:.6f}" for a in axis_local)

        save_body_poses = {
            "object_fixed":  body_poses["object_fixed"],
            "object_moving": (baseline_pos, baseline_quat),
        }
        save_joint_local = {"pos": joint_pos_local, "axis_str": axis_str}

        meshes_save = self.obj_meshes  # no hands in saved scene
        scales_save = {"fixed_mesh": self.fixed_scale,
                       "moving_mesh": self.moving_scale}
        xml = build_mjcf(meshes_save, scales_save, self.joint_type,
                         save_body_poses, {}, for_save=True,
                         save_joint_local=save_joint_local)
        return xml, joint_pos_local, axis_local


# ---------------------------------------------------------------------------
# Viewer thread
# ---------------------------------------------------------------------------


def viewer_loop(state: SceneState):
    """Main thread. Re-opens the viewer on rebuilds (e.g. scale changes)."""
    while not state.shutdown:
        with state.lock:
            model = state.model
            data = state.data
        rebuild = False
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                with state.lock:
                    if state.shutdown:
                        break
                    if state.rebuild_pending:
                        state.rebuild_pending = False
                        rebuild = True
                        break
                    state.sync_mocap()
                viewer.sync()
                time.sleep(0.02)
        if not rebuild:
            state.shutdown = True
            return
        with state.lock:
            state.compile()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def make_ui(state: SceneState, save_dir: Path, provenance: dict, fps: float):
    POS_RANGE = 1.5
    n_frames = len(state.frame_indices)

    def info_text():
        body_poses, joint_pose, axis_world = state.compute_poses()
        fp, fq = body_poses["object_fixed"]
        mp, mq = body_poses["object_moving"]
        fi = state.frame_indices[state.current_frame_slot]
        hands_at = state.hand_traj.get(fi, [])
        lefts = sum(1 for h in hands_at if not h["is_right"])
        rights = sum(1 for h in hands_at if h["is_right"])
        return (
            f"**Frame** {fi:06d} ({state.current_frame_slot + 1}/{n_frames})  "
            f"left hand: {lefts}  right hand: {rights}  \n"
            f"**Fixed** pos=({fp[0]:+.3f}, {fp[1]:+.3f}, {fp[2]:+.3f})  \n"
            f"**Moving** pos=({mp[0]:+.3f}, {mp[1]:+.3f}, {mp[2]:+.3f})  \n"
            f"**Joint** type=`{state.joint_type}` "
            f"axis_world=({axis_world[0]:+.3f}, {axis_world[1]:+.3f}, {axis_world[2]:+.3f})  \n"
            f"**Drive** "
            f"{state.drive_hinge_deg:+.1f}° / {state.drive_slide_m:+.3f} m"
        )

    def on_change(fx, fy, fz, frx, fry, frz,
                  mx, my, mz, mrx, mry, mrz,
                  jpx, jpy, jpz, jrx, jry, jrz,
                  hpx, hpy, hpz, hrx, hry, hrz,
                  dh, ds, jt, frame_slot):
        with state.lock:
            state.fixed_pos = np.array([fx, fy, fz], dtype=np.float64)
            state.fixed_euler = np.array([frx, fry, frz], dtype=np.float64)
            state.moving_baseline_pos = np.array([mx, my, mz], dtype=np.float64)
            state.moving_baseline_euler = np.array([mrx, mry, mrz], dtype=np.float64)
            state.joint_pos = np.array([jpx, jpy, jpz], dtype=np.float64)
            state.joint_euler = np.array([jrx, jry, jrz], dtype=np.float64)
            state.hand_baseline_pos = np.array([hpx, hpy, hpz], dtype=np.float64)
            state.hand_baseline_euler = np.array([hrx, hry, hrz], dtype=np.float64)
            state.drive_hinge_deg = float(dh)
            state.drive_slide_m = float(ds)
            # Joint type lives in MJCF, so changing it triggers a rebuild.
            if jt != state.joint_type:
                state.joint_type = jt
                state.rebuild_pending = True
            state.current_frame_slot = int(frame_slot) % max(1, n_frames)
        return info_text()

    def on_apply_scale(fs, ms):
        with state.lock:
            state.fixed_scale = float(fs)
            state.moving_scale = float(ms)
            state.rebuild_pending = True
        return (f"**Rebuilding** with fixed scale {fs:.3f}, "
                f"moving scale {ms:.3f} (viewer will flicker).")

    def on_tick(frame_slot):
        return (int(frame_slot) + 1) % max(1, n_frames)

    def on_play_toggle(is_playing):
        new_playing = not bool(is_playing)
        return (new_playing,
                gr.Timer(active=new_playing),
                "Pause" if new_playing else "Play")

    def on_save(overwrite):
        try:
            with state.lock:
                xml, jp_local, axis_local = state.build_save_xml()
                payload = {
                    "fixed": {
                        "pos": state.fixed_pos.tolist(),
                        "euler_deg_xyz": state.fixed_euler.tolist(),
                        "quat_wxyz": euler_xyz_to_quat_wxyz(*state.fixed_euler).tolist(),
                        "scale": float(state.fixed_scale),
                        "mesh_obj": state.obj_meshes["fixed_mesh"],
                    },
                    "moving_baseline": {
                        "pos": state.moving_baseline_pos.tolist(),
                        "euler_deg_xyz": state.moving_baseline_euler.tolist(),
                        "quat_wxyz": euler_xyz_to_quat_wxyz(*state.moving_baseline_euler).tolist(),
                        "scale": float(state.moving_scale),
                        "mesh_obj": state.obj_meshes["moving_mesh"],
                    },
                    "joint": {
                        "type": state.joint_type,
                        "pos_world": state.joint_pos.tolist(),
                        "euler_world_deg_xyz": state.joint_euler.tolist(),
                        "pos_local": jp_local.tolist(),
                        "axis_local": axis_local.tolist(),
                    },
                    "drive_at_save": {
                        "hinge_deg": float(state.drive_hinge_deg),
                        "slide_m": float(state.drive_slide_m),
                    },
                    "provenance": provenance,
                }
            xml_path = save_dir / "scene.xml"
            json_path = save_dir / "transforms.json"
            if (xml_path.exists() or json_path.exists()) and not overwrite:
                return (f"**SKIP**: {xml_path.name} / {json_path.name} exist. "
                        f"Tick `overwrite` to replace.")
            xml_path.write_text(xml)
            json_path.write_text(json.dumps(payload, indent=2))
            return (f"**Saved** `{xml_path.name}` + `{json_path.name}` to "
                    f"`{save_dir}`. Test with "
                    f"`python -m mujoco.viewer --mjcf={xml_path}`.")
        except Exception as e:
            return f"**FAIL**: {type(e).__name__}: {e}"

    with gr.Blocks(title="Articulate4D MuJoCo scene") as demo:
        gr.Markdown("# Two-body articulation + hand trajectory (MuJoCo)")
        gr.Markdown(
            "Pos / euler / joint / drive / frame sliders update the viewer "
            "live. **Apply Scale** rebuilds the model so mesh scale takes "
            "effect. Small green sphere + capsule = joint position + axis."
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Fixed body (welded)")
                fx = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos X")
                fy = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos Y")
                fz = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos Z")
                frx = gr.Slider(-180, 180, 0.0, step=1.0, label="rot X (deg)")
                fry = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Y (deg)")
                frz = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Z (deg)")
                fixed_scale = gr.Slider(0.05, 3.0, 1.0, step=0.01,
                                         label="scale (Apply Scale to take effect)")
            with gr.Column():
                gr.Markdown("### Moving body (baseline pose)")
                mx = gr.Slider(-POS_RANGE, POS_RANGE, 0.25, step=0.005, label="pos X")
                my = gr.Slider(-POS_RANGE, POS_RANGE, 0.0,  step=0.005, label="pos Y")
                mz = gr.Slider(-POS_RANGE, POS_RANGE, 0.0,  step=0.005, label="pos Z")
                mrx = gr.Slider(-180, 180, 0.0, step=1.0, label="rot X (deg)")
                mry = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Y (deg)")
                mrz = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Z (deg)")
                moving_scale = gr.Slider(0.05, 3.0, 1.0, step=0.01,
                                          label="scale (Apply Scale to take effect)")
            with gr.Column():
                gr.Markdown("### Joint (axis = R(rot)@[0,0,1])")
                jpx = gr.Slider(-POS_RANGE, POS_RANGE, 0.125, step=0.005, label="pos X")
                jpy = gr.Slider(-POS_RANGE, POS_RANGE, 0.0,   step=0.005, label="pos Y")
                jpz = gr.Slider(-POS_RANGE, POS_RANGE, 0.0,   step=0.005, label="pos Z")
                jrx = gr.Slider(-180, 180, 0.0, step=1.0, label="rot X (deg)")
                jry = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Y (deg)")
                jrz = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Z (deg)")
                jt = gr.Dropdown(["hinge", "slide"], value=state.joint_type,
                                  label="joint type (rebuild)")
            with gr.Column():
                gr.Markdown("### Hand baseline\n"
                            "Applied after trajectory centering: "
                            "`p_world = R(rot)·(t_hawor − traj_centroid) + pos`")
                hpx = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos X")
                hpy = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos Y")
                hpz = gr.Slider(-POS_RANGE, POS_RANGE, 0.0, step=0.005, label="pos Z")
                hrx = gr.Slider(-180, 180, 0.0, step=1.0, label="rot X (deg)")
                hry = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Y (deg)")
                hrz = gr.Slider(-180, 180, 0.0, step=1.0, label="rot Z (deg)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Drive (preview articulation)")
                drive_h = gr.Slider(-180, 180, 0.0, step=0.5,
                                     label="hinge drive (deg) — used if type=hinge")
                drive_s = gr.Slider(-0.5, 0.5, 0.0, step=0.005,
                                     label="slide drive (m) — used if type=slide")
            with gr.Column():
                gr.Markdown("### Animation (one step per frame)")
                frame_slider = gr.Slider(
                    0, max(0, n_frames - 1), 0, step=1,
                    label=f"frame slot (0..{n_frames - 1})"
                )
                play_btn = gr.Button("Play", variant="secondary")
                playing = gr.State(False)
                timer = gr.Timer(value=1.0 / max(1.0, fps), active=False)
            with gr.Column():
                gr.Markdown("### Save")
                apply_btn = gr.Button("Apply Scale (rebuild)", variant="secondary")
                overwrite_cb = gr.Checkbox(
                    label="Overwrite mujoco/scene.xml if exists", value=True
                )
                save_btn = gr.Button("Save scene.xml + transforms.json",
                                      variant="primary")
                status = gr.Markdown("")

        info = gr.Markdown(info_text())

        live_inputs = [fx, fy, fz, frx, fry, frz,
                       mx, my, mz, mrx, mry, mrz,
                       jpx, jpy, jpz, jrx, jry, jrz,
                       hpx, hpy, hpz, hrx, hry, hrz,
                       drive_h, drive_s, jt, frame_slider]
        for sld in live_inputs:
            sld.change(on_change, inputs=live_inputs, outputs=info)

        apply_btn.click(on_apply_scale, inputs=[fixed_scale, moving_scale],
                        outputs=status)
        save_btn.click(on_save, inputs=[overwrite_cb], outputs=status)
        play_btn.click(on_play_toggle, inputs=[playing],
                       outputs=[playing, timer, play_btn])
        timer.tick(on_tick, inputs=[frame_slider], outputs=[frame_slider])
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
                   help="Joint type connecting moving body to ground "
                        "(default hinge)")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Play timer rate (default 10 Hz)")
    p.add_argument("--max-faces", type=int, default=150_000,
                   help="Quadric-decimate each object mesh down to this face "
                        "count before STL export. MuJoCo's STL decoder caps "
                        "at 200000, so keep this below it (default 150000).")
    p.add_argument("--port", type=int, default=7860, help="Gradio port")
    p.add_argument("--share", action="store_true",
                   help="Expose a public gradio URL")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    save_dir = scene_dir / "mujoco"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Object meshes
    obj_meshes_files = {}
    obj_meshes_provenance = {}
    for tag, label, cand, asset_name, out_name in [
        ("fixed",  args.label_fixed,  args.label_fixed_candidate,
         "fixed_mesh",  "object_fixed.stl"),
        ("moving", args.label_moving, args.label_moving_candidate,
         "moving_mesh", "object_moving.stl"),
    ]:
        glb = find_label_mesh(scene_dir, label, cand)
        verts, faces = load_glb(glb)
        print(f"  loaded {glb.name}: {len(verts)} verts, {len(faces)} faces")
        out = save_dir / out_name
        centroid = center_and_export_mesh(verts, faces, out,
                                          max_faces=args.max_faces)
        obj_meshes_files[asset_name] = out.name
        obj_meshes_provenance[tag] = {
            "label": label,
            "candidate": cand,
            "src": str(glb.relative_to(scene_dir)),
            "source_centroid": centroid.tolist(),
            "n_verts": int(len(verts)),
            "n_faces": int(len(faces)),
        }
        print(f"{tag}: {glb.relative_to(scene_dir)} -> {out.name} "
              f"({len(verts)} verts, source centroid "
              f"{centroid.round(3).tolist()})")

    # Hand trajectory
    traj, hand_faces, canon, traj_centroids = load_hand_trajectory(scene_dir)
    for k, c in traj_centroids.items():
        side = "right" if k else "left"
        print(f"hand_{side}: trajectory centroid {c.round(3).tolist()} "
              f"(subtracted from per-frame fits to centre near origin)")
    hand_meshes_files = {}
    for is_right, asset_name, out_name in [
        (False, "hand_left_mesh",  "hand_left.stl"),
        (True,  "hand_right_mesh", "hand_right.stl"),
    ]:
        if is_right not in canon:
            print(f"hand: no {'right' if is_right else 'left'} detection in "
                  f"trajectory, skipping")
            continue
        out = save_dir / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=canon[is_right], faces=hand_faces,
                        process=False).export(str(out), file_type="stl")
        hand_meshes_files[asset_name] = out.name
        print(f"hand_{'right' if is_right else 'left'}: canonical mesh -> "
              f"{out.name}")

    frame_indices = list_scene_frames(scene_dir)
    n_with_hands = sum(1 for fi in frame_indices if fi in traj)
    print(f"trajectory: {len(frame_indices)} frames in scene, "
          f"{n_with_hands} with HaWoR detections "
          f"({len(traj)} unique frames have data)")

    provenance = {
        "scene_dir": str(scene_dir),
        "objects": obj_meshes_provenance,
        "n_frames": len(frame_indices),
        "n_frames_with_hands": n_with_hands,
        "joint_type": args.joint,
    }

    state = SceneState(
        mesh_dir=save_dir,
        obj_meshes=obj_meshes_files,
        hand_meshes=hand_meshes_files,
        hand_traj=traj,
        canon_hands=canon,
        traj_centroids=traj_centroids,
        frame_indices=frame_indices,
        joint_type=args.joint,
    )

    demo = make_ui(state, save_dir, provenance, args.fps)
    demo.launch(server_port=args.port, share=args.share,
                prevent_thread_lock=True)
    try:
        viewer_loop(state)
    finally:
        state.shutdown = True
        try:
            demo.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
