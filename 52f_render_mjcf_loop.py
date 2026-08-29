#!/usr/bin/env python
"""Headless-render an MJCF, driving one joint through a repeating sweep.

Loads any standalone MuJoCo scene.xml — e.g. `data/<scene>/mujoco/scene.xml`
(stage 52c) or `data/<scene>/mujoco_anim/scene.xml` (stage 52e) — drives a
chosen hinge/slide joint back and forth between two bounds for
``--n-cycles``, and renders each step offscreen with ``mujoco.Renderer`` (no
display / passive viewer needed) straight to an mp4.

Unlike stage 52e (interactive `mujoco.viewer.launch_passive`, needs a display
or X forwarding), this is a pure batch/headless exporter — the way to get a
shareable clip of an articulation off a render-less server. The joint is
driven purely kinematically (set qpos + `mj_forward`, no `mj_step`/contacts),
matching how 52c/52e already animate their live views.

**No model inference, reads only the MJCF** (+ the mesh STL/OBJ files it
references, resolved relative to the MJCF's own directory via its
`<compiler meshdir="...">`).

Env: any with `mujoco` + `numpy` + `opencv-python` (e.g. `any4d`). Needs a
working OpenGL context for offscreen rendering; MuJoCo defaults to EGL on
Linux (`MUJOCO_GL=osmesa` as a software fallback if EGL isn't available).

Examples:
    # Auto joint + range (from the MJCF's own <joint range=...> if limited)
    python 52f_render_mjcf_loop.py --mjcf data/dryer/mujoco/scene.xml

    # Explicit hinge sweep, 3 cycles at 30 FPS
    python 52f_render_mjcf_loop.py --mjcf data/dryer/mujoco_anim/scene.xml \\
        --range-min 0 --range-max 75 --n-cycles 3 --fps 30

    # Slide joint (meters), sine motion, reuse a 52e-saved camera view
    python 52f_render_mjcf_loop.py --mjcf data/drawer/mujoco/scene.xml \\
        --joint articulation --range-min 0 --range-max 0.25 \\
        --motion sine --view-file data/drawer/mujoco_anim/view.json
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

PARENT_LINK_RGBA = (0.9, 0.9, 0.9, 1.0)  # gray
CHILD_LINK_RGBA = (1.0, 124 / 255, 54 / 255, 1.0)  # orange
LINK_SPECULAR = 0.2
LINK_SHININESS = 0.1
LINK_REFLECTANCE = 0.1
HEADLIGHT_AMBIENT = (0.3, 0.3, 0.3)
HEADLIGHT_DIFFUSE = (0.8, 0.8, 0.8)
HEADLIGHT_SPECULAR = (0.6, 0.6, 0.6)


def find_drivable_joints(model):
    """Return {name: (jid, type_str)} for every hinge/slide joint in the model."""
    out = {}
    for jid in range(model.njnt):
        jtype = model.jnt_type[jid]
        if jtype == mujoco.mjtJoint.mjJNT_HINGE:
            type_str = "hinge"
        elif jtype == mujoco.mjtJoint.mjJNT_SLIDE:
            type_str = "slide"
        else:
            continue
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
        )
        out[name] = (jid, type_str)
    return out


def resolve_joint(model, joint_name):
    drivable = find_drivable_joints(model)
    if not drivable:
        sys.exit("error: model has no hinge/slide joint to drive")
    if joint_name is not None:
        if joint_name not in drivable:
            sys.exit(
                f"error: joint '{joint_name}' not found among hinge/slide joints: "
                f"{sorted(drivable)}"
            )
        return joint_name, *drivable[joint_name]
    if len(drivable) > 1:
        sys.exit(
            f"error: model has {len(drivable)} hinge/slide joints, pass --joint to "
            f"pick one: {sorted(drivable)}"
        )
    name, (jid, type_str) = next(iter(drivable.items()))
    return name, jid, type_str


def color_joint_links(model, jid):
    """Color the links immediately before and after the selected joint."""
    child_body_id = int(model.jnt_bodyid[jid])
    parent_body_id = int(model.body_parentid[child_body_id])

    for geom_id, body_id in enumerate(model.geom_bodyid):
        if body_id == parent_body_id:
            model.geom_rgba[geom_id] = PARENT_LINK_RGBA
        elif body_id == child_body_id:
            model.geom_rgba[geom_id] = CHILD_LINK_RGBA

    parent_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id) or "world"
    )
    child_name = (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, child_body_id)
        or f"body_{child_body_id}"
    )
    print(
        f"colors: parent link '{parent_name}' = gray, child link '{child_name}' = orange"
    )
    return parent_body_id, child_body_id


def make_links_glossy(model, scene, link_body_ids):
    """Give the selected links a glossy finish in the current render scene."""
    for scene_geom in scene.geoms[: scene.ngeom]:
        geom_id = int(scene_geom.objid)
        if (
            scene_geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
            and geom_id >= 0
            and int(model.geom_bodyid[geom_id]) in link_body_ids
        ):
            scene_geom.specular = LINK_SPECULAR
            scene_geom.shininess = LINK_SHININESS
            scene_geom.reflectance = LINK_REFLECTANCE


def configure_lighting(model):
    """Add a bright camera-mounted fill light to keep the object visible."""
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = HEADLIGHT_AMBIENT
    model.vis.headlight.diffuse[:] = HEADLIGHT_DIFFUSE
    model.vis.headlight.specular[:] = HEADLIGHT_SPECULAR


def resolve_range(model, jid, type_str, range_min, range_max):
    """CLI overrides > the joint's own <joint limited range=...> > a fallback default.

    Returned in the joint's native units: radians for hinge, metres for slide.
    """
    if range_min is not None and range_max is not None:
        lo, hi = float(range_min), float(range_max)
        return (np.deg2rad(lo), np.deg2rad(hi)) if type_str == "hinge" else (lo, hi)
    if bool(model.jnt_limited[jid]):
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        print(
            f"note: using the MJCF's own joint range: [{lo:.4f}, {hi:.4f}] "
            f"({'rad' if type_str == 'hinge' else 'm'})"
        )
        return lo, hi
    lo_deg_or_m, hi_deg_or_m = (0.0, 80.0) if type_str == "hinge" else (0.0, 0.25)
    print(
        f"warning: joint has no <joint limited range=...> and no --range-min/max "
        f"given; defaulting to [{lo_deg_or_m}, {hi_deg_or_m}] "
        f"({'deg' if type_str == 'hinge' else 'm'})"
    )
    return (
        (np.deg2rad(lo_deg_or_m), np.deg2rad(hi_deg_or_m))
        if type_str == "hinge"
        else (lo_deg_or_m, hi_deg_or_m)
    )


def joint_phase(t, n_cycles, motion):
    """Value in [0, 1] at fraction t in [0, 1]; n_cycles full min->max->min sweeps.

    Both waveforms start and end each cycle at 0 so `range_min` bookends every
    sweep, whichever motion is picked.
    """
    cycle_pos = n_cycles * t
    if motion == "triangle":
        frac = cycle_pos - np.floor(cycle_pos)
        cycle_idx = int(np.floor(cycle_pos))
        return frac if cycle_idx % 2 == 0 else (1.0 - frac)
    if motion == "sine":
        return (1.0 - np.cos(2.0 * np.pi * cycle_pos)) / 2.0
    raise ValueError(f"unknown motion {motion}")


def setup_camera(model, data, view_file, azimuth, elevation, distance, lookat):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if view_file is not None and view_file.is_file():
        with open(view_file) as f:
            view = json.load(f)
        cam.azimuth = float(view["azimuth"])
        cam.elevation = float(view["elevation"])
        cam.distance = float(view["distance"])
        cam.lookat[:] = [float(x) for x in view["lookat"]]
        print(f"camera: loaded {view_file}")
        return cam
    mujoco.mjv_defaultFreeCamera(model, cam)
    if azimuth is not None:
        cam.azimuth = azimuth
    if elevation is not None:
        cam.elevation = elevation
    if distance is not None:
        cam.distance = distance
    if lookat is not None:
        cam.lookat[:] = lookat
    return cam


def render_loop(
    mjcf_path: Path,
    out_path: Path,
    joint_name,
    range_min,
    range_max,
    motion,
    n_cycles,
    seconds_per_cycle,
    fps,
    width,
    height,
    camera_name,
    view_file,
    azimuth,
    elevation,
    distance,
    lookat,
):
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    configure_lighting(model)

    name, jid, type_str = resolve_joint(model, joint_name)
    link_body_ids = color_joint_links(model, jid)
    qadr = int(model.jnt_qposadr[jid])
    lo, hi = resolve_range(model, jid, type_str, range_min, range_max)
    unit = "rad" if type_str == "hinge" else "m"
    print(f"driving joint '{name}' ({type_str}): [{lo:.4f}, {hi:.4f}] {unit}")

    cam = None
    if camera_name is None:
        cam = setup_camera(model, data, view_file, azimuth, elevation, distance, lookat)
        print(
            f"camera: free camera at azimuth={cam.azimuth:.1f}°, elevation={cam.elevation:.1f}°, "
            f"distance={cam.distance:.3f}m, lookat={cam.lookat}"
        )

    # The offscreen framebuffer defaults to 640x480 unless the MJCF sets
    # <visual><global offwidth=".." offheight=".."/></visual>; grow it to fit
    # the requested render size rather than requiring every source MJCF to
    # declare one.
    model.vis.global_.offwidth = max(width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(height, model.vis.global_.offheight)
    renderer = mujoco.Renderer(model, height=height, width=width)

    duration = n_cycles * seconds_per_cycle
    n_frames = max(2, int(round(duration * fps)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    for i in range(n_frames):
        t = i / (n_frames - 1) if n_frames > 1 else 0.0
        phase = joint_phase(t, n_cycles, motion)
        data.qpos[qadr] = lo + (hi - lo) * phase
        mujoco.mj_forward(model, data)

        if camera_name is not None:
            renderer.update_scene(data, camera=camera_name)
        else:
            renderer.update_scene(data, camera=cam)
        make_links_glossy(model, renderer.scene, link_body_ids)
        rgb = renderer.render()  # (H, W, 3) uint8 RGB
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    writer.release()
    renderer.close()
    print(f"wrote {n_frames} frames ({duration:.1f}s @ {fps} FPS) to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--mjcf", type=Path, required=True, help="Path to a scene.xml")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output mp4 path (default: <mjcf_dir>/render_loop.mp4)",
    )
    p.add_argument(
        "--joint",
        type=str,
        default=None,
        help="Joint name to drive (default: auto if the model has exactly "
        "one hinge/slide joint)",
    )
    p.add_argument(
        "--range-min",
        type=float,
        default=None,
        help="Sweep lower bound (deg for hinge, m for slide). Default: the "
        "MJCF's own <joint limited range=...> if set, else a fallback",
    )
    p.add_argument(
        "--range-max",
        type=float,
        default=None,
        help="Sweep upper bound (deg for hinge, m for slide)",
    )
    p.add_argument(
        "--motion",
        choices=["triangle", "sine"],
        default="triangle",
        help="Sweep waveform (default triangle)",
    )
    p.add_argument(
        "--n-cycles",
        type=float,
        default=3.0,
        help="Number of full min->max->min sweeps (default 3)",
    )
    p.add_argument(
        "--seconds-per-cycle",
        type=float,
        default=2.0,
        help="Duration of one sweep, in seconds (default 2.0)",
    )
    p.add_argument(
        "--fps", type=float, default=30.0, help="Render/output FPS (default 30)"
    )
    p.add_argument(
        "--width", type=int, default=1280, help="Render width (default 1280)"
    )
    p.add_argument(
        "--height", type=int, default=720, help="Render height (default 720)"
    )
    p.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Use a named <camera> from the MJCF instead of a free camera",
    )
    p.add_argument(
        "--view-file",
        type=Path,
        default=None,
        help="52e-style view.json (azimuth/elevation/distance/lookat) to "
        "reuse for the free camera",
    )
    p.add_argument(
        "--azimuth", type=float, default=None, help="Free-camera azimuth, deg"
    )
    p.add_argument(
        "--elevation", type=float, default=None, help="Free-camera elevation, deg"
    )
    p.add_argument(
        "--distance", type=float, default=None, help="Free-camera distance, m"
    )
    p.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Free-camera lookat point",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.mjcf.is_file():
        sys.exit(f"error: {args.mjcf} does not exist")
    out_path = (
        args.out if args.out is not None else args.mjcf.parent / "render_loop.mp4"
    )
    render_loop(
        mjcf_path=args.mjcf.resolve(),
        out_path=out_path,
        joint_name=args.joint,
        range_min=args.range_min,
        range_max=args.range_max,
        motion=args.motion,
        n_cycles=args.n_cycles,
        seconds_per_cycle=args.seconds_per_cycle,
        fps=args.fps,
        width=args.width,
        height=args.height,
        camera_name=args.camera,
        view_file=args.view_file,
        azimuth=args.azimuth,
        elevation=args.elevation,
        distance=args.distance,
        lookat=args.lookat,
    )


if __name__ == "__main__":
    main()
