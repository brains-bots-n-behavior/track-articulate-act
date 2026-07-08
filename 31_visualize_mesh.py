#!/usr/bin/env python
"""Stage 31: visualize the stage-30 sam3d mesh.glb (no pyglet / no display).

A small inspection helper for the meshes stage 30 writes. For each requested
label it loads

    data/<scene>/sam3d/<label>/[cand_NN_<kf>/]mesh.glb      (--mesh-source sam3d, default)

or the stage-52 result

    data/<scene>/aligned/<label>/mesh.glb                   (--mesh-source aligned)

prints geometry stats (vertex / face counts, bounds, watertight, vertex
colors), and writes a visualization. Two backends, neither of which needs an
X display or pyglet (trimesh's built-in windowed viewer requires pyglet<2,
which we deliberately avoid):

    --format html  (default)  self-contained three.js scene you can rotate in a
                              browser. Works fully headless — open the .html
                              locally (VS Code Remote-SSH forwards it for you).
    --format png              offscreen raster via pyrender (EGL/OSMesa). Real
                              shaded image, also headless.

Multiple labels each get their own file by default; --combine puts them all in
one scene (meaningful for --mesh-source aligned, where every mesh already sits
in the shared Any4D world frame — sam3d meshes are each in their own canonical
frame and will pile up near the origin).

This stage only reads sam3d/aligned; it writes only the visualization files. It
is not part of the core pipeline order — run it any time after stage 30 (or 52).

Examples:
    # Interactive HTML for every label (default), into <scene>/sam3d/_previews/
    python 31_visualize_mesh.py --scene-dir data/oven

    # One label, second candidate
    python 31_visualize_mesh.py --scene-dir data/oven --labels oven_door --candidate 1

    # Offscreen PNGs instead of HTML
    python 31_visualize_mesh.py --scene-dir data/oven --format png

    # Aligned meshes, together in the world frame, one combined file
    python 31_visualize_mesh.py --scene-dir data/oven --mesh-source aligned --combine

    # Any .glb directly
    python 31_visualize_mesh.py --mesh path/to/mesh.glb
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

# Distinct RGBA colors to tell labels apart when --combine'd and a mesh has no
# vertex colors of its own.
_PALETTE = [
    (220, 70, 70, 255),
    (70, 130, 220, 255),
    (80, 190, 100, 255),
    (230, 170, 50, 255),
    (170, 90, 210, 255),
    (60, 200, 200, 255),
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, default=None,
                   help="Scene folder data/<scene_id>/ (omit only with --mesh)")
    p.add_argument("--mesh", type=Path, default=None,
                   help="Visualize a single .glb/.ply/.obj file directly, ignoring "
                        "--scene-dir / label resolution")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only visualize these labels (default: all under the source dir)")
    p.add_argument("--mesh-source", choices=["sam3d", "aligned", "auto"],
                   default="sam3d",
                   help="Which mesh to load: 'sam3d' (stage 30, default), "
                        "'aligned' (stage 52, world frame), or 'auto' "
                        "(aligned if present else sam3d)")
    p.add_argument("--candidate", type=int, default=0,
                   help="sam3d candidate index when stage 30 ran --all-candidates (default 0)")
    p.add_argument("--all-candidates", action="store_true",
                   help="Visualize every cand_*/mesh.glb for each label (sam3d source only)")
    p.add_argument("--combine", action="store_true",
                   help="Put all meshes in one scene/file instead of one each")
    p.add_argument("--format", choices=["html", "png"], default="html",
                   help="Output format: interactive 'html' (default) or offscreen 'png'")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <source_root>/_previews, or the "
                        "mesh's folder for --mesh)")
    p.add_argument("--resolution", type=int, nargs=2, default=(1280, 960),
                   metavar=("W", "H"), help="PNG resolution for --format png (default 1280 960)")
    p.add_argument("--gl", choices=["egl", "osmesa"], default="egl",
                   help="OpenGL backend for --format png offscreen rendering "
                        "(default 'egl'; use 'osmesa' if EGL is unavailable)")
    p.add_argument("--no-axis", action="store_true",
                   help="Don't add the XYZ origin axis marker to the scene")
    p.add_argument("--background", type=float, nargs=4, default=None,
                   metavar=("R", "G", "B", "A"),
                   help="Background RGBA in [0,1] (default: light gray)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mesh path resolution (mirrors stage 52's find_mesh_dir conventions)
# ---------------------------------------------------------------------------
def find_sam3d_mesh(label_dir: Path, candidate_idx: int):
    """[(name, path)] for one sam3d label: direct mesh.glb, else cand_<idx>_*,
    else first cand_*."""
    direct = label_dir / "mesh.glb"
    if direct.is_file():
        return [(label_dir.name, direct)]
    matches = sorted(label_dir.glob(f"cand_{candidate_idx:02d}_*"))
    if not matches:
        matches = sorted(label_dir.glob("cand_*"))
    for c in matches:
        mp = c / "mesh.glb"
        if mp.is_file():
            return [(f"{label_dir.name}/{c.name}", mp)]
    return []


def find_all_sam3d_candidates(label_dir: Path):
    """[(name, path)] for every cand_*/mesh.glb (or the direct one)."""
    direct = label_dir / "mesh.glb"
    if direct.is_file():
        return [(label_dir.name, direct)]
    out = []
    for c in sorted(label_dir.glob("cand_*")):
        mp = c / "mesh.glb"
        if mp.is_file():
            out.append((f"{label_dir.name}/{c.name}", mp))
    return out


def collect_meshes(args):
    """Return (out_dir_default, [(name, Path), ...]) of mesh files to visualize."""
    if args.mesh is not None:
        mp = args.mesh.resolve()
        if not mp.is_file():
            sys.exit(f"error: --mesh {mp} does not exist")
        return mp.parent, [(mp.stem, mp)]

    if args.scene_dir is None:
        sys.exit("error: pass --scene-dir (or --mesh for a single file)")
    scene_dir = args.scene_dir.resolve()

    source = args.mesh_source
    if source == "auto":
        source = "aligned" if (scene_dir / "aligned").is_dir() else "sam3d"
    root = scene_dir / ("aligned" if source == "aligned" else "sam3d")
    if not root.is_dir():
        sys.exit(f"error: {root} does not exist (run stage "
                 f"{'52' if source == 'aligned' else '30'} first)")
    print(f"mesh source: {source}  ({root})")

    label_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "_previews")
    if args.labels:
        names = {d.name for d in label_dirs}
        unknown = [l for l in args.labels if l not in names]
        if unknown:
            sys.exit(f"error: --labels not found under {root}: {unknown}")
        label_dirs = [d for d in label_dirs if d.name in args.labels]
    if not label_dirs:
        sys.exit(f"error: no label folders under {root}")

    out = []
    for d in label_dirs:
        if source == "aligned":
            mp = d / "mesh.glb"
            found = [(d.name, mp)] if mp.is_file() else []
        elif args.all_candidates:
            found = find_all_sam3d_candidates(d)
        else:
            found = find_sam3d_mesh(d, args.candidate)
        if not found:
            print(f"  [{d.name}] SKIP: no mesh.glb")
        out.extend(found)

    if not out:
        sys.exit("error: nothing to visualize")
    return root / "_previews", out


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(str(path), force="mesh")
    if not isinstance(m, trimesh.Trimesh) or m.faces.shape[0] == 0:
        raise RuntimeError(f"{path} did not load as a triangle mesh")
    return m


def describe(name: str, m: trimesh.Trimesh):
    lo, hi = m.bounds
    ext = hi - lo
    has_vc = (m.visual is not None
              and getattr(m.visual, "kind", None) == "vertex"
              and m.visual.vertex_colors is not None)
    print(f"  [{name}]")
    print(f"      vertices: {len(m.vertices):>8d}   faces: {len(m.faces):>8d}")
    print(f"      bounds  : min {np.round(lo, 4).tolist()}  max {np.round(hi, 4).tolist()}")
    print(f"      extent  : {np.round(ext, 4).tolist()}  (diag {np.linalg.norm(ext):.4f})")
    print(f"      watertight: {m.is_watertight}   vertex_colors: {bool(has_vc)}")


def _has_vertex_colors(m: trimesh.Trimesh) -> bool:
    return (m.visual is not None
            and getattr(m.visual, "kind", None) == "vertex"
            and m.visual.vertex_colors is not None)


def prepare(items, combine: bool):
    """Copy meshes, assigning a palette color per index for any mesh lacking
    vertex colors when combining (so labels are distinguishable)."""
    out = []
    for i, (name, m) in enumerate(items):
        mesh = m.copy()
        if combine and not _has_vertex_colors(mesh):
            mesh.visual.vertex_colors = _PALETTE[i % len(_PALETTE)]
        out.append((name, mesh))
    return out


def make_trimesh_scene(meshes, add_axis: bool, background):
    scene = trimesh.Scene()
    for name, m in meshes:
        scene.add_geometry(m, geom_name=name)
    if add_axis:
        diag = float(np.linalg.norm(scene.extents)) if scene.extents is not None else 1.0
        scene.add_geometry(trimesh.creation.axis(origin_size=diag * 0.01,
                                                 axis_length=diag * 0.5))
    if background is not None:
        scene.background = (np.array(background, dtype=np.float64) * 255).astype(np.uint8)
    return scene


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def write_html(meshes, out_path: Path, add_axis: bool, background):
    from trimesh.viewer import scene_to_html
    scene = make_trimesh_scene(meshes, add_axis, background)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(scene_to_html(scene))
    print(f"  wrote {out_path}")


def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """4x4 camera-to-world pose (pyrender/OpenGL convention: camera looks -Z)."""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-12)
    up = np.asarray(up, float)
    if abs(np.dot(fwd, up)) > 0.99:           # avoid a degenerate basis
        up = np.array([0.0, 0.0, 1.0])
    x = np.cross(fwd, up); x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(x, fwd)
    M = np.eye(4)
    M[:3, 0] = x
    M[:3, 1] = y
    M[:3, 2] = -fwd                           # camera +Z points back
    M[:3, 3] = eye
    return M


def write_png(meshes, out_path: Path, resolution, add_axis, background, gl):
    # Select the headless GL backend BEFORE importing pyrender/OpenGL.
    os.environ["PYOPENGL_PLATFORM"] = gl
    try:
        import pyrender
    except Exception as e:
        sys.exit(f"error: --format png needs pyrender ({type(e).__name__}: {e}). "
                 f"Use --format html, or `pip install pyrender`.")

    render_meshes = list(meshes)
    if add_axis:
        diag0 = 1.0
        for _, m in meshes:
            diag0 = max(diag0, float(np.linalg.norm(m.extents)))
        render_meshes = render_meshes + [("_axis", trimesh.creation.axis(
            origin_size=diag0 * 0.01, axis_length=diag0 * 0.5))]

    bg = (0.93, 0.93, 0.93, 1.0) if background is None else tuple(background)
    scene = pyrender.Scene(bg_color=bg, ambient_light=(0.35, 0.35, 0.35))
    all_pts = []
    for name, m in render_meshes:
        scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        all_pts.append(m.bounds)
    all_pts = np.concatenate(all_pts, axis=0)
    center = all_pts.mean(axis=0)
    diag = float(np.linalg.norm(all_pts.max(0) - all_pts.min(0))) or 1.0

    yfov = np.pi / 4.0
    dist = (diag / 2.0) / np.tan(yfov / 2.0) * 1.5
    eye = center + np.array([1.0, 0.8, 1.0]) / np.sqrt(2.61) * dist
    cam_pose = _look_at(eye, center)
    scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=resolution[0] / resolution[1]),
              pose=cam_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=cam_pose)

    try:
        r = pyrender.OffscreenRenderer(int(resolution[0]), int(resolution[1]))
        color, _ = r.render(scene)
        r.delete()
    except Exception as e:
        hint = "try --gl osmesa" if gl == "egl" else "try --gl egl"
        sys.exit(f"error: offscreen render failed with --gl {gl} "
                 f"({type(e).__name__}: {e}).\n       {hint}, or use --format html.")

    from PIL import Image
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).save(out_path)
    print(f"  wrote {out_path}")


def main():
    args = parse_args()
    out_dir = args.out.resolve() if args.out is not None else None
    default_out, files = collect_meshes(args)
    if out_dir is None:
        out_dir = default_out

    print(f"\n{len(files)} mesh(es) to visualize:")
    items = []
    for name, path in files:
        try:
            m = load_mesh(path)
        except Exception as e:
            print(f"  [{name}] FAIL: {e}")
            continue
        describe(name, m)
        items.append((name, m))
    if not items:
        sys.exit("error: no meshes loaded successfully")

    ext = args.format
    groups = ([("combined", prepare(items, True))]
              if (args.combine or len(items) == 1)
              else [(name, prepare([(name, m)], False)) for name, m in items])

    print(f"\nwriting {ext} -> {out_dir}")
    for gname, gmeshes in groups:
        safe = gname.replace("/", "__")
        out_path = out_dir / f"{safe}.{ext}"
        if ext == "html":
            write_html(gmeshes, out_path, add_axis=not args.no_axis,
                       background=args.background)
        else:
            write_png(gmeshes, out_path, args.resolution, add_axis=not args.no_axis,
                      background=args.background, gl=args.gl)

    if ext == "html":
        print("\nopen the .html in a browser (VS Code Remote-SSH forwards it, "
              "or scp it locally).")
    print("done.")


if __name__ == "__main__":
    main()
