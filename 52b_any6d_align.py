#!/usr/bin/env python
"""Stage 52b: Any6D-based mesh alignment  [PLACEHOLDER — NOT YET IMPLEMENTED].

Alternative to stage 52's silhouette/ICP alignment. Instead of optimizing the
mesh pose against the mask + MoGe depth, this stage reuses the 6D object pose
already estimated by **stage 32 (Any6D)** and bakes it into stage 52's output
layout so stage 51 picks it up automatically.

Rationale: Any6D registers the sam3d mesh directly to the metric pointcloud at
the label's keyframe and (when `any4d/cameras.npz` exists) bakes the result to
the world frame. For rigid objects that is often a cleaner placement than the
silhouette optimizer, and it avoids re-solving a pose stage 32 already computed.

Intended contract (to match stage 52 so stage 51 `--mesh-source auto` works):

    Reads:
        any6d/<label>/pose.json        # pose_object_to_world (+ keyframe), stage 32
        any6d/<label>/mesh_world.glb   # mesh already placed in the world frame
        (fallback: sam3d/<label>/mesh.glb + pose_object_to_camera + cameras.npz)

    Writes:
        aligned/<label>/mesh.glb       # the mesh in the shared world frame
        aligned/<label>/align.json     # { "method": "any6d (52b)", "keyframe": <kf>,
                                       #   "pose_object_to_world": [[...]], ... }

    Flags (planned, mirroring stage 52 where sensible):
        --scene-dir PATH   (required)
        --labels LIST      subset (default: all labels with an any6d/ pose)
        --candidate INT    sam3d candidate index (default 0)
        --overwrite        replace existing aligned/<label>/

Prerequisites: run stage 30 (meshes) and stage 32 (Any6D pose) first. Skip
non-rigid parts (e.g. `hand`) — those still want the manual placement path.

TODO(implement):
  1. Discover labels under any6d/ that have a pose.json.
  2. Require pose_object_to_world (i.e. stage 32 saw cameras.npz); otherwise
     compose pose_object_to_camera with the keyframe cam->world from
     any4d/cameras.npz here.
  3. Load any6d/<label>/mesh_world.glb (already world-frame) OR apply the pose
     to sam3d/<label>/mesh.glb, and export aligned/<label>/mesh.glb.
  4. Write align.json with method="any6d (52b)", keyframe, and the pose matrix.
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only process these labels (default: all with an any6d pose)")
    p.add_argument("--candidate", type=int, default=0,
                   help="sam3d candidate index (default 0)")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing aligned/<label>/ outputs")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    any6d_root = scene_dir / "any6d"

    print("Stage 52b (Any6D mesh alignment) is a PLACEHOLDER and is not yet "
          "implemented.")
    if not any6d_root.is_dir():
        print(f"  (also: {any6d_root} does not exist — run stage 32 first once "
              f"this stage is implemented.)")
    print("  Planned behavior: bake the stage-32 Any6D pose into "
          "aligned/<label>/{mesh.glb, align.json} so stage 51 renders it.")
    print("  For now, use stage 52 (silhouette/ICP) for mesh alignment.")
    sys.exit(2)


if __name__ == "__main__":
    main()
