#!/usr/bin/env python
"""Stage 05: transfer SAM masks onto combined SAM3D meshes with SegviGen.

This stage adapts SegviGen's ``inference_full.py --two_d_map`` mode to the
Articulate4D scene layout. It reads the keyframe and label list recorded by a
stage-03 combined reconstruction, composes the per-label SAM masks into an RGB
part map, and asks SegviGen to paint those part-indicative colors onto the
combined mesh.

Default inputs:

    data/<scene>/masks/tracking.json
    data/<scene>/masks/<label>/<keyframe>.png
    data/<scene>/sam3d/combined/cand_<id>_<frame>/{mesh.glb,keyframe.txt,mask_labels.txt}

Outputs:

    data/<scene>/segvigen/combined/<frame>/
        guidance.png          # color-coded 2D mask map passed to SegviGen
        segmented_mesh.glb    # SegviGen's color-coded mesh
        pieces/<mask-name>.glb # one geometry file per segmentation mask
        input.vxz             # reusable TRELLIS.2 voxel intermediate
        metadata.json         # labels, colors, mask coverage, and provenance

Run this in the SegviGen/TRELLIS.2 environment (24 GB+ NVIDIA GPU). The
checkpoint must be SegviGen's *full segmentation with 2D guidance* checkpoint,
not its unguided or interactive checkpoint.

Example:
    python scripts/05_segvigen_segment.py \\
        --scene-dir data/office_chair \\
        --ckpt-path SegviGen/checkpoint/full_seg_w_2d_map.ckpt

Use ``--prepare-only`` to build and inspect guidance.png without loading any
GPU models. When masks overlap, the default ``smallest`` policy gives the
pixel to the smaller part so thin handles/lids are not swallowed by a body
mask; all overlap counts are recorded in metadata.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


_FALLBACK_COLORS = [
    (234, 40, 28),
    (61, 122, 160),
    (173, 210, 45),
    (255, 154, 0),
    (145, 84, 184),
    (0, 170, 173),
]


def parse_args():
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--ckpt-path", type=Path, default=None,
                   help="SegviGen full_seg_w_2d_map .ckpt (required unless --prepare-only)")
    p.add_argument("--segvigen-root", type=Path, default=repo_root / "SegviGen",
                   help="SegviGen checkout (default: <repo>/SegviGen)")
    p.add_argument("--pipeline-config", type=Path, default=None,
                   help="Optional local TRELLIS.2-4B pipeline.json; otherwise SegviGen "
                        "uses its local copy or the Hugging Face cache")
    p.add_argument("--mesh", type=Path, default=None,
                   help="Input whole-object GLB (default: sam3d/combined/mesh.glb)")
    p.add_argument("--candidate", type=int, default=None,
                   help="Only process this combined candidate index (default: all candidates)")
    p.add_argument("--keyframe", type=int, default=None,
                   help="Override the keyframe recorded beside the SAM3D mesh")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Labels to encode (default: mesh's mask_labels.txt, then all tracking labels)")
    p.add_argument("--overlap-policy", choices=["smallest", "first", "last", "error"],
                   default="smallest",
                   help="Owner of pixels present in multiple masks (default: smallest part)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output folder (default: <scene>/segvigen/combined)")
    p.add_argument("--prepare-only", action="store_true",
                   help="Only create guidance.png + metadata.json; do not run SegviGen")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace this stage's existing named output files")
    return p.parse_args()


def _read_json(path: Path):
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in {path}: {exc}") from exc


def validate_checkpoint(path: Path):
    """Reject Git LFS pointer files before loading the expensive pipeline."""
    with path.open("rb") as checkpoint_file:
        header = checkpoint_file.read(128)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"SegviGen checkpoint is a Git LFS pointer, not model weights: {path}. "
            "Download full_seg_w_2d_map.ckpt (about 7.86 GB) from "
            "https://huggingface.co/fenghora/SegviGen and replace this file"
        )


def resolve_meshes(scene_dir: Path, explicit_mesh: Path | None,
                   candidate: int | None):
    """Return all requested ``(mesh_path, provenance_dir)`` pairs."""
    if explicit_mesh is not None:
        mesh = explicit_mesh.expanduser().resolve()
        if not mesh.is_file():
            raise RuntimeError(f"input mesh does not exist: {mesh}")
        return [(mesh, mesh.parent)]

    combined = scene_dir / "sam3d_scaled" / "combined"
    pattern = "cand_*_*" if candidate is None else f"cand_{candidate:02d}_*"
    results = []
    for candidate_dir in sorted(combined.glob(pattern)):
        mesh = candidate_dir / "mesh.glb"
        if mesh.is_file():
            results.append((mesh, candidate_dir))
    if results:
        return results

    # Compatibility with scenes produced before multiple candidates existed.
    direct = combined / "mesh.glb"
    if direct.is_file() and candidate is None:
        return [(direct, combined)]
    raise RuntimeError(
        f"no requested combined SAM3D mesh found under {combined}; "
        "run stage 03 with --combined"
    )


def resolve_keyframe(mesh_dir: Path, override: int | None):
    if override is not None:
        if override < 0:
            raise RuntimeError("--keyframe must be non-negative")
        return override
    path = mesh_dir / "keyframe.txt"
    if not path.is_file():
        raise RuntimeError(f"missing {path}; pass --keyframe explicitly")
    raw = path.read_text().strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"invalid keyframe in {path}: {raw!r}") from exc
    if value < 0:
        raise RuntimeError(f"invalid negative keyframe in {path}: {value}")
    return value


def resolve_labels(mesh_dir: Path, requested, tracking: dict):
    if requested is not None:
        labels = list(requested)
    else:
        labels_file = mesh_dir / "mask_labels.txt"
        labels = (labels_file.read_text().split() if labels_file.is_file()
                  else list(tracking))
    labels = list(dict.fromkeys(labels))
    if not labels:
        raise RuntimeError("no labels selected")
    unknown = [label for label in labels if label not in tracking]
    if unknown:
        raise RuntimeError(f"labels not present in masks/tracking.json: {unknown}")
    return labels


def label_color(entry: dict, index: int):
    color = entry.get("color")
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        color = _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)]
    try:
        rgb = tuple(int(round(float(c))) for c in color[:3])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid tracking color {color!r}") from exc
    if any(c < 0 or c > 255 for c in rgb):
        raise RuntimeError(f"tracking color outside [0,255]: {rgb}")
    if rgb == (255, 255, 255):
        raise RuntimeError("label color cannot be white (reserved for background)")
    return rgb


def load_mask(path: Path, target_hw=None):
    if not path.is_file():
        raise RuntimeError(f"mask does not exist: {path}")
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., -1]
    mask = array > 0
    if target_hw is not None and mask.shape != target_hw:
        height, width = target_hw
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize(
            (width, height), Image.Resampling.NEAREST
        )
        mask = np.asarray(resized) > 0
    return mask


def find_mask(masks_root: Path, label: str, keyframe: int):
    label_dir = masks_root / label
    preferred = label_dir / f"{keyframe:06d}.png"
    if preferred.is_file():
        return preferred
    matches = [path for path in label_dir.glob("*.png")
               if path.stem.isdigit() and int(path.stem) == keyframe]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous mask names for {label} frame {keyframe}: {matches}")
    raise RuntimeError(f"missing mask for {label} at frame {keyframe} under {label_dir}")


def compose_guidance(masks, colors, policy: str):
    """Return RGB guidance, owner map, and number of multiply-covered pixels."""
    stack = np.stack(masks, axis=0)
    coverage = stack.sum(axis=0)
    overlap_pixels = int(np.count_nonzero(coverage > 1))
    if policy == "error" and overlap_pixels:
        raise RuntimeError(
            f"masks overlap at {overlap_pixels} pixels; choose another --overlap-policy"
        )

    owner = np.full(stack.shape[1:], -1, dtype=np.int32)
    if policy in ("last", "error"):
        order = range(len(masks))
    elif policy == "first":
        order = reversed(range(len(masks)))
    else:  # smaller masks are painted last and therefore win overlaps
        order = sorted(range(len(masks)), key=lambda i: int(masks[i].sum()), reverse=True)
    for index in order:
        owner[masks[index]] = index

    guidance = np.full((*owner.shape, 3), 255, dtype=np.uint8)
    for index, color in enumerate(colors):
        guidance[owner == index] = color
    return guidance, owner, overlap_pixels


def display_path(path: Path, scene_dir: Path):
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def safe_piece_name(label: str, used: set[str]):
    """Return a stable, filesystem-safe and unique filename stem for a label."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._") or "segment"
    candidate = stem
    suffix = 2
    while candidate.casefold() in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def export_mesh_pieces(mesh_path: Path, pieces_dir: Path, labels, colors):
    """Split a color-coded GLB by face color and export one GLB per label.

    SegviGen encodes the segmentation in the output appearance.  We sample that
    appearance at the vertices, classify each triangle by its mean RGB color,
    and keep white as a background class which is not exported.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "trimesh is required to split the SegviGen result into pieces"
        ) from exc

    asset = trimesh.load(mesh_path, force="scene", process=False)
    # dump() applies scene-graph transforms while retaining each geometry's
    # visual, so exported pieces remain in the same coordinate system.
    geometries = list(asset.dump(concatenate=False))
    if not geometries:
        raise RuntimeError(f"segmented mesh contains no geometry: {mesh_path}")

    palette = np.asarray([*colors, (255, 255, 255)], dtype=np.float32)
    per_label = [[] for _ in labels]
    face_counts = [0 for _ in labels]

    for geometry in geometries:
        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.faces) == 0:
            continue
        try:
            vertex_rgb = np.asarray(geometry.visual.to_color().vertex_colors[:, :3],
                                    dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                "could not sample colors from the segmented mesh; its material "
                "is not supported by trimesh"
            ) from exc
        face_rgb = vertex_rgb[np.asarray(geometry.faces)].mean(axis=1)
        distances = ((face_rgb[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        owners = distances.argmin(axis=1)

        for index in range(len(labels)):
            face_indices = np.flatnonzero(owners == index)
            if not len(face_indices):
                continue
            part = geometry.submesh([face_indices], append=True, repair=False)
            if part is not None and len(part.faces):
                per_label[index].append(part)
                face_counts[index] += len(part.faces)

    pieces_dir.mkdir(parents=True, exist_ok=True)
    used_names = set()
    records = []
    for index, label in enumerate(labels):
        filename = safe_piece_name(label, used_names) + ".glb"
        output = pieces_dir / filename
        if per_label[index]:
            scene = trimesh.Scene()
            for part_index, part in enumerate(per_label[index]):
                scene.add_geometry(part, geom_name=f"{label}_{part_index}")
            scene.export(output)
        records.append({
            "label": label,
            "mesh": str(output),
            "faces": face_counts[index],
            "written": output.is_file(),
        })
    return records


def main():
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    if not scene_dir.is_dir():
        sys.exit(f"error: scene directory does not exist: {scene_dir}")
    if args.candidate is not None and args.candidate < 0:
        sys.exit("error: --candidate must be non-negative")

    tracking_path = scene_dir / "masks" / "tracking.json"
    if not tracking_path.is_file():
        sys.exit(f"error: {tracking_path} does not exist (run stage 02 with keyframe_candidates in prompts.json first)")

    try:
        mesh_jobs = resolve_meshes(scene_dir, args.mesh, args.candidate)
        tracking = _read_json(tracking_path)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    checkpoint = None
    segvigen_root = None
    inference_script = None
    pipeline_config = None
    if not args.prepare_only:
        if args.ckpt_path is None:
            sys.exit("error: --ckpt-path is required unless --prepare-only is set")
        checkpoint = args.ckpt_path.expanduser().resolve()
        if not checkpoint.is_file():
            sys.exit(f"error: SegviGen checkpoint does not exist: {checkpoint}")
        try:
            validate_checkpoint(checkpoint)
        except RuntimeError as exc:
            sys.exit(f"error: {exc}")
        segvigen_root = args.segvigen_root.expanduser().resolve()
        inference_script = segvigen_root / "inference_full.py"
        if not inference_script.is_file():
            sys.exit(f"error: SegviGen inference script not found: {inference_script}")
        if args.pipeline_config is not None:
            pipeline_config = args.pipeline_config.expanduser().resolve()
            if not pipeline_config.is_file():
                sys.exit(f"error: pipeline config does not exist: {pipeline_config}")

    out_root = (args.out_dir.expanduser().resolve() if args.out_dir is not None
                else scene_dir / "segvigen" / "combined")
    seen_frames = set()
    for mesh_path, mesh_dir in mesh_jobs:
        try:
            keyframe = resolve_keyframe(mesh_dir, args.keyframe)
            if keyframe in seen_frames:
                raise RuntimeError(
                    f"multiple meshes resolve to frame {keyframe:06d}; output would collide"
                )
            seen_frames.add(keyframe)
            labels = resolve_labels(mesh_dir, args.labels, tracking)
            mask_paths = [find_mask(scene_dir / "masks", label, keyframe)
                          for label in labels]
            masks = []
            target_hw = None
            for path in mask_paths:
                mask = load_mask(path, target_hw)
                if target_hw is None:
                    target_hw = mask.shape
                masks.append(mask)
            colors = [label_color(tracking[label], i)
                      for i, label in enumerate(labels)]
            guidance, owner, overlap_pixels = compose_guidance(
                masks, colors, args.overlap_policy
            )
        except RuntimeError as exc:
            sys.exit(f"error processing {mesh_path}: {exc}")

        process_mesh(
            args, scene_dir, mesh_path, keyframe, labels, mask_paths, masks,
            colors, guidance, owner, overlap_pixels,
            out_root / f"{keyframe:06d}", checkpoint, segvigen_root,
            inference_script, pipeline_config,
        )

    print(f"processed {len(mesh_jobs)} mesh(es) for scene {scene_dir.name}")


def process_mesh(args, scene_dir, mesh_path, keyframe, labels, mask_paths, masks,
                 colors, guidance, owner, overlap_pixels, out_dir, checkpoint,
                 segvigen_root, inference_script, pipeline_config):
    """Prepare and optionally segment one frame-specific combined mesh."""
    guidance_path = out_dir / "guidance.png"
    input_vxz_path = out_dir / "input.vxz"
    output_mesh_path = out_dir / "segmented_mesh.glb"
    pieces_dir = out_dir / "pieces"
    metadata_path = out_dir / "metadata.json"
    used_piece_names = set()
    expected_piece_paths = [
        pieces_dir / (safe_piece_name(label, used_piece_names) + ".glb")
        for label in labels
    ]
    named_outputs = [guidance_path, input_vxz_path, output_mesh_path, metadata_path,
                     *expected_piece_paths]
    existing = [path for path in named_outputs if path.exists()]
    if existing and not args.overwrite:
        sys.exit("error: output(s) already exist; pass --overwrite to replace: "
                 + ", ".join(str(path) for path in existing))
    if args.overwrite:
        for path in existing:
            if not path.is_file():
                sys.exit(f"error: refusing to replace non-file output: {path}")
            path.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(guidance, mode="RGB").save(guidance_path)

    label_records = []
    for index, (label, color, mask_path, mask) in enumerate(
            zip(labels, colors, mask_paths, masks)):
        label_records.append({
            "label": label,
            "color_rgb": list(color),
            "mask": display_path(mask_path, scene_dir),
            "mask_pixels": int(mask.sum()),
            "assigned_pixels": int(np.count_nonzero(owner == index)),
        })
    metadata = {
        "stage": "05_segvigen_segment",
        "status": "prepared" if args.prepare_only else "running",
        "scene": scene_dir.name,
        "source_mesh": display_path(mesh_path, scene_dir),
        "source_keyframe": keyframe,
        "guidance_map": display_path(guidance_path, scene_dir),
        "segmented_mesh": display_path(output_mesh_path, scene_dir),
        "overlap_policy": args.overlap_policy,
        "overlap_pixels": overlap_pixels,
        "background_pixels": int(np.count_nonzero(owner < 0)),
        "labels": label_records,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"scene: {scene_dir.name}")
    print(f"mesh: {mesh_path}")
    print(f"keyframe: {keyframe:06d}")
    for record in label_records:
        print(f"  [{record['label']}] color={record['color_rgb']} "
              f"mask={record['mask_pixels']} assigned={record['assigned_pixels']}")
    print(f"overlap: {overlap_pixels} pixels ({args.overlap_policy} policy)")
    print(f"guidance: {guidance_path}")

    if args.prepare_only:
        print("prepare-only: SegviGen inference skipped.")
        print(f"metadata: {metadata_path}")
        return

    command = [
        sys.executable,
        str(inference_script),
        "--ckpt_path", str(checkpoint),
        "--glb", str(mesh_path),
        "--input_vxz", str(input_vxz_path),
        "--img", str(guidance_path),
        "--export_glb", str(output_mesh_path),
        "--two_d_map",
    ]
    if pipeline_config is not None:
        command.extend(["--pipeline_config", str(pipeline_config)])

    print("running SegviGen 2D-guided inference ...")
    try:
        subprocess.run(command, cwd=segvigen_root, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"error: SegviGen inference failed with exit code {exc.returncode}")
    if not output_mesh_path.is_file():
        sys.exit(f"error: SegviGen returned successfully but did not write {output_mesh_path}")

    try:
        piece_records = export_mesh_pieces(output_mesh_path, pieces_dir, labels, colors)
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    for record in piece_records:
        record["mesh"] = display_path(Path(record["mesh"]), scene_dir)
        if record["written"]:
            print(f"piece [{record['label']}]: {record['mesh']} "
                  f"({record['faces']} faces)")
        else:
            print(f"warning: no faces classified as [{record['label']}]")

    metadata["status"] = "complete"
    metadata["checkpoint"] = str(checkpoint)
    metadata["mesh_pieces"] = piece_records
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"segmented mesh: {output_mesh_path}")
    print(f"metadata: {metadata_path}")
    print("done.")


if __name__ == "__main__":
    main()
