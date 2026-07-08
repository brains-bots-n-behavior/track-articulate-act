#!/usr/bin/env python
"""Stage 30: per-object 3D reconstruction with SAM 3D Objects.

For each label in masks/tracking.json with a non-null keyframe, run SAM 3D
Objects on (frames/<kf>.jpg, masks/<label>/<kf>.png) and write:

    data/<scene>/sam3d/<label>/
        keyframe.txt           # the frame index used (6-digit, no extension)
        splat.ply              # output["gs"].save_ply()
        mesh.glb               # output["glb"].export()   (basic vertex-color mesh)
        pose.json              # rotation_quat_wxyz, translation, scale
        input_rgba.png         # (only with --save-input) the RGBA fed to the model

When --all-candidates is set, each candidate gets its own subfolder:
    data/<scene>/sam3d/<label>/cand_00_<kf>/...
    data/<scene>/sam3d/<label>/cand_01_<kf>/...

Background removal:
    By default the background outside the mask is erased (filled with black,
    matching SAM 3D's internal rembg) so only the masked object is shown to the
    model. SAM 3D embeds the mask in the image's alpha channel but still feeds a
    full-resolution RGB branch with the background intact; erasing it removes
    that distractor (useful for parts of an articulated object, where the rest
    of the object would otherwise stay visible). Tune with --bg-mode /
    --bg-dilate / --bg-feather, or keep the original image with --bg-mode none.

    --bg-mode dim is a softer alternative to a solid fill: instead of replacing
    the surrounding pixels it keeps them but darkens them by --bg-dim (the ROI
    inside the mask stays at full brightness). This preserves a little context
    while still biasing the reconstruction toward the masked region, so the
    geometry is less likely to bleed into adjacent parts than with a hard
    black fill that drops all surrounding cues.

Run inside the `sam3d-objects` conda env.

Example:
    python 30_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects

    # Just one label, only the top candidate:
    python 30_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --labels mug

    # All candidates, force re-run:
    python 30_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --all-candidates --overwrite
"""

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

# RGB fill colors used to erase the background outside the mask.
_BG_COLORS = {
    "black": (0, 0, 0),       # matches the model's internal rembg (image * mask)
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--sam3d-repo", type=Path, required=True,
                   help="Path to the sam-3d-objects/ repo (needed for notebook/inference.py and checkpoints)")
    p.add_argument("--checkpoint-tag", type=str, default="hf",
                   help="Name of the checkpoint folder under <sam3d-repo>/checkpoints/ (default 'hf')")
    p.add_argument("--labels", nargs="*", default=None,
                   help="Only process these labels (default: all labels with a keyframe)")
    p.add_argument("--all-candidates", action="store_true",
                   help="Reconstruct every entry in keyframe_candidates (subfolders), not just the top one")
    p.add_argument("--candidate-idx", type=int, default=None,
                   help="Pick a specific candidate index (0=top). Mutually exclusive with --all-candidates.")
    p.add_argument("--seed", type=int, default=42, help="Diffusion seed")
    p.add_argument("--compile", action="store_true",
                   help="Pass compile=True to the SAM 3D pipeline (slower first run, faster afterwards)")
    p.add_argument("--overwrite", action="store_true",
                   help="Replace existing sam3d/<label>/ outputs")
    p.add_argument("--skip-mesh", action="store_true",
                   help="Save only splat.ply + pose.json (skip mesh.glb export)")
    p.add_argument("--bg-mode", choices=["black", "white", "gray", "dim", "none"],
                   default="black",
                   help="Erase the background outside the mask before reconstruction "
                        "by filling it with this color (default 'black', matching the "
                        "model's internal rembg). 'dim' keeps the surrounding pixels "
                        "but darkens them by --bg-dim (ROI stays full brightness). "
                        "'none' keeps the original background.")
    p.add_argument("--bg-dim", type=float, default=0.3,
                   help="Brightness multiplier for the surrounding when --bg-mode dim "
                        "(0.0 = full black, 1.0 = unchanged; default 0.3).")
    p.add_argument("--bg-dilate", type=int, default=0,
                   help="Grow the mask by N pixels before erasing the background, "
                        "to keep a thin margin around the object (default 0)")
    p.add_argument("--bg-feather", type=int, default=0,
                   help="Soft-blend the mask edge into the fill over N pixels "
                        "(Gaussian; default 0 = hard edge)")
    p.add_argument("--save-input", action="store_true",
                   help="Also save the preprocessed RGBA fed to the model as "
                        "<out_dir>/input_rgba.png (for inspection)")
    p.add_argument("--no-cudnn", action="store_true",
                   help="Disable cuDNN so convs use the native CUDA fallback. Use "
                        "this if you hit 'CUDA error: unrecognized error code' in a "
                        "conv (cuDNN-vs-driver mismatch in the sam3d-objects env, "
                        "e.g. torch 2.5.1+cu121/cuDNN 9.1 on a CUDA-13 driver).")
    p.add_argument("--blas-lib", choices=["default", "cublas", "cublaslt"],
                   default="default",
                   help="Override torch's cuBLAS backend (default: leave torch's "
                        "own routing alone). 'cublaslt' can dodge a broken legacy "
                        "cublasGemmEx on some builds, but can also FAIL skinny GEMMs "
                        "that default routing handles — try both if you hit "
                        "CUBLAS_STATUS_EXECUTION_FAILED.")
    return p.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    """Read a JPEG/PNG, return uint8 RGB (drop alpha if present)."""
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB")).astype(np.uint8)
    return arr


def load_binary_mask(path: Path, target_hw=None) -> np.ndarray:
    """Read an 8-bit mask, threshold > 0, optionally resize to (H, W)."""
    with Image.open(path) as im:
        m = np.array(im)
    if m.ndim == 3:
        m = m[..., -1]
    m = m > 0
    if target_hw is not None and m.shape != target_hw:
        # match the frame's H/W; nearest-neighbor to keep it binary
        H, W = target_hw
        m_u8 = (m.astype(np.uint8) * 255)
        m_u8 = np.array(Image.fromarray(m_u8).resize((W, H), Image.NEAREST))
        m = m_u8 > 0
    return m


def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Grow a boolean mask by ~`px` pixels. Tries scipy, falls back to PIL."""
    if px <= 0:
        return mask
    try:
        from scipy.ndimage import binary_dilation
        return binary_dilation(mask, iterations=int(px))
    except Exception:
        # PIL fallback: a (2*px+1) max filter grows the mask by px on each side.
        m_u8 = (mask.astype(np.uint8) * 255)
        size = 2 * int(px) + 1
        grown = Image.fromarray(m_u8).filter(ImageFilter.MaxFilter(size))
        return np.array(grown) > 0


def apply_background(image: np.ndarray, mask: np.ndarray, mode: str,
                     dilate_px: int = 0, feather_px: int = 0,
                     dim_factor: float = 0.3) -> np.ndarray:
    """Suppress the background (outside `mask`) of an RGB image.

    The object region (`mask`, optionally dilated by `dilate_px` to keep a thin
    margin) is preserved at full brightness; everything else is either replaced
    by a solid `mode` fill color, or — when `mode == "dim"` — kept but darkened
    to `dim_factor` of its original brightness so some context survives. An
    optional `feather_px` Gaussian soft edge blends the transition. `mode ==
    "none"` returns the image unchanged. Returns uint8 HxWx3.
    """
    if mode == "none":
        return image
    if mode != "dim" and mode not in _BG_COLORS:
        raise ValueError(f"unknown bg-mode '{mode}'; choose from "
                         f"{['none', 'dim'] + sorted(_BG_COLORS)}")

    keep = _dilate_mask(mask, dilate_px)
    alpha = keep.astype(np.float32)                    # 1 inside object, 0 outside
    if feather_px > 0:
        # Soft-blend the edge: blur the binary alpha into [0, 1].
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=float(feather_px)))
        alpha = np.array(alpha_img).astype(np.float32) / 255.0
    alpha = alpha[..., None]                            # HxWx1

    if mode == "dim":
        # Keep the surrounding pixels, just darkened — preserves context cues.
        bg = image.astype(np.float32) * float(dim_factor)
    else:
        bg = np.empty_like(image)
        bg[:] = _BG_COLORS[mode]
        bg = bg.astype(np.float32)
    out = image.astype(np.float32) * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def tensor_to_list(t):
    """Convert a torch tensor or numpy array to a plain python list."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t).reshape(-1).tolist()


def save_pose(output: dict, out_path: Path):
    """Pull rotation/translation/scale from the inference output dict and dump JSON."""
    pose = {
        "rotation_quat_wxyz": tensor_to_list(output["rotation"]),
        "translation": tensor_to_list(output["translation"]),
        "scale": tensor_to_list(output["scale"]),
    }
    with open(out_path, "w") as f:
        json.dump(pose, f, indent=2)


def run_one(inference, frame_path: Path, mask_path: Path, out_dir: Path,
            seed: int, save_mesh: bool, bg_mode: str = "black",
            bg_dilate: int = 0, bg_feather: int = 0, bg_dim: float = 0.3,
            save_input: bool = False):
    """
    Run SAM 3D Objects on a single (frame, mask), write outputs into out_dir.
    Returns a short summary string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image_rgb(frame_path)            # HxWx3 uint8
    mask = load_binary_mask(mask_path, target_hw=image.shape[:2])  # HxW bool

    if not mask.any():
        raise RuntimeError(f"empty mask at {mask_path}")

    # Suppress the background so the masked object dominates the model's input.
    image = apply_background(image, mask, bg_mode, bg_dilate, bg_feather, bg_dim)

    # Optionally dump the exact RGBA the model receives (RGB after bg fill,
    # mask in the alpha channel) for visual inspection.
    if save_input:
        rgba = np.concatenate(
            [image, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
        Image.fromarray(rgba, "RGBA").save(out_dir / "input_rgba.png")

    output = inference(image, mask, seed=seed)

    # Splat
    splat_path = out_dir / "splat.ply"
    output["gs"].save_ply(str(splat_path))

    # Mesh (already a trimesh.Trimesh inside output["glb"])
    if save_mesh:
        glb = output.get("glb")
        if glb is None:
            print(f"    warning: no glb in output (decode_formats may exclude 'mesh')")
        else:
            mesh_path = out_dir / "mesh.glb"
            glb.export(str(mesh_path))

    # Pose
    save_pose(output, out_dir / "pose.json")

    # Keyframe record (the frame stem, e.g. '000042')
    (out_dir / "keyframe.txt").write_text(frame_path.stem + "\n")

    return f"{splat_path.name}" + (", mesh.glb" if save_mesh else "")


def main():
    args = parse_args()
    if args.all_candidates and args.candidate_idx is not None:
        sys.exit("error: --all-candidates and --candidate-idx are mutually exclusive")

    if args.no_cudnn:
        # The sam3d-objects env's torch/cuDNN aborts some convs with "CUDA error:
        # unrecognized error code" on this cluster's CUDA-13 driver. Disabling
        # cuDNN routes convs through the native CUDA path.
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled (convs use the native CUDA fallback).")

    if args.blas_lib != "default":
        # Optional cuBLAS backend override (see --blas-lib help). Not forced by
        # default: forcing cuBLASLt fixes the `sam3` env but breaks skinny
        # fp16 GEMMs in sam3d-objects. Must run before any CUDA GEMM.
        try:
            torch.backends.cuda.preferred_blas_library(args.blas_lib)
            print(f"preferred_blas_library set to '{args.blas_lib}'")
        except Exception as e:
            print(f"warning: could not set preferred_blas_library={args.blas_lib}: {e}")

    scene_dir = args.scene_dir.resolve()
    sam3d_repo = args.sam3d_repo.resolve()
    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"
    tracking_path = masks_root / "tracking.json"
    out_root = scene_dir / "sam3d"

    if not tracking_path.is_file():
        sys.exit(f"error: {tracking_path} does not exist (run stages 10 & 20 first)")
    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    # Locate checkpoint & inference wrapper
    config_path = sam3d_repo / "checkpoints" / args.checkpoint_tag / "pipeline.yaml"
    if not config_path.is_file():
        sys.exit(f"error: checkpoint config not found at {config_path}")
    notebook_dir = sam3d_repo / "notebook"
    if not (notebook_dir / "inference.py").is_file():
        sys.exit(f"error: {notebook_dir / 'inference.py'} not found")

    # Make the notebook's inference.py importable
    sys.path.insert(0, str(notebook_dir))
    from inference import Inference  # noqa: E402

    # Load tracking.json
    with open(tracking_path) as f:
        tracking = json.load(f)

    # Filter labels
    labels = sorted(tracking.keys())
    if args.labels:
        unknown = [l for l in args.labels if l not in tracking]
        if unknown:
            sys.exit(f"error: --labels not in tracking.json: {unknown}")
        labels = [l for l in labels if l in args.labels]

    if not labels:
        sys.exit("error: nothing to process")

    # Build the inference pipeline once
    print(f"scene: {scene_dir.name}")
    print(f"checkpoint: {config_path}")
    print(f"labels to process: {labels}")
    if args.bg_mode == "none":
        print("background: kept (original image)")
    elif args.bg_mode == "dim":
        print(f"background: dimmed x{args.bg_dim:g} "
              f"(dilate={args.bg_dilate}, feather={args.bg_feather})")
    else:
        print(f"background: erased -> '{args.bg_mode}' "
              f"(dilate={args.bg_dilate}, feather={args.bg_feather})")
    print(f"loading SAM 3D Objects pipeline (compile={args.compile}) ...")
    inference = Inference(str(config_path), compile=args.compile)
    print("pipeline ready.\n")

    save_mesh = not args.skip_mesh

    for label in labels:
        entry = tracking[label]
        kf = entry.get("keyframe")
        candidates = entry.get("keyframe_candidates") or ([kf] if kf is not None else [])

        if not candidates:
            print(f"[{label}] SKIP: no keyframe (stage 20 didn't pick one)")
            continue

        # Decide which candidates to run
        if args.all_candidates:
            run_list = list(enumerate(candidates))
            multi = True
        elif args.candidate_idx is not None:
            if args.candidate_idx >= len(candidates):
                print(f"[{label}] SKIP: --candidate-idx {args.candidate_idx} out of range "
                      f"(have {len(candidates)} candidates)")
                continue
            run_list = [(args.candidate_idx, candidates[args.candidate_idx])]
            multi = False
        else:
            run_list = [(0, candidates[0])]
            multi = False

        for cand_idx, frame_idx in run_list:
            stem = f"{frame_idx:06d}"
            frame_path = frames_dir / f"{stem}.jpg"
            mask_path = masks_root / label / f"{stem}.png"

            if not frame_path.is_file():
                print(f"[{label}] cand{cand_idx} kf={stem}: SKIP (missing {frame_path})")
                continue
            if not mask_path.is_file():
                print(f"[{label}] cand{cand_idx} kf={stem}: SKIP (missing {mask_path})")
                continue

            if multi:
                out_dir = out_root / label / f"cand_{cand_idx:02d}_{stem}"
            else:
                out_dir = out_root / label

            if out_dir.exists():
                if args.overwrite:
                    shutil.rmtree(out_dir)
                else:
                    print(f"[{label}] cand{cand_idx} kf={stem}: SKIP "
                          f"({out_dir} exists; --overwrite to replace)")
                    continue

            print(f"[{label}] cand{cand_idx} kf={stem}: running ...")
            try:
                summary = run_one(
                    inference, frame_path, mask_path, out_dir,
                    seed=args.seed, save_mesh=save_mesh,
                    bg_mode=args.bg_mode, bg_dilate=args.bg_dilate,
                    bg_feather=args.bg_feather, bg_dim=args.bg_dim,
                    save_input=args.save_input,
                )
                print(f"[{label}] cand{cand_idx} kf={stem}: OK -> {out_dir}/{{{summary}}}")
            except Exception as e:
                print(f"[{label}] cand{cand_idx} kf={stem}: FAIL ({type(e).__name__}: {e})")
                traceback.print_exc()
                # don't tear down the pipeline; just move on
                # leave the partial out_dir for inspection unless overwrite was set
                continue

    print("\ndone.")


if __name__ == "__main__":
    main()
