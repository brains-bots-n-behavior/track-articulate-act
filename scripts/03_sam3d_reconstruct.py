#!/usr/bin/env python
"""Stage 03: per-part or combined-object 3D reconstruction with SAM 3D Objects.

For each label in masks/tracking.json, run SAM 3D Objects on every frame in
``keyframe_candidates`` using (frames/<kf>.jpg, masks/<label>/<kf>.png) and
write each reconstruction to its own candidate subfolder:

    data/<scene>/sam3d/<label>/
        cand_00_<kf>/
            keyframe.txt       # the frame index used (6-digit, no extension)
            splat.ply          # output["gs"].save_ply()
            mesh.glb           # output["glb"].export() (vertex-color mesh)
            pose.json          # rotation_quat_wxyz, translation, scale
            input_rgba.png     # only with --save-input

Use --candidate-idx to reconstruct just one candidate instead.

With --combined, the masks for all selected labels are unioned at a shared
keyframe and one reconstruction is written to data/<scene>/sam3d/combined/.
Shared candidate frames are ranked by how many labels nominated the frame,
then by their candidate ranks. Use --labels to control which masks participate.

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

Point map source:
    SAM 3D Objects conditions its sparse-structure stage on a point map. By
    default it runs its own MoGe depth model on the (background-suppressed)
    keyframe to produce one. With --pointmap-source da3 the point map is built
    instead from the depth + intrinsics stage 00 already wrote to
    data/<scene>/da3/, by back-projecting da3/depth/<idx>.npy through the
    matching K from da3/intrinsics.npz. That reuses the clip-wide,
    multi-view-consistent DA3 geometry rather than a single-image monocular
    estimate, so every part of an articulated object is conditioned on the same
    scene reconstruction.

    da3/ is indexed by POSITION in sorted(frames/*.jpg), not by frame stem (see
    stage 00: it saves depth[t] as f"{frame_indices[t]:06d}.npy" where
    frame_indices counts from 0 over the sorted frame list). Those agree only
    when the clip starts at frames/000000.jpg; for clips starting at 000001 the
    da3 index is the stem minus one. --da3-index position (the default) follows
    stage 00; --da3-index stem matches the frame stem directly.

Run inside the `sam3d-objects` conda env.

Example:
    python scripts/03_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects

    # Just one label, only the top candidate:
    python scripts/03_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --labels mug --candidate-idx 0

    # All candidates (the default), force re-run:
    python scripts/03_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --overwrite

    # Reconstruct the whole object from the union of all label masks:
    python scripts/03_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --combined

    # Condition on the existing DA3 geometry instead of SAM 3D's own MoGe:
    python scripts/03_sam3d_reconstruct.py \\
        --scene-dir data/kitchen_pour_01 \\
        --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \\
        --pointmap-source da3
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
    p.add_argument("--combined", action="store_true",
                   help="Reconstruct one whole object from the union of all selected "
                        "label masks, instead of reconstructing each label separately. "
                        "Outputs go under sam3d/combined/.")
    p.add_argument("--all-candidates", action="store_true",
                   help="Deprecated no-op: all keyframe_candidates are reconstructed by default")
    p.add_argument("--candidate-idx", type=int, default=None,
                   help="Reconstruct only this candidate index (0=top), instead of all candidates")
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
    p.add_argument("--pointmap-source", choices=["moge", "da3"], default="moge",
                   help="Where the point map that conditions SAM 3D comes from. "
                        "'moge' (default) runs SAM 3D's own depth model on the "
                        "keyframe; 'da3' back-projects the depth + intrinsics "
                        "stage 00 wrote to <scene>/da3/ instead.")
    p.add_argument("--da3-dir", type=Path, default=None,
                   help="DA3 product folder for --pointmap-source da3 "
                        "(default: <scene-dir>/da3)")
    p.add_argument("--da3-index", choices=["position", "stem"], default="position",
                   help="How a frame stem maps to a da3/depth/<idx>.npy. "
                        "'position' (default) uses the frame's position in "
                        "sorted(frames/*.jpg), which is what stage 00 wrote; "
                        "'stem' uses the 6-digit stem directly. The two agree "
                        "only when the clip starts at frames/000000.jpg.")
    p.add_argument("--da3-infer-intrinsics", action="store_true",
                   help="With --pointmap-source da3, let SAM 3D infer intrinsics "
                        "from the point map instead of using DA3's own K.")
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


class DA3PointmapLoader:
    """Builds SAM 3D point maps from the DA3 products stage 00 wrote.

    Stage 00 stores a metric z-depth map per frame plus the full-resolution
    pinhole K it belongs to. Back-projecting the depth through K gives a dense
    camera-space point map in the same RDF frame (+x right, +y down, +z forward)
    that SAM 3D's own depth model works in, so it can be handed straight to the
    pipeline with pointmap_convention="camera".
    """

    def __init__(self, da3_dir: Path, frames_dir: Path, index_mode: str = "position",
                 use_intrinsics: bool = True):
        self.da3_dir = da3_dir
        self.depth_dir = da3_dir / "depth"
        self.index_mode = index_mode
        self.use_intrinsics = use_intrinsics

        if not self.depth_dir.is_dir():
            raise FileNotFoundError(f"{self.depth_dir} does not exist (run stage 00 first)")

        # Stage 00 enumerates sorted(frames/*.jpg) and names its outputs by
        # position in that list, so rebuild the same list to invert the mapping.
        self.frame_stems = [path.stem for path in sorted(frames_dir.glob("*.jpg"))]

        # Needed either way: K is what turns the z-depth map into 3D points.
        # `use_intrinsics` only decides whether it is *also* handed to SAM 3D
        # instead of letting it infer intrinsics from the point map.
        intr_path = da3_dir / "intrinsics.npz"
        if not intr_path.is_file():
            raise FileNotFoundError(f"{intr_path} does not exist (run stage 00 first)")
        npz = np.load(intr_path)
        self.intrinsics = npz["intrinsics"]
        self.intr_frames = [int(i) for i in npz["frame_indices"].tolist()]

    def da3_index(self, stem: str) -> int:
        """Map a frames/<stem>.jpg to its index in the da3 products."""
        if self.index_mode == "stem":
            return int(stem)
        try:
            return self.frame_stems.index(stem)
        except ValueError:
            raise FileNotFoundError(
                f"frames/{stem}.jpg is not in the frame list da3 was built from")

    def __call__(self, frame_path: Path, target_hw=None) -> dict:
        """Return {'pointmap', 'intrinsics', 'info'} for one frame.

        `pointmap` is (H, W, 3) float32 in DA3's camera frame, NaN where the
        depth is invalid. `intrinsics` is the matching *normalized* 3x3 K (pixel
        units divided by the depth map's width/height) or None when SAM 3D
        should infer it. The point map is left at its native resolution; the
        pipeline resamples it onto the image grid.
        """
        idx = self.da3_index(frame_path.stem)
        depth_path = self.depth_dir / f"{idx:06d}.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(f"{depth_path} does not exist")

        depth = np.load(depth_path).astype(np.float32)
        if depth.ndim != 2:
            raise ValueError(f"{depth_path}: expected (H, W) depth, got {depth.shape}")
        H, W = depth.shape

        if idx not in self.intr_frames:
            raise KeyError(f"da3 index {idx} missing from {self.da3_dir/'intrinsics.npz'}")
        K = self.intrinsics[self.intr_frames.index(idx)].astype(np.float64)

        pointmap = back_project_depth(depth, K)

        # SAM 3D wants intrinsics normalized by the image size (principal point
        # in [0, 1]), which also makes them resolution-independent.
        intrinsics = None
        if self.use_intrinsics:
            intrinsics = K.copy()
            intrinsics[0] /= W
            intrinsics[1] /= H
            intrinsics = intrinsics.astype(np.float32)

        valid = int(np.isfinite(pointmap[..., 2]).sum())
        info = {
            "source": "da3",
            "da3_dir": str(self.da3_dir),
            "depth": str(depth_path.relative_to(self.da3_dir.parent)),
            "da3_index": idx,
            "frame_stem": frame_path.stem,
            "index_mode": self.index_mode,
            "resolution_hw": [H, W],
            "valid_pixels": valid,
            "valid_fraction": round(valid / float(H * W), 4),
            "intrinsics_source": "da3" if self.use_intrinsics else "inferred",
            "convention": "camera",
        }
        if target_hw is not None and tuple(target_hw) != (H, W):
            if abs((W / H) - (target_hw[1] / target_hw[0])) > 1e-3:
                print(f"    warning: da3 depth is {W}x{H} but the frame is "
                      f"{target_hw[1]}x{target_hw[0]} — differing aspect ratios "
                      f"will misalign the point map with the image")
            info["frame_resolution_hw"] = list(target_hw)

        return {"pointmap": pointmap, "intrinsics": intrinsics, "info": info}


def back_project_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Z-depth (H, W) + pinhole K -> (H, W, 3) camera-space points, NaN where invalid.

    Same RDF convention as stage 00 / MoGe: +x right, +y down, +z forward.
    Pixels with non-finite or non-positive depth become NaN so the downstream
    (NaN-aware) normalizer ignores them rather than treating them as the origin.
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = np.arange(W, dtype=np.float32)[None, :]
    v = np.arange(H, dtype=np.float32)[:, None]
    z = depth.astype(np.float32)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z

    pointmap = np.stack([np.broadcast_to(x, depth.shape),
                         np.broadcast_to(y, depth.shape), z], axis=-1).astype(np.float32)
    invalid = ~np.isfinite(z) | (z <= 0)
    if invalid.any():
        pointmap[invalid] = np.nan
    return pointmap


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


def run_one(inference, frame_path: Path, mask_paths, out_dir: Path,
            seed: int, save_mesh: bool, bg_mode: str = "black",
            bg_dilate: int = 0, bg_feather: int = 0, bg_dim: float = 0.3,
            save_input: bool = False, pointmap_loader=None):
    """
    Run SAM 3D Objects on a frame and the union of one or more masks.

    `pointmap_loader`, when given, is called as loader(frame_path, (H, W)) and
    must return {'pointmap', 'intrinsics', 'info'}; its point map replaces the
    one SAM 3D's own depth model would have estimated.

    Returns a short summary string.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image_rgb(frame_path)            # HxWx3 uint8
    if isinstance(mask_paths, (str, Path)):
        mask_paths = [Path(mask_paths)]
    else:
        mask_paths = [Path(path) for path in mask_paths]
    if not mask_paths:
        raise RuntimeError("no masks supplied")

    mask = np.zeros(image.shape[:2], dtype=bool)
    for mask_path in mask_paths:
        mask |= load_binary_mask(mask_path, target_hw=image.shape[:2])

    if not mask.any():
        raise RuntimeError(f"combined mask is empty: {mask_paths}")

    # Suppress the background so the masked object dominates the model's input.
    image = apply_background(image, mask, bg_mode, bg_dilate, bg_feather, bg_dim)

    # Optionally dump the exact RGBA the model receives (RGB after bg fill,
    # mask in the alpha channel) for visual inspection.
    if save_input:
        rgba = np.concatenate(
            [image, (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
        Image.fromarray(rgba, "RGBA").save(out_dir / "input_rgba.png")

    # Optional external point map (e.g. the DA3 depth stage 00 already ran).
    pointmap_bundle = None
    if pointmap_loader is not None:
        pointmap_bundle = pointmap_loader(frame_path, image.shape[:2])

    if pointmap_bundle is None:
        output = inference(image, mask, seed=seed)
    else:
        output = inference(
            image, mask, seed=seed,
            pointmap=pointmap_bundle["pointmap"],
            pointmap_convention=pointmap_bundle["info"]["convention"],
            pointmap_intrinsics=pointmap_bundle["intrinsics"],
        )

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
    (out_dir / "mask_labels.txt").write_text(
        "\n".join(path.parent.name for path in mask_paths) + "\n")

    # Provenance for the conditioning geometry, so runs can be told apart later.
    if pointmap_bundle is not None:
        with open(out_dir / "pointmap.json", "w") as f:
            json.dump(pointmap_bundle["info"], f, indent=2)

    return f"{splat_path.name}" + (", mesh.glb" if save_mesh else "")


def combined_candidates(tracking: dict, labels):
    """Rank shared frames nominated by the selected labels.

    More label votes win; ties favor a lower total candidate rank and then an
    earlier frame. This uses the keyframe picker results without privileging
    whichever label happens to sort first.
    """
    scores = {}
    for label in labels:
        entry = tracking[label]
        keyframe = entry.get("keyframe")
        candidates = entry.get("keyframe_candidates") or (
            [keyframe] if keyframe is not None else [])
        for rank, frame_idx in enumerate(candidates):
            frame_idx = int(frame_idx)
            votes, rank_sum = scores.get(frame_idx, (0, 0))
            scores[frame_idx] = (votes + 1, rank_sum + rank)
    return sorted(scores, key=lambda frame_idx: (
        -scores[frame_idx][0], scores[frame_idx][1], frame_idx))


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
        sys.exit(f"error: {tracking_path} does not exist (run stage 02 with keyframe_candidates in prompts.json first)")
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
    print(f"reconstruction: {'combined mask' if args.combined else 'per label'}")
    if args.bg_mode == "none":
        print("background: kept (original image)")
    elif args.bg_mode == "dim":
        print(f"background: dimmed x{args.bg_dim:g} "
              f"(dilate={args.bg_dilate}, feather={args.bg_feather})")
    else:
        print(f"background: erased -> '{args.bg_mode}' "
              f"(dilate={args.bg_dilate}, feather={args.bg_feather})")
    pointmap_loader = None
    if args.pointmap_source == "da3":
        da3_dir = (args.da3_dir or (scene_dir / "da3")).resolve()
        try:
            pointmap_loader = DA3PointmapLoader(
                da3_dir, frames_dir, index_mode=args.da3_index,
                use_intrinsics=not args.da3_infer_intrinsics,
            )
        except (FileNotFoundError, KeyError) as e:
            sys.exit(f"error: --pointmap-source da3: {e}")
        print(f"point map: da3 ({da3_dir}), index by {args.da3_index}, "
              f"intrinsics {'from da3' if not args.da3_infer_intrinsics else 'inferred'}")
    else:
        print("point map: SAM 3D's own depth model (MoGe)")
    print(f"loading SAM 3D Objects pipeline (compile={args.compile}) ...")
    inference = Inference(str(config_path), compile=args.compile)
    print("pipeline ready.\n")

    save_mesh = not args.skip_mesh

    if args.combined:
        candidates = combined_candidates(tracking, labels)
        if not candidates:
            sys.exit("error: selected labels have no keyframe candidates")

        if args.candidate_idx is not None:
            if args.candidate_idx < 0 or args.candidate_idx >= len(candidates):
                sys.exit(f"error: --candidate-idx {args.candidate_idx} out of range "
                         f"for combined reconstruction (have {len(candidates)})")
            run_list = [(args.candidate_idx, candidates[args.candidate_idx])]
            multi = False
        else:
            run_list = list(enumerate(candidates))
            multi = True

        for cand_idx, frame_idx in run_list:
            stem = f"{frame_idx:06d}"
            frame_path = frames_dir / f"{stem}.jpg"
            mask_paths = [masks_root / label / f"{stem}.png" for label in labels]
            missing = [path for path in mask_paths if not path.is_file()]
            tag = f"[combined] cand{cand_idx} kf={stem}"

            if not frame_path.is_file():
                print(f"{tag}: SKIP (missing {frame_path})")
                continue
            if missing:
                print(f"{tag}: SKIP (missing masks: "
                      f"{', '.join(str(path) for path in missing)})")
                continue

            out_dir = (out_root / "combined" / f"cand_{cand_idx:02d}_{stem}"
                       if multi else out_root / "combined")
            if out_dir.exists():
                if args.overwrite:
                    shutil.rmtree(out_dir)
                else:
                    print(f"{tag}: SKIP ({out_dir} exists; --overwrite to replace)")
                    continue

            print(f"{tag}: running with {len(mask_paths)} masks ...")
            try:
                summary = run_one(
                    inference, frame_path, mask_paths, out_dir,
                    seed=args.seed, save_mesh=save_mesh,
                    bg_mode=args.bg_mode, bg_dilate=args.bg_dilate,
                    bg_feather=args.bg_feather, bg_dim=args.bg_dim,
                    save_input=args.save_input, pointmap_loader=pointmap_loader,
                )
                print(f"{tag}: OK -> {out_dir}/{{{summary}}}")
            except Exception as e:
                print(f"{tag}: FAIL ({type(e).__name__}: {e})")
                traceback.print_exc()

        print("\ndone.")
        return

    for label in labels:
        entry = tracking[label]
        kf = entry.get("keyframe")
        candidates = entry.get("keyframe_candidates") or ([kf] if kf is not None else [])

        if not candidates:
            print(f"[{label}] SKIP: no keyframe (set keyframe_candidates in prompts.json before segmentation)")
            continue

        # Decide which candidates to run
        if args.candidate_idx is not None:
            if args.candidate_idx < 0 or args.candidate_idx >= len(candidates):
                print(f"[{label}] SKIP: --candidate-idx {args.candidate_idx} out of range "
                      f"(have {len(candidates)} candidates)")
                continue
            run_list = [(args.candidate_idx, candidates[args.candidate_idx])]
            multi = False
        else:
            run_list = list(enumerate(candidates))
            multi = True

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
                    save_input=args.save_input, pointmap_loader=pointmap_loader,
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
