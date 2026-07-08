#!/usr/bin/env python
"""Stage 10: SAM3 video segmentation + tracking.

Consumes:
    data/<scene>/frames/*.jpg              (canonical RGB frames from stage 00)

Produces:
    data/<scene>/masks/<label>/<frame>.png    (8-bit, 0/255, single channel)
    data/<scene>/masks/tracking.json          (prompt + provenance per label)

The script accepts either a single text prompt (--prompt) or a JSON file with
multiple prompts (--prompts-json). Multiple text prompts are run sequentially
with reset_session between them. Each prompt produces one folder per detected
SAM3 instance, named after the prompt's `label`. If a prompt detects multiple
instances, suffixes `_1`, `_2`, ... are appended (e.g. `person_1`, `person_2`).
Labels must be unique across prompts and contain no spaces (caller's
responsibility).

Run inside the `sam3` conda env.

Example:
    python 10_sam3_segment.py \\
        --scene-dir data/kitchen_pour_01 \\
        --prompt "blue mug"

    python 10_sam3_segment.py \\
        --scene-dir data/kitchen_pour_01 \\
        --prompts-json prompts.json \\
        --version sam3.1

prompts.json schema:
    {
      "prompts": [
        # --- Concept (text) prompts: segment ALL instances of a concept ---
        {"text": "person",   "frame_index": 0, "label": "person"},
        {"text": "blue mug", "frame_index": 0, "label": "mug"},

        # --- Interactive (PVS) prompts: segment ONE part from geometry ---
        # Use these to separate adjacent parts of an articulated object,
        # where a text concept grabs the whole thing (e.g. a fridge door vs
        # its body). Provide a box and/or points and omit `text`. Coordinates
        # are absolute image pixels; author them with 05_pick_prompts.py.
        {
          "frame_index": 0,
          "label": "lid",
          "box": [x1, y1, x2, y2],            # xyxy, optional
          "positive_points": [[x, y], ...],   # kept inside the part
          "negative_points": [[x, y], ...]    # excluded from the part
        }
      ]
    }

A prompt is INTERACTIVE (PVS) when it carries any of box / positive_points /
negative_points; otherwise it is a CONCEPT (text) prompt. Interactive prompts
build the object purely from the geometry via the SAM2 tracker path, so
negative points genuinely carve the boundary. The text concept is NOT used for
an interactive prompt even if present -- the concept detector re-asserts the
whole object on every frame during propagation, which is exactly why a few
negative clicks layered on a text mask can't remove a sub-part.

Notes on the geometry:
  * A box is encoded as two SAM2 corner points (labels 2 / 3).
  * At most 16 geometry points survive (the model keeps the first 8 + last 8),
    so keep clicks modest (<= 8 positive and <= 8 negative).
  * Coordinates are normalized to [0, 1] before being sent (the model rescales
    by its internal image_size); passing raw pixels would misplace every click.
"""

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_predictor
from sam3.visualization_utils import COLORS


# ---------------------------------------------------------------------------
# Workaround: legacy cuBLAS aborts on non-contiguous GEMMs in this env.
#
# On torch 2.10+cu128 (A100, driver 580 / CUDA 13), a matmul whose input has
# non-standard strides goes through the legacy cuBLAS path (cublasGemmEx /
# cublasGemmStridedBatchedEx) and aborts with CUBLAS_STATUS_INVALID_VALUE —
# for both bf16 AND fp16. SAM3's text encoder feeds such tensors into several
# matmuls (MHA in-projection, the final `pooled @ text_projection`, ...).
# Routing matmuls through cuBLASLt instead avoids the broken path. Verified:
# a standalone non-contiguous bf16 F.linear fails on "cublas"/default but
# passes on "cublaslt". Must be set before any CUDA GEMM runs.
# ---------------------------------------------------------------------------
try:
    torch.backends.cuda.preferred_blas_library("cublaslt")
except Exception as e:  # pragma: no cover - depends on torch/CUDA build
    print(f"warning: could not set preferred_blas_library=cublaslt: {e}")


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Path to data/<scene_id>/ (must contain frames/)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Single text prompt shortcut (applied at frame 0)")
    p.add_argument("--prompts-json", type=Path, default=None,
                   help="JSON file with a 'prompts' list (overrides --prompt)")
    p.add_argument("--version", choices=["sam3", "sam3.1"], default="sam3.1",
                   help="SAM3 model version (default: sam3.1)")
    p.add_argument("--mask-threshold", type=float, default=0.5,
                   help="Threshold applied to soft masks before saving (default 0.5)")
    p.add_argument("--overwrite", action="store_true",
                   help="Delete existing masks/ before running")
    p.add_argument("--compile", action="store_true",
                   help="Pass compile=True to the SAM3 predictor (slower first run, faster afterwards)")
    return p.parse_args()


def load_prompts(args):
    if args.prompts_json is not None:
        with open(args.prompts_json) as f:
            cfg = json.load(f)
        prompts = cfg["prompts"]
    elif args.prompt is not None:
        prompts = [{"text": args.prompt, "frame_index": 0, "label": args.prompt}]
    else:
        sys.exit("error: must provide either --prompt or --prompts-json")

    for p in prompts:
        p.setdefault("frame_index", 0)
        has_geom = bool(p.get("positive_points") or p.get("negative_points")
                        or p.get("box"))
        text = (p.get("text") or "").strip()
        if not text and not has_geom:
            sys.exit(f"error: prompt has neither 'text' nor points/box: {p}")
        if not p.get("label"):
            if text:
                p["label"] = text
            else:
                sys.exit(f"error: interactive prompt (no 'text') must set a "
                         f"'label' (it is used as the output folder name): {p}")
        if " " in p["label"] or "/" in p["label"]:
            sys.exit(f"error: label '{p['label']}' contains a space or slash; "
                     f"labels are used as folder names and must be filesystem-safe")
        for key in ("positive_points", "negative_points"):
            pts = p.get(key)
            if pts is None:
                continue
            if (not isinstance(pts, list)
                    or any(not (isinstance(xy, (list, tuple)) and len(xy) == 2)
                           for xy in pts)):
                sys.exit(f"error: prompt '{p['label']}' field '{key}' must be a "
                         f"list of [x, y] pairs (absolute image pixels), got: {pts}")
        box = p.get("box")
        if box is not None:
            if not (isinstance(box, (list, tuple)) and len(box) == 4):
                sys.exit(f"error: prompt '{p['label']}' field 'box' must be "
                         f"[x1, y1, x2, y2] in absolute pixels, got: {box}")

    seen_labels = [p["label"] for p in prompts]
    if len(set(seen_labels)) != len(seen_labels):
        dupes = [l for l in seen_labels if seen_labels.count(l) > 1]
        sys.exit(f"error: duplicate labels: {sorted(set(dupes))}")
    return prompts


def list_frame_paths(frames_dir: Path):
    paths = sorted(glob.glob(str(frames_dir / "*.jpg")))
    if not paths:
        sys.exit(f"error: no *.jpg frames found in {frames_dir}")
    # sanity: filenames should be parseable as ints (e.g. 000042.jpg)
    try:
        sorted(paths, key=lambda p: int(Path(p).stem))
    except ValueError:
        print(f"warning: frame names are not pure integers; lexicographic sort used")
    return paths


def resize_mask_to(mask: np.ndarray, target_hw: tuple, thresh: float) -> np.ndarray:
    """Resize a binary or soft mask to (H, W) and return uint8 in {0, 255}."""
    H, W = target_hw
    if mask.dtype == bool:
        mask_f = mask.astype(np.float32)
    else:
        mask_f = mask.astype(np.float32)
    if mask_f.shape[:2] != (H, W):
        mask_f = cv2.resize(mask_f, (W, H), interpolation=cv2.INTER_NEAREST)
    return ((mask_f > thresh).astype(np.uint8) * 255)


def propagate(predictor, session_id, start_frame_idx=None):
    # `start_frame_index` is required for interactive (points-only) prompts:
    # propagation derives its start frame from the detector's per-frame outputs,
    # which the SAM2 tracker path never fills, so without it the predictor
    # raises "No prompts are received on any frames". Direction defaults to
    # "both", so it tracks forward then backward from this frame.
    req = dict(type="propagate_in_video", session_id=session_id)
    if start_frame_idx is not None:
        req["start_frame_index"] = int(start_frame_idx)
    out = {}
    for resp in predictor.handle_stream_request(request=req):
        out[resp["frame_index"]] = resp["outputs"]
    return out


# SAM2 point-label convention used by the tracker prompt encoder:
#   1 = positive click, 0 = negative click,
#   2 = box top-left corner, 3 = box bottom-right corner
_LBL_POS, _LBL_NEG, _LBL_BOX_TL, _LBL_BOX_BR = 1, 0, 2, 3
# The tracker keeps only the first 8 + last 8 input points (see
# sam3_tracking_predictor.py: max_point_num_in_prompt_enc=16).
_MAX_GEOM_POINTS = 16


def _ids_to_list(ids):
    if ids is None:
        return []
    return ids.tolist() if hasattr(ids, "tolist") else list(ids)


def run_one_prompt(predictor, session_id, prompt, n_frames, orig_hw):
    """Reset the session and segment one prompt, returning per-frame outputs.

    A prompt is INTERACTIVE (PVS) when it carries box / positive_points /
    negative_points; otherwise it is a CONCEPT (text) prompt. Interactive
    prompts build the object purely from geometry via the SAM2 tracker path
    (no text concept), so negative points actually carve the part out -- a
    text concept would re-assert the whole object during propagation.
    """
    predictor.handle_request(
        request=dict(type="reset_session", session_id=session_id)
    )
    has_geom = bool(prompt.get("positive_points")
                    or prompt.get("negative_points")
                    or prompt.get("box"))
    if has_geom:
        return _run_interactive_prompt(predictor, session_id, prompt, orig_hw)
    return _run_concept_prompt(predictor, session_id, prompt)


def _run_concept_prompt(predictor, session_id, prompt):
    """Text concept prompt: detect & track every instance of the concept."""
    resp = predictor.handle_request(request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=int(prompt["frame_index"]),
        text=prompt["text"],
    ))
    ids = _ids_to_list(resp["outputs"]["out_obj_ids"])
    print(f"  concept '{prompt['text']}' at frame {prompt['frame_index']}: "
          f"SAM3 detected ids {ids}")
    if not ids:
        print(f"  warning: no objects found for '{prompt['text']}', skipping")
        return None
    return propagate(predictor, session_id, int(prompt["frame_index"]))


def _run_interactive_prompt(predictor, session_id, prompt, orig_hw):
    """PVS prompt: build a single object from a box and/or pos/neg points.

    Coordinates are normalized to [0, 1] (divided by the original frame size)
    and sent with rel_coordinates=True, the model's native convention -- it
    rescales by its internal image_size (1008). Passing raw original-resolution
    pixels (as an earlier version did) lands every click off-target.
    """
    H, W = orig_hw
    frame_index = int(prompt["frame_index"])
    pos = list(prompt.get("positive_points") or [])
    neg = list(prompt.get("negative_points") or [])
    box = prompt.get("box")

    # A box must be the first prompt; encode it as two corner points (2 / 3).
    pts, labels = [], []
    if box is not None:
        x1, y1, x2, y2 = box
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        pts += [[x1 / W, y1 / H], [x2 / W, y2 / H]]
        labels += [_LBL_BOX_TL, _LBL_BOX_BR]
    for x, y in pos:
        pts.append([x / W, y / H])
        labels.append(_LBL_POS)
    for x, y in neg:
        pts.append([x / W, y / H])
        labels.append(_LBL_NEG)

    if len(pts) > _MAX_GEOM_POINTS:
        half = _MAX_GEOM_POINTS // 2
        print(f"  warning: {len(pts)} geometry points exceed the model cap of "
              f"{_MAX_GEOM_POINTS}; only the first {half} + last {half} are "
              f"used (middle clicks dropped). Trim to <=8 positive and "
              f"<=8 negative.")

    resp = predictor.handle_request(request=dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=frame_index,
        obj_id=0,
        points=pts,
        point_labels=labels,
        rel_coordinates=True,
    ))
    ids = _ids_to_list(resp["outputs"].get("out_obj_ids"))
    box_str = " + box" if box is not None else ""
    print(f"  PVS '{prompt['label']}' at frame {frame_index}: "
          f"{len(pos)} positive + {len(neg)} negative click(s){box_str} "
          f"-> obj ids {ids}")
    if not ids:
        print(f"  warning: interactive prompt '{prompt['label']}' produced no "
              f"object, skipping")
        return None
    return propagate(predictor, session_id, frame_index)


def save_masks_and_track(outputs_per_frame, prompt, orig_hw, frame_stems,
                         masks_root, mask_threshold, color_idx_start):
    """
    Save every per-(sam3_obj_id) masklet to <label>/<frame>.png.
    Folder name is `prompt["label"]` (with `_1`, `_2`, ... suffix when the
    prompt detects multiple instances).
    Returns a dict keyed by folder name with provenance fields.
    """
    # 1. Discover every SAM3 obj_id that appears in any frame
    sam3_ids = set()
    for out in outputs_per_frame.values():
        ids = out["out_obj_ids"]
        sam3_ids.update(ids.tolist() if hasattr(ids, "tolist") else list(ids))
    sam3_ids = sorted(sam3_ids)

    # 2. Map each SAM3 id to a label-derived folder name
    base_label = prompt["label"]
    if len(sam3_ids) == 1:
        id_map = {sam3_ids[0]: base_label}
    else:
        id_map = {sid: f"{base_label}_{i + 1}" for i, sid in enumerate(sam3_ids)}

    # 3. Stats per folder
    is_interactive = bool(prompt.get("positive_points")
                          or prompt.get("negative_points") or prompt.get("box"))
    stats = {}
    for i, (sid, name) in enumerate(id_map.items()):
        color_idx = color_idx_start + i
        stats[name] = {
            "prompt": prompt.get("text") or "",
            "mode": "interactive" if is_interactive else "concept",
            "label": base_label,
            "color": list((COLORS[color_idx % len(COLORS)] * 255).astype(int).tolist()),
            "first_frame": None,
            "last_frame": None,
            "n_frames_visible": 0,
            "sam3_obj_id_local": int(sid),
        }

    # 4. Pre-create per-object folders
    for name in id_map.values():
        (masks_root / name).mkdir(parents=True, exist_ok=True)

    # 5. Iterate over every frame, write a PNG per object
    H, W = orig_hw
    n_frames = len(frame_stems)
    for frame_idx in range(n_frames):
        stem = frame_stems[frame_idx]
        out = outputs_per_frame.get(frame_idx)
        present_ids = {}
        if out is not None:
            ids_arr = out["out_obj_ids"]
            ids_list = (ids_arr.tolist() if hasattr(ids_arr, "tolist") else list(ids_arr))
            for arr_idx, sid in enumerate(ids_list):
                present_ids[sid] = arr_idx

        for sid, name in id_map.items():
            mask_path = masks_root / name / f"{stem}.png"
            if sid in present_ids:
                arr_idx = present_ids[sid]
                raw_mask = out["out_binary_masks"][arr_idx]
                if hasattr(raw_mask, "cpu"):
                    raw_mask = raw_mask.cpu().numpy()
                mask_u8 = resize_mask_to(np.asarray(raw_mask), (H, W), mask_threshold)
                if mask_u8.any():
                    stats[name]["n_frames_visible"] += 1
                    if stats[name]["first_frame"] is None:
                        stats[name]["first_frame"] = frame_idx
                    stats[name]["last_frame"] = frame_idx
            else:
                mask_u8 = np.zeros((H, W), dtype=np.uint8)
            cv2.imwrite(str(mask_path), mask_u8)

    # 6. Add keyframe placeholder for stage 20
    for name in stats:
        stats[name]["keyframe"] = None

    return stats, id_map


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"

    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist (run stage 00 first)")

    if masks_root.exists():
        if args.overwrite:
            print(f"removing existing {masks_root}")
            shutil.rmtree(masks_root)
        else:
            sys.exit(f"error: {masks_root} already exists; pass --overwrite to replace")
    masks_root.mkdir(parents=True)

    # Load frame metadata
    frame_paths = list_frame_paths(frames_dir)
    frame_stems = [Path(p).stem for p in frame_paths]
    n_frames = len(frame_paths)
    with Image.open(frame_paths[0]) as im:
        orig_W, orig_H = im.size
    print(f"scene: {scene_dir.name}")
    print(f"frames: {n_frames}, resolution: {orig_W}x{orig_H}")

    prompts = load_prompts(args)
    print(f"prompts ({len(prompts)}):")
    for p in prompts:
        has_geom = bool(p.get("positive_points") or p.get("negative_points")
                        or p.get("box"))
        mode = "PVS" if has_geom else "concept"
        print(f"  - [{mode}] label='{p['label']}' "
              f"text='{p.get('text') or ''}' frame={p['frame_index']}")

    # Build predictor
    print(f"building SAM3 predictor (version={args.version}, compile={args.compile}) ...")
    predictor = build_sam3_predictor(version=args.version, compile=args.compile)

    # Open session on the JPEG folder
    print(f"opening session on {frames_dir}")
    resp = predictor.handle_request(
        request=dict(type="start_session", resource_path=str(frames_dir))
    )
    session_id = resp["session_id"]

    # Run each prompt with reset between, merge into one tracking dict
    tracking = {}
    color_counter = 0
    try:
        for p_idx, prompt in enumerate(prompts):
            print(f"\n[{p_idx + 1}/{len(prompts)}] running prompt: "
                  f"'{prompt.get('text') or prompt['label']}'")
            outputs_per_frame = run_one_prompt(predictor, session_id, prompt,
                                                n_frames, orig_hw=(orig_H, orig_W))
            if outputs_per_frame is None:
                continue
            stats, id_map = save_masks_and_track(
                outputs_per_frame,
                prompt,
                (orig_H, orig_W),
                frame_stems,
                masks_root,
                args.mask_threshold,
                color_counter,
            )
            for name, s in stats.items():
                if name in tracking:
                    sys.exit(f"error: folder name '{name}' collides with an earlier "
                             f"prompt; labels must be unique across prompts")
                tracking[name] = s
            color_counter += len(id_map)
            print(f"  -> wrote folders {list(id_map.values())}")

        # Close session, shutdown predictor
        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id)
        )
    finally:
        try:
            predictor.shutdown()
        except Exception as e:
            print(f"warning: predictor.shutdown() raised: {e}")

    # Write tracking.json
    tracking_path = masks_root / "tracking.json"
    with open(tracking_path, "w") as f:
        json.dump(tracking, f, indent=2)
    print(f"\nwrote {tracking_path} ({len(tracking)} objects)")
    print(f"masks layout:")
    for name in sorted(tracking):
        s = tracking[name]
        print(f"  {name}: prompt='{s['prompt']}' "
              f"frames=[{s['first_frame']}..{s['last_frame']}] "
              f"visible={s['n_frames_visible']}")


if __name__ == "__main__":
    main()
