#!/usr/bin/env python
"""Stage 20: pick the best keyframe(s) per object.

For each label in masks/tracking.json, score every visible frame by mask
quality + image sharpness, then write:
    tracking[label]["keyframe"]            = top-1 frame index
    tracking[label]["keyframe_candidates"] = list of top-N frame indices,
                                              temporally spaced

Scoring (per frame where the mask is non-empty):
    area_frac    = mask_pixels / (H * W)
    sharpness    = variance of Laplacian of the grayscale crop inside the bbox
    on_boundary  = bbox touches the image edge (within --edge-margin px)
    centeredness = 1 - normalized L2 distance from mask centroid to image center

Filter rules (in this order):
    - mask non-empty
    - area_frac in [--area-min, --area-max]
    - not on_boundary  (object likely truncated by frame edge)

Score: sharpness * sqrt(area_frac) * centeredness
The frame with the highest score wins.

If everything is filtered out:
    - relax the boundary check
    - relax the area band
    - finally, fall back to the visible frame closest to the median area

Candidate selection picks the top frame, then greedily adds the next-highest-
scoring frame that is at least --min-spacing frames away from every already-
chosen candidate. This keeps candidates temporally diverse so stage 30 can
sample different viewing angles.

Manual override:
    --manual mug=42 person=18
overrides the scoring for the given labels. The override becomes the keyframe
and the only candidate. Frame must be a visible frame for that label.

Inputs:  data/<scene>/frames/*.jpg
         data/<scene>/masks/<label>/*.png
         data/<scene>/masks/tracking.json

Output:  updates tracking.json in place
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import median

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--area-min", type=float, default=0.005,
                   help="Minimum mask area fraction (default 0.5%%)")
    p.add_argument("--area-max", type=float, default=0.60,
                   help="Maximum mask area fraction (default 60%%)")
    p.add_argument("--edge-margin", type=int, default=2,
                   help="Pixels from the image edge that count as 'on boundary' (default 2)")
    p.add_argument("--n-candidates", type=int, default=3,
                   help="Number of temporally-spaced keyframe candidates to emit (default 3)")
    p.add_argument("--min-spacing", type=int, default=5,
                   help="Minimum frames between candidates (default 5)")
    p.add_argument("--manual", nargs="*", default=[],
                   help="Manual overrides like 'label=frame_index' (space-separated list)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print picks but do not modify tracking.json")
    return p.parse_args()


def parse_manual(items):
    out = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"error: bad --manual entry '{item}', expected label=frame")
        label, frame = item.split("=", 1)
        try:
            out[label] = int(frame)
        except ValueError:
            sys.exit(f"error: --manual frame '{frame}' is not an integer")
    return out


def list_visible_frames(label_dir: Path):
    """Return sorted list of (frame_idx, frame_stem, mask_path) where mask has pixels."""
    visible = []
    for mp in sorted(label_dir.glob("*.png")):
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if not (m > 0).any():
            continue
        try:
            frame_idx = int(mp.stem)
        except ValueError:
            continue
        visible.append((frame_idx, mp.stem, mp))
    return visible


def score_frame(mask_path: Path, frame_path: Path, edge_margin: int):
    """Return dict with area_frac, sharpness, on_boundary, centeredness, or None if unreadable."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    H, W = mask.shape
    bool_mask = mask > 0
    n_pix = int(bool_mask.sum())
    if n_pix == 0:
        return None

    area_frac = n_pix / float(H * W)

    ys, xs = np.where(bool_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    on_boundary = (
        x0 <= edge_margin
        or y0 <= edge_margin
        or x1 >= W - 1 - edge_margin
        or y1 >= H - 1 - edge_margin
    )

    cy, cx = ys.mean(), xs.mean()
    img_cy, img_cx = (H - 1) / 2.0, (W - 1) / 2.0
    norm_dist = np.hypot((cy - img_cy) / H, (cx - img_cx) / W)
    centeredness = max(0.0, 1.0 - 2.0 * norm_dist)  # 1 at center, ~0 at corner

    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    crop = img[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        sharpness = 0.0
    else:
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var())

    return {
        "area_frac": area_frac,
        "sharpness": sharpness,
        "on_boundary": bool(on_boundary),
        "centeredness": float(centeredness),
    }


def composite_score(s):
    return s["sharpness"] * np.sqrt(max(s["area_frac"], 1e-9)) * s["centeredness"]


def pick_spaced_candidates(scored_sorted, n, min_spacing):
    """
    Greedy: walk down the score-sorted list and accept a frame only if it is
    at least `min_spacing` frames away from every already-accepted candidate.
    Returns frame_indices in score-descending order (top score first).
    """
    chosen = []
    for s in scored_sorted:
        f = s["frame_idx"]
        if all(abs(f - c) >= min_spacing for c in chosen):
            chosen.append(f)
            if len(chosen) >= n:
                break
    return chosen


def rank_visible(label, visible, frames_dir, area_min, area_max, edge_margin):
    """
    Score every visible frame, apply the same 3-pass relaxation as before, and
    return (sorted_pool, reason). sorted_pool is the *final* pool of scored
    frames sorted by composite score (best first); reason explains which pass
    produced it.
    """
    scored = []
    for frame_idx, stem, mp in visible:
        fp = frames_dir / f"{stem}.jpg"
        if not fp.exists():
            continue
        s = score_frame(mp, fp, edge_margin)
        if s is None:
            continue
        s["frame_idx"] = frame_idx
        scored.append(s)

    if not scored:
        return [], "no scorable frames"

    # Pass 1: strict
    strict = [s for s in scored
              if not s["on_boundary"]
              and area_min <= s["area_frac"] <= area_max]
    if strict:
        return sorted(strict, key=composite_score, reverse=True), "strict"

    # Pass 2: allow boundary touch
    relaxed = [s for s in scored if area_min <= s["area_frac"] <= area_max]
    if relaxed:
        return sorted(relaxed, key=composite_score, reverse=True), "relaxed-boundary"

    # Pass 3: ignore area band, still avoid boundary if possible
    inboard = [s for s in scored if not s["on_boundary"]]
    if inboard:
        return sorted(inboard, key=composite_score, reverse=True), "relaxed-area"

    # Fallback: order by closeness to median area
    areas = [s["area_frac"] for s in scored]
    med = median(areas)
    fallback = sorted(scored, key=lambda s: abs(s["area_frac"] - med))
    return fallback, "fallback-median-area"


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    frames_dir = scene_dir / "frames"
    masks_root = scene_dir / "masks"
    tracking_path = masks_root / "tracking.json"

    if not tracking_path.is_file():
        sys.exit(f"error: {tracking_path} does not exist (run stage 10 first)")
    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    with open(tracking_path) as f:
        tracking = json.load(f)

    manual = parse_manual(args.manual)
    unknown = [lbl for lbl in manual if lbl not in tracking]
    if unknown:
        sys.exit(f"error: --manual labels not in tracking.json: {unknown}")

    print(f"scene: {scene_dir.name}")
    print(f"area band: [{args.area_min:.3%}, {args.area_max:.0%}], "
          f"edge margin: {args.edge_margin}px")
    print(f"candidates: top-{args.n_candidates}, min spacing {args.min_spacing} frames")
    print(f"labels: {len(tracking)}\n")

    if args.n_candidates < 1:
        sys.exit(f"error: --n-candidates must be >= 1, got {args.n_candidates}")

    for label in sorted(tracking):
        label_dir = masks_root / label
        if not label_dir.is_dir():
            print(f"  {label}: SKIP (no folder at {label_dir})")
            tracking[label]["keyframe"] = None
            tracking[label]["keyframe_candidates"] = []
            continue

        visible = list_visible_frames(label_dir)
        if not visible:
            print(f"  {label}: SKIP (no visible frames)")
            tracking[label]["keyframe"] = None
            tracking[label]["keyframe_candidates"] = []
            continue

        if label in manual:
            f = manual[label]
            visible_idxs = {v[0] for v in visible}
            if f not in visible_idxs:
                sys.exit(f"error: manual keyframe {f} for label '{label}' is not "
                         f"a visible frame (visible range "
                         f"{min(visible_idxs)}..{max(visible_idxs)})")
            tracking[label]["keyframe"] = f
            tracking[label]["keyframe_candidates"] = [f]
            print(f"  {label}: manual -> {f:06d}")
            continue

        sorted_pool, reason = rank_visible(
            label, visible, frames_dir,
            args.area_min, args.area_max, args.edge_margin,
        )
        if not sorted_pool:
            tracking[label]["keyframe"] = None
            tracking[label]["keyframe_candidates"] = []
            print(f"  {label}: FAIL ({reason})")
            continue

        candidates = pick_spaced_candidates(
            sorted_pool, args.n_candidates, args.min_spacing
        )
        tracking[label]["keyframe"] = candidates[0]
        tracking[label]["keyframe_candidates"] = candidates
        cand_str = ",".join(f"{c:06d}" for c in candidates)
        print(f"  {label}: top={candidates[0]:06d} candidates=[{cand_str}] "
              f"({reason}, visible={len(visible)}, scored={len(sorted_pool)})")

    if args.dry_run:
        print("\n--dry-run: tracking.json not modified")
    else:
        with open(tracking_path, "w") as f:
            json.dump(tracking, f, indent=2)
        print(f"\nupdated {tracking_path}")


if __name__ == "__main__":
    main()
