"""Command-line interface for the video-driven Stage 09 pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from .data import VideoScene
from .pipeline import PipelineConfig, estimate_label, write_results


DESCRIPTION = """Stage 09: video-driven revolute/prismatic joint estimation.

Uses moving-part 3D tracks to initialize both one-DoF joint types, refines
both with DINOv2 features and masks, selects on held-out video, then estimates
one scalar state per frame. Video-only initialization remains an ablation.
Robust DA3 depth consistency and an Any6D initial object pose are available
as explicit opt-in signals.
"""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=DESCRIPTION,
    )
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Scene folder, e.g. data/stapler")
    parser.add_argument("--static-label",
                        help="Fixed part label (auto-inferred for two-part scenes)")
    parser.add_argument("--moving-labels", "--labels", nargs="*", default=None,
                        help="Moving parts (default: every non-static mesh piece)")
    parser.add_argument("--candidate", type=int, default=0,
                        help="SAM3D combined-mesh candidate index")
    parser.add_argument("--device", default="cuda",
                        help="Torch device; nvdiffrast currently requires CUDA")
    parser.add_argument("--max-render-side", type=int, default=336)
    parser.add_argument("--keyframes", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--model-margin", type=float, default=0.05)
    parser.add_argument("--dino-model", default="dinov2_vitl14_reg")
    parser.add_argument("--dino-checkpoint", type=Path)
    parser.add_argument("--feature-dimensions", type=int, default=32)
    parser.add_argument("--no-features", action="store_true",
                        help="Mask-only ablation; DINO is enabled by default")
    parser.add_argument("--use-depth", action="store_true",
                        help="Add robust DA3 relative-depth consistency")
    parser.add_argument("--lambda-depth", type=float, default=0.25,
                        help="Weight for the optional robust depth loss")
    parser.add_argument("--depth-huber-delta", type=float, default=0.05,
                        help="Huber transition for relative depth error")
    pose_group = parser.add_mutually_exclusive_group()
    pose_group.add_argument(
        "--pose-initializer", choices=["sam3d", "any6d"], default="sam3d",
        help="Initial shared object pose: sam3d/combined/pose.json or "
             "any6d/combined/pose.json")
    pose_group.add_argument(
        "--use-any6d-pose", action="store_true",
        help="Compatibility alias for --pose-initializer any6d")
    parser.add_argument("--pose-iters", type=int, default=180)
    parser.add_argument("--track-iters", type=int, default=180,
                        help="Sparse pose iterations for --joint-initializer video only")
    parser.add_argument("--state-iters", type=int, default=180)
    parser.add_argument("--joint-iters", type=int, default=350)
    parser.add_argument("--global-iters", type=int, default=120)
    parser.add_argument("--validation-iters", type=int, default=100)
    parser.add_argument("--dense-iters", type=int, default=60)
    parser.add_argument("--no-dense-states", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ransac-iterations", type=int, default=64)
    parser.add_argument("--lambda-feature", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--lambda-boundary", type=float, default=0.15)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--lambda-shape", type=float, default=0.1)
    joint_group = parser.add_mutually_exclusive_group()
    joint_group.add_argument(
        "--joint-initializer", choices=["tracks", "video"], default="tracks",
        help="Initialize joints from Stage-08 3D tracks (default) or sparse image-based poses")
    joint_group.add_argument(
        "--use-track-prior", dest="joint_initializer", action="store_const", const="tracks",
        help="Compatibility alias for --joint-initializer tracks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate everything without writing JSON")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.lambda_depth < 0:
        raise SystemExit("error: --lambda-depth must be non-negative")
    if args.depth_huber_delta <= 0:
        raise SystemExit("error: --depth-huber-delta must be positive")
    if args.device.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise SystemExit("error: PyTorch is unavailable; activate the trellis2 environment") from exc
        if not torch.cuda.is_available():
            raise SystemExit(
                "error: CUDA is unavailable; Stage 09 uses nvdiffrast's CUDA renderer. "
                "Activate the trellis2 environment on a GPU worker."
            )
    try:
        pose_initializer = "any6d" if args.use_any6d_pose else args.pose_initializer
        scene = VideoScene(
            args.scene_dir, args.static_label, args.moving_labels,
            candidate=args.candidate,
            use_any6d_pose=pose_initializer == "any6d")
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    config = PipelineConfig(
        device=args.device, max_render_side=args.max_render_side,
        keyframes=args.keyframes, validation_fraction=args.validation_fraction,
        dino_model=args.dino_model, dino_checkpoint=args.dino_checkpoint,
        feature_dimensions=args.feature_dimensions, use_features=not args.no_features,
        pose_iters=args.pose_iters, track_iters=args.track_iters,
        state_iters=args.state_iters, joint_iters=args.joint_iters,
        global_iters=args.global_iters, validation_iters=args.validation_iters,
        dense_iters=args.dense_iters, dense_states=not args.no_dense_states,
        batch_size=args.batch_size, ransac_iterations=args.ransac_iterations,
        model_margin=args.model_margin, lambda_feature=args.lambda_feature,
        lambda_mask=args.lambda_mask, lambda_boundary=args.lambda_boundary,
        lambda_smooth=args.lambda_smooth, lambda_shape=args.lambda_shape,
        use_depth=args.use_depth, lambda_depth=args.lambda_depth,
        depth_huber_delta=args.depth_huber_delta,
        joint_initializer=args.joint_initializer, seed=args.seed,
    )
    print(f"scene:        {scene.root.name}")
    print(f"static:       {scene.static_label}")
    print(f"moving:       {scene.moving_labels}")
    print(f"canonical:    {scene.reference_frame:06d}")
    print(f"pose init:    {scene.pose_initializer_kind} ({scene.pose_source})")
    signals = ["masks"] if args.no_features else ["DINOv2", "masks"]
    if args.use_depth:
        signals.append("DA3 depth")
    print(f"measurements: {' + '.join(signals)}")
    print(f"depth loss:   {'enabled' if args.use_depth else 'disabled'}")
    print(f"joint init:   {args.joint_initializer}")
    results = {}
    for label in scene.moving_labels:
        print(f"\n[{label}]")
        try:
            result = estimate_label(scene, label, config)
        except Exception as exc:
            print(f"  FAIL ({type(exc).__name__}: {exc})")
            traceback.print_exc()
            continue
        results[label] = result
        print(f"  selected={result['type']}  confidence={result['type_confidence']}  "
              f"margin={result['model_selection_margin']:.3f}")
        print(f"  axis={json.dumps([round(x, 6) for x in result['axis_direction']])}")
        if result["axis_point"] is not None:
            print(f"  pivot={json.dumps([round(x, 6) for x in result['axis_point']])}")
    if not results:
        raise SystemExit("error: no moving label completed successfully")
    write_results(scene, results, dry_run=args.dry_run)
    if args.dry_run:
        print("\n--dry-run: outputs were not written")
    else:
        print(f"\nwrote {scene.root / 'joints' / 'joints.json'}")
