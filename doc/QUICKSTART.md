# Quickstart

The script prefixes are the stage numbers: **00 → 01 → 02 → 03 → 04 → 05 →
06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14**. Run the stages needed for your
workflow in this order. Joint estimation has separate mesh-based (07),
video-based (09), and point-track-based (10) options.

Run commands from the repository root. Use each model's environment from
[ENVIRONMENT_SETUP_GUIDE.md](ENVIRONMENT_SETUP_GUIDE.md); CPU processing stages
also need their imported dependencies installed. For the flags supported by a
script, use `python scripts/<filename>.py --help` in its environment.

```bash
SCENE=data/kitchen_pour_01
DA3_ROOT=/path/to/depth-anything-3
SAM3D_REPO=/path/to/sam-3d-objects
SEGVIGEN_ROOT=/path/to/SegviGen
SEGVIGEN_CKPT=/path/to/full_seg_w_2d_map.ckpt
TRACKCRAFT_REPO=/path/to/TrackCraft3r
TRACKCRAFT_CKPT=/path/to/trackcraft3r/model.safetensors
HAWOR_REPO=/path/to/HaWoR
```

Prepare the RGB frames in `$SCENE/frames/` with consistent numeric filenames,
for example `000000.jpg`, `000001.jpg`, and so on. Use a zero-based contiguous
sequence so DA3's sorted-frame indices match the frame stems used downstream.

## Stage 00 — depth and cameras

Run in the DA3 environment. Writes depth, intrinsics, camera poses, and a
reference point map under `$SCENE/da3/`.

```bash
python scripts/00_da3_depth_cameras.py \
    --scene-dir "$SCENE" --da3-root "$DA3_ROOT" --overwrite
```

Stage 08 reads these depth and camera products to lift point tracks into the
world frame. DA3 depth and camera translations share a consistent scene scale;
that scale is not guaranteed to be metric.

## Stage 01 — pick prompts and candidate frames

Use the interactive picker to label static and moving parts, select candidate
frames, and write `$SCENE/prompts.json`:

```bash
python scripts/01_pick_prompts.py --scene-dir "$SCENE"
```

Alternatively, write the JSON directly. Replace these labels, descriptions,
and candidate frame indices with values appropriate for your clip:

```json
{
  "prompts": [
    {
      "text": "cabinet body",
      "frame_index": 0,
      "label": "body",
      "motion": "static",
      "keyframe_candidates": [0, 10, 20]
    },
    {
      "text": "cabinet door",
      "frame_index": 0,
      "label": "door",
      "motion": "moving",
      "keyframe_candidates": [0, 10, 20]
    }
  ]
}
```

Labels must be unique and contain no spaces or `/`. To isolate adjacent parts,
use the picker's point/box prompts with `text` left blank. Candidate frames
must exist in the clip and show the relevant parts. Stage 02 copies the
candidates into `masks/tracking.json`, which stage 03 reads.

## Stage 02 — segment and track masks

Run in the SAM3 environment. Writes per-label masks and `masks/tracking.json`.

```bash
python scripts/02_sam3_segment.py \
    --scene-dir "$SCENE" --prompts-json "$SCENE/prompts.json" --overwrite
```

## Stage 03 — reconstruct meshes

Run in the SAM 3D Objects environment. For the SegviGen and static-registration
workflow in stages 04–07, reconstruct a combined object from the selected masks:

```bash
python scripts/03_sam3d_reconstruct.py \
    --scene-dir "$SCENE" --sam3d-repo "$SAM3D_REPO" \
    --combined --pointmap-source da3 --overwrite
```

Writes candidates under `sam3d/combined/`. All candidate frames are processed
by default; use `--candidate-idx 0` for only the first candidate. Omit
`--combined` for per-label reconstruction, optionally using `--labels`.

## Stage 04 — scale meshes against depth and masks

Uses stage 03 meshes, stage 02 masks, and stage 00 geometry. Writes scaled
meshes and poses under `sam3d_scaled/`.

```bash
python scripts/04_sacle_mesh.py --scene-dir "$SCENE"
```

## Stage 05 — segment the combined mesh with SegviGen

Run in the SegviGen environment. Transfers stage 02 part masks onto the
stage 03 combined reconstruction and writes mesh pieces under
`segvigen/combined/`.

```bash
python scripts/05_segvigen_segment.py \
    --scene-dir "$SCENE" --segvigen-root "$SEGVIGEN_ROOT" \
    --ckpt-path "$SEGVIGEN_CKPT"
```

Use `--prepare-only` to inspect the guidance images before running inference.

## Stage 06 — register static geometry

Registers stage 05 pieces across candidate frames using stage 04 scaled
meshes and static-part masks. Writes `registered_static/`.

```bash
python scripts/06_register_static.py --scene-dir "$SCENE"
```

Static/moving labels come from `masks/tracking.json`; override them with
`--static-labels` and `--moving-labels` when needed.

## Stage 07 — estimate joints from registered meshes

Reads stage 06 registered parts. Writes `simple_joint/joints.json` and a mesh
with joint arrows under `simple_joint/`.

```bash
python scripts/07_simple_joint.py --scene-dir "$SCENE"
```

## Stage 08 — track points with TrackCraft3R

Run in the TrackCraft3R environment. Reads stages 00 and 02 and writes
per-label world-space point tracks and scene flow under `trackcraft/`.
Choose a window that shows the object's motion and fits inside the clip:

```bash
REF_FRAME=0
python scripts/08_trackcraft_flow.py \
    --scene-dir "$SCENE" --trackcraft-repo "$TRACKCRAFT_REPO" \
    --checkpoint "$TRACKCRAFT_CKPT" \
    --start-idx "$REF_FRAME" --num-frames 12 --frame-stride 5 --overwrite
```

This example samples frames 0 through 55. Adjust the count or stride for a
shorter clip, or omit `--num-frames` to use all remaining strided frames.

## Stage 09 — estimate joints from video (optional)

The CLI delegates to `articulation_estimation.cli`. Run in an environment
with its DINOv2 and CUDA rendering dependencies installed:

```bash
python scripts/09_estimate_joint.py --scene-dir "$SCENE"
```

Its active implementation initializes joints from stage 08 tracks by default
and refines them using video, masks, and mesh rendering. The historical
scene-flow helper functions retained in the file do not define its active CLI.
Stage 10 provides the standalone point-track estimator.

## Stage 10 — estimate joints from point tracks

Reads stage 08 tracks and stage 02 moving-label metadata. Fits revolute or
prismatic motion and writes `pointtrack_joint/joints.json` plus per-label data.

```bash
python scripts/10_pointrack_to_joint.py --scene-dir "$SCENE"
```

Use `--joint-type revolute` or `--joint-type prismatic` to force a joint family;
the default is `auto`.

## Stage 11 — align per-label meshes (optional legacy path)

Reads stage 03 per-label meshes and stage 02 masks. This script still expects
the older geometry layout at `any4d/moge/depth`, `any4d/moge/intrinsics.npz`,
and `any4d/cameras.npz`; it does not directly consume stage 00's default
`da3/` layout. Run it only with that legacy bundle available:

```bash
python scripts/11_align_meshes.py --scene-dir "$SCENE" --overwrite
```

Writes world-space meshes and fit diagnostics under `aligned/`. The combined
mesh workflow above uses stages 04 and 06 for scaling and registration.

## Stage 12 — track hands with HaWoR

Run in the HaWoR environment. Writes MANO meshes and joints to
`hawor/per_frame/`, using stage 00 cameras for world-space output when available.

```bash
python scripts/12_hawor_hands.py \
    --scene-dir "$SCENE" --hawor-repo "$HAWOR_REPO" --overwrite
```

Use `--detected-only` to omit infilled detections, `--img-focal` to override the
focal length, or `--no-world` for camera-space output only.

## Stage 13 — scale hands into the scene

Reads stage 12 hands. By default, fits them against the stage 04 scaled
combined object mesh and writes corrected hands under `hawor_scaled/`.

```bash
python scripts/13_scale_hawor.py --scene-dir "$SCENE"
```

Use `--dry-run` to inspect the fit without saving. `--scale-source depth` or
`--scale-source pointmap` uses stage 00 geometry instead of the object mesh.

## Stage 14 — test hand-driven articulation in MuJoCo

Reads stage 06 registered meshes, stage 07 simple joints, and stage 13 scaled
hands. This script also imports the legacy `61_render_all.py` helper, which
is missing from this checkout. Once that helper is available beside the script:

```bash
python scripts/14_mujoco_retarget.py --scene-dir "$SCENE" --play
```

Use `mjpython` on macOS. `--export-video` renders offscreen, and `--dry-run`
prepares the scene without opening the viewer.
