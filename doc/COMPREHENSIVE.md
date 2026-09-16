# ArtAnything — Comprehensive Guide

Reference for the current scripts, stages `00`–`14`. For a shorter sequence of
commands, use the [quickstart](QUICKSTART.md). Install each model and its assets
using the [environment setup guide](ENVIRONMENT_SETUP_GUIDE.md). All commands
below run from the repository root.

## Workflow and data layout

Run the stages needed for your workflow in ascending filename order. The main
mesh workflow is `00 → 01 → 02 → 03 → 04 → 05 → 06 → 07`. Point tracking and
video-based fitting use stages 08–10. Hand processing uses 12–13; stage 14
combines those hands with registered meshes and simple joints for simulation.
Stage 11 is an optional alignment path with a different geometry input layout.

Stages 07, 09, and 10 are distinct joint estimators. They write separate result
directories and do not all feed one another. Stage 14 specifically consumes
stage 07 joints.

All paths in the table are relative to `data/<scene>/`. External model checkouts
and checkpoint caches are configured separately.

| Stage | Script | Reads | Writes |
| --- | --- | --- | --- |
| `00` | [`00_da3_depth_cameras.py`](../scripts/00_da3_depth_cameras.py) | `frames/` | `da3/` |
| `01` | [`01_pick_prompts.py`](../scripts/01_pick_prompts.py) | `frames/` | `prompts.json` |
| `02` | [`02_sam3_segment.py`](../scripts/02_sam3_segment.py) | `frames/`, `prompts.json` | `masks/`, `masks/tracking.json` |
| `03` | [`03_sam3d_reconstruct.py`](../scripts/03_sam3d_reconstruct.py) | Frames, masks, candidate metadata; optional `da3/` | `sam3d/<label>/cand_<id>_<frame>/` |
| `04` | [`04_sacle_mesh.py`](../scripts/04_sacle_mesh.py) | `sam3d/`, masks, `da3/` | `sam3d_scaled/` |
| `05` | [`05_segvigen_segment.py`](../scripts/05_segvigen_segment.py) | Combined SAM3D candidates and part masks | `segvigen/combined/<frame>/` |
| `06` | [`06_register_static.py`](../scripts/06_register_static.py) | SegviGen pieces, scaled combined meshes, mask metadata | `registered_static/` |
| `07` | [`07_simple_joint.py`](../scripts/07_simple_joint.py) | `registered_static/` | `simple_joint/` |
| `08` | [`08_trackcraft_flow.py`](../scripts/08_trackcraft_flow.py) | Frames, masks, `da3/` | `trackcraft/` |
| `09` | [`09_estimate_joint.py`](../scripts/09_estimate_joint.py) | Combined reconstruction, SegviGen pieces, frames, masks, cameras; tracks by default | `joints/` |
| `10` | [`10_pointrack_to_joint.py`](../scripts/10_pointrack_to_joint.py) | `trackcraft/`, moving-label metadata | `pointtrack_joint/` |
| `11` | [`11_align_meshes.py`](../scripts/11_align_meshes.py) | Per-label SAM3D meshes, masks, older `any4d/` geometry bundle | `aligned/` |
| `12` | [`12_hawor_hands.py`](../scripts/12_hawor_hands.py) | Frames, HaWoR assets; `da3/` for camera/world geometry | `hawor/` |
| `13` | [`13_scale_hawor.py`](../scripts/13_scale_hawor.py) | `hawor/`, scaled object meshes or DA3 geometry | `hawor_scaled/` |
| `14` | [`14_mujoco_retarget.py`](../scripts/14_mujoco_retarget.py) | Registered meshes, simple joints, scaled hands, external render helper | Interactive simulation or MP4 export |

Each stage has its own environment requirements. The tables below list selected
current options and parser defaults; `—` means no explicit value is supplied
and the script may derive one at runtime. Every stage accepts `--scene-dir`.
For the full CLI, use the entry point's `--help` in its environment, subject to
the stage 14 dependency noted below.

## Prepare a scene

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

### Details and data contracts

DA3 writes z-depth in the camera's right/down/forward convention. Camera
quaternions are stored in XYZW order as camera-to-world rotations; translations
use the same scene scale. `cameras.npz` includes `frame_indices`, intrinsics,
rotations, and translations. `depth/<frame>.npy` contains full-resolution depth.
DA3 does not produce point correspondences or scene flow; stage 08 does.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--scene-dir` | `required` | Scene folder data/&lt;scene_id&gt;/ (reads frames/*.jpg) |
| `--da3-root` | `—` | Path to the depth-anything-3 checkout (added to sys.path). Skip if depth_anything_3 is pip-installed. |
| `--model-name` | `depth-anything/DA3NESTED-GIANT-LARGE` | DA3 hub model id. Smaller: da3-large / da3-base. |
| `--process-res` | `504` | DA3 processing resolution (upper-bound resize). |
| `--ref-frame` | `0` | Frame index used as the reference for pointmap_ref / config.ref_frame (default: first processed frame). |
| `--start-idx` | `—` | Inclusive start frame index (default: 0). |
| `--end-idx` | `—` | Exclusive end frame index (default: end of clip). |
| `--out-name` | `da3` | Output subfolder under the scene dir (default: da3). |
| `--no-pointmap-ref` | `False` | Skip the dense reference point map to save disk space; retain it when using stage 13 point-map scaling. |
| `--overwrite` | `False` | Replace the existing output folder. |

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

### Details and data contracts

The picker records prompt type, static/moving status, and candidate frames.
Coordinates are absolute image pixels. A concept prompt contains `text`;
an interactive prompt omits text and supplies `box`, `positive_points`, and/or
`negative_points`. A box is `[x0, y0, x1, y1]`; point lists contain `[x, y]`
pairs. Keep at most eight positive and eight negative points.

The first candidate becomes the label's `keyframe`. Use several visible frames
spanning object motion for the multi-frame registration workflow. A separate
keyframe-selection script is not required.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--scene-dir` | `required` | Scene folder data/&lt;scene_id&gt;/ |
| `--frame-index` | `—` | Frame to open initially (default: first frame found) |
| `--prompts-json` | `—` | Output path (default: &lt;scene&gt;/prompts.json) |
| `--port` | `7860` | Gradio port |
| `--share` | `False` | Expose a public gradio URL |

## Stage 02 — segment and track masks

Run in the SAM3 environment. Writes per-label masks and `masks/tracking.json`.

```bash
python scripts/02_sam3_segment.py \
    --scene-dir "$SCENE" --prompts-json "$SCENE/prompts.json" --overwrite
```

### Details and data contracts

Stage 02 preserves `motion` and `keyframe_candidates` from the prompts.
Missing motion defaults to `static`; missing candidates leave no reconstruction
keyframe. Give moving parts explicit motion labels. For static labels, the
script subtracts the union of moving-part masks on each frame.

Review masks before reconstruction. A concept can produce multiple instances;
check the resulting labels in `tracking.json` instead of assuming they always
match the input label exactly.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--scene-dir` | `required` | Path to data/&lt;scene_id&gt;/ (must contain frames/) |
| `--prompt` | `—` | Single text prompt shortcut (applied at frame 0) |
| `--prompts-json` | `—` | JSON file with a 'prompts' list (overrides --prompt) |
| `--version` | `sam3.1` | SAM3 model version (default: sam3.1) Choices: `sam3`, `sam3.1`. |
| `--mask-threshold` | `0.5` | Threshold applied to soft masks before saving (default 0.5) |
| `--compile` | `False` | Pass compile=True to the SAM3 predictor (slower first run, faster afterwards) |
| `--overwrite` | `False` | Delete existing masks/ before running |

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

### Details and data contracts

Each candidate contains `mesh.glb`, `splat.ply`, `pose.json`, and
`keyframe.txt`; `--save-input` additionally writes `input_rgba.png`. Combined
reconstruction also records the contributing labels for mask transfer.

The default point-map source is SAM3D's own MoGe model. The example chooses DA3
explicitly to reuse stage 00 geometry. All candidates are processed by default.
With nonzero-based frame names, verify `--da3-index`: `position` uses the frame's
position in the sorted image list, while `stem` uses its numeric filename.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--sam3d-repo` | `required` | Path to the sam-3d-objects/ repo (needed for notebook/inference.py and checkpoints) |
| `--checkpoint-tag` | `hf` | Name of the checkpoint folder under &lt;sam3d-repo&gt;/checkpoints/ (default 'hf') |
| `--labels` | `—` | Only process these labels (default: all labels with a keyframe) |
| `--combined` | `False` | Reconstruct one whole object from the union of all selected label masks, instead of reconstructing each label separately. Outputs go under sam3d/combined/. |
| `--candidate-idx` | `—` | Reconstruct only this candidate index (0=top), instead of all candidates |
| `--skip-mesh` | `False` | Save only splat.ply + pose.json (skip mesh.glb export) |
| `--pointmap-source` | `moge` | Where the point map that conditions SAM 3D comes from. 'moge' (default) runs SAM 3D's own depth model on the keyframe; 'da3' back-projects the depth + intrinsics stage 00 wrote to &lt;scene&gt;/da3/ instead. Choices: `moge`, `da3`. |
| `--da3-index` | `position` | How a frame stem maps to a da3/depth/&lt;idx&gt;.npy. 'position' (default) uses the frame's position in sorted(frames/*.jpg), which is what stage 00 wrote; 'stem' uses the 6-digit stem directly. The two agree only when the clip starts at frames/000000.jpg. Choices: `position`, `stem`. |
| `--bg-mode` | `black` | Erase the background outside the mask before reconstruction by filling it with this color (default 'black', matching the model's internal rembg). 'dim' keeps the surrounding pixels but darkens them by --bg-dim (ROI stays full brightness). 'none' keeps the original background. Choices: `black`, `white`, `gray`, `dim`, `none`. |
| `--bg-dim` | `0.3` | Brightness multiplier for the surrounding when --bg-mode dim (0.0 = full black, 1.0 = unchanged; default 0.3). |
| `--bg-dilate` | `0` | Grow the mask by N pixels before erasing the background, to keep a thin margin around the object (default 0) |
| `--bg-feather` | `0` | Soft-blend the mask edge into the fill over N pixels (Gaussian; default 0 = hard edge) |
| `--save-input` | `False` | Also save the preprocessed RGBA fed to the model as &lt;out_dir&gt;/input_rgba.png (for inspection) |
| `--overwrite` | `False` | Replace existing sam3d/&lt;label&gt;/ outputs |

## Stage 04 — scale meshes against depth and masks

Uses stage 03 meshes, stage 02 masks, and stage 00 geometry. Writes scaled
meshes and poses under `sam3d_scaled/`.

```bash
python scripts/04_sacle_mesh.py --scene-dir "$SCENE"
```

### Details and data contracts

This resolves scale/depth ambiguity against segmented DA3 depth, then refines
image size against the silhouette. The output mirrors the candidate layout;
use the scaled pose with the scaled mesh. `--dry-run` computes corrections
without saving them.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-name` | `sam3d` | Stage-03 input directory name within the scene |
| `--out-name` | `sam3d_scaled` | Output directory name within the scene |
| `--labels` | `—` | Only process these Stage-03 labels (for example combined) |
| `--candidate-idx` | `—` | Only process this candidate index |
| `--min-valid-pixels` | `50` | Minimum valid masked DA3 pixels required |
| `--silhouette-min-scale` | `0.25` | Smallest post-depth uniform scale to search |
| `--silhouette-max-scale` | `4.0` | Largest post-depth uniform scale to search |
| `--copy-splat` | `False` | Also copy splat.ply when present |
| `--dry-run` | `False` | Compute and report corrections without writing files |
| `--overwrite` | `False` | Replace an existing mirrored output directory |

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

### Details and data contracts

Each frame output includes a guidance image, a segmented mesh, per-label
`pieces/<label>.glb`, and metadata. Use the full segmentation checkpoint with
2D guidance. The `smallest` overlap policy assigns overlapping pixels to the
smaller mask, helping retain thin parts. A Git LFS pointer is not a usable
checkpoint; the script checks for that condition.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--ckpt-path` | `—` | SegviGen full_seg_w_2d_map .ckpt (required unless --prepare-only) |
| `--segvigen-root` | `repo_root / 'SegviGen'` | SegviGen checkout (default: &lt;repo&gt;/SegviGen) |
| `--mesh` | `—` | Input whole-object GLB (default: sam3d/combined/mesh.glb) |
| `--candidate` | `—` | Only process this combined candidate index (default: all candidates) |
| `--keyframe` | `—` | Override the keyframe recorded beside the SAM3D mesh |
| `--labels` | `—` | Labels to encode (default: mesh's mask_labels.txt, then all tracking labels) |
| `--overlap-policy` | `smallest` | Owner of pixels present in multiple masks (default: smallest part) Choices: `smallest`, `first`, `last`, `error`. |
| `--out-dir` | `—` | Output folder (default: &lt;scene&gt;/segvigen/combined) |
| `--prepare-only` | `False` | Only create guidance.png + metadata.json; do not run SegviGen |
| `--overwrite` | `False` | Replace this stage's existing named output files |

## Stage 06 — register static geometry

Registers stage 05 pieces across candidate frames using stage 04 scaled
meshes and static-part masks. Writes `registered_static/`.

```bash
python scripts/06_register_static.py --scene-dir "$SCENE"
```

Static/moving labels come from `masks/tracking.json`; override them with
`--static-labels` and `--moving-labels` when needed.

### Details and data contracts

For each candidate, registration first relates SegviGen geometry to the scaled
SAM3D reconstruction and then uses static surfaces to place frames into one
reference coordinate system. Moving pieces inherit the same frame transform.
The earliest candidate is the default reference.

Outputs are `registered_meshes.glb` and `metadata.json`. The alternate
`--register-sam3d` mode registers full meshes, but stage 07 expects moving-part
records and rejects metadata produced by that mode.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--static-labels` | `—` | Override labels marked static in masks/tracking.json |
| `--moving-labels` | `—` | Override labels marked moving in masks/tracking.json |
| `--register-sam3d` | `False` | Register full meshes from sam3d_scaled/combined instead of the default SegviGen static/moving pieces |
| `--reference-frame` | `—` | Candidate frame ID to use as the registration reference (default: earliest available frame) |
| `--static-rotation-source` | `sam3d` | Rotation initializer for cross-frame static registration: 'sam3d' preserves the relative SAM3D rotation (default), 'identity' assumes pose-corrected meshes share canonical orientation, and 'auto' selects identity only for a clear near-180-degree SAM3D conflict Choices: `sam3d`, `identity`, `auto`. |
| `--samples` | `5000` | Static-surface samples used for registration (default: 5000) |
| `--iterations` | `100` | Fixed-rotation scale/translation registration iterations (default: 100) |
| `--out-dir` | `—` | Output directory (default: &lt;scene&gt;/registered_static) |
| `--overwrite` | `False` | Replace existing output files |

## Stage 07 — estimate joints from registered meshes

Reads stage 06 registered parts. Writes `simple_joint/joints.json` and a mesh
with joint arrows under `simple_joint/`.

```bash
python scripts/07_simple_joint.py --scene-dir "$SCENE"
```

### Details and data contracts

The estimator analyzes PCA motion of the registered pieces and compares
revolute and prismatic fits. Outputs are `joints.json` and `joint_mesh.glb` with
axis arrows. A prismatic axis has no unique axis point; a display anchor does
not imply a uniquely observed physical location.

These joints use stage 06's reference mesh coordinates. They are not directly
interchangeable with stage 10's DA3-world joint coordinates.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--moving-labels` | `—` | Moving-label subset (default: Stage-06 moving_labels) |
| `--pca-gap-threshold` | `0.12` | Normalized eigenvalue gap below which PCA axes are treated as degenerate and parallel-transported (default: 0.12) |
| `--min-rotation-deg` | `8.0` | Minimum observable PCA rotation span required to select a revolute joint (default: 8 degrees) |
| `--revolute-residual-ratio` | `0.8` | Revolute residual must be at most this fraction of the prismatic residual (default: 0.80) |
| `--out-dir` | `—` | Output directory (default: &lt;scene&gt;/simple_joint) |
| `--dry-run` | `False` | Estimate and print joints without writing GLB or JSON |
| `--overwrite` | `False` | Replace existing joint_mesh.glb and joints.json |

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

### Details and data contracts

For each label, the bundle contains `pts3d_ref.npy`, `pixel_ij.npy`,
`ref_mask.png`, and `scene_flow/<frame>.npy`. Tracks and flow are in DA3 world
coordinates. The scene-level `config.json` records the tracking window;
`pointmap_ref.npy` is recomputed at that window's reference frame.

Tracked indices follow `start_idx + k * frame_stride`. Every sampled frame
needs matching RGB, depth, cameras, and relevant masks. The reference frame
must cover the geometry used by downstream fitting.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--trackcraft-repo` | `required` | Path to the TrackCraft3r checkout (added to sys.path) |
| `--checkpoint` | `required` | TrackCraft3R model checkpoint (.safetensors) |
| `--labels` | `—` | Only process these labels (default: all in tracking.json) |
| `--start-idx` | `0` | Scene frame index of the window's first (reference) frame |
| `--num-frames` | `—` | Frames per model run (default: all remaining strided frames) |
| `--frame-stride` | `5` | Sample every Nth frame (default 5) |
| `--height` | `480` | Height |
| `--width` | `832` | Width |
| `--no-track-video` | `False` | Do not write trackcraft/point_tracks_video.mp4 |
| `--overwrite` | `False` | Replace any existing per-label tracking outputs |

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

### Details and data contracts

The active CLI is defined in
[`articulation_estimation/cli.py`](../articulation_estimation/cli.py).
It fits both joint families, refines them with DINOv2 features and mask losses,
selects a model on held-out video, and estimates joint states over time.
`--joint-initializer video` enables initialization without TrackCraft3R tracks.
The old numerical helpers in the entry-point file do not define this CLI.

**Input layout requirement:** the current loader reads pieces directly from
`segvigen/combined/pieces/`. Stage 05 writes candidate pieces under
`segvigen/combined/<frame>/pieces/`. Prepare the selected candidate's pieces at
the loader's expected path and pair them with the corresponding SAM3D candidate
before running this stage.

Outputs include `joints/joints.json`, per-label `joint.json`, and canonical
`moving_mesh.glb` / `static_mesh.glb` when a mesh payload is available.
`--pose-initializer any6d` consumes an externally prepared Any6D pose; this
checkout does not include an Any6D pose-generation entry point.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--static-label` | `—` | Fixed part label (auto-inferred for two-part scenes) |
| `--moving-labels` | `—` | Moving parts (default: every non-static mesh piece) |
| `--candidate` | `0` | SAM3D combined-mesh candidate index |
| `--joint-initializer` | `tracks` | Initialize joints from Stage-08 3D tracks (default) or sparse image-based poses Choices: `tracks`, `video`. |
| `--pose-initializer` | `sam3d` | Initial shared object pose: sam3d/combined/pose.json or any6d/combined/pose.json Choices: `sam3d`, `any6d`. |
| `--device` | `cuda` | Torch device; nvdiffrast currently requires CUDA |
| `--max-render-side` | `336` | Maximum image side used for rendering during fitting. |
| `--keyframes` | `16` | Number of keyframes sampled for fitting. |
| `--dino-model` | `dinov2_vitl14_reg` | DINOv2 feature model name. |
| `--dino-checkpoint` | `—` | Optional local DINOv2 checkpoint. |
| `--no-features` | `False` | Mask-only ablation; DINO is enabled by default |
| `--use-depth` | `False` | Add robust DA3 relative-depth consistency |
| `--no-dense-states` | `False` | Skip dense per-frame joint-state fitting. |
| `--dry-run` | `False` | Estimate everything without writing JSON |

## Stage 10 — estimate joints from point tracks

Reads stage 08 tracks and stage 02 moving-label metadata. Fits revolute or
prismatic motion and writes `pointtrack_joint/joints.json` plus per-label data.

```bash
python scripts/10_pointrack_to_joint.py --scene-dir "$SCENE"
```

Use `--joint-type revolute` or `--joint-type prismatic` to force a joint family;
the default is `auto`.

### Details and data contracts

This standalone estimator fits rigid transforms to tracked points and then
fits revolute/prismatic motion to the trajectories. Automatic revolute
classification requires enough observed rotation and a better point residual.
The results live separately from stage 09 under `pointtrack_joint/` and use
DA3 world coordinates. Revolute states are radians; prismatic states use scene
units.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--moving-labels` | `—` | Moving labels to fit (default: labels marked moving in masks/tracking.json) |
| `--joint-type` | `auto` | Automatically classify or force one joint family Choices: `auto`, `prismatic`, `revolute`. |
| `--min-rotation-deg` | `8.0` | Minimum observed pose rotation required to select revolute |
| `--out-dir` | `—` | Output directory (default: &lt;scene&gt;/pointtrack_joint) |
| `--dry-run` | `False` | Estimate and print without writing JSON |
| `--overwrite` | `False` | Replace existing output JSON files |

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

### Details and data contracts

This is a current optional script with an older input contract. It applies
SAM3D pose initialization, optional coarse scale/translation fitting,
silhouette/chamfer/ICP refinement, then camera-to-world baking. Each output
label contains `mesh.glb` and `align.json`.

Do not point this script at `da3/` expecting it to infer the older directory
layout. Stages 04 and 06 are the scaling and registration path for the current
combined-object workflow.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--labels` | `—` | Only align these labels (default: all) |
| `--candidate` | `0` | Candidate subfolder to align; 0 selects the first candidate. |
| `--coarse` | `False` | Re-fit scale (Y-height ratio) + translation against MoGe target points (mesh_alignment.py-style). DEFAULT off: sam3d's pose.json is already a strong prior, and the height-ratio heuristic over-shrinks meshes whose back is occluded in the keyframe. Useful for sam3d_body-style canonical meshes without a usable pose.json. |
| `--no-refine` | `False` | Skip the reprojection refinement step |
| `--render-factor` | `4` | Image/mask/K downscale factor used during refinement (default 4 -&gt; renders at 270x480 for a 1080p clip) |
| `--max-iters` | `300` | Nelder-Mead max iterations (default 300) |
| `--chamfer-weight` | `2.0` | Weight on silhouette-mask chamfer-on-distance-transform term (smoother than IoU; helps the optimizer move even when IoU is locally piecewise-constant) |
| `--overwrite` | `False` | Replace existing aligned/&lt;label&gt;/ outputs |

## Stage 12 — track hands with HaWoR

Run in the HaWoR environment. Writes MANO meshes and joints to
`hawor/per_frame/`, using stage 00 cameras for world-space output when available.

```bash
python scripts/12_hawor_hands.py \
    --scene-dir "$SCENE" --hawor-repo "$HAWOR_REPO" --overwrite
```

Use `--detected-only` to omit infilled detections, `--img-focal` to override the
focal length, or `--no-world` for camera-space output only.

### Details and data contracts

The HaWoR pipeline performs hand detection/tracking, motion estimation, SLAM,
and infilling. `per_frame/<frame>.npz` stores vertices, joints, handedness,
validity, camera translation, and boxes; world-space fields are added when
camera poses are available. `faces.npy` and `faces_left.npy` provide topology.
An overlay video is produced unless disabled.

Stage 00 intrinsics help keep image-space projections consistent, but HaWoR's
own estimated depth scale can still differ. Stage 13 corrects hand placement
against scene surfaces.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--hawor-repo` | `required` | Path to the HaWoR/ checkout (needs weights/ and _DATA/) |
| `--checkpoint` | `—` | HaWoR ckpt (default: &lt;repo&gt;/weights/hawor/checkpoints/hawor.ckpt) |
| `--infiller-weight` | `—` | Infiller weights (default: &lt;repo&gt;/weights/hawor/checkpoints/infiller.pt) |
| `--img-focal` | `—` | Pinhole focal in px. Default: median DA3 focal if da3/intrinsics.npz is present, else HaWoR's own estimate (~600 fallback). |
| `--ignore-da3-focal` | `False` | Do not auto-load the DA3 focal; let HaWoR estimate it. (--ignore-moge-focal is retained as a compatibility alias.) |
| `--hand` | `auto` | Hand to save and show in the overlay. Default 'auto' chooses the side seen in the most frames, breaking a tie by mean detector confidence. Choices: `auto`, `left`, `right`. |
| `--detected-only` | `False` | Save only hands actually detected at a frame; drop in-filled poses. Default keeps the full trajectory and marks in-filled hands with valid=False. |
| `--no-world` | `False` | Skip applying DA3 cam2world; save camera frame only. |
| `--no-overlay` | `False` | Do not write hawor/overlay.mp4. |
| `--recompute` | `False` | Also clear the _work cache, forcing tracking/SLAM to rerun. |
| `--overwrite` | `False` | Overwrite per_frame/faces/config (keeps the _work SLAM cache for fast re-derivation; add --recompute to clear it). |

## Stage 13 — scale hands into the scene

Reads stage 12 hands. By default, fits them against the stage 04 scaled
combined object mesh and writes corrected hands under `hawor_scaled/`.

```bash
python scripts/13_scale_hawor.py --scene-dir "$SCENE"
```

Use `--dry-run` to inspect the fit without saving. `--scale-source depth` or
`--scale-source pointmap` uses stage 00 geometry instead of the object mesh.

### Details and data contracts

Object mode uses the nearest scaled mesh candidate as the target surface.
Depth and point-map modes use stage 00 products instead. The default is
per-frame scaling with temporal regularization; unreliable corrections are
interpolated and the trajectory is smoothed. Global mode uses a robust scale
per hand side. Outputs mirror the hand products in `hawor_scaled/` and include
calibration/provenance information.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--input-name` | `hawor` | Stage-12 input directory name within the scene |
| `--out-name` | `hawor_scaled` | Output directory name within the scene |
| `--scale-mode` | `per-frame` | One robust scale per side, or temporally regularized measured-frame scales Choices: `global`, `per-frame`. |
| `--scale-source` | `object` | Target surface for the hand: the Stage-04 object mesh, the per-frame DA3 depth map, or the DA3 reference point map Choices: `object`, `depth`, `pointmap`. |
| `--object-root` | `sam3d_scaled` | Object-mesh directory name within the scene |
| `--object-label` | `combined` | Stage-03/04 object label used as the contact surface |
| `--object-candidate-idx` | `—` | Use only this object candidate instead of nearest keyframe |
| `--contact-offset` | `0.0` | Keep the hand this far in front of the object surface |
| `--min-contact-pixels` | `50` | Minimum hand pixels carrying a target-surface depth sample |
| `--temporal-regularization` | `True` | Interpolate unreliable per-frame corrections and smooth the scale trajectory; use --no-temporal-regularization for the original independent-frame behavior |
| `--temporal-window` | `9` | Odd centered window, in frames, for robust temporal scale smoothing; 1 performs interpolation without smoothing |
| `--dry-run` | `False` | Measure and report corrections without writing files |
| `--overwrite` | `False` | Replace an existing output directory |

## Stage 14 — test hand-driven articulation in MuJoCo

Reads stage 06 registered meshes, stage 07 simple joints, and stage 13 scaled
hands. This script also imports the legacy `61_render_all.py` helper, which
is missing from this checkout. Once that helper is available beside the script:

```bash
python scripts/14_mujoco_retarget.py --scene-dir "$SCENE" --play
```

Use `mjpython` on macOS. `--export-video` renders offscreen, and `--dry-run`
prepares the scene without opening the viewer.

### Details and data contracts

The default velocity drive projects hand motion onto the estimated joint while
contact is present. `--drive-mode contact-force` instead uses contact impulses.
There are no actuators and gravity is zero. Meshes and joint axes come from the
stage 06/07 reference frame.

The missing `61_render_all.py` helper is imported at module load time, so even
`--help` and `--dry-run` need it. The example is conditional on that dependency
being supplied. With it available, video export uses DA3 camera calibration;
OpenCV and Pillow are required in addition to MuJoCo, NumPy, and trimesh.

### Selected options

| Option | Default | Meaning |
| --- | --- | --- |
| `--hawor-name` | `hawor_scaled` | HaWoR directory inside the scene |
| `--label` | `—` | Moving label to simulate; repeat for multiple labels (default: all) |
| `--mesh-frame` | `first` | Canonical registered moving mesh attached to each joint Choices: `first`, `last`. |
| `--hands` | `all` | HaWoR hand surfaces that can contact the moving part Choices: `all`, `left`, `right`. |
| `--hand-coordinate-source` | `auto` | Prefer saved world hands or force DA3 camera conversion Choices: `auto`, `world`, `camera`. |
| `--drive-mode` | `velocity` | velocity: the part tracks the contacting hand's joint-relative speed and stops when contact ends; contact-force: the part is a free inertia pushed by contact impulses Choices: `velocity`, `contact-force`. |
| `--physics-hz` | `240.0` | MuJoCo integration frequency |
| `--play` | `False` | Start replay immediately instead of paused |
| `--export-video` | `—` | Render an MP4 offscreen through the DA3 camera instead of opening the viewer; without PATH, write mujoco_retarget/da3_camera.mp4 inside the scene |
| `--no-render-all-copy` | `False` | Skip the extra copy at &lt;scene&gt;/render_all/mujoco-retarget.mp4. |
| `--dry-run` | `False` | Validate and summarize inputs without importing MuJoCo |

## Inspecting and rerunning results

Inspect masks and candidate frames before spending time on reconstruction.
After stage 02, check the selected frames and motion labels:

```bash
jq 'to_entries | map({label: .key, motion: .value.motion,
                     keyframe: .value.keyframe,
                     candidates: .value.keyframe_candidates})' \
    "$SCENE/masks/tracking.json"
```

After fitting, inspect the result from the estimator you used: `simple_joint/`,
`joints/`, or `pointtrack_joint/`. Check the coordinate frame recorded in that
output before combining it with geometry from another branch.

Rerun downstream stages when prompts, masks, selected frames, or upstream
geometry change. Do not assume one global resume policy: use the specific
stage's `--overwrite`, candidate/label filters, and `--dry-run` where supported.
Stage 09's dry run performs estimation but skips writing results. Stage 05's
`--prepare-only` writes guidance without running model inference.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Stage 02 produces empty masks | Check the prompt frame and object visibility; inspect the saved labels and masks. |
| Negative points do not isolate an adjacent part | Use an interactive prompt without `text`, with points or a box on that part. |
| Stage 03 has no keyframes to reconstruct | Set `keyframe_candidates` in the prompt entries and rerun stage 02. |
| DA3 depth/cameras do not match RGB frames | Check zero-based frame naming and the stage 03 `--da3-index` setting. |
| Stage 05 rejects a checkpoint | Confirm the file contains actual model weights, not a Git LFS pointer, and is the full 2D-guided segmentation checkpoint. |
| Stage 06 cannot find scaled combined candidates | Reconstruct with stage 03 `--combined`, then run stage 04 on those candidates. |
| Stage 07 rejects registration metadata | Use stage 06's part-based registration; `--register-sam3d` produces full-mesh records. |
| Stage 08 runs out of memory | Reduce `--num-frames` or model input `--height` / `--width`; choose a window with useful motion. |
| Stage 09 cannot find mesh pieces | Check the `segvigen/combined/pieces/` layout requirement described above. |
| Stage 09 reports missing tracks | Run stage 08 with the mesh reference frame and at least two target frames, or choose `--joint-initializer video`. |
| Stage 10 selects prismatic motion unexpectedly | Check observed rotation and track quality before changing the rotation/residual thresholds or forcing a joint type. |
| Stage 11 cannot find geometry | It requires the older `any4d/` bundle; the current `da3/` output is not a direct substitute for that layout. |
| Hands project correctly but have the wrong scene depth | Check stage 12 focal settings and use stage 13 surface-based scaling. |
| Stage 14 fails before argument parsing | Its external `61_render_all.py` helper must be available beside the script. |

For model installation and compiled-extension errors, check the relevant
project's instructions in the [setup guide](ENVIRONMENT_SETUP_GUIDE.md).
