# Quickstart

Single scene, fresh run. For full flag reference and troubleshooting, see
[`README.md`](README.md).

Set these once and the rest works as-is.

```bash
SCENE=data/kitchen_pour_01
SAM3D_REPO=/home/jeremy/research/Articulate4D/sam-3d-objects
DA3_ROOT=/home/jeremy/research/Articulate4D/depth-anything-3      # stage 00
TRACKCRAFT_REPO=/home/jeremy/research/Articulate4D/TrackCraft3r   # stage 40
TRACKCRAFT_CKPT=/path/to/trackcraft3r/model.safetensors          # stage 40
HAWOR_REPO=/home/jeremy/research/Articulate4D/HaWoR   # stage 60
ANY6D_REPO=/home/jeremy/research/Articulate4D/Any6D   # only for stage 32
```

Stage order: **00 → (05) → 10 → 20 → 30 → 32 → 40 → 50 → 52 → 60 → 51 (replay)**.
(05 is the optional prompt picker that feeds stage 10; 32 / 50 / 52 / 60 are
independent — 32 needs 30 + 00.) Stage **00** (Depth-Anything-3)
needs only `frames/`, so it can run first; stage **40** (TrackCraft3R) provides
point tracking — together they replace Any4D, and **40 must run after 00**.
(The walkthrough below runs 00 alongside 40 for readability, but you can run it
up front.)

## 0. Normalize frame filenames (one-time per scene)

```bash
python scripts/pad_frame_names.py -n  "$SCENE/frames"   # preview
python scripts/pad_frame_names.py     "$SCENE/frames"   # apply
```

## 1. Write your prompts

Write it into the scene folder — `$SCENE/prompts.json` is the canonical
location (where stage 05's picker saves and what stage 10 reads):

```bash
cat > "$SCENE/prompts.json" <<'EOF'
{
  "prompts": [
    {"text": "mug",   "frame_index": 0, "label": "mug"},
    {"text": "spoon", "frame_index": 0, "label": "spoon"}
  ]
}
EOF
```

Rules: labels are **unique**, **no spaces**, **no `/`**. Two kinds of entry:
a **concept prompt** has `text` (no geometry) and segments every instance of
the concept; an **interactive (PVS) prompt** omits `text` and carries
`box` (xyxy px) and/or `positive_points` / `negative_points` (xy px), and is
segmented purely from that geometry. `frame_index`→`0`; `label`→`text` for
concept prompts but is **required** for interactive ones. See the full schema
in `README.md` (stage 10).

To separate adjacent parts (e.g. a laptop base vs. its lid), use an
**interactive prompt** — a text concept grabs the whole object and negative
clicks can't carve a sub-part out of it. The **stage 05** gradio picker writes
these for you (leave `text` blank; box + positive/negative points; ≤8 each):

```bash
python scripts/05_pick_prompts.py --scene-dir "$SCENE"   # writes $SCENE/prompts.json
```

Then point stage 10 at that file (`--prompts-json "$SCENE/prompts.json"`).

## 2. Stage 10 — segment

```bash
conda activate sam3
python scripts/10_sam3_segment.py \
    --scene-dir "$SCENE" \
    --prompts-json "$SCENE/prompts.json" \
    --overwrite
```

## 3. Stage 20 — pick keyframes

```bash
python scripts/20_pick_keyframes.py --scene-dir "$SCENE"
```

Tune for clip length if needed:
- short (<30 frames): add `--min-spacing 2`
- long  (>200 frames): add `--n-candidates 5 --min-spacing 20`

To override a pick: `--manual mug=42 spoon=87`.

## 4. Stage 30 — reconstruct meshes

```bash
conda deactivate && conda activate sam3d-objects
export TORCH_CUDA_ARCH_LIST="8.0+PTX"
export CUDA_HOME=$CONDA_PREFIX

python scripts/30_sam3d_reconstruct.py \
    --scene-dir   "$SCENE" \
    --sam3d-repo  "$SAM3D_REPO" \
    --overwrite
```

Variants:
- Compare candidates: append `--all-candidates`
- Single label:       append `--labels mug`
- Splat only:         append `--skip-mesh`
- Background: erased to black by default; `--bg-mode none` keeps the original,
  `--bg-mode white --bg-dilate 3 --bg-feather 2` tunes the fill, `--save-input`
  dumps the masked RGBA the model sees.

## 5a. Stage 00 — Depth-Anything-3 depth + cameras/world frame

Pick a reference frame (usually one of the stage-20 keyframes for the object
you care about) — it seeds the dense reference pointmap:

```bash
jq '. | to_entries | map({label: .key, keyframe: .value.keyframe})' \
    "$SCENE/masks/tracking.json"
REF_FRAME=42   # ← replace
```

```bash
conda deactivate && conda activate da3

python scripts/00_da3_depth_cameras.py \
    --scene-dir "$SCENE" \
    --da3-root  "$DA3_ROOT" \
    --ref-frame "$REF_FRAME" \
    --overwrite
```

Writes the geometry bundle to `$SCENE/any4d/` (depth, `cameras.npz`, intrinsics,
`pointmap_ref.npy`). Variants: `--start-idx 20 --end-idx 80` for a subrange,
`--model-name da3-large` for a smaller model.

## 5b. Stage 40 — TrackCraft3R point tracking

Runs TrackCraft3R on a fixed window starting at `--start-idx` and converts the
tracks into per-label scene flow (world frame). The window's first frame is the
reference, so set `--start-idx` to your `REF_FRAME`:

```bash
conda deactivate && conda activate trackcraft

python scripts/40_trackcraft_flow.py \
    --scene-dir       "$SCENE" \
    --trackcraft-repo "$TRACKCRAFT_REPO" \
    --checkpoint      "$TRACKCRAFT_CKPT" \
    --start-idx "$REF_FRAME" --num-frames 12 --frame-stride 5 \
    --overwrite
```

Variants:
- Wider temporal span: `--frame-stride 10` (window = num-frames × frame-stride)
- Single label:        append `--labels mug`

Quick check that the bundle is reusable (RGB + depth + ref pointmap + flow):

```bash
python scripts/41_replay_in_rerun.py --scene-dir "$SCENE"
```

## 6. Stage 50 — estimate joint type + axis

```bash
python scripts/50_estimate_joint.py --scene-dir "$SCENE"
```

Procrustes per frame + an LM step on the frames temporally closest to
`ref_frame`. Variants:
- Strictly previous frames, wider window:
  `--refine-side prev --refine-window 12`
- Tighten the coarse-fit θ band:
  `--theta-min 5 --theta-max 60 --frame-rms-quantile 0.5`
- Skip refinement for non-rigid labels (e.g. a hand):
  `--labels hand --no-refine`

Sanity-check the recovered joints:

```bash
jq '. | to_entries | map({label: .key,
                          type: .value.type,
                          axis: .value.axis_direction,
                          point: .value.axis_point})' \
    "$SCENE/any4d/joints.json"
```

## 7. Stage 52 — align sam3d meshes to image + mask

```bash
python scripts/52_align_meshes.py --scene-dir "$SCENE" --overwrite
```

Default pipeline per label: apply `pose.json` (with PyTorch3D→RDF flip) →
silhouette IoU + DT-chamfer + ICP refinement → bake camera-to-world for the
keyframe. Saves an aligned GLB per label in the shared world frame.

Variants:
- Higher rendering fidelity (slower): `--render-factor 2 --max-iters 600`
- Enable mesh_alignment.py-style rescale for sam3d-body-like meshes: `--coarse`
- Single label:                       `--labels laptop_up`

For rigid objects, **stage 32 (Any6D)** is an object-level alternative that
registers the mesh directly to the stage-00 depth — use it when the silhouette
optimizer struggles.

Sanity-check:

```bash
jq '{kf: .keyframe,
     iou_pose: .step_A_pose_json.iou_full,
     iou_refined: .step_C_refine.iou_full_after}' \
    "$SCENE/aligned/<label>/align.json"
```

### Stage 32 — Any6D 6D object pose

Per-object 6D pose: Any6D registers each stage-30 mesh to the stage-00
**metric** depth (`any4d/moge/depth`) at the label's keyframe and returns a 6D
object→camera pose (also baked to the shared world frame when
`any4d/cameras.npz` exists). Needs stages **30 + 00** (independent of 40/50, so
run it any time after those); runs in the `any6d` GPU env. Restrict to rigid
objects — skip `hand`/non-object parts. (Note: DA3 depth is not guaranteed
metric — see the stage-00 scale caveat if the poses look off.)

**Headless-safe:** opens no GUI and renders nothing to a display (Any6D's
refiner uses an offscreen CUDA rasterizer). All results are plain data files
under `any6d/<label>/` — `scp`/`rsync` that folder to a workstation for
rendering. Keep `--debug 0` (the default).

```bash
conda deactivate && conda activate any6d
python scripts/32_any6d_pose.py \
    --scene-dir  "$SCENE" \
    --any6d-repo "$ANY6D_REPO" \
    --labels <object_label>
```

Sanity-check the pose on the server (no display needed):

```bash
jq '{kf: .keyframe, t_cam: [.pose_object_to_camera[0][3],
                            .pose_object_to_camera[1][3],
                            .pose_object_to_camera[2][3]]}' \
    "$SCENE/any6d/<label>/pose.json"
```

Then transfer the folder and eyeball the posed mesh on a workstation:

```bash
rsync -a "$SCENE/any6d/" workstation:/path/to/any6d/
# on the workstation:
python scripts/31_visualize_mesh.py --mesh /path/to/any6d/<label>/mesh_world.glb
```

### (Optional) Stage 33 — check Any6D poses against the point cloud

Overlay **all** posed Any6D objects on the scene point cloud in one Rerun view,
to confirm each one sits where the real object is. Runs in the stage-41/51 env
(`numpy` + `pillow` + `rerun-sdk`); reads only data files, writes nothing.

```bash
conda deactivate && conda activate <replay-env>   # any env with numpy + pillow + rerun-sdk
python scripts/33_visualize_any6d.py --scene-dir "$SCENE"
```

- Tightest per-object check (uses the MoGe metric depth at each object's own
  keyframe): `--point-source moge` (or `both` to also see the global cloud).
- Headless server (no display): `--save-rrd "$SCENE/any6d/preview.rrd"`, then
  `scp` it and open with `rerun preview.rrd` on a workstation.

## 8. Stage 60 — HaWoR hand tracking

Video-temporal two-hand tracker (detect/track → motion → DROID-SLAM → in-fill).
Writes per-frame MANO mesh + 3D keypoints to `hawor/per_frame/<frame>.npz`; if
`any4d/cameras.npz` exists it bakes the camera-to-world transform so
`verts_world` / `joints_world` are ready for the scene-authoring track.

```bash
conda deactivate && conda activate hawor
python scripts/60_hawor_hands.py \
    --scene-dir   "$SCENE" \
    --hawor-repo  "$HAWOR_REPO" \
    --overwrite
```

Variants:
- Detected only:    `--detected-only` (drop in-filled hands)
- Force focal:      `--img-focal F` (override the auto MoGe focal)
- Skip world bake:  `--no-world` (camera-frame output only)
- Save subrange:    `--start-idx 20 --end-idx 80` (inference still runs the full clip)

Sanity-check:

```bash
jq '{n: .total_hands,
     frames: .n_frames_with_hands,
     world: .world_frame_baked}' "$SCENE/hawor/config.json"
ls "$SCENE/hawor/per_frame/" | head
```

## 9. Stage 51 — full replay (flow + trajectories + joints + meshes)

```bash
python scripts/51_replay_with_joints.py --scene-dir "$SCENE"
```

Auto-picks `aligned/` if it exists, else `sam3d/`. Scrub the `stable_time`
timeline; you should see RGB + ref pointcloud + MoGe depth + per-label
scene-flow arrows + 3D point trajectories + joint axes + the aligned meshes
sitting where the actual objects are.

Variants:
- Pre-alignment view: `--mesh-source sam3d` (meshes will pile near origin)
- Skip overlays: `--no-meshes`, `--no-joints`, `--no-scene-flow`, `--no-trajectories`
- Single label: `--labels laptop_up`

## 10. Final sanity check

```bash
jq '. | to_entries | map({label: .key,
                          keyframe: .value.keyframe,
                          candidates: .value.keyframe_candidates,
                          visible: .value.n_frames_visible})' \
    "$SCENE/masks/tracking.json"

ls "$SCENE/sam3d/" "$SCENE/any4d/" "$SCENE/aligned/" "$SCENE/hawor/" 2>/dev/null
jq . "$SCENE/any4d/config.json"
jq '. | to_entries | map({label: .key, type: .value.type})' \
    "$SCENE/any4d/joints.json"
jq '{n: .total_hands, frames: .n_frames_with_hands}' \
    "$SCENE/hawor/config.json" 2>/dev/null
```

## 11. (Optional) Author a two-body articulation scene

A separate track that turns the sam3d meshes (stage 30) + HaWoR hands
(stage 60) into a hinge/slide scene with the hand trajectory replayed on top.
It does **not** need stages 50/52. Pick one editor, then optionally animate it
in MuJoCo. See `README.md` for the full slider reference.

```bash
# Interactive editor — MuJoCo viewer (writes mujoco/scene.xml + transforms.json)
python scripts/52c_mujoco_scene.py \
    --scene-dir "$SCENE" \
    --label-fixed laptop_base --label-moving laptop_up --joint hinge

# ...or the Rerun editor (adds scene flow + pointcloud; writes rerun/transforms.json,
# and supports joint keyframes for 52e)
python scripts/52d_rerun_scene.py \
    --scene-dir "$SCENE" \
    --label-fixed laptop_base --label-moving laptop_up --joint hinge

# Replay a saved 52d scene as a real animated MuJoCo articulation
python scripts/52e_mujoco_animate.py --scene-dir "$SCENE"
```
