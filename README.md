# Articulate4D Pipeline — Command Guide

Per-stage isolated tools, separate conda envs, one shared scene folder on disk.
Stages run in order: **(05) → 10 → 20 → 30 → 00 → 40 → 50 → 52 → 60 → 51 (replay)**.
Each stage reads and writes only files under `data/<scene_id>/`. Stage 05 (the
prompt picker) is an optional pre-step that feeds stage 10. Stages 50/52/60
are independent of each other and can be reordered or skipped.

**Geometry + tracking (replaces Any4D).** Depth, per-frame cameras, and the
scene **world frame** now come from **stage 00** (`00_da3_depth_cameras.py`,
Depth-Anything-3); point tracking / scene flow comes from **stage 40**
(`40_trackcraft_flow.py`, TrackCraft3R). Together they produce exactly the
files the old Any4D stage 40 did, so everything downstream is unchanged. Run
**00 before 40** (40 reads 00's depth + cameras). The output folder keeps the
name **`any4d/`** for compatibility with the downstream stages — it is just the
shared geometry/tracking bundle, no longer produced by Any4D.

```
data/<scene>/
├── frames/                 # stage 00 puts JPEGs here (use pad_frame_names.py to normalize)
├── prompts.json            # stage 05 writes here (point/box prompts); stage 10 reads it
├── masks/                  # stage 10 writes here; stage 20 updates tracking.json in place
├── sam3d/                  # stage 30 writes here (splat.ply, mesh.glb, pose.json per label)
├── any4d/                  # shared geometry/tracking bundle (name kept for compat)
│   ├── config.json         # stage 00 writes; stage 40 updates (tracker, ref_frame)
│   ├── cameras.npz         # stage 00 (DA3): per-frame cam->world quats + trans
│   ├── pointmap_ref.npy    # stage 00 (DA3); stage 40 recomputes at its ref frame
│   ├── moge/{depth, mask, intrinsics.npz}   # stage 00 (DA3): z-depth + K, full res
│   ├── <label>/{ref_mask.png, pixel_ij.npy, pts3d_ref.npy, scene_flow/<frame>.npy}  # stage 40 (TrackCraft3R)
│   └── joints.json         # stage 50 writes here (plus joint.json per label)
├── aligned/                # stage 52 writes here (mesh.glb in the world frame + align.json)
├── any6d/                  # stage 32 writes here (Any6D 6D pose per label)
│   └── <label>/{pose.txt, pose.json, K.txt, final_mesh.glb, mesh_world.glb}
├── wilor/                  # stage 60 writes here (per-frame hand mesh + joints); 60b rescales in place
│   ├── config.json
│   ├── faces.npy           # shared MANO topology
│   └── per_frame/<frame>.npz  # verts, joints, is_right, cam_t, bbox; +verts_world if cam pose known
├── hawor/                  # stage 60c writes here (video-temporal alt to wilor/; same npz schema)
│   ├── config.json
│   ├── faces.npy           # shared MANO topology (faces_left.npy for left-hand winding)
│   ├── _work/              # cached HaWoR seq intermediates (tracks, SLAM, params)
│   └── per_frame/<frame>.npz  # verts, joints, is_right, valid, cam_t, bbox; +verts_world if cam pose known
├── mujoco/                 # stage 52c writes here (scene.xml + transforms.json — interactive MuJoCo editor)
├── rerun/                  # stage 52d writes here (transforms.json — interactive Rerun editor)
└── mujoco_anim/            # stage 52e writes here (scene.xml + STLs — animated MuJoCo replay of a 52d scene)
```

Stages **52c / 52d / 52e** are an optional *scene-authoring* track (build a
two-body articulation + hand-trajectory scene). They read stage-30 sam3d
meshes and stage-60 WiLoR hands directly, so they need 30 + 60 but not the
50/52 fit. 52d's `transforms.json` feeds 52e.

| stage | script | env | what it does |
|-------|--------|-----|--------------|
| pre | `pad_frame_names.py` | any | normalize frame filenames |
| 00 | `00_da3_depth_cameras.py` | DA3 env | Depth-Anything-3 depth + per-frame cameras/world frame → `any4d/` (Any4D substitute; run before 40) |
| 05 | `05_pick_prompts.py` | any (+ gradio) | interactive click/box prompt picker → `prompts.json` (optional input for stage 10) |
| 10 | `10_sam3_segment.py` | `sam3` | text-prompt video segmentation |
| 20 | `20_pick_keyframes.py` | any (numpy + cv2) | score visible frames, pick per-label keyframes |
| 30 | `30_sam3d_reconstruct.py` | `sam3d-objects` | per-label 3D reconstruction (splat + mesh + pose) |
| 31 | `31_visualize_mesh.py` | any (trimesh; + pyrender for `--format png`) | headless HTML / PNG render of a stage-30 (or stage-52) `mesh.glb` |
| 32 | `32_any6d_pose.py` | `any6d` (GPU; nvdiffrast + open3d + FoundationPose) | per-label Any6D 6D pose: register sam3d mesh to MoGe metric depth at the keyframe (needs 30 + 40) |
| 40 | `40_trackcraft_flow.py` | TrackCraft3R env | TrackCraft3R point tracking → per-label scene flow (reads stage 00; Any4D substitute) |
| 41 | `41_replay_in_rerun.py` | rerun + numpy | minimal replay of the stage-40 bundle |
| 50 | `50_estimate_joint.py` | any (numpy + scipy) | per-label joint type + axis estimation |
| 51 | `51_replay_with_joints.py` | rerun | stage-41 replay + joint axes + sam3d / aligned meshes |
| 52 | `52_align_meshes.py` | any (numpy + cv2 + trimesh + scipy) | align sam3d mesh to image/mask at the keyframe (automatic) |
| 52b | `52b_any6d_align.py` | any (trimesh) | **[placeholder]** bake the stage-32 Any6D pose into `aligned/` (alternative to 52's silhouette/ICP) |
| 52c | `52c_mujoco_scene.py` | any (mujoco + gradio + trimesh) | interactive MuJoCo editor: two-body joint scene + WiLoR hand replay → `mujoco/scene.xml` |
| 52d | `52d_rerun_scene.py` | any (rerun + gradio + trimesh) | Rerun version of 52c (+ scene flow / pointcloud) → `rerun/transforms.json` |
| 52e | `52e_mujoco_animate.py` | any (mujoco + trimesh) | replay a saved 52d scene as a real animated MuJoCo articulation |
| 60 | `60_wilor_hands.py` | `wilor` (torch 2.12+cu130) | per-frame WiLoR hand mesh + 3D keypoints |
| 60b | `60b_rescale_wilor_focal.py` | any (numpy) | post-fix legacy WiLoR npz to the real MoGe focal length (depth correction, in place) |
| 60c | `60c_hawor_hands.py` | `hawor` (torch 2.0.1+cu118) | video-temporal HaWoR hands (track + SLAM + infill) → `hawor/`, same npz schema as 60 |

---

## Prerequisite — normalize frame filenames

If your frames have inconsistent zero-padding (`70.jpg`, `00070.jpg`,
`000070.jpg`), pad them to a fixed width first. Both stage 10 and stage 30
rely on `frames/<frame>.jpg` and `masks/<label>/<frame>.png` lining up.

```bash
# Always preview first
python scripts/pad_frame_names.py -n data/<scene>/frames

# Then actually rename to 6-digit
python scripts/pad_frame_names.py data/<scene>/frames

# For mask folders later (after stage 10), if needed
python scripts/pad_frame_names.py -e png data/<scene>/masks/<label>
```

Flags: `-n` dry-run, `-v` per-file echo, `-e EXT` extension (default `jpg`),
`-w N` width (default `6`).

The Python version (`pad_frame_names.py`) is the one to use. The bash
equivalent crashed SSH on big dirs because of per-file forks; Python's
`os.scandir` + `os.rename` has zero subprocesses.

---

## Stage 00 — Depth-Anything-3 depth + camera/world frame

**Env:** DA3 env (Depth-Anything-3 + torch)
**Reads:** `frames/*.jpg`
**Writes:** the scene-level geometry bundle under `any4d/` (`moge/depth`,
`moge/mask`, `moge/intrinsics.npz`, `cameras.npz`, `frame_indices.npy`,
`pointmap_ref.npy`, `config.json`)

Replaces the *depth + camera* half of the old Any4D stage 40. Runs DA3 once
over the clip and emits Any4D's exact on-disk schema so stages 40, 52, 60, 51,
41, 33, 32 consume it unchanged. Depth is z-depth at full resolution (same
convention as MoGe); DA3's world-to-camera extrinsics are inverted to the
**camera-to-world** quaternion (XYZW) + translation that the consumers expect.
DA3 has no temporal correspondence, so it does **not** produce scene flow —
that comes from **stage 40** (TrackCraft3R). Run stage 00 first.

Setup (one-time): `git clone https://github.com/ByteDance-Seed/depth-anything-3
&& cd depth-anything-3 && pip install -e .`

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | Reads `frames/*.jpg` |
| `--da3-root PATH` | — | Path to the depth-anything-3 checkout (skip if pip-installed) |
| `--model-name STR` | `depth-anything/DA3NESTED-GIANT-LARGE` | DA3 hub id (smaller: `da3-large` / `da3-base`) |
| `--process-res INT` | `504` | DA3 processing resolution |
| `--ref-frame INT` | `0` | Frame used for `pointmap_ref` / `config.ref_frame` |
| `--start-idx / --end-idx` | full clip | Frame range to process |
| `--out-name STR` | `any4d` | Output subfolder (drop-in for stage 40) |
| `--no-pointmap-ref` | off | Skip the dense `pointmap_ref.npy` |
| `--overwrite` | off | Replace the output folder |

### Invocations

```bash
conda activate da3
python scripts/00_da3_depth_cameras.py \
    --scene-dir data/kitchen_pour_01 \
    --da3-root  /path/to/depth-anything-3 \
    --ref-frame 42 --overwrite
```

**Scale caveat:** DA3 depth/pose are not guaranteed to match MoGe's metric
scale. Stages that only need a self-consistent scene scale (52, 51) are fine;
stage 32 (Any6D) expects *metric* depth — sanity-check its translations if you
feed it a DA3 bundle.

### Quick sanity check

```bash
jq '{source, ref_frame, full_resolution_wh}' data/<scene>/any4d/config.json
ls data/<scene>/any4d/                     # cameras.npz, pointmap_ref.npy, moge/
python -c "import numpy as np; d=np.load('data/<scene>/any4d/cameras.npz'); print(d['cam_quats_xyzw'].shape, d['cam_trans'].shape)"
```

---

## Stage 05 — interactive prompt picker (optional, feeds stage 10)

**Env:** any with `gradio` + `numpy` + `opencv-python`
**Reads:** `data/<scene>/frames/*.jpg` (the only required input)
**Writes (append):** `data/<scene>/prompts.json` (or `--prompts-json PATH`)

A gradio click/box picker for cases where stage 10's text-only prompts can't
separate adjacent parts of an articulated object (e.g. `laptop_base` vs
`laptop_up`). Run it on the machine that holds `data/`, open the printed URL
locally (VS Code Remote-SSH auto-forwards the port), then per prompt:

1. Drag the **frame slider** (or Prev/Next) to the frame this prompt should
   target. Each saved entry records whatever frame was active — different
   prompts can use different frames.
2. Click the image to drop **green (positive)** / **red (negative)** points,
   or switch the radio to **box** and click two opposite corners for a blue
   bbox.
3. Fill in `label` (filesystem-safe — no spaces, no `/`) and either:
   - **leave `text` blank** for an *interactive (PVS) part prompt* — box +
     positive points on the part, negative points on everything to exclude
     (the recommended way to split a sub-part), or
   - **fill `text`** for a *concept prompt* (whole object, no geometry).
   Then **Append to prompts.json**. Keep interactive prompts to **≤8 positive
   and ≤8 negative** clicks (the tracker uses at most 16 points).

The live JSON preview shows exactly what will be written. All coordinates are
written as absolute image pixels; stage 10 normalizes them to [0, 1] before
sending them to SAM3 (the model rescales by its internal `image_size`).

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--frame-index INT` | first frame found | Frame to open on initially (snaps to nearest available) |
| `--prompts-json PATH` | `<scene>/prompts.json` | Output file (appended/merged) |
| `--port INT` | `7860` | Gradio port |
| `--share` | off | Public gradio URL |

A "replace existing entry with the same label" checkbox (on by default) lets
you re-pick a label without manually editing the file. The script refuses to
overwrite a `prompts.json` it can't parse.

### Invocations

```bash
# Scan all frames, open on frame 0, save to <scene>/prompts.json
python scripts/05_pick_prompts.py --scene-dir data/macbook-all

# Open already positioned on frame 42, custom output, custom port
python scripts/05_pick_prompts.py \
    --scene-dir data/macbook-all \
    --frame-index 42 \
    --prompts-json /tmp/my_prompts.json \
    --port 7871
```

Then hand the file to stage 10: `--prompts-json data/<scene>/prompts.json`.

---

## Stage 10 — SAM3 video segmentation

**Env:** `sam3`
**Reads:** `data/<scene>/frames/*.jpg`, plus a text prompt (CLI or JSON)
**Writes:** `data/<scene>/masks/<label>/*.png` + `data/<scene>/masks/tracking.json`

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | `data/<scene_id>/` |
| `--prompt TEXT` | — | Single text-prompt shortcut at frame 0 |
| `--prompts-json PATH` | — | JSON file with a `prompts` list (overrides `--prompt`) |
| `--version {sam3, sam3.1}` | `sam3.1` | Which model checkpoint to download |
| `--mask-threshold FLOAT` | `0.5` | Threshold applied to soft masks before saving |
| `--overwrite` | off | Delete existing `masks/` before running |
| `--compile` | off | `compile=True` to the predictor (slower first run, faster after) |

### Constraints (script enforces these)

- Labels must be **unique across prompts**
- Labels must contain **no spaces** and **no `/`** (they are used as folder names)

### Invocations

**Single text prompt** (most common):
```bash
conda activate sam3
python scripts/10_sam3_segment.py \
    --scene-dir data/kitchen_pour_01 \
    --prompt "mug"
```

**Multiple prompts via JSON**:

Write the file inside the scene folder (`data/<scene>/prompts.json` is the
canonical location — it's where stage 05's picker saves, and what stage 10
reads by default convention):

```bash
cat > data/kitchen_pour_01/prompts.json <<'EOF'
{
  "prompts": [
    {"text": "person",   "frame_index": 0, "label": "person"},
    {"text": "blue mug", "frame_index": 0, "label": "mug"},
    {"text": "spoon",    "frame_index": 0, "label": "spoon"}
  ]
}
EOF

python scripts/10_sam3_segment.py \
    --scene-dir data/kitchen_pour_01 \
    --prompts-json data/kitchen_pour_01/prompts.json \
    --version sam3.1 \
    --overwrite
```

### Full prompts.json schema

Each entry is one of two kinds:

- **Concept (text) prompt** — has `text`, no geometry. SAM3 detects and tracks
  *every instance* of the text concept.
- **Interactive (PVS) prompt** — has any of `box` / `positive_points` /
  `negative_points`, and `text` is **omitted**. Stage 10 builds the object
  *purely from the geometry* (SAM2-style tracking), so negative points actually
  carve the part out. This is the reliable way to separate adjacent parts of an
  articulated object — a text concept re-asserts the whole object on every
  frame during propagation, so negative clicks layered on top of it can't
  remove a sub-part.

`frame_index` defaults to `0`. `label` defaults to `text` for concept prompts
and is **required** for interactive prompts (it is the output folder name).
Coordinates are absolute image pixels. A box counts as 2 points, and the
tracker keeps at most 16 points (first 8 + last 8) — keep clicks to ≤8 positive
and ≤8 negative.

```jsonc
{
  "prompts": [
    // Concept, minimal: text only (frame_index -> 0, label -> "mug")
    {"text": "mug"},

    // Concept on a chosen frame, with an explicit label
    {"text": "person", "frame_index": 29, "label": "person"},

    // Interactive part: box (xyxy) + clicks, NO text. Positive points stay
    // inside the part; negatives (e.g. on the body) are carved out.
    {"frame_index": 76, "label": "dryer_body",
     "box": [120, 90, 980, 1040],
     "positive_points": [[400, 600], [620, 720]],
     "negative_points": [[820, 300]]},

    // Interactive part: clicks only, no box
    {"frame_index": 29, "label": "dryer_door",
     "positive_points": [[860, 360]],
     "negative_points": [[400, 600], [300, 800]]}
  ]
}
```

**With `torch.compile` speedup** (worth it for multi-prompt runs):
```bash
python scripts/10_sam3_segment.py \
    --scene-dir data/kitchen_pour_01 \
    --prompts-json data/kitchen_pour_01/prompts.json \
    --compile --overwrite
```

### Notes on naming

- Single instance per prompt → folder is exactly `<label>` (`masks/mug/`)
- Multiple instances → suffixed `<label>_1`, `<label>_2`, …
  (`masks/person_1/`, `masks/person_2/`)

### Quick sanity check

```bash
jq 'keys' data/<scene>/masks/tracking.json
ls data/<scene>/masks/<label> | head
```

---

## Stage 20 — pick keyframes

**Env:** any (pure numpy + cv2 + stdlib — usually run in the same env you already have)
**Reads:** `frames/`, `masks/<label>/*.png`, `masks/tracking.json`
**Writes:** updates `masks/tracking.json` in place, adding `keyframe` and `keyframe_candidates` per label

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--area-min FLOAT` | `0.005` | Reject masks smaller than 0.5% of the frame |
| `--area-max FLOAT` | `0.60` | Reject masks larger than 60% of the frame |
| `--edge-margin INT` | `2` | Pixels from image edge counted as "on boundary" |
| `--n-candidates INT` | `3` | Number of temporally-spaced candidates to emit |
| `--min-spacing INT` | `5` | Min frame distance between any two candidates |
| `--manual LIST` | — | Manual override: `label=frame_idx label=frame_idx ...` |
| `--dry-run` | off | Print picks but don't write tracking.json |

### Invocations

**Default**:
```bash
python scripts/20_pick_keyframes.py --scene-dir data/kitchen_pour_01
```

**Short clip (<30 frames) — relax spacing or you'll only get 1 candidate**:
```bash
python scripts/20_pick_keyframes.py \
    --scene-dir data/kitchen_pour_01 \
    --min-spacing 2
```

**Long clip (>200 frames) — spread candidates out**:
```bash
python scripts/20_pick_keyframes.py \
    --scene-dir data/kitchen_pour_01 \
    --n-candidates 5 --min-spacing 20
```

**Manual override** (keyframe must be a visible frame for that label):
```bash
python scripts/20_pick_keyframes.py \
    --scene-dir data/kitchen_pour_01 \
    --manual mug=42 spoon=87
```

**Preview without writing**:
```bash
python scripts/20_pick_keyframes.py \
    --scene-dir data/kitchen_pour_01 \
    --dry-run
```

### Quick sanity check

```bash
jq '. | to_entries | map({label: .key, keyframe: .value.keyframe, candidates: .value.keyframe_candidates})' \
    data/<scene>/masks/tracking.json
```

Open `frames/<keyframe>.jpg` next to `masks/<label>/<keyframe>.png` — if the
mask looks clean and the object is roughly centered, the picker did its job.

---

## Stage 30 — SAM 3D Objects reconstruction

**Env:** `sam3d-objects`
**Reads:** `frames/<kf>.jpg`, `masks/<label>/<kf>.png`, `masks/tracking.json` (+ SAM 3D checkpoints)
**Writes:** `data/<scene>/sam3d/<label>/{splat.ply, mesh.glb, pose.json, keyframe.txt}` (+ `input_rgba.png` with `--save-input`)

### Background removal

By default stage 30 **erases the background** outside the mask (fills it with
black) before reconstruction, so only the masked object is shown to the model.
SAM 3D embeds the mask in the alpha channel but still feeds a full-resolution
RGB branch with the background intact — erasing it removes that distractor,
which matters most for **parts of an articulated object** (otherwise the rest
of the object stays visible in that branch). Black matches SAM 3D's internal
`rembg` (`image * mask`). Use `--bg-mode none` to restore the old behavior.

`--bg-mode dim` is a softer middle ground: instead of replacing the surrounding
pixels it **keeps them but darkens them** by `--bg-dim` (default `0.3`), while
the ROI inside the mask stays at full brightness. This preserves a little
context (lighting, contact cues) while still biasing the reconstruction toward
the masked region, so the geometry is less likely to extend into adjacent parts
than with a hard black fill.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--sam3d-repo PATH` | — *(required)* | Path to the `sam-3d-objects/` checkout |
| `--checkpoint-tag NAME` | `hf` | Folder under `<sam3d-repo>/checkpoints/` |
| `--labels LIST` | all | Process only these labels |
| `--all-candidates` | off | Run every entry in `keyframe_candidates`; outputs go in `cand_NN_<kf>/` subfolders |
| `--candidate-idx INT` | — | Pick a specific candidate (0 = top). Mutex with `--all-candidates`. |
| `--seed INT` | `42` | Diffusion seed |
| `--compile` | off | `compile=True` to the SAM 3D pipeline |
| `--overwrite` | off | Replace existing outputs |
| `--skip-mesh` | off | Only save `splat.ply` + `pose.json` (no `mesh.glb`) |
| `--bg-mode {black,white,gray,dim,none}` | `black` | Background treatment outside the mask (`dim` darkens it instead of replacing it; `none` keeps the original image) |
| `--bg-dim FLOAT` | `0.3` | Brightness multiplier for the surrounding when `--bg-mode dim` (`0`=black, `1`=unchanged) |
| `--bg-dilate INT` | `0` | Grow the mask by N px before erasing (keeps a margin around the object) |
| `--bg-feather INT` | `0` | Soft-blend the mask edge into the fill over N px (Gaussian) |
| `--save-input` | off | Also save the preprocessed RGBA fed to the model as `input_rgba.png` |

### Set this once per shell (helps avoid CUDA-arch errors)

```bash
export TORCH_CUDA_ARCH_LIST="8.0+PTX"          # A100
export CUDA_HOME=$CONDA_PREFIX
```

### Invocations

**Default** (top candidate per label, splat + mesh):
```bash
conda activate sam3d-objects
python scripts/30_sam3d_reconstruct.py \
    --scene-dir data/kitchen_pour_01 \
    --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects
```

**Single label, faster iteration**:
```bash
python scripts/30_sam3d_reconstruct.py \
    --scene-dir data/kitchen_pour_01 \
    --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \
    --labels mug
```

**Run all candidates** (compare results across multiple keyframes):
```bash
python scripts/30_sam3d_reconstruct.py \
    --scene-dir data/kitchen_pour_01 \
    --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \
    --all-candidates --overwrite
```

Layout becomes:
```
sam3d/mug/cand_00_000042/{splat.ply, mesh.glb, pose.json, keyframe.txt}
sam3d/mug/cand_01_000154/...
sam3d/mug/cand_02_000021/...
```

**Force a specific candidate** (top is `0`, second-best is `1`, etc.):
```bash
python scripts/30_sam3d_reconstruct.py \
    --scene-dir data/kitchen_pour_01 \
    --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \
    --candidate-idx 1 --overwrite
```

**Splat-only (skip mesh export)**:
```bash
python scripts/30_sam3d_reconstruct.py \
    --scene-dir data/kitchen_pour_01 \
    --sam3d-repo /home/jeremy/research/Articulate4D/sam-3d-objects \
    --skip-mesh
```

**Background control** (default is `black`). Keep the original image, or tune
the fill / margin and dump the masked input for inspection:
```bash
# Restore pre-feature behavior (full background):
python scripts/30_sam3d_reconstruct.py ... --bg-mode none

# White fill, 3px margin + soft edge, and save the RGBA the model sees:
python scripts/30_sam3d_reconstruct.py ... \
    --bg-mode white --bg-dilate 3 --bg-feather 2 --save-input

# Dim the surrounding to 25% brightness (keeps context, ROI stays normal):
python scripts/30_sam3d_reconstruct.py ... --bg-mode dim --bg-dim 0.25
```

### Quick sanity check

```bash
# Folder structure
ls data/<scene>/sam3d/

# Pose JSON
jq . data/<scene>/sam3d/<label>/pose.json

# Keyframe used
cat data/<scene>/sam3d/<label>/keyframe.txt

# (with --save-input) eyeball the masked image the model actually saw
# data/<scene>/sam3d/<label>/input_rgba.png
```

---

## Stage 31 — visualize the sam3d mesh.glb (headless, no pyglet)

**Env:** any with `trimesh` (+ `pyrender` only for `--format png`) — both
present in `sam3d-objects`
**Reads:** `sam3d/<label>/[cand_NN_<kf>/]mesh.glb` (default) or
`aligned/<label>/mesh.glb` (`--mesh-source aligned`)
**Writes:** one visualization file per mesh under `--out`
(default `<source>/_previews/`)

A small inspection helper — load the **triangle mesh** stage 30 produced (the
`mesh.glb`, *not* the `splat.ply` Gaussian splat), print geometry stats
(vertex / face counts, bounds, watertight, vertex colors), and write a
visualization. Both backends are **fully headless** — no X display and **no
pyglet** (trimesh's built-in windowed viewer hard-requires `pyglet<2`, which
this script deliberately avoids):

- `--format html` *(default)* — a self-contained three.js scene; open the
  `.html` in a browser (VS Code Remote-SSH forwards it, or scp it locally) and
  rotate/zoom interactively.
- `--format png` — offscreen raster via `pyrender` (EGL by default, `--gl
  osmesa` fallback). A real shaded image with an XYZ axis marker.

Not part of the core pipeline order; run it any time after stage 30 (or 52).

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — | Scene folder (omit only with `--mesh`) |
| `--mesh PATH` | — | Visualize a single `.glb`/`.ply`/`.obj` directly, skipping label resolution |
| `--labels LIST` | all | Subset of labels to visualize |
| `--mesh-source {sam3d,aligned,auto}` | `sam3d` | Which mesh to load (`auto` = aligned if present) |
| `--candidate INT` | `0` | sam3d candidate index when stage 30 ran `--all-candidates` |
| `--all-candidates` | off | Visualize every `cand_*/mesh.glb` per label (sam3d source) |
| `--combine` | off | One file for all meshes (meaningful for `aligned`; sam3d meshes pile near origin) |
| `--format {html,png}` | `html` | Interactive HTML or offscreen PNG |
| `--out DIR` | `<source>/_previews` | Output directory |
| `--resolution W H` | `1280 960` | PNG size for `--format png` |
| `--gl {egl,osmesa}` | `egl` | OpenGL backend for offscreen PNG |
| `--no-axis` | off | Drop the XYZ origin axis marker |
| `--background R G B A` | light gray | Background RGBA in [0,1] |

### Invocations

```bash
# Interactive HTML for every label (default) -> data/<scene>/sam3d/_previews/
python scripts/31_visualize_mesh.py --scene-dir data/<scene>

# One label, second candidate
python scripts/31_visualize_mesh.py --scene-dir data/<scene> --labels mug --candidate 1

# Offscreen PNGs instead of HTML
python scripts/31_visualize_mesh.py --scene-dir data/<scene> --format png

# The aligned meshes, together in the Any4D world frame, one combined file
python scripts/31_visualize_mesh.py --scene-dir data/<scene> \
    --mesh-source aligned --combine

# Any .glb directly
python scripts/31_visualize_mesh.py --mesh path/to/mesh.glb
```

No display or pyglet needed. If `--format png` fails to acquire an EGL context,
retry with `--gl osmesa`, or just use the default `--format html`.

---

## Stage 32 — Any6D 6D object pose

**Env:** `any6d` (GPU required — nvdiffrast + open3d + FoundationPose)
**Reads:** `sam3d/<label>/[cand_NN_<kf>/]{mesh.glb, keyframe.txt}`,
`frames/<kf>.jpg`, `masks/<label>/<kf>.png`,
`any4d/moge/{depth/<kf>.npy, intrinsics.npz}` (+ `moge/mask/<kf>.png` and
`any4d/cameras.npz` if present)
**Writes:** `data/<scene>/any6d/<label>/{pose.txt, pose.json, K.txt,
final_mesh.glb, mesh_world.glb}`

Adapts `Any6D/run_pose.py` (the SAM2/InstantMesh-free driver) to the scene
layout. For each label, Any6D registers the **stage-30 mesh** to the masked
metric pointcloud at that label's keyframe and returns the object→camera 6D
pose, while also rescaling the mesh to metric size via its oriented-bounding-box
ratio fit. **Not part of the core order** — it's an object-level alternative to
the stage-52 silhouette alignment, and needs stages 30 + 40 (it reuses the
MoGe metric depth + intrinsics; stage 52 it does not need).

The MoGe depth is **already metric** (meters), so unlike the Any6D demo there is
no `depth_scale` divisor — the `.npy` is fed straight in. Pixels are restricted
to valid metric depth (mask ∩ `depth>0` ∩ MoGe valid mask) so holes don't inject
`(0,0,0)` points into the OBB fit. If `any4d/cameras.npz` is present the pose is
also baked into the **Any4D world frame** (same cam→world convention as stages
52 / 60), and `mesh_world.glb` is the mesh placed there — ready to overlay
alongside `aligned/` meshes in a stage-51-style viewer. Without `cameras.npz`
it instead writes `mesh_cam.glb` (camera frame).

**Headless-safe.** The script opens no GUI and renders nothing to a display:
Any6D's refiner uses an offscreen CUDA rasterizer (nvdiffrast
`RasterizeCudaContext`, not a GL/display context) and its Open3D
`draw_geometries` calls are disabled. Everything is written as plain data files
under `any6d/<label>/` (poses + `.glb` meshes) so you can `scp`/`rsync` the
folder to a workstation for rendering. Keep `--debug 0` (the default) on a
render-less server.

### Outputs per label

| file | meaning |
|---|---|
| `pose.txt` | 4×4 object→camera transform (Any6D's raw output) |
| `pose.json` | structured: `pose_object_to_camera`, `pose_object_to_world` (if cam pose known), `cam_to_world`, `keyframe`, `K`, provenance |
| `K.txt` | the 3×3 intrinsics used |
| `final_mesh.glb` | Any6D's metric-rescaled mesh, in its own object frame |
| `mesh_world.glb` | that mesh placed by the estimated pose into the Any4D world frame (or `mesh_cam.glb` if no `cameras.npz`) |

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--any6d-repo PATH` | — *(required)* | Path to the `Any6D/` checkout (provides `estimater.py` + weights) |
| `--labels LIST` | all | Subset of sam3d labels to register |
| `--candidate INT` | `0` | sam3d candidate index when stage 30 ran `--all-candidates` |
| `--iteration INT` | `5` | Any6D refiner iterations |
| `--no-refinement` | off | Skip the render-and-compare refinement (coarse OBB fit only) |
| `--no-axis-align` | off | Disable the coarse OBB axis alignment |
| `--no-coarse` | off | Disable the coarse OBB scale/pose initialization |
| `--debug INT` | `0` | Any6D debug level. `0` = headless: write only the pose + mesh data files. `>0` adds intermediate disk artifacts (still no GUI); leave at `0` on a render-less server. |
| `--overwrite` | off | Replace existing `any6d/<label>/` |

### Invocations

```bash
conda activate any6d

# All object labels with a mesh
python scripts/32_any6d_pose.py \
    --scene-dir  data/dryer \
    --any6d-repo /home/jeremy/research/Articulate4D/Any6D

# One label, a specific sam3d candidate, more refiner iterations
python scripts/32_any6d_pose.py \
    --scene-dir  data/dryer \
    --any6d-repo /home/jeremy/research/Articulate4D/Any6D \
    --labels dryer_door --candidate 1 --iteration 8 --overwrite
```

### Quick sanity check

```bash
ls data/<scene>/any6d/
jq '{kf: .keyframe, t_cam: [.pose_object_to_camera[0][3],
                            .pose_object_to_camera[1][3],
                            .pose_object_to_camera[2][3]]}' \
    data/<scene>/any6d/<label>/pose.json

# Eyeball a single posed mesh in the world frame
python scripts/31_visualize_mesh.py --mesh data/<scene>/any6d/<label>/mesh_world.glb

# ...or all posed objects together, overlaid on the scene point cloud (stage 33)
python scripts/33_visualize_any6d.py --scene-dir data/<scene>
```

Labels whose sam3d folder has no matching `masks/<label>/<kf>.png` (e.g. stale
multi-instance `<label>_1` reconstructions, or an empty sam3d folder) are
**skipped with a message** rather than failing the run.

---

## Stage 33 — visualize Any6D poses against the point cloud

**Env:** any with `numpy` + `pillow` + `rerun-sdk` (the stage 41/51 env, e.g.
`any4d`)
**Reads:** `any6d/<label>/{mesh_world.glb, pose.json}` (stage 32),
`any4d/{config.json, cameras.npz, pointmap_ref.npy}` and
`any4d/moge/{depth,mask,intrinsics.npz}` (stage 40), `frames/*.jpg`
**Writes:** nothing under `data/` — only a Rerun recording (viewer, `.rrd`, or web)

The alignment check for stage 32. For every label with a `mesh_world.glb`, it
logs the posed mesh into Rerun **on top of a scene point cloud in the same
Any4D world frame** (no extra transform on the mesh — the stage-32 pose is
authoritative, exactly as stage 51 renders `aligned/` meshes), so you can
eyeball whether each object sits where the real geometry is. Same blueprint as
stages 41/51 (white background, RDF coordinates, no line grid). Labels with an
empty `any6d/<label>/` folder (Any6D couldn't pose them) are **skipped with a
message**; a label that only has `mesh_cam.glb` (stage 32 ran without
`cameras.npz`, so there is no world frame) is skipped too.

Two point-cloud sources (`--point-source`):

- `any4d` *(default)* — the Any4D reference pointmap (`pointmap_ref.npy`), one
  global cloud for the whole scene, coloured by the ref-frame RGB and masked to
  MoGe's valid pixels (the same cloud stages 41/51 draw). Best for a quick "does
  everything sit together" look. It is a **single frame** (`config.ref_frame`),
  so a part that *moved* between the ref frame and its own keyframe will not line
  up with it — use `moge` for those.
- `moge` — the MoGe **metric** depth at **each object's own keyframe**,
  unprojected to that frame's camera and baked into the world frame. This is the
  exact geometry Any6D registered to, per object, so a good pose overlays tightly
  even for a part that articulated between frames. One cloud per distinct
  keyframe.
- `both` — draw both (the `any4d` cloud dimmed toward gray so the meshes and
  `moge` clouds pop).

**Headless-safe.** Any6D itself renders nothing; run this wherever you can open
(or SSH-forward) a Rerun viewer. On a render-less server pass `--save-rrd PATH`
to write a recording you `scp` to a workstation and open with `rerun file.rrd`,
or `--serve` to serve it over the web.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | Scene folder `data/<scene_id>/` |
| `--labels LIST` | all | Subset of `any6d/` labels to visualize |
| `--point-source {any4d,moge,both}` | `any4d` | Which point cloud to check against |
| `--no-mask` | off | Don't mask the `any4d` cloud by MoGe's valid-pixel mask (show the full pointmap) |
| `--max-points INT` | `250000` | Subsample each cloud to at most this many points (`0` = keep all) |
| `--save-rrd PATH` | — | Write the recording to a `.rrd` file instead of spawning a viewer |
| `--serve` | off | Serve the recording over the web instead of a local viewer |
| `--port INT` | `9999` | Port for the spawned viewer |
| `--seed INT` | `0` | Subsampling RNG seed |

### Invocations

```bash
conda activate any4d   # any env with numpy + pillow + rerun-sdk

# Spawn the viewer: global Any4D cloud + all posed meshes
python scripts/33_visualize_any6d.py --scene-dir data/<scene>

# Check one object against the MoGe depth at its own keyframe
python scripts/33_visualize_any6d.py --scene-dir data/<scene> \
    --labels <object_label> --point-source moge

# Headless server: write a recording to open on a workstation
python scripts/33_visualize_any6d.py --scene-dir data/<scene> \
    --save-rrd data/<scene>/any6d/preview.rrd
```

### Quick sanity check

Each posed object should sit inside / on the point cloud where the real object
is. A gross offset means a bad stage-32 pose (or, for `--point-source any4d`, a
part that moved away from the reference frame — re-check with `moge`).

---

## Stage 40 — Any4D scene flow + MoGe depth

**Env:** `any4d`
**Reads:** `frames/*.jpg`, `masks/<label>/<ref>.png` (ref-frame masks),
`masks/tracking.json`
**Writes:** scene-level under `any4d/`, sparse per-label flow under `any4d/<label>/`

Runs Any4D once over the full `[start, end)` frame range with a single
`--ref-frame`, plus MoGe per-frame at FULL image resolution. Scene flow is
saved sparsely — only the pixels belonging to each label's mask at the ref
frame end up in `scene_flow/<frame>.npy`. Two resolutions are recorded in
`config.json`:

- `model_resolution_wh` — Any4D's internal (~518×336, patch-aligned). Pointmap,
  per-frame cameras, scene flow are stored at this resolution.
- `full_resolution_wh` — original clip resolution. MoGe depth + intrinsics
  are stored here so they can be used by later stages without a re-projection.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--any4d-repo PATH` | — *(required)* | Path to the `Any4D/` checkout |
| `--checkpoint PATH` | `<any4d-repo>/checkpoints/any4d_4v_combined.pth` | |
| `--ref-frame INT` | — *(required)* | Frame index used as Any4D's reference |
| `--start-idx INT` | `0` | Inclusive start index |
| `--end-idx INT` | end of clip | Exclusive end index |
| `--labels LIST` | all | Subset of labels to write flow for |
| `--machine STR` | `local` | Hydra `machine` override |
| `--data-norm-type` | `dinov2` | |
| `--no-amp` | off | Disable autocast (use fp32) |
| `--overwrite` | off | Replace existing `any4d/` |

### Invocations

Pick a sensible `--ref-frame` (usually one of the stage-20 keyframes for an
object you care about) and run:

```bash
conda activate any4d
python scripts/40_any4d_flow.py \
    --scene-dir  data/kitchen_pour_01 \
    --any4d-repo /path/to/Any4D \
    --ref-frame  42 \
    --overwrite
```

OOM on a long clip? Narrow the range:
```bash
python scripts/40_any4d_flow.py \
    --scene-dir  data/kitchen_pour_01 \
    --any4d-repo /path/to/Any4D \
    --ref-frame  42 --start-idx 20 --end-idx 80
```

### Quick sanity check

```bash
jq . data/<scene>/any4d/config.json
ls data/<scene>/any4d/                            # cameras.npz, pointmap_ref.npy, moge/, <labels>/
ls data/<scene>/any4d/<label>/scene_flow/ | head
```

---

## Stage 41 — minimal replay in Rerun

**Env:** any (rerun-sdk, numpy, pillow, opencv-python, matplotlib)
**Reads:** everything under `data/<scene>/any4d/` + RGB frames
**Writes:** nothing — opens the Rerun viewer

Visualizes RGB + ref pointcloud + MoGe depth + per-label scene-flow arrows +
per-point trajectories. A reusable sanity check that the stage-40 bundle
contains everything downstream stages need.

If you want joint axes and meshes overlaid as well, use **stage 51** instead.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--no-scene-flow` | off | Skip the scene-flow arrows |
| `--no-trajectories` | off | Skip 3D point polylines |
| `--max-tracks INT` | `200` | Trajectories per label |
| `--frame-time-step FLOAT` | `0.2` | Seconds per frame on the `stable_time` timeline |
| `--labels LIST` | all | Subset to replay |
| `--max-arrows INT` | `500` | Subsample flow arrows per frame per label |

### Invocations

```bash
python scripts/41_replay_in_rerun.py --scene-dir data/<scene>
```

---

## Stage 50 — joint type + axis estimation

**Env:** any (numpy + scipy)
**Reads:** `any4d/<label>/pts3d_ref.npy` + every `any4d/<label>/scene_flow/<frame>.npy`,
plus `any4d/config.json` for `ref_frame`
**Writes:** `any4d/joints.json` (scene summary) and `any4d/<label>/joint.json`
(per label)

Three-pass refinement for each label, treating the masked points as samples on
a single rigid movable part:

1. **Coarse — Procrustes per frame.** For each target frame, solve the rigid
   alignment `x_t = R_t · x_ref + d_t` via SVD Procrustes, extract `(axis, θ)`
   from `R_t`, recover an axis point from `(I − R_t) c = d_t`. Filter frames by
   `--theta-min / --theta-max` and by `--frame-rms-quantile`. Sign-aligned
   median aggregation across kept frames.
2. **Step 2 — local LM (Levenberg-Marquardt).** Pick the `--refine-window`
   frames temporally closest to `ref_frame` with `|θ| ≥ --refine-theta-min`,
   subsample to `--refine-points` points, and minimize
   `Σ ‖R(a, θ_t)(x_ref − c) + c − (x_ref + flow_t)‖²` over `(a, c, {θ_t})`.
   This avoids fitting MoGe noise in near-stationary windows.

A linear "slide algorithm" mode (`--method linear`) is kept for parity with
the algorithm in the design notes; it is biased for finite rotations and is
not the recommended default.

### CLI flags (most important)

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--labels LIST` | all | Subset to fit |
| `--method` | `procrustes` | `procrustes` (recommended) or `linear` |
| `--theta-min FLOAT` | `3.0` | (procrustes) min coarse θ per frame, degrees |
| `--theta-max FLOAT` | `90.0` | (procrustes) max coarse θ per frame, degrees |
| `--frame-rms-quantile FLOAT` | `0.75` | (procrustes) keep lower-RMS frames |
| `--refine / --no-refine` | on | Run the step-2 LM refinement |
| `--refine-window INT` | `8` | Frames closest to ref_frame to use in step 2 |
| `--refine-side` | `both` | `both` / `prev` / `next` |
| `--refine-theta-min FLOAT` | `1.5` | Skip near-stationary frames in step 2 |
| `--refine-points INT` | `4000` | LM point subsample |
| `--min-flow FLOAT` | `0.0` | Drop frames whose mean flow magnitude is below |
| `--dry-run` | off | Print results without writing JSON |

### Invocations

```bash
# Default — recommended
python scripts/50_estimate_joint.py --scene-dir data/<scene>

# Strictly previous-in-time frames, wider window
python scripts/50_estimate_joint.py --scene-dir data/<scene> \
    --refine-side prev --refine-window 12

# Skip refinement for non-rigid labels (e.g. a moving hand)
python scripts/50_estimate_joint.py --scene-dir data/<scene> \
    --labels hand --no-refine
```

### Quick sanity check

```bash
jq '. | to_entries | map({label: .key,
                          type: .value.type,
                          axis_dir: .value.axis_direction,
                          axis_point: .value.axis_point})' \
    data/<scene>/any4d/joints.json
```

Open `data/<scene>/any4d/<label>/joint.json` to see step-by-step results
(`coarse_axis_*`, `step2_axis_*`, `refine.thetas_deg`, etc.).

---

## Stage 52 — sam3d mesh alignment to image + mask

**Env:** any (numpy + opencv-python + trimesh + scipy)
**Reads:** `sam3d/<label>/[cand_NN_<kf>/]{mesh.glb, pose.json, keyframe.txt}`,
`masks/<label>/<kf>.png`, `any4d/moge/depth/<kf>.npy`,
`any4d/moge/intrinsics.npz`, `any4d/cameras.npz`
**Writes:** `aligned/<label>/mesh.glb` (in Any4D world frame) and
`aligned/<label>/align.json`

Per label, using each label's recorded keyframe:

1. **Apply pose.json** as the initial camera-frame transform, then rotate
   180° about Z to convert from sam3d's PyTorch3D-style camera frame
   (X-left, Y-up) to RDF (X-right, Y-down).
2. *(off by default)* **Coarse rescale** — re-fit isotropic scale via the
   Y-height ratio and translation via centroid offset against MoGe target
   points inside the mask (mesh_alignment.py-style). The height heuristic
   over-shrinks meshes whose backside is occluded in the keyframe, so for
   sam3d-objects we skip it; enable via `--coarse` for sam3d-body-style
   canonical meshes.
3. **Reprojection refinement.** Nelder-Mead over
   `(Δrotation, Δtranslation, Δlog_scale)` with a physically-sized initial
   simplex. Cost combines:
   - `1 − IoU(silhouette, mask)`
   - chamfer-on-distance-transform between silhouette and mask (smooth term —
     critical because IoU alone is piecewise-constant under small pixel shifts)
   - mean 3D ICP distance from sampled mesh verts to MoGe pointmap.
   Renders at `--render-factor` downsampled resolution with `--max-faces`
   random face subsample.
4. **Bake camera-to-world.** Apply the keyframe's `cam_quats_xyzw` +
   `cam_trans` from `any4d/cameras.npz`, so the saved mesh sits in Any4D's
   world frame and stage 51 can render it without any extra transform.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--labels LIST` | all | Subset to align |
| `--candidate INT` | `0` | When `sam3d` ran with `--all-candidates`, which one |
| `--overwrite` | off | Replace existing `aligned/<label>/` |
| `--coarse / --no-coarse` | off | mesh_alignment.py-style scale+translate step |
| `--no-refine` | off | Skip the Nelder-Mead refinement |
| `--render-factor INT` | `4` | Downsample factor for silhouette rendering |
| `--max-faces INT` | `15000` | Face subsample during refinement |
| `--max-iters INT` | `300` | Nelder-Mead cap |
| `--iou-weight FLOAT` | `1.0` | Silhouette IoU cost weight |
| `--chamfer-weight FLOAT` | `2.0` | DT-based silhouette chamfer weight |
| `--icp-weight FLOAT` | `1.0` | 3D ICP cost weight |

### Invocations

```bash
python scripts/52_align_meshes.py --scene-dir data/<scene> --overwrite

# Single label, more refinement iterations
python scripts/52_align_meshes.py --scene-dir data/<scene> \
    --labels laptop_up --max-iters 600 --overwrite

# Larger image rendering for fine-grained refinement (slower)
python scripts/52_align_meshes.py --scene-dir data/<scene> \
    --render-factor 2 --overwrite
```

### Quick sanity check

```bash
ls data/<scene>/aligned/
jq '{kf: .keyframe,
     iou_pose: .step_A_pose_json.iou_full,
     iou_refined: .step_C_refine.iou_full_after}' \
    data/<scene>/aligned/<label>/align.json
```

---

## Stage 52b — manual gradio-based alignment

**Env:** any with `gradio`, `numpy`, `opencv-python`, `trimesh`
**Reads:** same inputs as stage 52
**Writes:** same outputs as stage 52 (`aligned/<label>/{mesh.glb, align.json}`),
but with `align.json.method = "manual (52b)"`

Use when stage 52's automatic refinement misbehaves — typical case is a hand
(non-rigid; rigid silhouette/ICP can converge to spurious local minima) or
when you want to deliberately place the hand so it visibly contacts another
object in the scene.

### Initial placement (done once at startup)

1. Apply sam3d's `pose.json` with the PyTorch3D→RDF flip.
2. Project the result; compute the mesh 2D centroid and the mask 2D centroid;
   shift XY in 3D so the projections coincide.
3. Sample MoGe depth at the mask 2D centroid; translate Z so the mesh's
   centroid sits at that depth.

### Sliders

| group | sliders | range |
|---|---|---|
| Orientation | Rotation X / Y / Z (deg) | ±180 |
| Depth + scale | Depth (camera +Z, m), Scale multiplier | depth ∈ [init−1, init+2], scale ∈ [0.3, 3.0] |
| Fine translation | Tx, Ty, Tz (m) | ±0.5 (Tz is *added* on top of Depth) |

The "Depth + scale" pair is the right knob for putting a hand in contact with
an object: drop the hand mesh to the object's depth, then dial the scale up
or down so the projection still matches the mask. Scale and depth jointly
determine apparent size — adjust them together to slide the hand back and
forth in 3D without disturbing its 2D footprint.

### Overlay (live)

- **Red Lambertian-shaded** mesh — translucent or fully opaque, opacity
  controlled by the **Mesh opacity** slider (`0.10`–`1.00`, default `0.85`).
  Per-face shading + painter's-algorithm depth ordering gives proper 3D
  shape perception rather than a flat fill.
- **Red outline** around the silhouette so the boundary stays sharp at low
  opacity.
- **Green** outline — the label's mask.
- **Blue** shaded fill — context mesh (only if `--context-label` is set).
- IoU + centroid + depth + scale + α displayed beneath the image.

### Performance

The live render uses quadric-decimated geometry (`--decimate-faces`,
default 30 000) so each slider update lands in ~70 ms even for a 500 K-face
sam3d mesh. The save path uses the **full** original mesh — decimation is
preview-only. Falls back to the full mesh (or `--max-render-faces` random
subsample) if the `fast-simplification` package isn't installed.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--label STR` | — *(required)* | Which sam3d label to align manually |
| `--candidate INT` | `0` | sam3d candidate index |
| `--context-label STR` | — | Render `aligned/<context>/mesh.glb` as blue (e.g. the object the hand should touch) |
| `--decimate-faces INT` | `30000` | Target face count for the topology-preserving live-render mesh (needs `fast-simplification`; falls back gracefully). Save uses full geometry. |
| `--max-render-faces INT` | `0` | Random-subsample fallback if decimation is unavailable; `0` = full mesh |
| `--render-factor INT` | `2` | Downscale factor for live rasterization |
| `--port INT` | `7860` | Gradio port |
| `--share` | off | Public gradio URL |

### Invocations

```bash
# Align the hand, with the laptop lid as visual context
python scripts/52b_align_meshes_manual.py \
    --scene-dir data/macbook-all \
    --label hand \
    --context-label laptop_up

# Pick a different sam3d candidate
python scripts/52b_align_meshes_manual.py \
    --scene-dir data/macbook-all \
    --label hand \
    --candidate 1 --port 7870
```

When saved, stage 51 picks up the result automatically (`--mesh-source auto`
prefers `aligned/`).

---

# Scene-authoring track (52c / 52d / 52e)

An optional branch for building a **two-body articulation scene** (one body
welded to ground, one attached via a hinge or slide joint) with the full
WiLoR hand trajectory played back on top. It reads stage-30 sam3d meshes and
stage-60 WiLoR hands **directly** — it does *not* require the stage-50/52
joint fit or aligned meshes. Typical flow:

```
30 + 60  →  52c (MuJoCo edit)  or  52d (Rerun edit)  →  52e (animated MuJoCo replay of a 52d scene)
```

52c and 52d share the same gradio slider UI; pick whichever viewer you
prefer. 52e consumes the `rerun/transforms.json` that 52d writes.

---

## Stage 52c — interactive MuJoCo articulation editor

**Env:** any with `mujoco` + `gradio` + `numpy` + `trimesh`
**Reads:** `sam3d/<label-fixed>/[cand_NN_<kf>/]mesh.glb`,
`sam3d/<label-moving>/...mesh.glb`, `wilor/per_frame/<frame>.npz`,
`wilor/faces.npy`, `frames/*.jpg`
**Writes:** `mujoco/scene.xml` + `mujoco/transforms.json`

A gradio slider panel beside a live MuJoCo passive viewer. You position the
two bodies and the joint, scrub the WiLoR hand trajectory, and save a real
MJCF.

- **Fixed body** — pos (m), euler XYZ (deg), scale. Welded to ground.
- **Moving body baseline** — its rest pose at joint angle 0: pos, euler XYZ,
  scale.
- **Joint** — pos + orientation euler XYZ. The world-frame axis is
  `R_joint @ [0,0,1]`; a green sphere + capsule marks it.
- **Drive** — separate hinge (deg) and slide (m) sliders; the one matching
  `--joint` moves the body about the axis (visual preview only — the saved
  MJCF carries the real joint).
- **Animation** — frame slider over every `frames/*.jpg`; Play/Pause runs a
  `gr.Timer` at `--fps`. Each frame, `hand_left` / `hand_right` are placed by
  best-fit rigid pose from the canonical MANO mesh to that frame's WiLoR
  verts; a laterality with no detection that frame is moved off-screen.
- **Apply Scale** rebuilds the model (MuJoCo bakes mesh scale at compile
  time).

`scene.xml` welds the fixed body and gives the moving body a real
`<joint type="hinge|slide">` in its local frame (hands are a trajectory, so
they're omitted from the MJCF). `transforms.json` records every slider value
plus provenance.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--label-fixed STR` | — *(required)* | sam3d label welded to ground |
| `--label-fixed-candidate INT` | `0` | sam3d candidate for the fixed body |
| `--label-moving STR` | — *(required)* | sam3d label attached via the joint |
| `--label-moving-candidate INT` | `0` | sam3d candidate for the moving body |
| `--joint {hinge,slide}` | `hinge` | Joint connecting moving body to ground |
| `--fps FLOAT` | `10.0` | Play-timer rate |
| `--max-faces INT` | `150000` | Quadric-decimate each mesh before STL export (MuJoCo's STL decoder caps at 200000 — keep below it) |
| `--port INT` | `7860` | Gradio port |
| `--share` | off | Public gradio URL |

### Invocations

```bash
# Laptop lid hinge, full hand trajectory
python scripts/52c_mujoco_scene.py \
    --scene-dir data/macbook-all \
    --label-fixed laptop_base \
    --label-moving laptop_up \
    --joint hinge

# Drawer (slide), 30 FPS playback
python scripts/52c_mujoco_scene.py \
    --scene-dir data/<scene> \
    --label-fixed cabinet --label-moving drawer \
    --joint slide --fps 30
```

---

## Stage 52d — Rerun version of the editor (+ scene flow / pointcloud)

**Env:** any with `rerun-sdk` + `gradio` + `numpy` + `trimesh` + `pillow`
**Reads:** same sam3d meshes + WiLoR hands as 52c, **plus** the stage-40
bundle: `any4d/{config.json, pointmap_ref.npy, moge/mask/*}` and
`any4d/<label>/{pts3d_ref.npy, scene_flow/<frame>.npy}`
**Writes:** `rerun/transforms.json`

Same slider grid as 52c, but the viewer is Rerun rather than MuJoCo: meshes
are uploaded once as static archetypes and slider edits only re-log cheap
`Transform3D`s, so updates are immediate. It additionally renders the Any4D
reference pointcloud and per-label scene-flow arrows for context. No real
joint constraint is enforced — the drive slider applies the motion manually.

Unlike 52c's single static drive snapshot, **52d supports joint keyframes**:
a sorted list of `(frame_slot, joint_value)` pairs that 52d interpolates
piecewise-linearly along the trajectory. These get saved into
`transforms.json` and are what 52e replays by default.

`transforms.json` uses the same schema as 52c (minus the MJCF-specific
fields), and auto-loads at startup if it already exists.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--label-fixed STR` | — *(required)* | sam3d label welded to ground |
| `--label-fixed-candidate INT` | `0` | |
| `--label-moving STR` | — *(required)* | sam3d label attached via the joint |
| `--label-moving-candidate INT` | `0` | |
| `--joint {hinge,slide}` | `hinge` | |
| `--frame-time-step FLOAT` | `0.2` | Seconds per frame on the `stable_time` timeline (matches stage 41) |
| `--max-faces INT` | `150000` | Quadric-decimate each mesh (`0` = none; Rerun handles big meshes but slows above ~200k faces) |
| `--no-scene-flow` | off | Don't render Any4D scene-flow arrows |
| `--no-pointcloud` | off | Don't render the Any4D reference pointcloud |
| `--max-arrows INT` | `500` | Subsample scene-flow arrows per label per frame |
| `--load-transforms PATH` | `<scene>/rerun/transforms.json` if present | transforms.json to restore at startup |
| `--no-load-transforms` | off | Skip auto-load; start from pointcloud-anchored defaults |
| `--port INT` | `7860` | Gradio port |
| `--rerun-port INT` | `9999` | Rerun viewer port |
| `--share` | off | Public gradio URL |

### Invocations

```bash
python scripts/52d_rerun_scene.py \
    --scene-dir data/macbook-all \
    --label-fixed laptop_base --label-moving laptop_up \
    --joint hinge --fps 15
```

---

## Stage 52e — animated MuJoCo replay of a saved 52d scene

**Env:** any with `mujoco` + `numpy` + `trimesh`
**Reads:** `rerun/transforms.json` (or `--transforms PATH`), the sam3d
`mesh.glb`s named in its provenance, `wilor/per_frame/*.npz`,
`wilor/faces.npy`, `frames/*.jpg`
**Writes (under `mujoco_anim/`):** `scene.xml` (a re-runnable MJCF),
`object_fixed.stl`, `object_moving.stl`, `hand_left.stl` /
`hand_right.stl` (only the lateralities the trajectory contains)

Rebuilds a 52d scene as a **real** MuJoCo articulation — fixed body welded to
ground, moving body on a hinge/slide joint at the saved pose, hand mocap
bodies driven through the full WiLoR trajectory. The joint motion is
generated on the fly: by default it replays the saved `joint_keyframes` if
present, otherwise a linear sweep `0 → drive_at_save`.

The emitted MJCF is standalone — re-open it any time with
`python -m mujoco.viewer --mjcf=data/<scene>/mujoco_anim/scene.xml`. Press
**V** in the viewer to save the current orbit camera to `view.json` (auto-
loaded next run).

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--transforms PATH` | `<scene>/rerun/transforms.json` | The 52d scene to replay |
| `--fps FLOAT` | `15.0` | Playback rate |
| `--joint-motion {keyframes,static,linear,triangle,sine}` | `keyframes` if saved, else `linear` | How the joint drives over the trajectory |
| `--joint-amplitude FLOAT` | `1.0` | Scale factor on `drive_at_save` |
| `--n-cycles FLOAT` | `1.0` | Cycles for `triangle` / `sine` motion |
| `--max-faces INT` | `150000` | Decimate object meshes (MuJoCo STL caps at 200000) |
| `--once` | off | Play through once then exit (default: loop) |
| `--view-file PATH` | `<scene>/mujoco_anim/view.json` | Saved orbit-camera state; auto-loaded if present |
| `--reset-view` | off | Skip auto-loading the saved view this run |

### Invocations

```bash
# Default: replay saved keyframes (or a linear 0 -> drive sweep)
python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer

# Two open/close cycles at 30 FPS
python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer \
    --joint-motion triangle --n-cycles 2 --fps 30

# Hold the joint static at 1.5x the saved drive while the hand plays
python scripts/52e_mujoco_animate.py --scene-dir data-final/dryer \
    --joint-motion static --joint-amplitude 1.5
```

---

## Stage 60 — WiLoR hand tracking

**Env:** `wilor` (project memory: torch 2.12+cu130, on remote dsailogin)
**Reads:** `frames/*.jpg`, `<wilor-repo>/pretrained_models/{wilor_final.ckpt,
model_config.yaml, detector.pt}`, optionally `any4d/cameras.npz` for
world-frame baking
**Writes:** `wilor/{config.json, faces.npy, per_frame/<frame>.npz}`

Per frame: YOLO hand detection → WiLoR fit per hand → compact npz with the
778-vertex MANO mesh and 21 3D keypoints, already in camera frame (RDF,
matches MoGe / Any4D). If `any4d/cameras.npz` is present the script also
bakes the camera-to-world transform for each frame and adds `verts_world` /
`joints_world` fields ready for stage-51 rendering.

### Per-frame npz fields

| key | shape | dtype | meaning |
|---|---|---|---|
| `verts` | `(n_hands, 778, 3)` | float32 | mesh verts in camera frame (cam_t already applied) |
| `joints` | `(n_hands, 21, 3)` | float32 | MANO keypoints in camera frame |
| `is_right` | `(n_hands,)` | bool | right=True, left=False |
| `cam_t` | `(n_hands, 3)` | float32 | translation applied to canonical verts |
| `bbox` | `(n_hands, 4)` | float32 | YOLO detection bbox, xyxy in image coords |
| `yolo_conf` | `(n_hands,)` | float32 | YOLO detection confidence |
| `focal_length` | `()` | float32 | per-frame WiLoR focal (auto-computed) |
| `img_size_wh` | `(2,)` | int32 | image width, height |
| `frame_idx` | `()` | int32 | source frame index |
| `verts_world` | `(n_hands, 778, 3)` | float32 | *(if cam pose known)* world frame |
| `joints_world` | `(n_hands, 21, 3)` | float32 | *(if cam pose known)* world frame |

Shared MANO topology is at `wilor/faces.npy` (`(Nf, 3)` int32) — load once,
reuse across frames. Frames with zero detections are simply omitted.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--wilor-repo PATH` | — *(required)* | Path to `WiLoR/` checkout |
| `--checkpoint` | `wilor_final.ckpt` | filename under `pretrained_models/` |
| `--model-config` | `model_config.yaml` | |
| `--detector` | `detector.pt` | |
| `--start-idx INT` | `0` | Inclusive frame index start |
| `--end-idx INT` | end of clip | Exclusive end |
| `--detector-conf FLOAT` | `0.3` | YOLO confidence threshold |
| `--rescale-factor FLOAT` | `2.0` | Bbox padding factor for the crop |
| `--max-hands INT` | none | Cap to top-N detections per frame |
| `--batch-size INT` | `16` | WiLoR fit batch within a single frame |
| `--fast` | off | FP16 + `torch.compile` + layer drop for speed |
| `--no-world` | off | Skip applying Any4D cam2world; save camera frame only |
| `--save-obj` | off | Also dump per-hand `.obj` to `wilor/obj/` |
| `--overwrite` | off | Replace `wilor/` |

### Invocations

```bash
conda activate wilor
python scripts/60_wilor_hands.py \
    --scene-dir data/<scene> \
    --wilor-repo /path/to/WiLoR \
    --overwrite

# Subrange + fast inference for a long clip
python scripts/60_wilor_hands.py \
    --scene-dir data/<scene> \
    --wilor-repo /path/to/WiLoR \
    --start-idx 20 --end-idx 80 --fast --overwrite

# Cap to two hands per frame and also export .obj for ad-hoc viewing
python scripts/60_wilor_hands.py \
    --scene-dir data/<scene> \
    --wilor-repo /path/to/WiLoR \
    --max-hands 2 --save-obj --overwrite

# Force camera-frame output only (no Any4D dependency)
python scripts/60_wilor_hands.py \
    --scene-dir data/<scene> \
    --wilor-repo /path/to/WiLoR \
    --no-world --overwrite
```

### Quick sanity check

```bash
jq '{n: .total_hands,
     frames: .n_frames_with_hands,
     world: .world_frame_baked,
     focal: .focal_length}' \
    data/<scene>/wilor/config.json

ls data/<scene>/wilor/per_frame/ | head
python -c "import numpy as np; \
    d=np.load('data/<scene>/wilor/per_frame/000042.npz'); \
    print({k: d[k].shape if hasattr(d[k],'shape') else d[k] for k in d.files})"
```

### Loading in your own code

```python
import numpy as np
faces = np.load("data/<scene>/wilor/faces.npy")          # (Nf, 3)
d = np.load("data/<scene>/wilor/per_frame/000042.npz")
verts_world = d["verts_world"]   # (n_hands, 778, 3)
joints = d["joints_world"]       # (n_hands, 21, 3)
is_right = d["is_right"]         # (n_hands,) bool
```

(Rerun integration: `rr.log("hands/<frame>/<n>", rr.Mesh3D(vertex_positions=
verts_world[n], triangle_indices=faces))`.)

---

## Stage 60b — rescale legacy WiLoR output to the real MoGe focal

**Env:** any with `numpy`
**Reads:** `wilor/per_frame/*.npz` (in place), `any4d/moge/intrinsics.npz`
(real focals), optionally `any4d/cameras.npz` (to re-bake world frame)
**Writes:** overwrites each per-frame npz in place

A one-off post-fix for `wilor/` folders produced **before** stage 60 learned
to use the MoGe focal. The old code wrote `cam_t` / `verts` / `joints` with
WiLoR's *nominal* focal (e.g. `5000/256 × 1920 ≈ 37500` px for a 1920-wide
clip), so hands landed ~30× too deep. This script rescales camera-frame Z by
`focal_real / focal_wilor` (projection onto the image is preserved, so the
hand still aligns), updates `focal_length`, records `focal_length_orig` for
auditing, and — if `any4d/cameras.npz` exists — re-bakes `verts_world` /
`joints_world`.

It is **idempotent**: frames already carrying `focal_length_orig` are skipped,
as are frames with no MoGe focal. Current stage 60 already applies the real
focal, so you only need this for older runs.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--dry-run` | off | Print the per-frame correction without writing |

### Invocations

```bash
# Preview the correction
python scripts/60b_rescale_wilor_focal.py --scene-dir data-final/dryer --dry-run

# Apply it
python scripts/60b_rescale_wilor_focal.py --scene-dir data-final/dryer
```

---

## Stage 60c — HaWoR hand tracking (video-temporal alternative to 60)

**Env:** `hawor` (project memory: torch 2.0.1+cu118, on remote dsailogin) — GPU required
**Reads:** `frames/*.jpg`, `<hawor-repo>/weights/hawor/checkpoints/{hawor.ckpt,
infiller.pt}`, `<hawor-repo>/_DATA/...` (MANO), optionally
`any4d/cameras.npz` (world-frame baking) and `any4d/moge/intrinsics.npz` (focal)
**Writes:** `hawor/{config.json, faces.npy, faces_left.npy, _work/, per_frame/<frame>.npz}`

A drop-in alternative to stage 60. Instead of fitting each frame independently,
HaWoR runs a full video pipeline — detect/track → per-frame motion estimation →
masked DROID-SLAM (camera trajectory + metric scale) → a transformer in-filler
that completes **both** hands across the clip — giving a temporally smooth,
two-hand (left=0, right=1) trajectory. This stage re-projects those hands into
the per-frame **camera frame** (RDF, same as MoGe / Any4D / WiLoR) and writes
the **same npz schema as stage 60**, so `hawor/` is consumable anywhere `wilor/`
is. If `any4d/cameras.npz` exists it bakes the **identical** cam→world transform
WiLoR uses, producing `verts_world` / `joints_world` in the Any4D world frame.

SLAM/tracking intermediates are cached under `hawor/_work/`, so re-deriving the
npz format (e.g. with different `--start-idx`/`--detected-only`) is cheap;
`--overwrite` keeps that cache, `--recompute` clears it.

### Per-frame npz fields

Same as stage 60, **except** `yolo_conf` is replaced by `valid` (HaWoR has no
detector confidence). Frames with zero hands are omitted.

| key | shape | dtype | meaning |
|---|---|---|---|
| `verts` | `(n_hands, 778, 3)` | float32 | mesh verts in camera frame (RDF) |
| `joints` | `(n_hands, 21, 3)` | float32 | MANO keypoints in camera frame |
| `is_right` | `(n_hands,)` | bool | right=True, left=False |
| `valid` | `(n_hands,)` | bool | **True=detected this frame, False=in-filled** |
| `cam_t` | `(n_hands, 3)` | float32 | camera-frame wrist (joint 0) position |
| `bbox` | `(n_hands, 4)` | float32 | xyxy in image coords, from projected verts |
| `focal_length` | `()` | float32 | focal used for the clip (single value) |
| `img_size_wh` | `(2,)` | int32 | image width, height |
| `frame_idx` | `()` | int32 | source frame index |
| `verts_world` | `(n_hands, 778, 3)` | float32 | *(if cam pose known)* Any4D world frame |
| `joints_world` | `(n_hands, 21, 3)` | float32 | *(if cam pose known)* Any4D world frame |

`hawor/faces.npy` is standard MANO topology (matches `wilor/faces.npy`). The
left hand is genuinely posed in 3D (not reflected like WiLoR's), so it shares
those faces with **reversed winding** — `hawor/faces_left.npy` holds the
corrected-winding faces for rendering left-hand normals.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--hawor-repo PATH` | — *(required)* | Path to `HaWoR/` checkout (needs `weights/`, `_DATA/`) |
| `--checkpoint` | `<repo>/weights/hawor/checkpoints/hawor.ckpt` | |
| `--infiller-weight` | `<repo>/weights/hawor/checkpoints/infiller.pt` | |
| `--img-focal FLOAT` | median MoGe focal, else HaWoR estimate | pinhole focal in px |
| `--ignore-moge-focal` | off | Don't auto-load the MoGe focal; let HaWoR estimate |
| `--start-idx INT` | start of clip | Only *save* frames with index ≥ this (inference still runs on the whole clip — SLAM needs continuity) |
| `--end-idx INT` | end of clip | Only *save* frames with index < this |
| `--detected-only` | off | Drop in-filled hands; keep only detected ones |
| `--no-world` | off | Skip Any4D cam2world; camera frame only |
| `--overwrite` | off | Replace `per_frame/`+`faces`+`config` (keeps `_work/` cache) |
| `--recompute` | off | Also clear `_work/`, forcing track/SLAM to rerun |

### Invocations

```bash
conda activate hawor

# Default: video-temporal hands for the whole clip, MoGe focal, world-frame baked
python scripts/60c_hawor_hands.py \
    --scene-dir data/oven \
    --hawor-repo /home/jeremy/research/Articulate4D/HaWoR

# Only detected hands (drop in-filled), forced focal, fresh run
python scripts/60c_hawor_hands.py \
    --scene-dir data/oven \
    --hawor-repo /home/jeremy/research/Articulate4D/HaWoR \
    --detected-only --img-focal 1500 --overwrite
```

> **Note — scale vs. WiLoR:** HaWoR depth comes from its own SLAM metric scale;
> even with the shared MoGe focal the absolute depth can differ from a WiLoR run
> of the same clip. Both stages share the camera/world conventions, so they
> overlay correctly in image space, but don't assume identical metric Z.

---

## Stage 51 — full replay in Rerun (joints + meshes)

**Env:** same as stage 41 (rerun-sdk, numpy, pillow, opencv-python,
matplotlib, trimesh)
**Reads:** stage-40 bundle, stage-50 joints, stage-52 aligned meshes (or
stage-30 sam3d meshes if no `aligned/` exists)
**Writes:** nothing — opens the Rerun viewer

Reuses stage 41's replay (imported via `importlib`, no duplication), then
overlays:

- **Joint axes** under `pred/joints/<label>` — revolute as a line through
  the projection of the part centroid onto the axis (with a marker + half-
  length arrow), prismatic as an arrow at the part centroid. Logged static.
- **GLB meshes** under `pred/meshes/<label>` — pulled from `aligned/` by
  default (already in world frame), or from `sam3d/` (renders near origin,
  pre-alignment). Static.

### CLI flags (most important)

| flag | default | meaning |
|---|---|---|
| `--scene-dir PATH` | — *(required)* | |
| `--no-scene-flow` | off | Skip flow arrows |
| `--no-trajectories` | off | Skip 3D point polylines |
| `--no-joints` | off | Skip the joint-axis overlay |
| `--no-meshes` | off | Skip the GLB mesh overlay |
| `--mesh-source` | `auto` | `aligned` / `sam3d` / `auto` (= `aligned` if present) |
| `--mesh-candidate INT` | `0` | Candidate index when source is `sam3d` |
| `--max-tracks INT` | `200` | Trajectories per label |
| `--max-arrows INT` | `500` | Flow arrows per frame per label |
| `--labels LIST` | all | Subset |

### Invocations

```bash
# After stages 40 + 50 + 52
python scripts/51_replay_with_joints.py --scene-dir data/<scene>

# Before alignment (mesh placement will be near origin)
python scripts/51_replay_with_joints.py --scene-dir data/<scene> --mesh-source sam3d

# Just joints, no meshes
python scripts/51_replay_with_joints.py --scene-dir data/<scene> --no-meshes

# Single label
python scripts/51_replay_with_joints.py --scene-dir data/<scene> --labels laptop_up
```

---

## Resuming after a partial run

- **Stage 05**: appends/merges into `prompts.json`; re-running a label
  replaces its entry (replace checkbox on by default). Safe to run repeatedly.
- **Stage 10**: re-run with `--overwrite` to redo from scratch, or change
  prompts and rerun. (It always wipes `masks/` if you pass `--overwrite`.)
- **Stage 20**: idempotent — just re-run. Use `--manual` to fix bad picks.
- **Stage 30**: omits `--overwrite` by default, so already-built objects are
  skipped. Add `--labels foo bar` to redo a subset, plus `--overwrite` to
  replace.
- **Stage 32**: per-label; skips labels whose `any6d/<label>/` already exists
  unless you pass `--overwrite`. Add `--labels foo` for a subset. Labels with no
  matching mask at the keyframe are skipped with a message.
- **Stage 40**: requires `--overwrite` to replace an existing `any4d/`. There
  is no per-label partial mode — the full Any4D forward pass is single-shot.
- **Stage 50**: idempotent — re-run with new flags to retune. Updates the
  per-label `joint.json` and scene `joints.json` in place.
- **Stage 52**: per-label; skips labels whose `aligned/<label>/` already
  exists unless you pass `--overwrite`. Add `--labels foo` for a subset.
- **Stage 60**: requires `--overwrite` to replace an existing `wilor/`. Use
  `--start-idx` / `--end-idx` to resume / cover a subrange and merge
  manually if needed.
- **Stage 60b**: idempotent — already-corrected frames (carrying
  `focal_length_orig`) are skipped, so re-running is a no-op.
- **Stage 52c / 52d**: interactive editors; "Save" overwrites
  `mujoco/` / `rerun/`. 52d auto-loads the existing `transforms.json` on
  startup so you can resume editing.
- **Stage 52e**: pure consumer of `rerun/transforms.json` — re-run any time
  with different `--joint-motion` / `--fps` to regenerate `mujoco_anim/`.
- **Stage 41 / 51**: pure replay — re-run any time.

---

## Troubleshooting

| symptom | first thing to check |
|---|---|
| Stage 10: empty mask folder for a prompt | The prompt didn't detect anything on `frame_index=0`. Try another `frame_index` in the JSON. |
| Stage 10: mask still covers regions you marked **negative** (negatives have no effect) | The prompt has both `text` and points. A text concept re-asserts the whole object every frame, so negatives layered on it can't carve a sub-part. Make it an **interactive (PVS) prompt**: remove `text` and keep only `box` + `positive_points` + `negative_points`. Stage 10 then segments purely from the geometry. Keep ≤8 positive and ≤8 negative (the tracker drops points past 16). |
| Stage 10: `CUBLAS_STATUS_INVALID_VALUE` in the text encoder (`cublasGemmEx`/`cublasGemmStridedBatchedEx`) | Legacy-cuBLAS bug on some torch/CUDA builds (seen on torch 2.10+cu128) — non-contiguous GEMMs abort for both bf16 and fp16. The script forces `torch.backends.cuda.preferred_blas_library("cublaslt")` at startup to route around it; if it still fails, confirm cuBLASLt works: `python - <<<'import torch,torch.nn.functional as F; torch.backends.cuda.preferred_blas_library("cublaslt"); x=torch.randn(4,13,1024,device="cuda",dtype=torch.bfloat16).transpose(0,1); F.linear(x,torch.randn(3072,1024,device="cuda",dtype=torch.bfloat16)); torch.cuda.synchronize(); print("ok")'`. If that fails too, reinstall a torch build without the regression (e.g. a cu121/cu124 wheel). |
| Stage 10: `--version sam3.1` → `init_state() got an unexpected keyword argument 'offload_state_to_cpu'` | Base-predictor-vs-sam3.1-model API skew inside the vendored `sam3` checkout (not this script). Use `--version sam3`, or align the `sam3` package with the sam3.1 model API. |
| Stage 20: `keyframe: null` for a label | Visible-frame count was 0 or all frames failed the area band. Inspect `n_frames_visible` in tracking.json; widen `--area-min` / `--area-max`. |
| Stage 30: kaolin import error | `notebook/inference.py:25-26` imports kaolin unconditionally. Either fix the env or guard those two imports with `try/except`. |
| Stage 30: `CUDA error: unrecognized error code` (cuDNN) or `CUBLAS_STATUS_EXECUTION_FAILED` (`cublasGemmEx`/`cublasLtMatmul`) in MoGe / DINO | The CUDA-13 driver is a *major* version ahead of this env's bundled CUDA 12.x libs (torch 2.5.1+cu121), so specific cuDNN conv + cuBLAS fp16 conv-GEMM kernels abort (a plain fp16 matmul of the same dims still works — it's kernel-specific, not all of fp16). **Fix: bump the bundled libs** — `pip install --no-deps -U nvidia-cublas-cu12 nvidia-cudnn-cu12` (keeps torch 2.5.1 + pinned spconv/flash_attn/kaolin), then run clean. Stop-gaps if you can't bump: `--no-cudnn` (forces conv→GEMM) and/or `--blas-lib cublaslt` — but note forcing cuBLASLt can itself fail skinny GEMMs here, so try `--blas-lib cublas`/`default` too. Last resort: reinstall torch on cu128+/cu130 (risks the pinned extensions). Not related to `--bg-mode`. |
| Stage 30: gsplat / pytorch3d build crash | Re-check `TORCH_CUDA_ARCH_LIST="8.0+PTX"` and `CUDA_HOME=$CONDA_PREFIX` in the current shell. |
| Stage 32: a label is silently skipped | No `masks/<label>/<kf>.png` (the sam3d mesh's keyframe has no matching mask — common for stale multi-instance `<label>_1` reconstructions) or no MoGe depth at that frame. Re-run stages 10/30 against the current masks, or pass `--labels` to target the objects that do line up. |
| Stage 32: pose is wildly off-scale or flipped | Any6D's coarse OBB fit can latch onto the wrong axis on near-symmetric parts. Try `--no-axis-align` or raise `--iteration`. The mesh fed in is the stage-30 reconstruction — a bad mesh propagates here; transfer `final_mesh.glb` / `mesh_world.glb` and eyeball them with stage 31 on a workstation. |
| Stage 32: hands/empty parts produce garbage | It registers whatever sam3d label you point it at; restrict to rigid objects with `--labels` (skip `hand` and non-object parts). |
| Stage 40: OOM on a long clip | Narrow with `--start-idx` / `--end-idx`, or `--no-amp` if it's a precision issue. |
| Stage 50: axis sits meters from the part | Linear/coarse-only fit on cumulative flow is biased for finite rotations. Make sure `--refine` is on (default) and `ref_frame` is set in `any4d/config.json`. If `joint.json` shows `coarse_axis_*` very different from the final `axis_*`, step 2 fixed it. |
| Stage 50: step 2 makes things worse for a label | Likely non-rigid (e.g. a hand). Pass `--no-refine` for that label, or use `--method linear` for a quick visual sanity check. |
| Stage 52: refined IoU equals initial IoU | Old code path. Make sure `--chamfer-weight > 0` (default 2.0); pure IoU is piecewise-constant and Nelder-Mead gets stuck. |
| Stage 52: pose.json gives IoU 0 | Coordinate convention. sam3d's `pose.json` is in PyTorch3D camera frame; stage 52 handles the X-Y flip — if you're rendering elsewhere, mirror manually. |
| Stage 51: mesh sits near the origin | You're rendering `sam3d/` (unaligned). Run stage 52, then re-run 51 (default `--mesh-source auto` will switch to `aligned/`). |
| Stage 60: hand sits ~30× too deep | Legacy run with the nominal WiLoR focal. Run `60b_rescale_wilor_focal.py` to rescale in place (current stage 60 already uses the real MoGe focal). |
| Stage 52c / 52e: STL fails to load in MuJoCo | sam3d meshes exceed MuJoCo's 200000-face STL cap. Keep `--max-faces` below it (default 150000). |
| Stage 52c / 52d: gradio UI loads but viewer is empty | Confirm both `--label-fixed` and `--label-moving` have a `sam3d/<label>/mesh.glb` (or `cand_NN_*/mesh.glb`), and `wilor/` exists for the hand trajectory. |
| Stage 52e: "no transforms.json" | Save a scene from stage 52d first (writes `rerun/transforms.json`), or point `--transforms` at one. |
| `pad_frame_names.sh` kills SSH | Use the Python version instead — `pad_frame_names.py`. |

---

## Copy-paste recipe

See [`QUICKSTART.md`](QUICKSTART.md) for a single-scene, fresh-run recipe that
walks through all stages end-to-end.
