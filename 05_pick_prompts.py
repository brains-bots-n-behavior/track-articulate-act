#!/usr/bin/env python
"""Stage 05: gradio-based picker for SAM3 click + box refinement prompts.

Runs *before* stage 10 (segmentation): it authors the prompts.json that stage
10 consumes.

Use this when stage 10's text-only prompts can't separate adjacent parts of
an articulated object (e.g. laptop_base vs laptop_up). Run the picker on
the machine that holds data/, open the printed gradio URL in your local
browser (VS Code's Remote-SSH auto-forwards the port), click on the frame
to drop positive / negative points, switch the radio to **box** and click
two opposite corners for an optional bounding box. Save writes one entry
into `data/<scene>/prompts.json`, then stage 10 reads it.

Two kinds of prompt come out of this picker:
  * CONCEPT prompt  -- fill `text`, no geometry. Stage 10 segments every
    instance of the text concept.
  * INTERACTIVE (PVS) prompt -- leave `text` blank and pick a box + positive
    points (on the part) + negative points (on everything to exclude). Stage
    10 builds the object purely from this geometry, so the negatives genuinely
    carve the part out. This is the reliable way to split sub-parts; a text
    concept re-asserts the whole object during propagation, so negative clicks
    layered on top of it can't remove a sub-part. Keep clicks modest -- the
    tracker uses at most 16 points (first 8 + last 8; a box counts as 2).

The **frame slider** at the top of the UI iterates every JPEG under
`data/<scene>/frames/`, so each prompt entry can target a different frame
(`prompts.json[i].frame_index` follows whichever frame was selected when
that entry was saved).

Reads:
    data/<scene>/frames/*.jpg               (the only required input)

Writes (append):
    data/<scene>/prompts.json               (or --prompts-json PATH)

Coordinate convention:
    All coordinates are written as absolute image pixels. Stage 10
    normalizes boxes to xywh in [0, 1] before handing them to SAM3.1; for
    points it sets `rel_coordinates=False` so the pixels go through raw.

Run inside any env with gradio + numpy + opencv-python.

Examples:
    # Default: scan all frames, open on frame 0, save to <scene>/prompts.json
    python scripts/05_pick_prompts.py --scene-dir data/macbook-all

    # Open the slider already positioned on frame 42, custom output path
    python scripts/05_pick_prompts.py \\
        --scene-dir data/macbook-all \\
        --frame-index 42 \\
        --prompts-json /tmp/my_prompts.json \\
        --port 7871
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


_GREEN = (35, 200, 65)
_RED = (220, 50, 50)
_BLUE = (50, 110, 230)


def draw_overlay(image, pos_pts, neg_pts, box, pending):
    """Return a copy of `image` with the picked annotations rendered on top."""
    out = image.copy()
    for x, y in pos_pts:
        cv2.circle(out, (int(x), int(y)), 7, _GREEN, -1)
        cv2.circle(out, (int(x), int(y)), 8, (255, 255, 255), 1)
    for x, y in neg_pts:
        cv2.circle(out, (int(x), int(y)), 7, _RED, -1)
        cv2.circle(out, (int(x), int(y)), 8, (255, 255, 255), 1)
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), _BLUE, 2)
    if pending is not None:
        x, y = pending
        cv2.drawMarker(out, (int(x), int(y)), _BLUE,
                       markerType=cv2.MARKER_CROSS,
                       markerSize=20, thickness=2)
    return out


def load_frame_rgb(frames_dir: Path, frame_idx: int):
    fp = frames_dir / f"{frame_idx:06d}.jpg"
    if not fp.is_file():
        return None
    bgr = cv2.imread(str(fp))
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Prompt entry assembly
# ---------------------------------------------------------------------------


def build_entry(text, label, frame_index, pos_pts, neg_pts, box):
    text = (text or "").strip()
    d = {"frame_index": int(frame_index), "label": (label or "").strip()}
    # `text` is omitted for interactive (PVS) prompts so stage 10 segments the
    # part purely from the geometry instead of the (whole-object) text concept.
    if text:
        d = {"text": text, **d}
    if pos_pts:
        d["positive_points"] = [[int(x), int(y)] for x, y in pos_pts]
    if neg_pts:
        d["negative_points"] = [[int(x), int(y)] for x, y in neg_pts]
    if box is not None:
        x1, y1, x2, y2 = box
        d["box"] = [int(min(x1, x2)), int(min(y1, y2)),
                    int(max(x1, x2)), int(max(y1, y2))]
    return d


def load_existing(path: Path):
    if not path.is_file():
        return {"prompts": []}
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        return {"prompts": [], "_load_error": f"{type(e).__name__}: {e}"}
    if not isinstance(d, dict) or "prompts" not in d:
        return {"prompts": [d] if isinstance(d, dict) else []}
    return d


def append_entry(prompts_path: Path, entry: dict, replace_same_label: bool):
    existing = load_existing(prompts_path)
    if "_load_error" in existing:
        return (f"**FAIL**: couldn't parse existing `{prompts_path.name}` "
                f"({existing['_load_error']}); refusing to overwrite.")
    entries = list(existing.get("prompts", []))
    same_idx = [i for i, p in enumerate(entries) if p.get("label") == entry["label"]]
    if same_idx and not replace_same_label:
        return (f"**SKIP**: a prompt with label `{entry['label']}` already exists "
                f"at index {same_idx[0]}. Tick the replace checkbox to overwrite.")
    if same_idx:
        entries[same_idx[0]] = entry
        verb = "replaced"
    else:
        entries.append(entry)
        verb = "appended"
    out = {"prompts": entries}
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps(out, indent=2))
    return (f"**OK**: {verb} `{entry['label']}` (frame {entry['frame_index']:06d}) "
            f"→ `{prompts_path}` ({len(entries)} prompt(s) total). "
            f"Run stage 10 with `--prompts-json {prompts_path}`.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def list_frame_indices(frames_dir: Path):
    fis = sorted(int(p.stem) for p in frames_dir.glob("*.jpg")
                 if p.stem.isdigit())
    return fis


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    p.add_argument("--scene-dir", type=Path, required=True,
                   help="Scene folder data/<scene_id>/")
    p.add_argument("--frame-index", type=int, default=None,
                   help="Frame to open initially (default: first frame found)")
    p.add_argument("--prompts-json", type=Path, default=None,
                   help="Output path (default: <scene>/prompts.json)")
    p.add_argument("--port", type=int, default=7860, help="Gradio port")
    p.add_argument("--share", action="store_true",
                   help="Expose a public gradio URL")
    return p.parse_args()


def main():
    args = parse_args()
    scene_dir = args.scene_dir.resolve()
    frames_dir = scene_dir / "frames"
    if not frames_dir.is_dir():
        sys.exit(f"error: {frames_dir} does not exist")

    frame_indices = list_frame_indices(frames_dir)
    if not frame_indices:
        sys.exit(f"error: no integer-named .jpg frames under {frames_dir}")

    # Resolve initial slot from --frame-index (closest match wins if absent).
    if args.frame_index is None:
        init_slot = 0
    elif args.frame_index in frame_indices:
        init_slot = frame_indices.index(args.frame_index)
    else:
        # nearest available
        init_slot = int(np.argmin([abs(fi - args.frame_index)
                                    for fi in frame_indices]))
        print(f"warning: --frame-index {args.frame_index} not in frames/; "
              f"snapping to nearest available frame "
              f"{frame_indices[init_slot]:06d}")

    init_frame_idx = frame_indices[init_slot]
    init_rgb = load_frame_rgb(frames_dir, init_frame_idx)
    if init_rgb is None:
        sys.exit(f"error: failed to read frames/{init_frame_idx:06d}.jpg")
    H0, W0 = init_rgb.shape[:2]
    print(f"frames: {len(frame_indices)} "
          f"(range {frame_indices[0]:06d}..{frame_indices[-1]:06d})")
    print(f"opening on slot {init_slot} → frame {init_frame_idx:06d}  "
          f"({W0}x{H0})")

    prompts_path = (args.prompts_json.resolve()
                    if args.prompts_json is not None
                    else scene_dir / "prompts.json")
    existing = load_existing(prompts_path)
    existing_labels = [p.get("label") for p in existing.get("prompts", [])]
    print(f"prompts.json: {prompts_path}")
    if existing_labels:
        print(f"  existing labels: {existing_labels}")

    n_slots = len(frame_indices)

    with gr.Blocks(title="SAM3 prompt picker") as demo:
        gr.Markdown(f"# SAM3 prompt picker — `{scene_dir.name}`")
        gr.Markdown(
            "Drag the **frame slider** to pick which frame each prompt "
            "targets (entries store the frame_index that was active when "
            "you saved them). Click the image to drop a **green "
            "(positive)** or **red (negative)** point. Switch to **box** "
            "mode and click two opposite corners for a blue bbox.  \n"
            "**To separate a sub-part** (e.g. a fridge door from its body), "
            "leave `text` **blank** and use a box + positive points on the "
            "part and negative points on the rest — stage 10 then segments "
            "it purely from your clicks (an interactive/PVS prompt), so the "
            "negatives actually carve. Keep it to **≤8 positive and ≤8 "
            "negative** (the tracker drops points beyond 16).  \n"
            "For a whole object, just fill `text` (a concept prompt). Always "
            "fill `label`, then **Append to prompts.json**.  \n"
            f"_Output:_ `{prompts_path}`  •  "
            + (f"existing labels: `{', '.join(existing_labels)}`"
               if existing_labels else "no existing prompts.")
        )

        # ---- shared state ----
        cur_image_state = gr.State(init_rgb)         # base RGB image (numpy)
        cur_frame_state = gr.State(init_frame_idx)   # int, the actual frame idx
        pos_state = gr.State([])      # list of (x, y)
        neg_state = gr.State([])      # list of (x, y)
        box_state = gr.State(None)    # (x1, y1, x2, y2) or None
        pending_state = gr.State(None)  # pending box corner or None

        # ---- frame slider row ----
        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0, maximum=max(0, n_slots - 1), value=init_slot,
                step=1,
                label=f"frame slot (0..{n_slots - 1}) — "
                      f"{n_slots} frame(s) available"
            )
            with gr.Column(scale=1, min_width=120):
                prev_btn = gr.Button("◀ Prev")
                next_btn = gr.Button("Next ▶")
            with gr.Column(scale=2, min_width=180):
                frame_md = gr.Markdown(
                    f"**frame_index:** `{init_frame_idx:06d}`  "
                    f"(slot {init_slot}/{max(0, n_slots - 1)})"
                )

        # ---- main row ----
        with gr.Row():
            with gr.Column(scale=3):
                img_view = gr.Image(value=init_rgb, label="Click to pick",
                                    interactive=False, height=600)
                status_md = gr.Markdown("_Click on the image to add a point._")
            with gr.Column(scale=2):
                mode = gr.Radio(["positive point", "negative point", "box"],
                                value="positive point", label="Click mode")
                text_in = gr.Textbox(
                    "", label="text (concept prompt; leave BLANK for an "
                    "interactive part defined by points/box)")
                label_in = gr.Textbox(
                    "", label="label (filesystem-safe; no spaces or `/`)"
                )
                with gr.Row():
                    undo_btn = gr.Button("Undo last", variant="secondary")
                    clear_pts_btn = gr.Button("Clear points",
                                              variant="secondary")
                    clear_box_btn = gr.Button("Clear box",
                                              variant="secondary")
                clear_all_btn = gr.Button("Clear everything",
                                          variant="secondary")
                json_preview = gr.Code(language="json",
                                       label="prompt entry JSON")
                replace_cb = gr.Checkbox(
                    value=True,
                    label="replace existing entry with the same label"
                )
                append_btn = gr.Button("Append to prompts.json",
                                       variant="primary")
                save_status = gr.Markdown("")

        # ---- helpers used by callbacks ----

        def render(base, pos, neg, box, pending, text, label, frame_idx):
            overlay = draw_overlay(base, pos, neg, box, pending)
            status_lines = [
                f"frame `{frame_idx:06d}`  •  "
                f"**positive** ({len(pos)})  •  "
                f"**negative** ({len(neg)})  •  "
                f"**box**: " + (str(tuple(box)) if box else "—"),
            ]
            if pending is not None:
                status_lines.append(
                    f"_pending box corner_: `{pending}` — click once more "
                    f"to finalize"
                )
            entry = build_entry(text, label, frame_idx, pos, neg, box)
            return (overlay,
                    "  \n".join(status_lines),
                    json.dumps(entry, indent=2))

        def frame_label_md(slot, frame_idx):
            return (f"**frame_index:** `{frame_idx:06d}`  "
                    f"(slot {int(slot)}/{max(0, n_slots - 1)})")

        # ---- callbacks ----

        def on_frame_change(slot, text, label):
            """Switch to a new frame: reload the image, reset all picks."""
            slot = int(round(float(slot)))
            slot = max(0, min(n_slots - 1, slot))
            fi = frame_indices[slot]
            new_image = load_frame_rgb(frames_dir, fi)
            if new_image is None:
                # Should not happen — list_frame_indices already proved the
                # file exists at scan time. Fall back to a blank notice.
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(), gr.update(),
                        gr.update())
            overlay, status, j = render(new_image, [], [], None, None,
                                         text, label, fi)
            return (new_image, fi, overlay, [], [], None, None,
                    status, j, frame_label_md(slot, fi))

        def on_prev(slot, text, label):
            new_slot = max(0, int(round(float(slot))) - 1)
            return (new_slot, *on_frame_change(new_slot, text, label))

        def on_next(slot, text, label):
            new_slot = min(n_slots - 1, int(round(float(slot))) + 1)
            return (new_slot, *on_frame_change(new_slot, text, label))

        def on_click(mode_v, pos, neg, box, pending, text, label,
                     base_image, cur_fi, evt: gr.SelectData):
            x, y = int(evt.index[0]), int(evt.index[1])
            h, w = base_image.shape[:2]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            if mode_v == "positive point":
                pos = pos + [(x, y)]
            elif mode_v == "negative point":
                neg = neg + [(x, y)]
            else:  # box
                if pending is None:
                    pending = (x, y)
                else:
                    px, py = pending
                    box = (min(x, px), min(y, py), max(x, px), max(y, py))
                    pending = None
            overlay, status, j = render(base_image, pos, neg, box, pending,
                                         text, label, cur_fi)
            return overlay, pos, neg, box, pending, status, j

        def on_undo(mode_v, pos, neg, box, pending, text, label,
                    base_image, cur_fi):
            if mode_v == "positive point" and pos:
                pos = pos[:-1]
            elif mode_v == "negative point" and neg:
                neg = neg[:-1]
            else:
                if pending is not None:
                    pending = None
                else:
                    box = None
            overlay, status, j = render(base_image, pos, neg, box, pending,
                                         text, label, cur_fi)
            return overlay, pos, neg, box, pending, status, j

        def on_clear_pts(box, pending, text, label, base_image, cur_fi):
            overlay, status, j = render(base_image, [], [], box, pending,
                                         text, label, cur_fi)
            return overlay, [], [], status, j

        def on_clear_box(pos, neg, text, label, base_image, cur_fi):
            overlay, status, j = render(base_image, pos, neg, None, None,
                                         text, label, cur_fi)
            return overlay, None, None, status, j

        def on_clear_all(text, label, base_image, cur_fi):
            overlay, status, j = render(base_image, [], [], None, None,
                                         text, label, cur_fi)
            return overlay, [], [], None, None, status, j

        def on_text_label_change(text, label, pos, neg, box, pending,
                                 base_image, cur_fi):
            _, _, j = render(base_image, pos, neg, box, pending,
                             text, label, cur_fi)
            return j

        def on_append(text, label, pos, neg, box, replace_v, cur_fi):
            text = (text or "").strip()
            label = (label or "").strip()
            has_geom = bool(pos or neg or box is not None)
            if not label:
                return "**ERROR**: `label` is required (it becomes a folder name)."
            if " " in label or "/" in label:
                return ("**ERROR**: `label` must have no spaces and no `/` "
                        "(it becomes a folder name).")
            if not text and not has_geom:
                return ("**ERROR**: provide `text`, or pick points / a box for "
                        "an interactive (PVS) part prompt.")
            entry = build_entry(text, label, cur_fi, pos, neg, box)
            msg = append_entry(prompts_path, entry, replace_v)
            # Warn about the tracker's 16-point cap (box counts as 2 points).
            n_geom = len(pos) + len(neg) + (2 if box is not None else 0)
            if has_geom and n_geom > 16:
                msg += ("  \n⚠️ **{} geometry points** exceed the model cap of "
                        "16 (it keeps only the first 8 + last 8 — middle clicks "
                        "are dropped). Trim to ≤8 positive and ≤8 negative."
                        .format(n_geom))
            if has_geom and not text:
                msg += ("  \n_Interactive prompt: stage 10 segments this part "
                        "purely from the geometry (no text concept)._")
            return msg

        # ---- wiring ----

        # Slider drives a frame change on release (avoids reloading the image
        # for every intermediate value while scrubbing).
        frame_slider.release(
            on_frame_change,
            inputs=[frame_slider, text_in, label_in],
            outputs=[cur_image_state, cur_frame_state, img_view,
                     pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview, frame_md],
        )
        prev_btn.click(
            on_prev,
            inputs=[frame_slider, text_in, label_in],
            outputs=[frame_slider,
                     cur_image_state, cur_frame_state, img_view,
                     pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview, frame_md],
        )
        next_btn.click(
            on_next,
            inputs=[frame_slider, text_in, label_in],
            outputs=[frame_slider,
                     cur_image_state, cur_frame_state, img_view,
                     pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview, frame_md],
        )

        img_view.select(
            on_click,
            inputs=[mode, pos_state, neg_state, box_state, pending_state,
                    text_in, label_in, cur_image_state, cur_frame_state],
            outputs=[img_view, pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview],
        )
        undo_btn.click(
            on_undo,
            inputs=[mode, pos_state, neg_state, box_state, pending_state,
                    text_in, label_in, cur_image_state, cur_frame_state],
            outputs=[img_view, pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview],
        )
        clear_pts_btn.click(
            on_clear_pts,
            inputs=[box_state, pending_state, text_in, label_in,
                    cur_image_state, cur_frame_state],
            outputs=[img_view, pos_state, neg_state, status_md, json_preview],
        )
        clear_box_btn.click(
            on_clear_box,
            inputs=[pos_state, neg_state, text_in, label_in,
                    cur_image_state, cur_frame_state],
            outputs=[img_view, box_state, pending_state, status_md,
                     json_preview],
        )
        clear_all_btn.click(
            on_clear_all,
            inputs=[text_in, label_in, cur_image_state, cur_frame_state],
            outputs=[img_view, pos_state, neg_state, box_state, pending_state,
                     status_md, json_preview],
        )
        text_in.change(
            on_text_label_change,
            inputs=[text_in, label_in, pos_state, neg_state, box_state,
                    pending_state, cur_image_state, cur_frame_state],
            outputs=[json_preview],
        )
        label_in.change(
            on_text_label_change,
            inputs=[text_in, label_in, pos_state, neg_state, box_state,
                    pending_state, cur_image_state, cur_frame_state],
            outputs=[json_preview],
        )
        append_btn.click(
            on_append,
            inputs=[text_in, label_in, pos_state, neg_state, box_state,
                    replace_cb, cur_frame_state],
            outputs=[save_status],
        )

        # initial JSON preview
        def on_load(base_image, cur_fi):
            _, _, j = render(base_image, [], [], None, None, "", "", cur_fi)
            return j
        demo.load(on_load,
                  inputs=[cur_image_state, cur_frame_state],
                  outputs=[json_preview])

    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
