#!/usr/bin/env python
"""Stage 62b: encode Stage 62's Rerun layout into a single MP4.

The Rerun viewer has no built-in "export the viewport as video" command, so
this stage drives one for us:

1. Stage 62 logs the scene to a temporary ``.rrd``, with ``--hide-panels`` and
   ``--hide-view-titles`` forced on, so the recording's blueprint carries no
   top/blueprint/selection/time panel and no view title text.
2. ``rerun --headless`` loads that recording offscreen (no window, no X11
   grab), keeping its gRPC control server alive.
3. ``rerun viewer-mcp`` is driven over stdio JSON-RPC to step the ``frame``
   timeline one ID at a time and save a PNG of each rendered frame.
4. The viewer's own pane rectangles are read back from its widget tree, so
   every frame is cropped to just the view contents -- the thin per-view button
   strip above each pane is removed -- and the panes are repacked into one
   canvas that preserves Stage 62's layout.
5. The canvas frames are encoded to H.264 with ffmpeg, or to ``mp4v`` with
   OpenCV when ffmpeg is unavailable.

Every argument this stage does not recognize is forwarded verbatim to Stage
62, so the scene selection, joint styling, and pane switches are exactly the
ones documented there.

Examples::

    python scripts/62b_rerun_video.py --scene-dir data/trashbin
    python scripts/62b_rerun_video.py --scene-dir data/trashbin \
        --output render_all/trashbin.mp4 --window-size 2560x1440
    python scripts/62b_rerun_video.py --scene-dir data/trashbin \
        --mesh-frame last --no-depth --gutter 8
    python scripts/62b_rerun_video.py --scene-dir data/trashbin \
        --media-side bottom --scene-share 1.4 --rgb-overlays none

Needs ``rerun-sdk`` (for the ``rerun`` CLI its wheel installs), numpy, Pillow,
and either ffmpeg or OpenCV.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIEWER_READY_TIMEOUT = 120.0
SCREENSHOT_TIMEOUT = 120.0


def _load_stage62():
    """Load the numeric Stage-62 script as a module."""
    path = SCRIPT_DIR / "62_render_alll_rerun.py"
    spec = importlib.util.spec_from_file_location(
        "articulate4d_stage62_render_all_rerun", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Stage 62 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE62 = _load_stage62()


@dataclass(frozen=True)
class Rect:
    """One viewer rectangle, in the viewer's logical points."""

    x: float
    y: float
    w: float
    h: float


def _terminate(process: subprocess.Popen, timeout: float = 15.0):
    """Stop a `rerun` launcher and the viewer binary it forks.

    The `rerun` on PATH is a Python shim that execs the real CLI as a child,
    so signalling only the shim orphans a live viewer holding its gRPC port.
    Both are spawned into their own session, so the group signal reaches both.
    """
    if process.poll() is not None:
        return
    for send_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), send_signal)
        except (ProcessLookupError, PermissionError):
            try:
                process.kill()
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


class ViewerMCP:
    """Minimal JSON-RPC client for ``rerun viewer-mcp`` over stdio."""

    def __init__(self):
        self.process = subprocess.Popen(
            ["rerun", "viewer-mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            start_new_session=True,
        )
        self._ids = itertools.count(1)
        self.rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "articulate4d-stage62b", "version": "1"},
        })
        self._write({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _write(self, message: dict):
        if self.process.poll() is not None:
            raise RuntimeError("the rerun viewer-mcp process exited")
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def rpc(self, method: str, params: dict):
        message_id = next(self._ids)
        self._write({"jsonrpc": "2.0", "id": message_id,
                     "method": method, "params": params})
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("the rerun viewer-mcp process closed stdout")
            message = json.loads(line)
            if message.get("id") != message_id:
                continue
            if "error" in message:
                raise RuntimeError(f"viewer-mcp {method}: {message['error']}")
            return message["result"]

    def tool(self, name: str, arguments: dict | None = None):
        result = self.rpc("tools/call",
                          {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise RuntimeError(f"viewer-mcp {name}: {result}")
        return result

    def tool_json(self, name: str, arguments: dict | None = None):
        result = self.tool(name, arguments)
        structured = result.get("structuredContent")
        if structured is not None:
            return structured
        for block in result.get("content", []):
            if block.get("type") == "text":
                return json.loads(block["text"])
        raise RuntimeError(f"viewer-mcp {name} returned no readable payload")

    def close(self):
        _terminate(self.process)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("render_all/stage62_rerun.mp4"),
        help="MP4 path; relative paths resolve inside the scene directory",
    )
    parser.add_argument(
        "--window-size", default="1920x1080", metavar="WxH",
        help=("Offscreen viewer size, which sets the video resolution; the "
              "packed output is a little smaller once chrome is cropped"),
    )
    parser.add_argument(
        "--fps", type=float,
        help="Video FPS; defaults to the Stage 62 playback FPS",
    )
    parser.add_argument(
        "--gutter", type=int, default=0,
        help="Pixels of padding inserted between the packed panes",
    )
    parser.add_argument(
        "--pad-color", default="0,0,0", metavar="R,G,B",
        help="Fill color for gutters and for any short column's remainder",
    )
    parser.add_argument(
        "--crf", type=int, default=18,
        help="ffmpeg H.264 quality; lower is better, 18 is near-lossless",
    )
    parser.add_argument(
        "--encoder", choices=("auto", "ffmpeg", "opencv"), default="auto",
        help="Video encoder; 'auto' prefers ffmpeg and falls back to OpenCV",
    )
    parser.add_argument(
        "--viewer-port", type=int,
        help="gRPC port for the headless viewer; default picks a free one",
    )
    parser.add_argument(
        "--keep-frames", type=Path, metavar="DIR",
        help="Keep the packed PNG frames in DIR instead of a temporary tree",
    )
    parser.add_argument(
        "--keep-rrd", type=Path, metavar="PATH",
        help="Write the intermediate recording here and keep it",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing --output video",
    )
    return parser.parse_known_args(argv)


def _parse_window_size(text: str) -> tuple[int, int]:
    parts = text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"--window-size must look like 1920x1080, got {text!r}")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("--window-size values must be positive")
    return width, height


def _parse_color(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise ValueError(f"--pad-color must look like R,G,B, got {text!r}")
    channels = tuple(int(part) for part in parts)
    if not all(0 <= channel <= 255 for channel in channels):
        raise ValueError("--pad-color channels must be within 0..255")
    return channels


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _stage62_args(forwarded: list[str], rrd_path: Path):
    """Stage 62's own namespace, with the video-capture switches forced on."""
    for flag in ("--save-rrd", "--hide-panels", "--hide-view-titles"):
        if flag in forwarded:
            raise ValueError(f"{flag} is set by Stage 62b; drop it")
    return STAGE62.parse_args(forwarded + [
        "--hide-panels",
        "--hide-view-titles",
        "--save-rrd", str(rrd_path),
        "--overwrite",
    ])


def _spawn_viewer(rrd_path: Path, port: int, window: tuple[int, int]):
    width, height = window
    return subprocess.Popen(
        ["rerun", "--headless", "--hide-welcome-screen",
         "--port", str(port),
         "--window-size", f"{width}x{height}",
         str(rrd_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _connect_viewer(viewer, port: int) -> ViewerMCP:
    """Attach to the headless viewer once it serves the loaded recording."""
    client = ViewerMCP()
    endpoint = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + VIEWER_READY_TIMEOUT
    last_error = None
    while time.monotonic() < deadline:
        if viewer.poll() is not None:
            raise RuntimeError(
                f"the headless viewer exited with code {viewer.returncode}")
        try:
            client.tool("connect", {"endpoint": endpoint})
            state = client.tool_json("viewer_state")
            if state.get("recordings"):
                return client
            last_error = "viewer has no recording yet"
        except RuntimeError as exc:
            last_error = str(exc)
        time.sleep(0.5)
    client.close()
    raise RuntimeError(
        f"the headless viewer was not ready within {VIEWER_READY_TIMEOUT:.0f}s"
        f" ({last_error})")


def _pane_rects(client: ViewerMCP) -> list[Rect]:
    """Read the content rectangle of every open view from the widget tree."""
    nodes = client.tool_json("query_tree", {"limit": 500})["nodes"]
    rects = []
    for node in nodes:
        bounds = node.get("bounds")
        if node.get("role") != "Pane" or not bounds:
            continue
        rect = Rect(float(bounds["x"]), float(bounds["y"]),
                    float(bounds["w"]), float(bounds["h"]))
        if rect.w >= 1.0 and rect.h >= 1.0:
            rects.append(rect)
    if not rects:
        raise RuntimeError(
            "the viewer reported no view panes; the blueprint may be empty")
    # egui reports one Pane per view, but a parent node can repeat the same
    # rectangle, so keep the first of each geometry.
    unique = {}
    for rect in rects:
        unique.setdefault(
            (round(rect.x, 1), round(rect.y, 1),
             round(rect.w, 1), round(rect.h, 1)), rect)
    return sorted(unique.values(), key=lambda rect: (rect.x, rect.y))


def _split_groups(rects: list[Rect], axis: str) -> list[list[Rect]]:
    """Cut the rects at every full gap along one axis, keeping their order."""
    if axis == "x":
        low, high = (lambda r: r.x), (lambda r: r.x + r.w)
    else:
        low, high = (lambda r: r.y), (lambda r: r.y + r.h)
    ordered = sorted(rects, key=low)
    groups = [[ordered[0]]]
    edge = high(ordered[0])
    for rect in ordered[1:]:
        if low(rect) >= edge - 0.5:
            groups.append([rect])
            edge = high(rect)
        else:
            groups[-1].append(rect)
            edge = max(edge, high(rect))
    return groups


def _to_pixels(rect: Rect, scale: float) -> tuple[int, int, int, int]:
    return (int(round(rect.x * scale)), int(round(rect.y * scale)),
            int(round(rect.w * scale)), int(round(rect.h * scale)))


def _pack_panes(rects: list[Rect], gutter: int, scale: float):
    """Lay the panes out in pixel space with all viewer chrome removed.

    Rerun's viewport is a tree of horizontal and vertical splits, so the panes
    can always be recovered by repeatedly cutting the set at a gap that runs
    clean across it. Recursing on those cuts drops the button strip above every
    pane while preserving the nesting -- which keeps `--media-side bottom` and
    friends laid out the way the blueprint asked for, not just the default.
    """
    if len(rects) == 1:
        source = _to_pixels(rects[0], scale)
        return [(source, 0, 0)], source[2], source[3]

    for axis in ("x", "y"):
        groups = _split_groups(rects, axis)
        if len(groups) < 2:
            continue
        placements = []
        cursor = 0
        across = 0
        for group in groups:
            sub, width, height = _pack_panes(group, gutter, scale)
            for source, offset_x, offset_y in sub:
                placements.append(
                    (source, offset_x + cursor, offset_y) if axis == "x"
                    else (source, offset_x, offset_y + cursor))
            cursor += (width if axis == "x" else height) + gutter
            across = max(across, height if axis == "x" else width)
        along = cursor - gutter
        return ((placements, along, across) if axis == "x"
                else (placements, across, along))

    # Overlapping panes cannot be cut apart; keep their captured offsets.
    origin_x = min(rect.x for rect in rects)
    origin_y = min(rect.y for rect in rects)
    placements = []
    width = height = 0
    for rect in rects:
        source = _to_pixels(rect, scale)
        offset_x = int(round((rect.x - origin_x) * scale))
        offset_y = int(round((rect.y - origin_y) * scale))
        placements.append((source, offset_x, offset_y))
        width = max(width, offset_x + source[2])
        height = max(height, offset_y + source[3])
    return placements, width, height


def _canvas_size(width: int, height: int) -> tuple[int, int]:
    """H.264 in 4:2:0 needs even dimensions."""
    return width + width % 2, height + height % 2


def _compose(image: np.ndarray, placements, canvas_hw, pad_color):
    canvas = np.empty((canvas_hw[0], canvas_hw[1], 3), dtype=np.uint8)
    canvas[:, :] = np.asarray(pad_color, dtype=np.uint8)
    height, width = image.shape[:2]
    for (source_x, source_y, source_w, source_h), dest_x, dest_y in placements:
        x0, y0 = max(source_x, 0), max(source_y, 0)
        x1, y1 = min(source_x + source_w, width), min(source_y + source_h, height)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = image[y0:y1, x0:x1, :3]
        end_y = min(dest_y + patch.shape[0], canvas.shape[0])
        end_x = min(dest_x + patch.shape[1], canvas.shape[1])
        canvas[dest_y:end_y, dest_x:end_x] = patch[
            :end_y - dest_y, :end_x - dest_x]
    return canvas


def _read_png(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


def capture_frames(client: ViewerMCP, frame_ids, frames_dir: Path,
                   gutter: int, pad_color):
    """Step the frame timeline, screenshot each ID, and pack the panes."""
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_path = frames_dir / "_viewer.png"
    placements = canvas_hw = None
    written = []
    for index, frame_id in enumerate(frame_ids):
        client.tool("set_time", {"timeline": "frame", "time": int(frame_id)})
        # A headless viewer never supersamples -- it caps pixels_per_point at
        # its own scale of 1 -- so --window-size alone sets the output size.
        client.tool("screenshot", {"save_path": str(raw_path)})
        image = _read_png(raw_path)
        if placements is None:
            rects = _pane_rects(client)
            scale = image.shape[1] / float(
                max(rect.x + rect.w for rect in rects))
            placements, packed_w, packed_h = _pack_panes(
                rects, gutter, scale)
            canvas_w, canvas_h = _canvas_size(packed_w, packed_h)
            canvas_hw = (canvas_h, canvas_w)
            print(f"panes: {len(rects)}; captured "
                  f"{image.shape[1]}x{image.shape[0]}; "
                  f"video {canvas_w}x{canvas_h}")
        canvas = _compose(image, placements, canvas_hw, pad_color)
        frame_path = frames_dir / f"frame_{index:06d}.png"
        Image.fromarray(canvas).save(frame_path)
        written.append(frame_path)
        percent = int(100 * (index + 1) / len(frame_ids))
        print(f"\rcapture {index + 1}/{len(frame_ids)} ({percent}%)",
              end="", flush=True)
    print()
    raw_path.unlink(missing_ok=True)
    return written, canvas_hw


def encode_video(frames, output: Path, fps: float, encoder: str, crf: int):
    """Encode packed PNGs, preferring ffmpeg's H.264 over OpenCV's mp4v."""
    output.parent.mkdir(parents=True, exist_ok=True)
    use_ffmpeg = encoder == "ffmpeg" or (
        encoder == "auto" and shutil.which("ffmpeg") is not None)
    if encoder == "ffmpeg" and shutil.which("ffmpeg") is None:
        raise RuntimeError("--encoder ffmpeg was requested but ffmpeg is absent")
    if use_ffmpeg:
        # capture_frames() names the packed frames frame_000000.png upward, so
        # the image2 demuxer gives one video frame per capture exactly.
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-framerate", f"{fps}",
             "-i", str(frames[0].parent / "frame_%06d.png"),
             "-frames:v", str(len(frames)),
             "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
             "-pix_fmt", "yuv420p", str(output)],
            check=True,
        )
        return "libx264"

    import cv2

    first = _read_png(frames[0])
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps),
        (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open {output} for writing")
    try:
        for path in frames:
            writer.write(_read_png(path)[:, :, ::-1])
    finally:
        writer.release()
    return "mp4v"


def main(argv=None):
    args, forwarded = parse_args(argv)
    try:
        window = _parse_window_size(args.window_size)
        pad_color = _parse_color(args.pad_color)
        if args.gutter < 0:
            raise ValueError("--gutter cannot be negative")
        if shutil.which("rerun") is None:
            raise RuntimeError(
                "Stage 62b needs the 'rerun' CLI from rerun-sdk on PATH")

        workspace = Path(tempfile.mkdtemp(prefix="stage62b-"))
        viewer = client = None
        try:
            rrd_path = (args.keep_rrd.expanduser().resolve()
                        if args.keep_rrd else workspace / "stage62.rrd")
            stage62_args = _stage62_args(forwarded, rrd_path)
            prepared = STAGE62.prepare_rerun_scene(stage62_args)
            scene_dir = prepared.scene.scene_dir
            output = args.output.expanduser()
            output = (output.resolve() if output.is_absolute()
                      else (scene_dir / output).resolve())
            if output.suffix.lower() != ".mp4":
                raise ValueError("--output PATH must end in .mp4")
            if output.exists() and not args.overwrite:
                raise FileExistsError(
                    f"{output} exists; pass --overwrite to replace it")
            fps = float(args.fps or prepared.scene.playback_fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("--fps must be positive")

            STAGE62.print_summary(prepared, stage62_args)
            STAGE62.log_rerun_recording(prepared, stage62_args)

            port = args.viewer_port or _free_port()
            viewer = _spawn_viewer(rrd_path, port, window)
            client = _connect_viewer(viewer, port)
            frames_dir = (args.keep_frames.expanduser().resolve()
                          if args.keep_frames else workspace / "frames")
            frames, _ = capture_frames(
                client, [int(value) for value in prepared.scene.frame_ids],
                frames_dir, args.gutter, pad_color)
            codec = encode_video(frames, output, fps, args.encoder, args.crf)
            print(f"wrote {output} ({len(frames)} frames, {fps:g} fps, {codec})")
            if args.keep_rrd:
                print(f"kept recording: {rrd_path}")
            if args.keep_frames:
                print(f"kept frames: {frames_dir}")
        finally:
            if client is not None:
                client.close()
            if viewer is not None:
                _terminate(viewer)
            keep = args.keep_frames is not None or args.keep_rrd is not None
            shutil.rmtree(workspace, ignore_errors=not keep)
        return 0
    except (FileNotFoundError, FileExistsError, ImportError, RuntimeError,
            ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
