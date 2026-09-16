#!/usr/bin/env python
"""Stage 04: match SAM3D meshes to segmented DA3 depth and silhouette.

This stage resolves SAM3D's scale/depth ambiguity and then refines its image
size against the corresponding segmentation silhouette.  For every Stage-03
result it computes

* ``z_mesh_front``: the RDF-camera z of the posed mesh's front-surface
  center, and
* ``z_depth``: the mean valid DA3 z-depth inside the corresponding SAM masks.

The front surface is formed with a winding-independent camera z-buffer.  Dense
mesh vertices and triangle centers are projected with the DA3 intrinsics, and
only the nearest mesh depth at each object-mask pixel is retained.  Its center
therefore has one sample per visible image pixel, like the segmented depth-map
mean.  Hidden/back geometry is deliberately excluded from the estimate; it
follows the same final whole-mesh transform but cannot pull the reference
center backward.

Under the pinhole camera model, apparent size is proportional to ``f*s/z``.
The mesh is therefore moved along its current camera ray and uniformly resized
with one shared factor

    depth_scale = z_depth / z_mesh_front

by multiplying both the SAM3D translation and scale by ``depth_scale``.  This
makes the front-surface center depth equal ``z_depth`` while preserving every
projected mesh point.  The per-frame intrinsics in ``da3/intrinsics.npz`` are
used both for the front-surface z-buffer projection and to construct the center
ray.  The predicted rotation is never modified.

After that depth placement, the mesh is projected back into the keyframe and
a second uniform scale is optimized about the depth-corrected pose translation.
The objective is full-frame intersection-over-union (IoU) between the triangle
silhouette and the raw union of the masks listed in ``mask_labels.txt``.  A
coarse-to-fine one-dimensional search changes scale only; it never changes
rotation or independently shifts the mesh in x/y.  Finally, the newly scaled
mesh's front depth is measured again and one camera-origin similarity is
applied to both translation and scale.  That last update restores ``z_depth``
exactly without changing the optimized projection.

Inputs (candidate and legacy direct layouts are both supported)::

    <scene>/sam3d/<label>/[cand_NN_<frame>/]{mesh.glb,pose.json,keyframe.txt}
    <scene>/masks/<object-label>/<frame>.png
    <scene>/da3/depth/<frame>.npy
    <scene>/da3/intrinsics.npz

Outputs mirror Stage 03 under ``<scene>/sam3d_scaled`` by default.  The GLB is
copied unchanged and its corrected pose is saved as ``pose.json``; downstream
code should continue to apply that pose to ``mesh.glb``.  ``depth_scale.json``
records both corrections and before/after silhouette IoU for auditing, and
``projection_silhouette.png`` stores the final binary projection.

Example::

    python scripts/04_sacle_mesh.py --scene-dir data/drawer
    python scripts/04_sacle_mesh.py --scene-dir data/drawer \
        --labels combined --candidate-idx 0 --overwrite

Run in an environment containing numpy, Pillow, trimesh, and OpenCV.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


_CANDIDATE_RE = re.compile(r"cand_(\d+)_(\d+)")
_MODEL_FROM_GLB = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])
_PYTORCH3D_TO_RDF = np.diag([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class Reconstruction:
    label: str
    source_dir: Path
    frame: int
    candidate: int | None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--scene-dir", type=Path, required=True,
                        help="Scene directory, for example data/drawer")
    parser.add_argument("--input-name", default="sam3d",
                        help="Stage-03 input directory name within the scene")
    parser.add_argument("--out-name", default="sam3d_scaled",
                        help="Output directory name within the scene")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Only process these Stage-03 labels (for example combined)")
    parser.add_argument("--candidate-idx", type=int, default=None,
                        help="Only process this candidate index")
    parser.add_argument("--min-valid-pixels", type=int, default=50,
                        help="Minimum valid masked DA3 pixels required")
    parser.add_argument("--silhouette-min-scale", type=float, default=0.25,
                        help="Smallest post-depth uniform scale to search")
    parser.add_argument("--silhouette-max-scale", type=float, default=4.0,
                        help="Largest post-depth uniform scale to search")
    parser.add_argument("--silhouette-search-max-side", type=int, default=512,
                        help="Maximum mask side used by the coarse silhouette search")
    parser.add_argument("--copy-splat", action="store_true",
                        help="Also copy splat.ply when present")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing mirrored output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and report corrections without writing files")
    return parser.parse_args(argv)


def read_json(path: Path) -> dict:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def frame_from_dir(path: Path) -> int:
    keyframe_path = path / "keyframe.txt"
    if keyframe_path.is_file():
        try:
            frame = int(keyframe_path.read_text().strip())
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"invalid keyframe in {keyframe_path}") from exc
        if frame < 0:
            raise RuntimeError(f"negative keyframe in {keyframe_path}")
        return frame
    match = _CANDIDATE_RE.fullmatch(path.name)
    if match:
        return int(match.group(2))
    raise RuntimeError(f"cannot determine keyframe for {path}")


def discover_reconstructions(root: Path, labels, candidate_idx):
    if not root.is_dir():
        raise RuntimeError(f"missing Stage-03 directory: {root}")
    available = sorted(path.name for path in root.iterdir() if path.is_dir())
    if labels is not None:
        unknown = sorted(set(labels) - set(available))
        if unknown:
            raise RuntimeError(f"unknown Stage-03 label(s): {unknown}")
        selected = list(dict.fromkeys(labels))
    else:
        selected = available

    records = []
    for label in selected:
        label_dir = root / label
        has_candidate_layout = False
        candidates = []
        for path in sorted(label_dir.iterdir() if label_dir.is_dir() else []):
            match = _CANDIDATE_RE.fullmatch(path.name)
            if not path.is_dir() or match is None:
                continue
            if not ((path / "mesh.glb").is_file()
                    and (path / "pose.json").is_file()):
                continue
            has_candidate_layout = True
            candidate = int(match.group(1))
            if candidate_idx is not None and candidate != candidate_idx:
                continue
            candidates.append(Reconstruction(
                label=label, source_dir=path,
                frame=frame_from_dir(path), candidate=candidate,
            ))

        # A Stage-03 run with --candidate-idx writes directly in the label
        # directory.  Prefer candidate folders when present so a stale legacy
        # direct result is not processed a second time.
        if candidates:
            records.extend(candidates)
        elif (not has_candidate_layout
              and (label_dir / "mesh.glb").is_file()
              and (label_dir / "pose.json").is_file()):
            records.append(Reconstruction(
                label=label, source_dir=label_dir,
                frame=frame_from_dir(label_dir), candidate=None,
            ))
    return records


def load_intrinsics(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise RuntimeError(f"missing DA3 intrinsics: {path}")
    try:
        with np.load(path) as archive:
            required = {"frame_indices", "intrinsics"}
            if not required.issubset(archive.files):
                raise RuntimeError(f"invalid intrinsics archive: {path}")
            frames = np.asarray(archive["frame_indices"]).reshape(-1)
            matrices = np.asarray(archive["intrinsics"])
    except OSError as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if len(frames) != len(matrices):
        raise RuntimeError(f"frame/intrinsics row count differs in {path}")

    result = {}
    for frame, value in zip(frames, matrices):
        K = np.asarray(value, dtype=np.float64).reshape(3, 3)
        if (not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-12
                or K[0, 0] <= 0 or K[1, 1] <= 0):
            raise RuntimeError(f"invalid intrinsics for frame {int(frame):06d}")
        result[int(frame)] = K
    return result


def quat_wxyz_to_R(value) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("SAM3D rotation quaternion is zero or non-finite")
    w, x, y, z = quaternion / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=np.float64)


def parse_pose(pose: dict, path: Path):
    try:
        rotation = quat_wxyz_to_R(pose["rotation_quat_wxyz"])
        translation = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
        scale = np.asarray(pose["scale"], dtype=np.float64).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid SAM3D pose in {path}") from exc
    if scale.size not in (1, 3):
        raise RuntimeError(f"SAM3D scale must have one or three values in {path}")
    if not np.isfinite(translation).all():
        raise RuntimeError(f"non-finite SAM3D translation in {path}")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError(f"invalid SAM3D scale in {path}: {scale}")
    scale_xyz = np.repeat(scale, 3) if scale.size == 1 else scale.copy()
    return rotation, translation, scale, scale_xyz


def glb_to_rdf_camera(rotation, translation, scale_xyz) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        _PYTORCH3D_TO_RDF
        @ rotation.T
        @ np.diag(scale_xyz)
        @ _MODEL_FROM_GLB
    )
    transform[:3, 3] = _PYTORCH3D_TO_RDF @ translation
    return transform


def load_mesh_geometry(path: Path, trimesh) -> tuple[np.ndarray, np.ndarray]:
    try:
        asset = trimesh.load(path, force="scene", process=False)
        meshes = [geometry for geometry in asset.dump(concatenate=False)
                  if isinstance(geometry, trimesh.Trimesh)
                  and len(geometry.vertices) > 0
                  and len(geometry.faces) > 0]
    except Exception as exc:
        raise RuntimeError(f"could not load mesh {path}: {exc}") from exc
    if not meshes:
        raise RuntimeError(f"mesh has no triangle geometry: {path}")
    vertex_parts = []
    face_parts = []
    offset = 0
    for mesh in meshes:
        mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)
        mesh_faces = np.asarray(mesh.faces, dtype=np.int64)
        vertex_parts.append(mesh_vertices)
        face_parts.append(mesh_faces + offset)
        offset += len(mesh_vertices)
    vertices = np.concatenate(vertex_parts, axis=0)
    faces = np.concatenate(face_parts, axis=0)
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"mesh contains non-finite vertices: {path}")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise RuntimeError(f"mesh contains invalid face indices: {path}")
    return vertices, faces


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def front_surface_center(vertices: np.ndarray, faces: np.ndarray,
                         transform: np.ndarray, K: np.ndarray,
                         valid_mask: np.ndarray, chunk_size: int = 200000):
    """Return the visible front-surface center from a sampled camera z-buffer.

    Generated meshes can have inconsistent winding, so face normals are not a
    reliable visibility signal.  Vertices plus triangle centers provide dense,
    deterministic surface samples; the nearest sample at each projected pixel
    defines the visible side independent of winding.  A 2x2 pixel splat closes
    sub-pixel gaps between samples on the high-resolution generated meshes.
    """
    camera_vertices = transform_points(vertices, transform)
    if not np.isfinite(camera_vertices).all():
        raise RuntimeError("posed mesh contains non-finite vertices")
    full_center = camera_vertices.mean(axis=0, dtype=np.float64)
    if valid_mask.ndim != 2:
        raise RuntimeError(f"front-surface mask must be HxW, got {valid_mask.shape}")
    height, width = valid_mask.shape
    z_buffer = np.full(height * width, np.inf, dtype=np.float64)

    def add_samples(samples: np.ndarray):
        projected = samples @ K.T
        finite = np.isfinite(projected).all(axis=1) & (samples[:, 2] > 1e-8)
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-8)
        base_cols = np.floor(uv[:, 0]).astype(np.int64)
        base_rows = np.floor(uv[:, 1]).astype(np.int64)
        # Splat to the four surrounding pixels.  The nearest-depth reduction
        # still removes hidden/back samples at every covered pixel.
        for col_offset, row_offset in ((0, 0), (1, 0), (0, 1), (1, 1)):
            cols = base_cols + col_offset
            rows = base_rows + row_offset
            inside = (
                finite
                & (cols >= 0) & (cols < width)
                & (rows >= 0) & (rows < height)
            )
            if not inside.any():
                continue
            rows_inside = rows[inside]
            cols_inside = cols[inside]
            in_object = valid_mask[rows_inside, cols_inside]
            if not in_object.any():
                continue
            indices = rows_inside[in_object] * width + cols_inside[in_object]
            np.minimum.at(z_buffer, indices, samples[inside][in_object, 2])

    for start in range(0, len(camera_vertices), chunk_size):
        add_samples(camera_vertices[start:start + chunk_size])
    for start in range(0, len(faces), chunk_size):
        triangles = camera_vertices[faces[start:start + chunk_size]]
        add_samples(triangles.mean(axis=1, dtype=np.float64))

    z_buffer = z_buffer.reshape(height, width)
    visible = valid_mask & np.isfinite(z_buffer)
    visible_count = int(visible.sum())
    if visible_count == 0:
        raise RuntimeError("mesh has no visible front-surface samples in the mask")
    rows, cols = np.nonzero(visible)
    z = z_buffer[rows, cols]
    pixel_h = np.column_stack((
        cols.astype(np.float64), rows.astype(np.float64), np.ones(visible_count)
    ))
    rays = pixel_h @ np.linalg.inv(K).T
    rays /= rays[:, 2:3]
    front_points = rays * z[:, None]
    front_center = front_points.mean(axis=0, dtype=np.float64)
    if not np.isfinite(front_center).all() or front_center[2] <= 0:
        raise RuntimeError(
            f"front-surface center is not in front of the camera: {front_center}")
    info = {
        "surface_sample_count": int(len(vertices) + len(faces)),
        "visible_pixel_count": visible_count,
        "target_mask_pixel_count": int(valid_mask.sum()),
        "visible_mask_coverage": float(visible_count / valid_mask.sum()),
        "visible_depth_min": float(z.min()),
        "visible_depth_max": float(z.max()),
    }
    return camera_vertices, full_center, front_center, info


def load_binary_mask(path: Path, target_hw) -> np.ndarray:
    try:
        with Image.open(path) as image:
            value = np.asarray(image)
    except OSError as exc:
        raise RuntimeError(f"could not read mask {path}: {exc}") from exc
    if value.ndim == 3:
        value = value[..., -1]
    height, width = target_hw
    if value.shape != (height, width):
        value = np.asarray(Image.fromarray(value).resize(
            (width, height), Image.Resampling.NEAREST))
    return value > 0


def resize_binary_mask(mask: np.ndarray, target_hw) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbor sampling."""
    height, width = target_hw
    if mask.shape == (height, width):
        return mask.copy()
    value = Image.fromarray(mask.astype(np.uint8) * 255).resize(
        (width, height), Image.Resampling.NEAREST)
    return np.asarray(value) > 0


def silhouette_stats(silhouette: np.ndarray, target: np.ndarray) -> dict:
    if silhouette.shape != target.shape:
        raise RuntimeError(
            f"silhouette/target shapes differ: {silhouette.shape}, {target.shape}")
    predicted_pixels = int(silhouette.sum())
    target_pixels = int(target.sum())
    intersection = int(np.count_nonzero(silhouette & target))
    union = predicted_pixels + target_pixels - intersection
    if target_pixels == 0:
        raise RuntimeError("segmentation-mask union is empty")
    if union == 0:
        raise RuntimeError("silhouette and segmentation-mask union are empty")
    return {
        "projected_pixels": predicted_pixels,
        "target_pixels": target_pixels,
        "intersection_pixels": intersection,
        "union_pixels": union,
        "iou": float(intersection / union),
        "precision": (float(intersection / predicted_pixels)
                      if predicted_pixels else 0.0),
        "recall": float(intersection / target_pixels),
    }


def silhouette_raster_size(image_hw, max_side: int):
    height, width = image_hw
    scale = min(1.0, float(max_side) / max(height, width))
    raster_height = max(1, int(round(height * scale)))
    raster_width = max(1, int(round(width * scale)))
    return raster_height, raster_width


def scale_intrinsics(K: np.ndarray, source_hw, target_hw) -> np.ndarray:
    source_height, source_width = source_hw
    target_height, target_width = target_hw
    resize = np.diag([
        target_width / source_width,
        target_height / source_height,
        1.0,
    ])
    return resize @ K


def rasterize_sample_silhouette(points: np.ndarray, K: np.ndarray,
                                  target_hw) -> np.ndarray:
    """Fast dense-sample silhouette used only by the coarse scale search."""
    height, width = target_hw
    homogeneous = points @ K.T
    valid = np.isfinite(homogeneous).all(axis=1) & (points[:, 2] > 1e-8)
    uv = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    finite = np.isfinite(uv).all(axis=1)
    uv = uv[finite]
    silhouette = np.zeros((height, width), dtype=bool)
    if not len(uv):
        return silhouette
    base_cols = np.floor(uv[:, 0]).astype(np.int64)
    base_rows = np.floor(uv[:, 1]).astype(np.int64)
    for col_offset, row_offset in ((0, 0), (1, 0), (0, 1), (1, 1)):
        cols = base_cols + col_offset
        rows = base_rows + row_offset
        inside = (
            (cols >= 0) & (cols < width)
            & (rows >= 0) & (rows < height)
        )
        silhouette[rows[inside], cols[inside]] = True
    return silhouette


def rasterize_triangle_silhouette(vertices: np.ndarray, faces: np.ndarray,
                                   K: np.ndarray, target_hw, cv2,
                                   chunk_size: int = 100000) -> np.ndarray:
    """Rasterize the exact in-frame union of all front-of-camera triangles.

    OpenCV applies an even-odd rule when multiple polygons are passed to one
    ``fillPoly`` call, which can cancel overlapping front/back triangles.  Draw
    each convex triangle independently so the result is a true set union.
    """
    height, width = target_hw
    homogeneous = vertices @ K.T
    valid_vertex = (
        np.isfinite(homogeneous).all(axis=1) & (vertices[:, 2] > 1e-8))
    uv = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    uv[valid_vertex] = (
        homogeneous[valid_vertex, :2] / homogeneous[valid_vertex, 2:3])
    silhouette = np.zeros((height, width), dtype=np.uint8)
    subpixel_shift = 4
    subpixel_factor = 1 << subpixel_shift

    for start in range(0, len(faces), chunk_size):
        face_chunk = faces[start:start + chunk_size]
        usable = valid_vertex[face_chunk].all(axis=1)
        if not usable.any():
            continue
        triangles = uv[face_chunk[usable]]
        on_image = (
            (triangles[:, :, 0].max(axis=1) >= 0)
            & (triangles[:, :, 1].max(axis=1) >= 0)
            & (triangles[:, :, 0].min(axis=1) < width)
            & (triangles[:, :, 1].min(axis=1) < height)
        )
        triangles = triangles[on_image]
        if not len(triangles):
            continue
        contours = np.rint(
            np.clip(triangles, -100000.0, 100000.0) * subpixel_factor
        ).astype(np.int32)
        for contour in contours:
            cv2.fillConvexPoly(
                silhouette, contour, 1,
                lineType=cv2.LINE_8, shift=subpixel_shift)
    return silhouette > 0


def silhouette_search_samples(camera_vertices: np.ndarray,
                              faces: np.ndarray,
                              max_samples: int = 750000) -> np.ndarray:
    """Use every vertex and fill the remaining budget with face centers."""
    if len(camera_vertices) >= max_samples or not len(faces):
        return camera_vertices
    center_count = min(max_samples - len(camera_vertices), len(faces))
    face_indices = np.linspace(
        0, len(faces) - 1, center_count, dtype=np.int64)
    centers = camera_vertices[faces[face_indices]].mean(
        axis=1, dtype=np.float64)
    return np.concatenate((camera_vertices, centers), axis=0)


def safe_silhouette_scale_range(camera_vertices: np.ndarray,
                                camera_translation: np.ndarray,
                                requested_min: float,
                                requested_max: float):
    """Cap the scale range before any vertex can cross the camera plane."""
    translation_z = float(camera_translation[2])
    near = max(1e-8, translation_z * 1e-7)
    if translation_z <= near:
        raise RuntimeError(
            f"depth-corrected pose translation is behind the camera: "
            f"z={translation_z}")
    if np.any(camera_vertices[:, 2] <= near):
        count = int(np.count_nonzero(camera_vertices[:, 2] <= near))
        raise RuntimeError(
            f"depth-corrected mesh crosses the camera plane "
            f"({count} vertices have z <= {near:.6g})")

    delta_z = camera_vertices[:, 2] - translation_z
    toward_camera = delta_z < 0
    camera_limit = np.inf
    if toward_camera.any():
        limits = ((translation_z - near) / -delta_z[toward_camera])
        camera_limit = float(np.min(limits))
    effective_max = min(requested_max, camera_limit * (1.0 - 1e-7))
    if effective_max <= requested_min:
        raise RuntimeError(
            "silhouette scale range is empty after camera-plane clipping: "
            f"[{requested_min:.6g}, {effective_max:.6g}]")
    return requested_min, effective_max, camera_limit


def _best_silhouette_scale(results: dict[float, dict]) -> float:
    return max(
        results,
        key=lambda scale: (
            results[scale]["iou"],
            -abs(float(np.log(scale))),
        ),
    )


def _evaluate_scale_grid(values, render, target, cache):
    for value in values:
        scale = float(value)
        key = float(round(scale, 12))
        if key in cache:
            continue
        silhouette = render(scale)
        cache[key] = silhouette_stats(silhouette, target)


def _refine_scale_grid(render, target, lower: float, upper: float,
                       coarse_steps: int, refine_rounds: int):
    cache = {}
    initial = np.geomspace(lower, upper, coarse_steps)
    if lower <= 1.0 <= upper:
        initial = np.append(initial, 1.0)
    _evaluate_scale_grid(initial, render, target, cache)

    for _ in range(refine_rounds):
        ordered = sorted(cache)
        best = _best_silhouette_scale(cache)
        index = ordered.index(best)
        bracket_lower = ordered[max(0, index - 1)]
        bracket_upper = ordered[min(len(ordered) - 1, index + 1)]
        if bracket_lower == bracket_upper:
            break
        _evaluate_scale_grid(
            np.geomspace(bracket_lower, bracket_upper, 7),
            render, target, cache)
    return _best_silhouette_scale(cache), cache


def fit_silhouette_scale(camera_vertices: np.ndarray, faces: np.ndarray,
                         camera_translation: np.ndarray, K: np.ndarray,
                         target_mask: np.ndarray, requested_min: float,
                         requested_max: float, search_max_side: int, cv2):
    """Fit one uniform pose-scale multiplier to raw segmentation-mask IoU."""
    if not target_mask.any():
        raise RuntimeError("segmentation-mask union is empty")
    lower, upper, camera_limit = safe_silhouette_scale_range(
        camera_vertices, camera_translation, requested_min, requested_max)
    local_vertices = camera_vertices - camera_translation

    search_hw = silhouette_raster_size(target_mask.shape, search_max_side)
    search_K = scale_intrinsics(K, target_mask.shape, search_hw)
    search_target = resize_binary_mask(target_mask, search_hw)
    search_samples = silhouette_search_samples(camera_vertices, faces)
    local_samples = search_samples - camera_translation

    def render_search(scale):
        points = camera_translation + scale * local_samples
        return rasterize_sample_silhouette(points, search_K, search_hw)

    search_best, search_cache = _refine_scale_grid(
        render_search, search_target, lower, upper,
        coarse_steps=13, refine_rounds=3)

    def render_full(scale):
        vertices = camera_translation + scale * local_vertices
        return rasterize_triangle_silhouette(
            vertices, faces, K, target_mask.shape, cv2)

    # Reduced rasterization finds the global basin cheaply.  Refine within a
    # five-percent neighborhood using the actual full-resolution triangles.
    local_lower = max(lower, search_best * np.exp(-0.05))
    local_upper = min(upper, search_best * np.exp(0.05))
    full_cache = {}
    full_masks = {}

    def evaluate_full(values):
        for value in values:
            scale = float(value)
            key = float(round(scale, 12))
            if key in full_cache:
                continue
            silhouette = render_full(scale)
            full_masks[key] = silhouette
            full_cache[key] = silhouette_stats(silhouette, target_mask)

    evaluate_full(np.geomspace(local_lower, local_upper, 7))
    evaluate_full([search_best])
    full_best = _best_silhouette_scale(full_cache)
    ordered = sorted(full_cache)
    index = ordered.index(full_best)
    bracket_lower = ordered[max(0, index - 1)]
    bracket_upper = ordered[min(len(ordered) - 1, index + 1)]
    if bracket_lower != bracket_upper:
        evaluate_full(np.geomspace(bracket_lower, bracket_upper, 3))
        full_best = _best_silhouette_scale(full_cache)

    evaluate_full([1.0])
    baseline_stats = full_cache[1.0]
    # The fitted result must never be worse than keeping the original
    # projection, even if the reduced-resolution basin estimate was imperfect.
    full_cache[1.0] = baseline_stats
    full_best = _best_silhouette_scale(full_cache)
    final_silhouette = full_masks[full_best]
    final_stats = full_cache[full_best]
    tolerance = 1e-6
    info = {
        "scale_factor": float(full_best),
        "requested_scale_range": [float(requested_min), float(requested_max)],
        "effective_scale_range": [float(lower), float(upper)],
        "camera_plane_scale_limit": (
            float(camera_limit) if np.isfinite(camera_limit) else None),
        "search_hw": [int(search_hw[0]), int(search_hw[1])],
        "search_surface_samples": int(len(search_samples)),
        "search_evaluations": int(len(search_cache)),
        "coarse_best_scale_factor": float(search_best),
        "full_resolution_evaluations": int(len(full_cache)),
        "baseline": baseline_stats,
        "final": final_stats,
        "hit_lower_bound": bool(
            abs(np.log(full_best / lower)) <= tolerance),
        "hit_upper_bound": bool(
            abs(np.log(upper / full_best)) <= tolerance),
        "target_touches_image_border": bool(
            target_mask[0].any() or target_mask[-1].any()
            or target_mask[:, 0].any() or target_mask[:, -1].any()),
    }
    return float(full_best), final_silhouette, info


def mask_labels_for(record: Reconstruction):
    path = record.source_dir / "mask_labels.txt"
    if path.is_file():
        labels = [line.strip() for line in path.read_text().splitlines()
                  if line.strip()]
        if labels:
            return list(dict.fromkeys(labels))
    if record.label != "combined":
        return [record.label]
    raise RuntimeError(f"combined reconstruction has no mask labels: {path}")


def masked_average_depth(scene_dir: Path, record: Reconstruction,
                         min_valid_pixels: int):
    depth_path = scene_dir / "da3" / "depth" / f"{record.frame:06d}.npy"
    if not depth_path.is_file():
        raise RuntimeError(f"missing DA3 depth: {depth_path}")
    try:
        depth = np.asarray(np.load(depth_path), dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read {depth_path}: {exc}") from exc
    if depth.ndim != 2:
        raise RuntimeError(f"expected HxW depth in {depth_path}, got {depth.shape}")

    labels = mask_labels_for(record)
    combined = np.zeros(depth.shape, dtype=bool)
    mask_paths = []
    for label in labels:
        mask_path = (scene_dir / "masks" / label
                     / f"{record.frame:06d}.png")
        if not mask_path.is_file():
            raise RuntimeError(f"missing object mask: {mask_path}")
        combined |= load_binary_mask(mask_path, depth.shape)
        mask_paths.append(mask_path)

    valid = combined & np.isfinite(depth) & (depth > 0)
    da3_valid_path = (scene_dir / "da3" / "mask"
                      / f"{record.frame:06d}.png")
    if da3_valid_path.is_file():
        valid &= load_binary_mask(da3_valid_path, depth.shape)
    count = int(valid.sum())
    if count < min_valid_pixels:
        raise RuntimeError(
            f"only {count} valid masked depth pixels at {record.frame:06d}; "
            f"need at least {min_valid_pixels}"
        )
    average = float(np.mean(depth[valid], dtype=np.float64))
    if not np.isfinite(average) or average <= 0:
        raise RuntimeError(f"invalid masked average depth at {record.frame:06d}")
    return (average, count, labels, mask_paths, depth.shape, depth_path,
            da3_valid_path, valid, combined)


def project_point(point: np.ndarray, K: np.ndarray) -> np.ndarray:
    if not np.isfinite(point).all() or point[2] <= 0:
        raise RuntimeError(f"mesh center is not in front of the camera: {point}")
    homogeneous = K @ point
    if abs(homogeneous[2]) < 1e-12:
        raise RuntimeError("mesh center has an invalid camera projection")
    return homogeneous[:2] / homogeneous[2]


def camera_ray(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
    ray = np.linalg.solve(K, np.array([pixel[0], pixel[1], 1.0]))
    if not np.isfinite(ray).all() or abs(ray[2]) < 1e-12:
        raise RuntimeError("intrinsics produced an invalid mesh-center ray")
    return ray / ray[2]


def relative_path(path: Path, scene_dir: Path) -> str:
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def process_one(scene_dir: Path, input_root: Path, output_root: Path,
                record: Reconstruction, intrinsics, args, trimesh, cv2):
    pose_path = record.source_dir / "pose.json"
    mesh_path = record.source_dir / "mesh.glb"
    pose = read_json(pose_path)
    rotation, translation, original_scale, scale_xyz = parse_pose(pose, pose_path)
    transform = glb_to_rdf_camera(rotation, translation, scale_xyz)
    vertices, faces = load_mesh_geometry(mesh_path, trimesh)

    (target_depth, valid_count, mask_labels, mask_paths, depth_hw,
     depth_path, da3_valid_path, valid_depth_mask,
     segmentation_mask) = masked_average_depth(
        scene_dir, record, args.min_valid_pixels)
    if record.frame not in intrinsics:
        raise RuntimeError(
            f"frame {record.frame:06d} is absent from da3/intrinsics.npz")
    K = intrinsics[record.frame]

    (camera_vertices, mesh_center, front_center,
     front_info) = front_surface_center(
        vertices, faces, transform, K, valid_depth_mask)
    if front_info["visible_pixel_count"] < args.min_valid_pixels:
        raise RuntimeError(
            f"only {front_info['visible_pixel_count']} masked pixels have a "
            f"visible mesh sample; need at least {args.min_valid_pixels}")
    full_mesh_depth = float(mesh_center[2])
    front_depth = float(front_center[2])
    if not np.isfinite(full_mesh_depth) or full_mesh_depth <= 0:
        raise RuntimeError(f"full mesh center depth is not positive: {full_mesh_depth}")

    # Use only the visible/front-side center as the depth reference.  The back
    # half is never part of this statistic, but receives the same whole-mesh
    # correction.  At the target depth the front center remains on its current
    # ray, so every mesh projection stays unchanged.
    front_pixel = project_point(front_center, K)
    front_ray = camera_ray(front_pixel, K)
    focal = float(np.sqrt(K[0, 0] * K[1, 1]))
    pose_scale = float(np.cbrt(np.prod(scale_xyz)))
    projected_scale_proxy = focal * pose_scale / front_depth
    target_pose_scale = projected_scale_proxy * target_depth / focal
    depth_scale = target_pose_scale / pose_scale
    direct_ratio = target_depth / front_depth
    if not np.isclose(depth_scale, direct_ratio, rtol=1e-12, atol=1e-12):
        raise RuntimeError("inconsistent pinhole depth/scale correction")

    depth_corrected_translation = translation * depth_scale
    depth_corrected_scale = original_scale * depth_scale
    depth_corrected_scale_xyz = scale_xyz * depth_scale
    depth_corrected_transform = glb_to_rdf_camera(
        rotation, depth_corrected_translation, depth_corrected_scale_xyz)
    depth_corrected_vertices = transform_points(
        vertices, depth_corrected_transform)
    depth_corrected_mesh_center = depth_corrected_vertices.mean(
        axis=0, dtype=np.float64)
    # Uniformly scaling the complete camera-space mesh preserves the selected
    # faces and their projected weights, so the front center scales exactly.
    depth_corrected_front_center = front_center * depth_scale
    depth_corrected_front_pixel = project_point(
        depth_corrected_front_center, K)
    desired_front_center = front_ray * target_depth

    tolerance = max(1e-8, abs(target_depth) * 1e-8)
    if abs(depth_corrected_front_center[2] - target_depth) > tolerance:
        raise RuntimeError(
            "corrected front-surface center does not match target depth: "
            f"{depth_corrected_front_center[2]} vs {target_depth}")
    if not np.allclose(
            depth_corrected_front_center, desired_front_center,
            rtol=1e-8, atol=1e-8):
        raise RuntimeError("corrected front-surface center moved off its camera ray")
    if not np.allclose(
            depth_corrected_front_pixel, front_pixel, rtol=1e-8, atol=1e-8):
        raise RuntimeError("depth/scale correction changed the mesh projection")
    if not np.allclose(
            depth_corrected_vertices, camera_vertices * depth_scale,
            rtol=1e-8, atol=1e-8):
        raise RuntimeError("corrected whole-mesh transform is not uniformly scaled")

    # The first coupled update preserves the Stage-03 projection.  Now hold its
    # translation fixed and optimize only pose scale against the raw (not
    # DA3-valid-clipped) segmentation union.
    depth_corrected_camera_translation = (
        _PYTORCH3D_TO_RDF @ depth_corrected_translation)
    (silhouette_scale, final_silhouette,
     silhouette_info) = fit_silhouette_scale(
        depth_corrected_vertices, faces,
        depth_corrected_camera_translation, K, segmentation_mask,
        args.silhouette_min_scale, args.silhouette_max_scale,
        args.silhouette_search_max_side, cv2)

    silhouette_scaled_scale = depth_corrected_scale * silhouette_scale
    silhouette_scaled_scale_xyz = (
        depth_corrected_scale_xyz * silhouette_scale)
    silhouette_scaled_transform = glb_to_rdf_camera(
        rotation, depth_corrected_translation,
        silhouette_scaled_scale_xyz)
    (silhouette_scaled_vertices, silhouette_scaled_mesh_center,
     silhouette_scaled_front_center,
     silhouette_front_info) = front_surface_center(
        vertices, faces, silhouette_scaled_transform, K, valid_depth_mask)
    if silhouette_front_info["visible_pixel_count"] < args.min_valid_pixels:
        raise RuntimeError(
            f"only {silhouette_front_info['visible_pixel_count']} masked pixels "
            "have a visible sample after silhouette fitting; need at least "
            f"{args.min_valid_pixels}")

    expected_silhouette_vertices = (
        depth_corrected_camera_translation
        + silhouette_scale * (
            depth_corrected_vertices - depth_corrected_camera_translation))
    if not np.allclose(
            silhouette_scaled_vertices, expected_silhouette_vertices,
            rtol=1e-8, atol=1e-8):
        raise RuntimeError("silhouette correction did not scale about pose translation")

    # Scaling about the pose translation can perturb front depth.  Restore the
    # DA3 target by scaling the entire fitted camera-space result about the
    # camera origin.  This changes depth but provably preserves every pixel.
    silhouette_front_depth = float(silhouette_scaled_front_center[2])
    post_silhouette_depth_scale = target_depth / silhouette_front_depth
    corrected_translation = (
        depth_corrected_translation * post_silhouette_depth_scale)
    corrected_scale = silhouette_scaled_scale * post_silhouette_depth_scale
    corrected_scale_xyz = (
        silhouette_scaled_scale_xyz * post_silhouette_depth_scale)
    corrected_transform = glb_to_rdf_camera(
        rotation, corrected_translation, corrected_scale_xyz)
    corrected_vertices = transform_points(vertices, corrected_transform)
    corrected_mesh_center = corrected_vertices.mean(axis=0, dtype=np.float64)
    corrected_front_center = (
        silhouette_scaled_front_center * post_silhouette_depth_scale)
    corrected_front_pixel = project_point(corrected_front_center, K)
    silhouette_front_pixel = project_point(silhouette_scaled_front_center, K)

    if not np.allclose(
            corrected_vertices,
            silhouette_scaled_vertices * post_silhouette_depth_scale,
            rtol=1e-8, atol=1e-8):
        raise RuntimeError("final depth restoration is not a camera-origin scale")
    if abs(corrected_front_center[2] - target_depth) > tolerance:
        raise RuntimeError(
            "final front-surface center does not match target depth: "
            f"{corrected_front_center[2]} vs {target_depth}")
    if not np.allclose(
            corrected_front_pixel, silhouette_front_pixel,
            rtol=1e-8, atol=1e-8):
        raise RuntimeError("final depth restoration changed the fitted projection")

    corrected_pose = dict(pose)
    corrected_pose["translation"] = corrected_translation.tolist()
    corrected_pose["scale"] = corrected_scale.tolist()
    metadata = {
        "stage": "04_sacle_mesh",
        "method": (
            "visible_front_zbuffer_depth_then_raw_mask_iou_uniform_scale_"
            "then_projection_preserving_depth_restore"
        ),
        "registration_used": True,
        "registration_type": "one_parameter_uniform_scale_silhouette_iou",
        "source_mesh": relative_path(mesh_path, scene_dir),
        "source_pose": relative_path(pose_path, scene_dir),
        "keyframe": int(record.frame),
        "candidate": record.candidate,
        "mask_labels": mask_labels,
        "mask_sources": [relative_path(path, scene_dir) for path in mask_paths],
        "depth_source": relative_path(depth_path, scene_dir),
        "da3_valid_mask": (relative_path(da3_valid_path, scene_dir)
                           if da3_valid_path.is_file() else None),
        "depth_hw": [int(depth_hw[0]), int(depth_hw[1])],
        "valid_masked_depth_pixels": valid_count,
        "intrinsics": K.tolist(),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "front_surface_definition": (
            "nearest projected mesh sample per valid object-mask pixel; "
            "samples are vertices and triangle centers with a 2x2 pixel splat"
        ),
        "front_surface_sample_count": front_info["surface_sample_count"],
        "front_visible_pixel_count": front_info["visible_pixel_count"],
        "front_target_mask_pixel_count": front_info["target_mask_pixel_count"],
        "front_visible_mask_coverage": front_info["visible_mask_coverage"],
        "front_visible_depth_min": front_info["visible_depth_min"],
        "front_visible_depth_max": front_info["visible_depth_max"],
        "original_mesh_center_rdf_camera": mesh_center.tolist(),
        "original_average_mesh_depth": full_mesh_depth,
        "original_front_surface_center_rdf_camera": front_center.tolist(),
        "original_average_front_surface_depth": front_depth,
        "average_segmented_depth": target_depth,
        "depth_scale_factor": float(depth_scale),
        "silhouette_scale_factor": float(silhouette_scale),
        "post_silhouette_depth_scale_factor": float(
            post_silhouette_depth_scale),
        "total_translation_factor": float(
            depth_scale * post_silhouette_depth_scale),
        "total_pose_scale_factor": float(
            depth_scale * silhouette_scale
            * post_silhouette_depth_scale),
        "front_center_pixel_uv": front_pixel.tolist(),
        "front_center_camera_ray_z1": front_ray.tolist(),
        "depth_corrected_mesh_center_rdf_camera": (
            depth_corrected_mesh_center.tolist()
        ),
        "depth_corrected_front_surface_center_rdf_camera": (
            depth_corrected_front_center.tolist()
        ),
        "depth_corrected_front_center_pixel_uv": (
            depth_corrected_front_pixel.tolist()
        ),
        "silhouette_scaled_mesh_center_rdf_camera": (
            silhouette_scaled_mesh_center.tolist()
        ),
        "silhouette_scaled_front_surface_center_rdf_camera": (
            silhouette_scaled_front_center.tolist()
        ),
        "silhouette_scaled_average_front_surface_depth": (
            silhouette_front_depth
        ),
        "silhouette_scaled_front_visible_pixel_count": (
            silhouette_front_info["visible_pixel_count"]
        ),
        "silhouette_scaled_front_visible_mask_coverage": (
            silhouette_front_info["visible_mask_coverage"]
        ),
        "corrected_mesh_center_rdf_camera": corrected_mesh_center.tolist(),
        "corrected_front_surface_center_rdf_camera": (
            corrected_front_center.tolist()
        ),
        "corrected_front_center_pixel_uv": corrected_front_pixel.tolist(),
        "original_translation": translation.tolist(),
        "depth_corrected_translation": depth_corrected_translation.tolist(),
        "corrected_translation": corrected_translation.tolist(),
        "original_scale": original_scale.tolist(),
        "depth_corrected_scale": depth_corrected_scale.tolist(),
        "silhouette_scaled_scale": silhouette_scaled_scale.tolist(),
        "corrected_scale": corrected_scale.tolist(),
        "silhouette_target_definition": (
            "raw union of mask_labels.txt segmentation masks, without DA3 "
            "valid-depth clipping"
        ),
        "silhouette_rasterization": (
            "full-resolution union of projected triangles; dense surface "
            "samples are used only for coarse search"
        ),
        "silhouette_iou_before": silhouette_info["baseline"]["iou"],
        "silhouette_iou_after": silhouette_info["final"]["iou"],
        "silhouette_fit": silhouette_info,
        "projection_silhouette": "projection_silhouette.png",
        "rotation_unchanged": True,
        "depth_correction_projection_unchanged": True,
        "post_silhouette_depth_restore_projection_unchanged": True,
        "projection_unchanged": bool(np.isclose(
            silhouette_scale, 1.0, rtol=0.0, atol=1e-12)),
    }

    output_dir = output_root / record.source_dir.relative_to(input_root)
    if args.dry_run:
        return output_dir, metadata
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} exists; pass --overwrite to replace")
        if not output_dir.is_dir():
            raise RuntimeError(f"refusing to replace non-directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(mesh_path, output_dir / "mesh.glb")
    for name in ("keyframe.txt", "mask_labels.txt", "input_rgba.png"):
        source = record.source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    if args.copy_splat and (record.source_dir / "splat.ply").is_file():
        shutil.copy2(record.source_dir / "splat.ply", output_dir / "splat.ply")
    Image.fromarray(
        final_silhouette.astype(np.uint8) * 255
    ).save(output_dir / "projection_silhouette.png")
    with (output_dir / "pose.json").open("w") as handle:
        json.dump(corrected_pose, handle, indent=2)
        handle.write("\n")
    with (output_dir / "depth_scale.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return output_dir, metadata


def main(argv=None):
    args = parse_args(argv)
    if args.candidate_idx is not None and args.candidate_idx < 0:
        sys.exit("error: --candidate-idx must be non-negative")
    if args.min_valid_pixels <= 0:
        sys.exit("error: --min-valid-pixels must be positive")
    if (not np.isfinite(args.silhouette_min_scale)
            or args.silhouette_min_scale <= 0):
        sys.exit("error: --silhouette-min-scale must be finite and positive")
    if (not np.isfinite(args.silhouette_max_scale)
            or args.silhouette_max_scale <= args.silhouette_min_scale):
        sys.exit(
            "error: --silhouette-max-scale must be finite and greater than "
            "--silhouette-min-scale")
    if args.silhouette_search_max_side < 64:
        sys.exit("error: --silhouette-search-max-side must be at least 64")
    if not args.input_name or not args.out_name:
        sys.exit("error: --input-name and --out-name cannot be empty")
    if Path(args.input_name).name != args.input_name:
        sys.exit("error: --input-name must be one directory name")
    if Path(args.out_name).name != args.out_name:
        sys.exit("error: --out-name must be one directory name")
    if args.input_name == args.out_name:
        sys.exit("error: input and output names must differ (source is preserved)")

    try:
        import cv2
        import trimesh
    except ImportError:
        sys.exit(
            "error: trimesh and OpenCV are required; activate an environment "
            "that provides them")

    scene_dir = args.scene_dir.expanduser().resolve()
    input_root = scene_dir / args.input_name
    output_root = scene_dir / args.out_name
    try:
        records = discover_reconstructions(
            input_root, args.labels, args.candidate_idx)
        if not records:
            raise RuntimeError("no complete Stage-03 mesh/pose results selected")
        intrinsics = load_intrinsics(scene_dir / "da3" / "intrinsics.npz")
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    print(f"scene: {scene_dir.name}")
    print(f"selected reconstructions: {len(records)}")
    print("method: masked mean DA3 z + coupled depth/scale + raw-mask "
          "silhouette IoU scale fit + projection-preserving depth restore")
    if args.dry_run:
        print("dry run: no files will be written")

    succeeded = 0
    skipped = 0
    failed = 0
    for record in records:
        tag = f"[{record.label}] frame={record.frame:06d}"
        if record.candidate is not None:
            tag += f" cand={record.candidate:02d}"
        output_dir = output_root / record.source_dir.relative_to(input_root)
        if output_dir.exists() and not args.overwrite and not args.dry_run:
            print(f"{tag}: SKIP ({output_dir} exists; --overwrite to replace)")
            skipped += 1
            continue
        try:
            output_dir, metadata = process_one(
                scene_dir, input_root, output_root, record,
                intrinsics, args, trimesh, cv2)
            print(
                f"{tag}: "
                f"z_front={metadata['original_average_front_surface_depth']:.6g} "
                f"z_mask={metadata['average_segmented_depth']:.6g} "
                f"depth_scale={metadata['depth_scale_factor']:.6g} "
                f"silhouette_scale={metadata['silhouette_scale_factor']:.6g} "
                f"IoU={metadata['silhouette_iou_before']:.3f}"
                f"->{metadata['silhouette_iou_after']:.3f} "
                f"-> {output_dir}"
            )
            succeeded += 1
        except FileExistsError as exc:
            print(f"{tag}: SKIP ({exc})")
            skipped += 1
        except Exception as exc:
            print(f"{tag}: FAIL ({type(exc).__name__}: {exc})", file=sys.stderr)
            traceback.print_exc()
            failed += 1

    print(f"done: {succeeded} succeeded, {skipped} skipped, {failed} failed")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
