"""Scene discovery and IO for video-driven joint estimation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .geometry import quat_wxyz_to_matrix, quat_xyzw_to_matrix


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
STATIC_NAME_HINTS = ("body", "base", "basis", "frame", "static", "cabinet")


@dataclass(frozen=True)
class Camera:
    K: np.ndarray
    R_world_to_camera: np.ndarray
    t_world_to_camera: np.ndarray


@dataclass
class MeshPart:
    label: str
    vertices: np.ndarray
    faces: np.ndarray


def _numeric_files(folder: Path, suffixes: Iterable[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not folder.is_dir():
        return out
    allowed = {s.lower() for s in suffixes}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in allowed:
            try:
                out[int(path.stem)] = path
            except ValueError:
                pass
    return out


def _load_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    # force="mesh" applies GLB scene-graph node transforms before concatenation.
    loaded = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path} contains no triangle mesh")
    return (np.asarray(loaded.vertices, dtype=np.float64).copy(),
            np.asarray(loaded.faces, dtype=np.int64).copy())


def _fit_similarity_from_correspondences(source: np.ndarray, target: np.ndarray,
                                         max_points: int = 30_000):
    """Fit ``target = source @ linear + translation`` with a proper similarity."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Any6D source/world meshes do not have matching vertices")
    if len(source) < 4:
        raise ValueError("Any6D meshes contain too few vertices")
    if len(source) > max_points:
        indices = np.linspace(0, len(source) - 1, max_points, dtype=np.int64)
        source, target = source[indices], target[indices]
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    unconstrained, *_ = np.linalg.lstsq(source_centered, target_centered, rcond=None)
    U, singular, Vt = np.linalg.svd(unconstrained)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1] = -1.0
    scale = float(np.dot(singular, correction) / 3.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Any6D mesh correspondence produced an invalid scale")
    linear = scale * (U @ np.diag(correction) @ Vt)
    translation = target_mean - source_mean @ linear
    residual = np.sqrt(np.mean(np.sum(
        (source @ linear + translation - target) ** 2, axis=1)))
    target_diag = np.linalg.norm(target.max(axis=0) - target.min(axis=0))
    if residual > max(1e-5, 1e-3 * target_diag):
        raise ValueError(
            f"Any6D mesh correspondence is inconsistent (RMS={residual:.4g})"
        )
    return linear, translation, float(residual)


def _infer_index_offset(disk_ids: set[int], record_ids: set[int]) -> int:
    """Infer the filename-to-camera index offset (the current data are 1-based on disk)."""
    if not disk_ids or not record_ids:
        return 0
    candidates = {0, min(disk_ids) - min(record_ids), max(disk_ids) - max(record_ids)}
    best = max(candidates, key=lambda off: sum((i - off) in record_ids for i in disk_ids))
    return int(best)


class VideoScene:
    """Validated meshes, images, masks, and cameras; tracks are loaded separately."""

    def __init__(self, scene_dir: Path, static_label: str | None,
                 moving_labels: list[str] | None, candidate: int = 0,
                 use_any6d_pose: bool = False):
        self.root = Path(scene_dir).resolve()
        self.frames = _numeric_files(self.root / "frames", IMAGE_SUFFIXES)
        self.mask_root = self.root / "masks"
        self.pieces_root = self.root / "segvigen" / "combined" / "pieces"
        self.sam_root = self._find_candidate(self.root / "sam3d" / "combined", candidate)
        if not self.frames:
            raise FileNotFoundError(f"no numeric RGB frames under {self.root / 'frames'}")
        for need in (self.mask_root, self.pieces_root, self.sam_root / "mesh.glb",
                     self.sam_root / "pose.json"):
            if not need.exists():
                raise FileNotFoundError(f"missing {need}")

        piece_labels = {p.stem for p in self.pieces_root.glob("*.glb")}
        mask_labels = {p.name for p in self.mask_root.iterdir() if p.is_dir()}
        self.labels = sorted(piece_labels & mask_labels)
        if len(self.labels) < 2:
            raise ValueError("need at least two labels present in both masks/ and "
                             "segvigen/combined/pieces/")
        self.static_label, self.moving_labels = self._resolve_labels(static_label, moving_labels)

        self.intr_path, self.cameras_path = self._camera_paths()
        intr = np.load(self.intr_path)
        self.intr_frame_ids = [int(x) for x in intr["frame_indices"].tolist()]
        self.intrinsics = np.asarray(intr["intrinsics"], dtype=np.float64)
        self._intr_pos = {fid: i for i, fid in enumerate(self.intr_frame_ids)}
        self.index_offset = _infer_index_offset(set(self.frames), set(self.intr_frame_ids))

        self._cameras = np.load(self.cameras_path) if self.cameras_path else None
        if self._cameras is not None:
            cam_ids = [int(x) for x in self._cameras["frame_indices"].tolist()]
            self._cam_pos = {fid: i for i, fid in enumerate(cam_ids)}
        else:
            self._cam_pos = {}

        self.reference_frame = self._read_reference_frame()
        self.pose_initializer_kind = "sam3d"
        self.pose_source = str((self.sam_root / "pose.json").relative_to(self.root))
        self._any6d_similarity = None
        if use_any6d_pose:
            self._any6d_similarity = self._load_any6d_similarity()
            self.pose_initializer_kind = "any6d"
        self.parts = {label: self._load_world_part(label) for label in self.labels}

    @staticmethod
    def _find_candidate(root: Path, candidate: int) -> Path:
        if (root / "mesh.glb").is_file():
            return root
        matches = sorted(root.glob(f"cand_{candidate:02d}_*")) or sorted(root.glob("cand_*"))
        for path in matches:
            if (path / "mesh.glb").is_file():
                return path
        return root

    def _resolve_labels(self, static_label: str | None,
                        moving_labels: list[str] | None) -> tuple[str, list[str]]:
        if static_label is None:
            hinted = [label for label in self.labels
                      if any(hint in label.lower().split("_") for hint in STATIC_NAME_HINTS)]
            if len(hinted) == 1:
                static_label = hinted[0]
            elif len(self.labels) == 2:
                scores = [sum(h in label.lower() for h in STATIC_NAME_HINTS)
                          for label in self.labels]
                static_label = self.labels[int(np.argmax(scores))]
            else:
                raise ValueError("cannot infer the static part; pass --static-label")
        if static_label not in self.labels:
            raise ValueError(f"static label {static_label!r} is not one of {self.labels}")
        if moving_labels is None:
            moving_labels = [label for label in self.labels if label != static_label]
        unknown = [label for label in moving_labels if label not in self.labels]
        if unknown:
            raise ValueError(f"moving labels not present in masks and pieces: {unknown}")
        if static_label in moving_labels:
            raise ValueError("the static label cannot also be a moving label")
        if not moving_labels:
            raise ValueError("no moving labels selected")
        return static_label, list(moving_labels)

    def _camera_paths(self) -> tuple[Path, Path | None]:
        intrinsics = self.root / "da3" / "intrinsics.npz"
        cameras = self.root / "da3" / "cameras.npz"
        if not intrinsics.is_file():
            raise FileNotFoundError(f"missing {intrinsics}; run stage 00 first")
        return intrinsics, cameras if cameras.is_file() else None

    def _read_reference_frame(self) -> int:
        keyframe_file = self.sam_root / "keyframe.txt"
        if keyframe_file.is_file():
            desired = int(keyframe_file.read_text().strip())
        else:
            tracking = self.mask_root / "tracking.json"
            records = json.loads(tracking.read_text()) if tracking.is_file() else {}
            desired = int(records.get(self.static_label, {}).get("keyframe", min(self.frames)))
        common = self.available_frames(self.static_label)
        if not common:
            raise ValueError(f"no usable frames for static label {self.static_label}")
        return min(common, key=lambda frame_id: abs(frame_id - desired))

    def record_id(self, frame_id: int) -> int:
        record = int(frame_id) - self.index_offset
        if record not in self._intr_pos:
            raise KeyError(f"frame {frame_id} has no matching camera/intrinsics record")
        return record

    def camera(self, frame_id: int) -> Camera:
        record = self.record_id(frame_id)
        K = self.intrinsics[self._intr_pos[record]].copy()
        if self._cameras is None or record not in self._cam_pos:
            return Camera(K, np.eye(3), np.zeros(3))
        i = self._cam_pos[record]
        R_cw = quat_xyzw_to_matrix(self._cameras["cam_quats_xyzw"][i])
        t_cw = np.asarray(self._cameras["cam_trans"][i], dtype=np.float64)
        R_wc = R_cw.T
        return Camera(K, R_wc, -R_wc @ t_cw)

    def camera_to_world(self, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
        camera = self.camera(frame_id)
        R_cw = camera.R_world_to_camera.T
        return R_cw, -R_cw @ camera.t_world_to_camera

    def available_frames(self, label: str) -> list[int]:
        masks = _numeric_files(self.mask_root / label, (".png", ".jpg", ".jpeg"))
        return sorted(fid for fid in set(self.frames) & set(masks)
                      if (fid - self.index_offset) in self._intr_pos)

    def paired_frames(self, moving_label: str) -> list[int]:
        return sorted(set(self.available_frames(self.static_label)) &
                      set(self.available_frames(moving_label)))

    def image(self, frame_id: int) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.frames[frame_id]).convert("RGB"))

    def mask(self, label: str, frame_id: int) -> np.ndarray:
        from PIL import Image

        paths = _numeric_files(self.mask_root / label, (".png", ".jpg", ".jpeg"))
        if frame_id not in paths:
            raise FileNotFoundError(f"missing mask for {label} frame {frame_id}")
        mask = np.asarray(Image.open(paths[frame_id]))
        if mask.ndim == 3:
            mask = mask[..., -1]
        return mask > 0

    def depth(self, frame_id: int) -> np.ndarray:
        path = self.root / "da3" / "depth" / f"{int(frame_id):06d}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing depth map {path}")
        depth = np.asarray(np.load(path), dtype=np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = np.nan
        return depth

    def _load_any6d_similarity(self):
        """Recover the canonical-SAM3D to Any6D-world similarity from saved meshes."""
        any6d_root = self.root / "any6d" / "combined"
        pose_path = any6d_root / "pose.json"
        world_path = any6d_root / "mesh_world.glb"
        for need in (pose_path, world_path):
            if not need.is_file():
                raise FileNotFoundError(
                    f"missing {need}; run legacy Any6D pose estimation for the combined object first"
                )
        record = json.loads(pose_path.read_text())
        pose_frame = int(record.get("keyframe", -1))
        if pose_frame != self.reference_frame:
            raise ValueError(
                f"Any6D keyframe {pose_frame} does not match Stage-09 canonical "
                f"frame {self.reference_frame}"
            )
        source_rel = record.get("mesh_source")
        if not source_rel:
            raise ValueError(f"{pose_path} has no mesh_source")
        source_path = self.root / source_rel
        if not source_path.is_file():
            raise FileNotFoundError(f"Any6D source mesh is missing: {source_path}")
        source_vertices, _ = _load_trimesh(source_path)
        world_vertices, _ = _load_trimesh(world_path)
        linear, translation, residual = _fit_similarity_from_correspondences(
            source_vertices, world_vertices)
        self.pose_source = str(pose_path.relative_to(self.root))
        return linear, translation, residual

    def _load_world_part(self, label: str) -> MeshPart:
        vertices, faces = _load_trimesh(self.pieces_root / f"{label}.glb")
        if self._any6d_similarity is not None:
            linear, translation, _ = self._any6d_similarity
            vertices_world = vertices @ linear + translation
            return MeshPart(label, vertices_world.astype(np.float32),
                            faces.astype(np.int32))
        pose = json.loads((self.sam_root / "pose.json").read_text())
        R = quat_wxyz_to_matrix(pose["rotation_quat_wxyz"])
        t = np.asarray(pose["translation"], dtype=np.float64)
        scale = float(np.asarray(pose["scale"]).reshape(-1)[0])
        vertices_cam = scale * (vertices @ R.T) + t
        # SAM3D uses PyTorch3D camera axes; convert to RDF with a proper 180° Z rotation.
        vertices_cam[:, :2] *= -1.0
        R_cw, t_cw = self.camera_to_world(self.reference_frame)
        vertices_world = vertices_cam @ R_cw.T + t_cw
        return MeshPart(label, vertices_world.astype(np.float32), faces.astype(np.int32))
