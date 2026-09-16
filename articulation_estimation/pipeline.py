"""Track-initialized joints refined against video through differentiable rendering."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .data import MeshPart, VideoScene
from .features import DINOFeatures
from .fitting import JointFit, fit_joint_hypotheses
from .geometry import (apply_rigid_torch, canonicalize_axis, so3_exp_torch,
                       so3_log_np, transform_matrix)
from .tracks import PointTrackSequence
from .rendering import (DifferentiableMeshRenderer, Observation, choose_render_size,
                        image_loss, resize_mask_and_boundary, sample_vertex_features)


@dataclass
class PipelineConfig:
    device: str = "cuda"
    max_render_side: int = 336
    keyframes: int = 16
    validation_fraction: float = 0.25
    dino_model: str = "dinov2_vitl14_reg"
    dino_checkpoint: Path | None = None
    feature_dimensions: int = 32
    use_features: bool = True
    pose_iters: int = 180
    track_iters: int = 180
    state_iters: int = 180
    joint_iters: int = 350
    global_iters: int = 120
    validation_iters: int = 100
    dense_iters: int = 60
    dense_states: bool = True
    batch_size: int = 4
    ransac_iterations: int = 64
    model_margin: float = 0.05
    lambda_feature: float = 1.0
    lambda_mask: float = 1.0
    lambda_boundary: float = 0.15
    lambda_smooth: float = 0.01
    lambda_shape: float = 0.1
    use_depth: bool = False
    lambda_depth: float = 0.25
    depth_huber_delta: float = 0.05
    joint_initializer: str = "tracks"
    use_track_prior: bool = False  # Compatibility alias for joint_initializer="tracks".
    seed: int = 0


@dataclass
class TrackedPose:
    frame_id: int
    transform: np.ndarray
    rotvec: np.ndarray
    translation: np.ndarray
    loss: float
    confidence: float = 1.0


@dataclass
class RefinedHypothesis:
    joint_type: str
    axis: np.ndarray
    pivot: np.ndarray | None
    states: dict[int, float]
    moving_vertices: object
    static_vertices: object
    train_loss: float
    validation_loss: float = np.inf
    validation_states: dict[int, float] | None = None


def _mask_descriptor(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return np.zeros(7, dtype=np.float64)
    cx, cy = xs.mean() / w, ys.mean() / h
    bw = (xs.max() - xs.min() + 1) / w
    bh = (ys.max() - ys.min() + 1) / h
    area = len(xs) / (h * w)
    x0, y0 = xs / w - cx, ys / h - cy
    return np.array([cx, cy, bw, bh, np.sqrt(area), np.mean(x0 * x0), np.mean(y0 * y0)])


def select_keyframes(scene: VideoScene, label: str, count: int,
                     frames: list[int] | None = None) -> list[int]:
    frames = scene.paired_frames(label) if frames is None else sorted(frames)
    if len(frames) <= count:
        return frames
    descriptors = np.stack([_mask_descriptor(scene.mask(label, frame)) for frame in frames])
    scale = descriptors.std(axis=0)
    descriptors = (descriptors - descriptors.mean(axis=0)) / np.where(scale > 1e-6, scale, 1.0)
    reference_pos = int(np.argmin(np.abs(np.asarray(frames) - scene.reference_frame)))
    chosen = [reference_pos, 0, len(frames) - 1]
    chosen = list(dict.fromkeys(chosen))
    while len(chosen) < count:
        distance = np.min(np.stack([
            np.linalg.norm(descriptors - descriptors[idx], axis=1) for idx in chosen
        ]), axis=0)
        distance[chosen] = -1.0
        chosen.append(int(np.argmax(distance)))
    return sorted(frames[idx] for idx in chosen)


def split_keyframes(keyframes: list[int], reference: int,
                    validation_fraction: float) -> tuple[list[int], list[int]]:
    if len(keyframes) < 5 or validation_fraction <= 0:
        return list(keyframes), []
    target = max(1, int(round(len(keyframes) * validation_fraction)))
    candidates = [frame for frame in keyframes if frame != reference]
    positions = np.linspace(0, len(candidates) - 1, target + 2)[1:-1]
    validation = sorted({candidates[int(round(pos))] for pos in positions})
    train = [frame for frame in keyframes if frame not in validation]
    if reference not in train:
        train.append(reference)
        train.sort()
    return train, validation


def _observation(scene: VideoScene, label: str, frame_id: int,
                 renderer: DifferentiableMeshRenderer, features=None,
                 use_depth: bool = False) -> Observation:
    import torch

    mask, boundary = resize_mask_and_boundary(scene.mask(label, frame_id),
                                              (renderer.h, renderer.w))
    depth_tensor = valid_tensor = None
    if use_depth:
        import cv2

        raw_depth = scene.depth(frame_id)
        resized = cv2.resize(raw_depth, (renderer.w, renderer.h),
                             interpolation=cv2.INTER_NEAREST)
        valid = np.isfinite(resized) & (resized > 0) & (mask.numpy() > 0.5)
        # Depth discontinuities and segmentation edges are the least reliable
        # pixels. Erode once before computing the robust depth residual.
        valid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=1) > 0
        clean_depth = np.where(valid, resized, 0.0).astype(np.float32)
        depth_tensor = torch.from_numpy(clean_depth)
        valid_tensor = torch.from_numpy(valid.astype(np.float32))
    return Observation(
        frame_id=frame_id,
        mask=mask.to(renderer.device),
        boundary_distance=boundary.to(renderer.device),
        camera=scene.camera(frame_id),
        features=None if features is None else features.to(renderer.device),
        depth=None if depth_tensor is None else depth_tensor.to(renderer.device),
        depth_valid=None if valid_tensor is None else valid_tensor.to(renderer.device),
    )


def _weights(config: PipelineConfig, features: bool = True) -> dict[str, float]:
    return {
        "mask": config.lambda_mask,
        "boundary": config.lambda_boundary,
        "feature": config.lambda_feature if features else 0.0,
        "depth": config.lambda_depth if config.use_depth else 0.0,
        "depth_delta": config.depth_huber_delta,
    }


def optimize_static_pose(scene: VideoScene, renderer: DifferentiableMeshRenderer,
                         parts: dict[str, MeshPart], keyframes: list[int],
                         config: PipelineConfig) -> tuple[dict[str, MeshPart], dict]:
    """Stage A: shared object pose and anisotropic scale from static-part masks."""
    import torch

    device = renderer.device
    static = parts[scene.static_label]
    vertices = torch.as_tensor(static.vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(static.faces, dtype=torch.int32, device=device)
    all_vertices = np.concatenate([part.vertices for part in parts.values()], axis=0)
    center = torch.as_tensor(all_vertices.mean(axis=0), dtype=torch.float32, device=device)
    observations = [_observation(scene, scene.static_label, frame, renderer,
                                 use_depth=config.use_depth)
                    for frame in keyframes]
    rotvec = torch.nn.Parameter(torch.zeros(3, device=device))
    translation = torch.nn.Parameter(torch.zeros(3, device=device))
    log_scale = torch.nn.Parameter(torch.zeros(3, device=device))
    optimizer = torch.optim.Adam([
        {"params": [rotvec], "lr": 2e-3},
        {"params": [translation], "lr": 2e-3},
        {"params": [log_scale], "lr": 2e-4},
    ])
    best = (np.inf, None)
    n_iters = max(0, int(config.pose_iters))
    for iteration in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        R = so3_exp_torch(rotvec)
        scaled = (vertices - center) * torch.exp(log_scale) + center
        posed = apply_rigid_torch(scaled, R, translation, center)
        batch = observations if len(observations) <= config.batch_size else [
            observations[(iteration * config.batch_size + j) % len(observations)]
            for j in range(config.batch_size)
        ]
        loss = torch.zeros((), device=device)
        for obs in batch:
            rendered = renderer.render(posed, faces, obs.camera,
                                       return_depth=obs.depth is not None)
            item, _ = image_loss(rendered, obs, _weights(config, features=False))
            loss = loss + item / len(batch)
        loss = loss + config.lambda_shape * log_scale.square().sum()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best[0]:
            best = (value, (rotvec.detach().clone(), translation.detach().clone(),
                            log_scale.detach().clone()))

    if best[1] is None:
        best = (np.nan, (rotvec.detach(), translation.detach(), log_scale.detach()))
    rot_best, trans_best, scale_best = best[1]
    R = so3_exp_torch(rot_best)
    baked: dict[str, MeshPart] = {}
    for label, part in parts.items():
        value = torch.as_tensor(part.vertices, dtype=torch.float32, device=device)
        value = (value - center) * torch.exp(scale_best) + center
        value = apply_rigid_torch(value, R, trans_best, center)
        baked[label] = MeshPart(label, value.detach().cpu().numpy(), part.faces.copy())
    return baked, {
        "loss": float(best[0]),
        "rotation_vector": rot_best.cpu().numpy().tolist(),
        "translation": trans_best.cpu().numpy().tolist(),
        "anisotropic_scale": torch.exp(scale_best).cpu().numpy().tolist(),
        "frames": [int(x) for x in keyframes],
    }


def prepare_features(scene: VideoScene, renderer: DifferentiableMeshRenderer,
                     keyframes: list[int], config: PipelineConfig):
    if not config.use_features:
        return None, {frame: None for frame in keyframes}
    extractor = DINOFeatures(config.dino_model, config.device,
                             (renderer.h, renderer.w), config.dino_checkpoint)
    raw = {frame: extractor.extract_patch_map(scene.image(frame)) for frame in keyframes}
    extractor.fit_pca(raw, config.feature_dimensions)
    return extractor, {frame: extractor.project(value) for frame, value in raw.items()}


def make_feature_mesh(scene: VideoScene, renderer: DifferentiableMeshRenderer,
                      part: MeshPart, label: str, reference_features):
    import torch

    vertices = torch.as_tensor(part.vertices, dtype=torch.float32, device=renderer.device)
    faces = torch.as_tensor(part.faces, dtype=torch.int32, device=renderer.device)
    if reference_features is None:
        return vertices, faces, None, None
    observed = _observation(scene, label, scene.reference_frame, renderer,
                            reference_features)
    attrs, valid = sample_vertex_features(renderer, vertices, faces, observed.camera,
                                          observed.features, observed.mask)
    return vertices, faces, attrs, valid


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros(2)
    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def _centroid_shift(scene: VideoScene, label: str, previous: int, target: int,
                    depth: float) -> np.ndarray:
    before = _mask_centroid(scene.mask(label, previous))
    after = _mask_centroid(scene.mask(label, target))
    delta = after - before
    camera = scene.camera(target)
    camera_shift = np.array([
        delta[0] * depth / camera.K[0, 0],
        delta[1] * depth / camera.K[1, 1],
        0.0,
    ])
    return camera.R_world_to_camera.T @ camera_shift


def track_sparse_poses(scene: VideoScene, label: str,
                       renderer: DifferentiableMeshRenderer, moving_mesh,
                       keyframes: list[int], feature_maps: dict[int, object],
                       config: PipelineConfig) -> dict[int, TrackedPose]:
    """Stage B: independent analysis-by-synthesis SE(3) tracking on keyframes."""
    import torch

    vertices, faces, attrs, attr_valid = moving_mesh
    center = vertices.mean(dim=0)
    reference = scene.reference_frame
    observations = {frame: _observation(
        scene, label, frame, renderer, feature_maps.get(frame),
        use_depth=config.use_depth) for frame in keyframes}
    order = [reference]
    order += sorted((frame for frame in keyframes if frame > reference))
    order += sorted((frame for frame in keyframes if frame < reference), reverse=True)
    tracked: dict[int, TrackedPose] = {
        reference: TrackedPose(reference, np.eye(4), np.zeros(3), np.zeros(3), 0.0)
    }

    for frame in order[1:]:
        candidates = [known for known in tracked if ((known < frame) == (reference < frame))]
        previous = min(candidates or list(tracked), key=lambda known: abs(known - frame))
        prior = tracked[previous]
        rot_init = prior.rotvec.copy()
        trans_init = prior.translation.copy()
        camera = scene.camera(frame)
        center_cam = camera.R_world_to_camera @ center.detach().cpu().numpy() + camera.t_world_to_camera
        trans_init += _centroid_shift(scene, label, previous, frame,
                                      max(float(center_cam[2]), 0.1))

        rotvec = torch.nn.Parameter(torch.as_tensor(rot_init, dtype=torch.float32,
                                                    device=renderer.device))
        translation = torch.nn.Parameter(torch.as_tensor(trans_init, dtype=torch.float32,
                                                         device=renderer.device))
        optimizer = torch.optim.Adam([
            {"params": [rotvec], "lr": 5e-3},
            {"params": [translation], "lr": 2e-3},
        ])
        best = (np.inf, None)
        obs = observations[frame]
        for _ in range(max(0, int(config.track_iters))):
            optimizer.zero_grad(set_to_none=True)
            R = so3_exp_torch(rotvec)
            posed = apply_rigid_torch(vertices, R, translation, center)
            rendered = renderer.render(
                posed, faces, obs.camera, attrs, attr_valid,
                return_depth=obs.depth is not None)
            loss, _ = image_loss(rendered, obs, _weights(config, attrs is not None))
            loss.backward()
            optimizer.step()
            value = float(loss.detach())
            if value < best[0]:
                best = (value, (rotvec.detach().clone(), translation.detach().clone()))
        if best[1] is None:
            best = (np.inf, (rotvec.detach(), translation.detach()))
        rv, tr = best[1]
        R_np = so3_exp_torch(rv).cpu().numpy()
        c_np, t_np = center.cpu().numpy(), tr.cpu().numpy()
        d_np = c_np + t_np - R_np @ c_np
        tracked[frame] = TrackedPose(
            frame, transform_matrix(R_np, d_np), rv.cpu().numpy(), t_np, float(best[0]))

    losses = np.array([tracked[frame].loss for frame in keyframes if frame != reference])
    finite = losses[np.isfinite(losses)]
    median = float(np.median(finite)) if len(finite) else 1.0
    mad = float(np.median(np.abs(finite - median))) if len(finite) else 1.0
    scale = max(1.4826 * mad, 1e-3)
    for frame in keyframes:
        if frame == reference:
            tracked[frame].confidence = 1.0
        else:
            tracked[frame].confidence = float(np.clip(
                np.exp(-max(tracked[frame].loss - median, 0.0) / scale), 0.05, 1.0))
    return tracked


def _joint_vertices(vertices, joint_type: str, axis, pivot, state):
    if joint_type == "prismatic":
        return vertices + state * axis
    R = so3_exp_torch(state * axis)
    return apply_rigid_torch(vertices, R, vertices.new_zeros(3), pivot)


def refine_hypothesis(fit: JointFit, frames: list[int], scene: VideoScene,
                      label: str, renderer: DifferentiableMeshRenderer,
                      moving_mesh, static_mesh, feature_maps: dict[int, object],
                      config: PipelineConfig) -> RefinedHypothesis:
    """Stage F: replace every free SE(3) by one shared explicit joint primitive."""
    import torch

    moving_v, moving_f, moving_attr, moving_valid = moving_mesh
    static_v, static_f, static_attr, static_valid = static_mesh
    center = torch.cat([moving_v, static_v], dim=0).mean(dim=0)
    observations_m = {frame: _observation(
        scene, label, frame, renderer, feature_maps.get(frame),
        use_depth=config.use_depth) for frame in frames}
    observations_s = {frame: _observation(
        scene, scene.static_label, frame, renderer, feature_maps.get(frame),
        use_depth=config.use_depth) for frame in frames}
    raw_axis = torch.nn.Parameter(torch.as_tensor(fit.axis, dtype=torch.float32,
                                                  device=renderer.device))
    raw_pivot = torch.nn.Parameter(torch.as_tensor(
        center.detach().cpu().numpy() if fit.pivot is None else fit.pivot,
        dtype=torch.float32, device=renderer.device))
    q = torch.nn.Parameter(torch.as_tensor(fit.states, dtype=torch.float32,
                                           device=renderer.device))
    common_rot = torch.nn.Parameter(torch.zeros(3, device=renderer.device))
    common_translation = torch.nn.Parameter(torch.zeros(3, device=renderer.device))
    common_log_scale = torch.nn.Parameter(torch.zeros((), device=renderer.device))
    reference_pos = frames.index(scene.reference_frame)
    with torch.no_grad():
        q[reference_pos] = 0.0

    # Evaluate the selected frame lazily so no KxV vertex batch is retained.
    def moving_at(index: int):
        axis = raw_axis / raw_axis.norm().clamp_min(1e-8)
        scale = torch.exp(common_log_scale)
        base = (moving_v - center) * scale + center
        delta = raw_pivot - center
        pivot = center + delta - axis * torch.dot(axis, delta)
        pivot = (pivot - center) * scale + center
        moved = _joint_vertices(base, fit.joint_type, axis, pivot, q[index])
        return apply_rigid_torch(moved, so3_exp_torch(common_rot),
                                 common_translation, center)

    def static_current():
        scale = torch.exp(common_log_scale)
        base = (static_v - center) * scale + center
        return apply_rigid_torch(base, so3_exp_torch(common_rot),
                                 common_translation, center)

    def objective(indices: list[int], include_static: bool):
        loss = torch.zeros((), device=renderer.device)
        for index in indices:
            frame = frames[index]
            rendered = renderer.render(moving_at(index), moving_f,
                                       observations_m[frame].camera,
                                       moving_attr, moving_valid,
                                       return_depth=observations_m[frame].depth is not None)
            item, _ = image_loss(rendered, observations_m[frame],
                                 _weights(config, moving_attr is not None))
            loss = loss + item / len(indices)
            if include_static:
                rendered_static = renderer.render(static_current(), static_f,
                                                   observations_s[frame].camera,
                                                   static_attr, static_valid,
                                                   return_depth=(
                                                       observations_s[frame].depth
                                                       is not None))
                static_item, _ = image_loss(
                    rendered_static, observations_s[frame],
                    _weights(config, static_attr is not None))
                loss = loss + 0.5 * static_item / len(indices)
        if len(q) >= 3:
            acceleration = q[2:] - 2.0 * q[1:-1] + q[:-2]
            loss = loss + config.lambda_smooth * acceleration.square().mean()
        loss = loss + config.lambda_shape * common_log_scale.square()
        return loss

    stages = [
        ("state", int(config.state_iters), [q], False),
        ("joint", int(config.joint_iters),
         [q, raw_axis] + ([] if fit.joint_type == "prismatic" else [raw_pivot]), False),
        ("global", int(config.global_iters),
         [q, raw_axis, common_rot, common_translation, common_log_scale] +
         ([] if fit.joint_type == "prismatic" else [raw_pivot]), True),
    ]
    generator = np.random.default_rng(config.seed + (0 if fit.joint_type == "prismatic" else 1))
    for stage_name, iterations, params, include_static in stages:
        if iterations <= 0:
            continue
        groups = []
        for parameter in params:
            if parameter is q:
                lr = 1e-2
            elif parameter is raw_axis or parameter is raw_pivot:
                lr = 2e-4
            elif parameter is common_log_scale:
                lr = 5e-5
            else:
                lr = 1e-4
            groups.append({"params": [parameter], "lr": lr})
        optimizer = torch.optim.Adam(groups)
        best = (np.inf, None)
        for iteration in range(iterations):
            if len(frames) <= config.batch_size:
                indices = list(range(len(frames)))
            else:
                indices = sorted(generator.choice(len(frames), config.batch_size,
                                                  replace=False).tolist())
            optimizer.zero_grad(set_to_none=True)
            loss = objective(indices, include_static)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                q[reference_pos] = 0.0
            value = float(loss.detach())
            if value < best[0]:
                best = (value, [parameter.detach().clone() for parameter in params])
        if best[1] is not None:
            with torch.no_grad():
                for parameter, value in zip(params, best[1]):
                    parameter.copy_(value)
                q[reference_pos] = 0.0

    with torch.no_grad():
        axis = raw_axis / raw_axis.norm().clamp_min(1e-8)
        scale = torch.exp(common_log_scale)
        common_R = so3_exp_torch(common_rot)
        base_moving = (moving_v - center) * scale + center
        base_static = (static_v - center) * scale + center
        base_moving = apply_rigid_torch(base_moving, common_R, common_translation, center)
        base_static = apply_rigid_torch(base_static, common_R, common_translation, center)
        axis_out = common_R @ axis
        if fit.joint_type == "revolute":
            delta = raw_pivot - center
            pivot = center + delta - axis * torch.dot(axis, delta)
            pivot = (pivot - center) * scale + center
            pivot_out = apply_rigid_torch(pivot, common_R, common_translation, center)
        else:
            pivot_out = None
        # q is applied after the optional common mesh scale in moving_at(), so
        # its physical value is already expressed in final world units.
        q_out = q
        axis_np, q_np = canonicalize_axis(axis_out.cpu().numpy(), q_out.cpu().numpy())
        pivot_np = None if pivot_out is None else pivot_out.cpu().numpy()
        train_loss = float(objective(list(range(len(frames))), include_static=False))
    return RefinedHypothesis(
        fit.joint_type, axis_np, pivot_np,
        {frame: float(value) for frame, value in zip(frames, q_np)},
        base_moving.detach(), base_static.detach(), train_loss)


def _apply_hypothesis(vertices, hypothesis: RefinedHypothesis, state):
    import torch

    axis = torch.as_tensor(hypothesis.axis, dtype=vertices.dtype, device=vertices.device)
    if hypothesis.joint_type == "prismatic":
        return vertices + state * axis
    pivot = torch.as_tensor(hypothesis.pivot, dtype=vertices.dtype, device=vertices.device)
    return _joint_vertices(vertices, "revolute", axis, pivot, state)


def _interpolate_state(states: dict[int, float], frame_id: int) -> float:
    frames = np.asarray(sorted(states), dtype=np.float64)
    values = np.asarray([states[int(frame)] for frame in frames], dtype=np.float64)
    return float(np.interp(frame_id, frames, values))


def optimize_one_state(hypothesis: RefinedHypothesis, frame_id: int, initial: float,
                       observation: Observation, renderer: DifferentiableMeshRenderer,
                       moving_faces, moving_attr, moving_valid,
                       config: PipelineConfig, iterations: int) -> tuple[float, float]:
    import torch

    state = torch.nn.Parameter(torch.tensor(float(initial), dtype=torch.float32,
                                            device=renderer.device))
    optimizer = torch.optim.Adam([state], lr=1e-2)
    best = (np.inf, float(initial))
    for _ in range(max(0, int(iterations))):
        optimizer.zero_grad(set_to_none=True)
        moved = _apply_hypothesis(hypothesis.moving_vertices, hypothesis, state)
        rendered = renderer.render(moved, moving_faces, observation.camera,
                                   moving_attr, moving_valid,
                                   return_depth=observation.depth is not None)
        loss, _ = image_loss(rendered, observation,
                             _weights(config, moving_attr is not None))
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best[0]:
            best = (value, float(state.detach()))
    if not np.isfinite(best[0]):
        with torch.no_grad():
            moved = _apply_hypothesis(hypothesis.moving_vertices, hypothesis, state)
            rendered = renderer.render(moved, moving_faces, observation.camera,
                                       moving_attr, moving_valid,
                                       return_depth=observation.depth is not None)
            loss, _ = image_loss(rendered, observation,
                                 _weights(config, moving_attr is not None))
        best = (float(loss), float(state.detach()))
    return best[1], best[0]


def validate_hypothesis(hypothesis: RefinedHypothesis, validation_frames: list[int],
                        scene: VideoScene, label: str,
                        renderer: DifferentiableMeshRenderer, moving_mesh,
                        feature_maps: dict[int, object], config: PipelineConfig):
    _, moving_faces, moving_attr, moving_valid = moving_mesh
    if not validation_frames:
        hypothesis.validation_loss = hypothesis.train_loss
        hypothesis.validation_states = {}
        return
    states, losses = {}, []
    for frame in validation_frames:
        observation = _observation(
            scene, label, frame, renderer, feature_maps.get(frame),
            use_depth=config.use_depth)
        initial = _interpolate_state(hypothesis.states, frame)
        state, loss = optimize_one_state(
            hypothesis, frame, initial, observation, renderer, moving_faces,
            moving_attr, moving_valid, config, config.validation_iters)
        states[frame] = state
        losses.append(loss)
    hypothesis.validation_states = states
    hypothesis.validation_loss = float(np.median(losses))
    hypothesis.states.update(states)


def recover_dense_states(hypothesis: RefinedHypothesis, frames: list[int],
                         scene: VideoScene, label: str,
                         renderer: DifferentiableMeshRenderer, moving_mesh,
                         extractor: DINOFeatures | None, config: PipelineConfig) -> dict[int, float]:
    _, moving_faces, moving_attr, moving_valid = moving_mesh
    states = dict(hypothesis.states)
    if not config.dense_states:
        return dict(sorted(states.items()))
    for frame in frames:
        if frame == scene.reference_frame:
            states[frame] = 0.0
            continue
        features = None
        if extractor is not None:
            features = extractor.project(extractor.extract_patch_map(scene.image(frame)))
        observation = _observation(scene, label, frame, renderer, features,
                                   use_depth=config.use_depth)
        initial = _interpolate_state(states, frame)
        state, _ = optimize_one_state(
            hypothesis, frame, initial, observation, renderer, moving_faces,
            moving_attr, moving_valid, config, config.dense_iters)
        states[frame] = state

    ordered = sorted(frames)
    values = np.asarray([states[frame] for frame in ordered], dtype=np.float64)
    if len(values) >= 3 and config.lambda_smooth > 0:
        D = np.zeros((len(values) - 2, len(values)), dtype=np.float64)
        for i in range(len(values) - 2):
            D[i, i:i + 3] = [1.0, -2.0, 1.0]
        strength = 10.0 * config.lambda_smooth
        values = np.linalg.solve(np.eye(len(values)) + strength * (D.T @ D), values)
    if scene.reference_frame in ordered:
        values -= values[ordered.index(scene.reference_frame)]
    return {frame: float(value) for frame, value in zip(ordered, values)}


def estimate_label(scene: VideoScene, label: str, config: PipelineConfig,
                   progress: Callable[[str], None] = print) -> dict:
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    frames = scene.paired_frames(label)
    if scene.reference_frame not in frames:
        raise ValueError(f"moving label {label} has no mask at canonical frame "
                         f"{scene.reference_frame}")
    initializer = "tracks" if config.use_track_prior else config.joint_initializer
    if initializer not in {"tracks", "video"}:
        raise ValueError(f"unknown joint initializer: {initializer}")
    if config.keyframes < 3 or not 0 <= config.validation_fraction < 1:
        raise ValueError("need keyframes >= 3 and 0 <= validation_fraction < 1")
    track_sequence = (PointTrackSequence(scene.root, label, scene.reference_frame)
                      if initializer == "tracks" else None)
    candidate_frames = (sorted(set(frames) & set(track_sequence.available_frames))
                        if track_sequence is not None else frames)
    keyframes = select_keyframes(scene, label, config.keyframes, candidate_frames)
    train_frames, validation_frames = split_keyframes(
        keyframes, scene.reference_frame, config.validation_fraction)
    if len(train_frames) < 3:
        raise ValueError("joint initialization needs at least 3 training keyframes; "
                         "extend track coverage, increase --keyframes, or reduce "
                         "--validation-fraction")
    # Validate and fit track poses before allocating the GPU renderer. Validation
    # trajectories never enter joint fitting; dense frames may lie outside tracks.
    track_poses = (track_sequence.estimate_poses(train_frames)
                   if track_sequence is not None else None)
    image_hw = scene.image(scene.reference_frame).shape[:2]
    render_hw = choose_render_size(image_hw, config.max_render_side)
    renderer = DifferentiableMeshRenderer(render_hw, image_hw, config.device)
    progress(f"  keyframes={keyframes}  train={train_frames}  val={validation_frames}")
    progress(f"  render={render_hw[1]}x{render_hw[0]}  frame-index-offset={scene.index_offset}")

    used_parts = {scene.static_label: scene.parts[scene.static_label], label: scene.parts[label]}
    baked, pose_diag = optimize_static_pose(scene, renderer, used_parts,
                                            train_frames, config)
    progress(f"  static pose loss={pose_diag['loss']:.5f}")
    extractor, feature_maps = prepare_features(scene, renderer, keyframes, config)
    if extractor is not None:
        progress(f"  DINO={config.dino_model}  PCA={config.feature_dimensions}D")

    moving_mesh = make_feature_mesh(scene, renderer, baked[label], label,
                                    feature_maps[scene.reference_frame])
    static_mesh = make_feature_mesh(scene, renderer, baked[scene.static_label],
                                    scene.static_label, feature_maps[scene.reference_frame])
    fit_frames = train_frames
    moving_vertices = baked[label].vertices
    center = moving_vertices.mean(axis=0)
    diameter = float(np.linalg.norm(moving_vertices.max(axis=0) - moving_vertices.min(axis=0)))
    track_diagnostics = None
    if track_poses is not None:
        tracked = {}
        for frame, pose in track_poses.items():
            rotation, translation = pose.transform[:3, :3], pose.transform[:3, 3]
            tracked[frame] = TrackedPose(
                frame, pose.transform, so3_log_np(rotation),
                translation - (np.eye(3) - rotation) @ center,
                pose.normalized_rms, pose.confidence)
        center, diameter = track_sequence.centroid, track_sequence.diameter
        track_diagnostics = {
            "path": str(track_sequence.root.relative_to(scene.root)),
            "track_reference_frame": track_sequence.track_reference_frame,
            "mesh_reference_frame": scene.reference_frame,
            "fit_frames": [int(frame) for frame in fit_frames],
            "coordinate_frame": "DA3 world (Stage-08 output)",
            "frame_index_convention": "image filename stems",
            "residual": "trimmed point-registration RMS / reference bounding-box diagonal",
            "frames": [{"frame": int(frame), "valid_points": pose.valid_points,
                        "inlier_points": pose.inlier_points,
                        "normalized_rms": pose.normalized_rms}
                       for frame, pose in track_poses.items()],
        }
    else:
        tracked = track_sparse_poses(scene, label, renderer, moving_mesh, keyframes,
                                     feature_maps, config)
    progress(f"  joint initializer={initializer}: " + ", ".join(
        f"{frame}:{tracked[frame].loss:.3f}" for frame in sorted(tracked)))
    transforms = np.stack([tracked[frame].transform for frame in fit_frames])
    confidences = np.asarray([tracked[frame].confidence for frame in fit_frames])
    fits = fit_joint_hypotheses(
        transforms, center, diameter, reference_index=fit_frames.index(scene.reference_frame),
        confidences=confidences, ransac_iterations=config.ransac_iterations,
        seed=config.seed)
    hypotheses = {}
    for joint_type in ("prismatic", "revolute"):
        progress(f"  {joint_type} pose-fit median={fits[joint_type].residual:.5f}")
        hypothesis = refine_hypothesis(
            fits[joint_type], fit_frames, scene, label, renderer,
            moving_mesh, static_mesh, feature_maps, config)
        validate_hypothesis(hypothesis, validation_frames, scene, label,
                            renderer, moving_mesh, feature_maps, config)
        hypotheses[joint_type] = hypothesis
        progress(f"  {joint_type} video loss: train={hypothesis.train_loss:.5f} "
                 f"validation={hypothesis.validation_loss:.5f}")

    selected_type = min(hypotheses, key=lambda kind: hypotheses[kind].validation_loss)
    selected = hypotheses[selected_type]
    other_type = "revolute" if selected_type == "prismatic" else "prismatic"
    denominator = max(selected.validation_loss, 1e-8)
    margin = abs(hypotheses[other_type].validation_loss - selected.validation_loss) / denominator
    low_confidence = margin < config.model_margin
    dense = recover_dense_states(selected, frames, scene, label, renderer,
                                 moving_mesh, extractor, config)
    axis, dense_values = canonicalize_axis(
        selected.axis, np.asarray([dense[frame] for frame in sorted(dense)]))
    dense = {frame: float(value) for frame, value in zip(sorted(dense), dense_values)}
    final_vertices = selected.moving_vertices.detach().cpu().numpy()
    final_static_vertices = selected.static_vertices.detach().cpu().numpy()

    candidates = {}
    for kind, hypothesis in hypotheses.items():
        candidates[kind] = {
            "axis_direction": hypothesis.axis.tolist(),
            "axis_point": None if hypothesis.pivot is None else hypothesis.pivot.tolist(),
            "pose_fit_residual": float(fits[kind].residual),
            "initial_joint": fits[kind].to_dict(fit_frames),
            "train_video_loss": float(hypothesis.train_loss),
            "validation_video_loss": float(hypothesis.validation_loss),
            "validation_states": {str(k): float(v) for k, v in
                                  (hypothesis.validation_states or {}).items()},
        }
    measurement = ["mask"]
    if config.use_features:
        measurement.insert(0, "DINOv2")
    if config.use_depth:
        measurement.append("DA3-depth")
    return {
        "version": 3,
        "method": "video_dino_mesh_joint_refinement",
        "measurement": "+".join(measurement),
        "joint_initializer": initializer,
        "track_initialization": track_diagnostics,
        "sparse_se3_loss_kind": ("normalized_track_rms" if initializer == "tracks"
                                 else "image_loss"),
        "type": selected_type,
        "type_confidence": "low" if low_confidence else "high",
        "model_selection_margin": float(margin),
        "model_selection_margin_required": float(config.model_margin),
        "axis_direction": axis.tolist(),
        "axis_point": None if selected.pivot is None else selected.pivot.tolist(),
        "state_units": ("radians" if selected_type == "revolute"
                        else "initializer world units"),
        "metric_scale_observable": False,
        "joint_states": [
            {"frame": int(frame), "q": float(value)} for frame, value in dense.items()
        ],
        "label": label,
        "static_label": scene.static_label,
        "coordinate_frame": "DA3 world axes with initializer-provided object scale",
        "part_centroid_world": final_vertices.mean(axis=0).tolist(),
        "part_bbox_diagonal": float(np.linalg.norm(
            final_vertices.max(axis=0) - final_vertices.min(axis=0))),
        "canonical_frame": int(scene.reference_frame),
        "keyframes": [int(x) for x in keyframes],
        "train_keyframes": [int(x) for x in train_frames],
        "validation_keyframes": [int(x) for x in validation_frames],
        "model_candidates": candidates,
        "static_pose_refinement": pose_diag,
        "sparse_se3": [
            {
                "frame": int(frame),
                "transform": tracked[frame].transform.tolist(),
                "loss": float(tracked[frame].loss),
                "confidence": float(tracked[frame].confidence),
            }
            for frame in sorted(tracked)
        ],
        "inputs": {
            "rgb": "frames/",
            "masks": [f"masks/{scene.static_label}/", f"masks/{label}/"],
            "meshes": [f"segvigen/combined/pieces/{scene.static_label}.glb",
                       f"segvigen/combined/pieces/{label}.glb"],
            "pose_initializer_kind": scene.pose_initializer_kind,
            "pose_initializer": scene.pose_source,
            "track_prior_enabled": initializer == "tracks",
            "tracks_used_in_refinement": False,
            "depth_used": bool(config.use_depth),
            "depth_loss_weight": float(config.lambda_depth) if config.use_depth else 0.0,
            "depth_huber_delta": (float(config.depth_huber_delta)
                                   if config.use_depth else None),
        },
        # Consumed by write_results() and deliberately omitted from JSON. These
        # q=0 meshes let joint-motion overlays replay the fitted articulation exactly.
        "_mesh_payload": {
            "moving_vertices": final_vertices.astype(np.float32),
            "moving_faces": moving_mesh[1].detach().cpu().numpy().astype(np.int32),
            "static_vertices": final_static_vertices.astype(np.float32),
            "static_faces": static_mesh[1].detach().cpu().numpy().astype(np.int32),
        },
    }


def compact_result(result: dict) -> dict:
    """Drop heavy diagnostics from the scene-level summary, retaining final states."""
    return {key: value for key, value in result.items()
            if key not in {"sparse_se3", "static_pose_refinement"}
            and not key.startswith("_")}


def write_results(scene: VideoScene, results: dict[str, dict], dry_run: bool = False):
    if dry_run:
        return
    output_root = scene.root / "joints"
    output_root.mkdir(parents=True, exist_ok=True)
    for label, result in results.items():
        label_root = output_root / label
        label_root.mkdir(parents=True, exist_ok=True)
        public_result = {key: value for key, value in result.items()
                         if not key.startswith("_")}
        payload = result.get("_mesh_payload")
        if payload is not None:
            import trimesh

            moving_path = label_root / "moving_mesh.glb"
            static_path = label_root / "static_mesh.glb"
            trimesh.Trimesh(
                vertices=payload["moving_vertices"], faces=payload["moving_faces"],
                process=False).export(str(moving_path))
            trimesh.Trimesh(
                vertices=payload["static_vertices"], faces=payload["static_faces"],
                process=False).export(str(static_path))
            public_result["canonical_meshes"] = {
                "moving": str(moving_path.relative_to(scene.root)),
                "static": str(static_path.relative_to(scene.root)),
                "joint_state": 0.0,
            }
        (label_root / "joint.json").write_text(
            json.dumps(public_result, indent=2) + "\n")
        # Keep the scene summary consistent with the per-label sidecar.
        result.clear()
        result.update(public_result)
    summary = {label: compact_result(result) for label, result in results.items()}
    (output_root / "joints.json").write_text(json.dumps(summary, indent=2) + "\n")
