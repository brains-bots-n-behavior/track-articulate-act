#!/usr/bin/env python
"""Stage 10: estimate one-DoF joints from TrackCraft3R point tracks.

For every requested moving label, this script reads the reference points and
frame-specific scene flow written by ``08_trackcraft_flow.py``.  It first fits
a robust rigid transform from the reference points to each tracked frame, then
fits both prismatic and revolute motion directly to the point trajectories.
The automatic classifier selects a revolute joint only when its point residual
is sufficiently better *and* the recovered rigid poses contain an observable
rotation.  This prevents a far-away hinge from explaining pure translation.

Reads::

    <scene>/trackcraft/config.json
    <scene>/trackcraft/<label>/pts3d_ref.npy
    <scene>/trackcraft/<label>/scene_flow/<frame>.npy
    <scene>/masks/tracking.json                    # moving-label selection

Writes::

    <scene>/pointtrack_joint/joints.json
    <scene>/pointtrack_joint/<label>/joint.json

All joint parameters are expressed in the same DA3 world coordinate system as
the Stage-08 points.  Prismatic state is in world units; revolute state is in
radians.  A prismatic axis has no observable position, so ``axis_point`` is
null and ``position`` is only a display anchor at the reference centroid.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path

import numpy as np


@dataclass
class TrackObservation:
    """One tracked point cloud and its robust reference-to-frame pose."""

    frame: int
    target: np.ndarray
    valid: np.ndarray
    transform: np.ndarray
    inliers: np.ndarray
    rigid_inlier_rms: float
    rigid_all_rms: float
    median_flow: float


@dataclass
class ModelFit:
    """One optimized joint hypothesis."""

    joint_type: str
    axis: np.ndarray
    pivot: np.ndarray | None
    states: np.ndarray
    frame_rms: np.ndarray
    residual_rms: float
    normalized_residual_rms: float
    solver_success: bool
    solver_nfev: int


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--scene-dir", type=Path, required=True,
        help="Scene containing trackcraft/ output from Stage 08",
    )
    parser.add_argument(
        "--moving-labels", "--labels", nargs="*", default=None,
        help=("Moving labels to fit (default: labels marked moving in "
              "masks/tracking.json)"),
    )
    parser.add_argument(
        "--joint-type", choices=("auto", "prismatic", "revolute"),
        default="auto", help="Automatically classify or force one joint family",
    )
    parser.add_argument(
        "--min-rotation-deg", type=float, default=8.0,
        help="Minimum observed pose rotation required to select revolute",
    )
    parser.add_argument(
        "--revolute-residual-ratio", type=float, default=0.80,
        help=("In auto mode, revolute residual must be at most this fraction "
              "of the prismatic residual"),
    )
    parser.add_argument(
        "--min-motion", type=float, default=0.0,
        help="Ignore target frames below this median point displacement",
    )
    parser.add_argument(
        "--trim-quantile", type=float, default=0.80,
        help="Fraction of lowest-residual tracks retained in robust fits",
    )
    parser.add_argument(
        "--rigid-iterations", type=int, default=5,
        help="Trimmed rigid-pose fitting iterations per frame",
    )
    parser.add_argument(
        "--max-fit-points", type=int, default=1000,
        help="Maximum common tracks used for nonlinear joint refinement",
    )
    parser.add_argument(
        "--fit-iterations", type=int, default=300,
        help="Maximum nonlinear evaluations for each joint hypothesis",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: <scene>/pointtrack_joint)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate and print without writing JSON")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output JSON files")
    return parser.parse_args(argv)


def read_json(path: Path):
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return value


def display_path(path: Path, scene_dir: Path):
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def safe_name(label: str):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def normalize(vector, message="vector is degenerate"):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError(message)
    return vector / norm


def skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def so3_exp(rotation_vector):
    """Stable Rodrigues exponential for a three-vector."""
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotation_vector))
    if theta < 1e-9:
        matrix = skew(rotation_vector)
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    matrix = skew(rotation_vector / theta)
    return (np.eye(3) + np.sin(theta) * matrix
            + (1.0 - np.cos(theta)) * matrix @ matrix)


def rotation_angle(rotation):
    cosine = np.clip(
        (np.trace(np.asarray(rotation, dtype=np.float64)) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.arccos(cosine))


def so3_log(rotation):
    """Stable logarithm of a proper 3-D rotation."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    theta = rotation_angle(rotation)
    vee = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ])
    if theta < 1e-7:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        values, vectors = np.linalg.eig(rotation)
        axis = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
        return theta * normalize(axis, "could not recover a near-pi rotation axis")
    return theta * vee / (2.0 * np.sin(theta))


def rigid_transform(source, target, valid, trim_quantile=0.8, iterations=5):
    """Fit target = R source + t with iterative fixed-fraction trimming."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    finite = (np.asarray(valid, dtype=bool)
              & np.isfinite(source).all(axis=1)
              & np.isfinite(target).all(axis=1))
    available = np.flatnonzero(finite)
    if len(available) < 6:
        raise RuntimeError(
            f"rigid pose needs at least 6 finite tracks; found {len(available)}")
    keep_count = max(6, int(np.ceil(trim_quantile * len(available))))
    active = available.copy()
    rotation, translation = np.eye(3), np.zeros(3)
    for _ in range(iterations):
        x, y = source[active], target[active]
        source_mean, target_mean = x.mean(axis=0), y.mean(axis=0)
        covariance = (x - source_mean).T @ (y - target_mean)
        u, singular, vt = np.linalg.svd(covariance)
        if singular[1] <= max(float(singular[0]), 1.0) * 1e-12:
            raise RuntimeError("tracked points are collinear; rigid rotation is undefined")
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
        rotation = vt.T @ correction @ u.T
        translation = target_mean - rotation @ source_mean
        distance = np.linalg.norm(
            source @ rotation.T + translation - target, axis=1)
        ranked = available[np.argsort(distance[available], kind="stable")]
        updated = ranked[:keep_count]
        if np.array_equal(np.sort(active), np.sort(updated)):
            active = updated
            break
        active = updated

    # Refit once on the final consensus set.
    x, y = source[active], target[active]
    source_mean, target_mean = x.mean(axis=0), y.mean(axis=0)
    u, singular, vt = np.linalg.svd((x - source_mean).T @ (y - target_mean))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ correction @ u.T
    translation = target_mean - rotation @ source_mean
    distance = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    inliers = np.zeros(len(source), dtype=bool)
    inliers[active] = True
    inlier_rms = float(np.sqrt(np.mean(distance[active] ** 2)))
    all_rms = float(np.sqrt(np.mean(distance[available] ** 2)))
    return transform, inliers, inlier_rms, all_rms


def load_label_tracks(label_dir: Path, reference_frame: int, trim_quantile: float,
                      rigid_iterations: int, min_motion: float):
    points_path = label_dir / "pts3d_ref.npy"
    flow_dir = label_dir / "scene_flow"
    if not points_path.is_file():
        raise RuntimeError(f"missing {points_path}")
    if not flow_dir.is_dir():
        raise RuntimeError(f"missing {flow_dir}")
    try:
        points = np.load(points_path, allow_pickle=False).astype(np.float64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not load {points_path}: {exc}") from exc
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 6:
        raise RuntimeError(
            f"{points_path} must have shape (N, 3) with N >= 6; got {points.shape}")
    reference_valid = np.isfinite(points).all(axis=1)
    if int(reference_valid.sum()) < 6:
        raise RuntimeError(f"{points_path} has fewer than 6 finite points")

    identity_inliers = reference_valid.copy()
    observations = [TrackObservation(
        frame=int(reference_frame), target=points.copy(), valid=reference_valid,
        transform=np.eye(4), inliers=identity_inliers,
        rigid_inlier_rms=0.0, rigid_all_rms=0.0, median_flow=0.0,
    )]
    skipped = []
    seen = {int(reference_frame)}
    flow_files = sorted(flow_dir.glob("*.npy"))
    if not flow_files:
        raise RuntimeError(f"no point-track flow files found in {flow_dir}")
    for path in flow_files:
        try:
            frame = int(path.stem)
        except ValueError:
            skipped.append({"file": path.name, "reason": "non-numeric frame name"})
            continue
        if frame in seen:
            skipped.append({"file": path.name, "reason": "duplicates reference frame"})
            continue
        try:
            flow = np.load(path, allow_pickle=False).astype(np.float64)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"could not load {path}: {exc}") from exc
        if flow.shape != points.shape:
            raise RuntimeError(
                f"{path} has shape {flow.shape}; expected {points.shape}")
        valid = reference_valid & np.isfinite(flow).all(axis=1)
        if int(valid.sum()) < 6:
            skipped.append({"frame": frame, "reason": "fewer than 6 finite tracks"})
            continue
        median_flow = float(np.median(np.linalg.norm(flow[valid], axis=1)))
        if median_flow < min_motion:
            skipped.append({
                "frame": frame,
                "reason": f"median displacement {median_flow:.8g} below --min-motion",
            })
            continue
        target = points + flow
        transform, inliers, inlier_rms, all_rms = rigid_transform(
            points, target, valid, trim_quantile, rigid_iterations)
        observations.append(TrackObservation(
            frame=frame, target=target, valid=valid, transform=transform,
            inliers=inliers, rigid_inlier_rms=inlier_rms,
            rigid_all_rms=all_rms, median_flow=median_flow,
        ))
        seen.add(frame)
    observations.sort(key=lambda observation: observation.frame)
    if len(observations) < 3:
        raise RuntimeError(
            "joint estimation needs the reference and at least two target frames")
    reference_index = next(
        index for index, observation in enumerate(observations)
        if observation.frame == reference_frame)
    return points, observations, reference_index, skipped


def canonicalize(axis, states):
    """Choose a deterministic axis sign while preserving predicted motion."""
    axis = normalize(axis).copy()
    states = np.asarray(states, dtype=np.float64).copy()
    if states[-1] < states[0]:
        axis *= -1.0
        states *= -1.0
    elif abs(float(states[-1] - states[0])) < 1e-10:
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0.0:
            axis *= -1.0
            states *= -1.0
    return axis, states


def initial_prismatic(points, observations, reference_index):
    center = points[np.isfinite(points).all(axis=1)].mean(axis=0)
    centers = np.stack([
        observation.transform[:3, :3] @ center + observation.transform[:3, 3]
        for observation in observations
    ])
    displacement = centers - centers[reference_index]
    useful = displacement[np.linalg.norm(displacement, axis=1) > 1e-9]
    if len(useful):
        _, _, vt = np.linalg.svd(useful, full_matrices=False)
        axis = vt[0]
    else:
        axis = np.array([1.0, 0.0, 0.0])
    states = displacement @ axis
    return canonicalize(axis, states)


def initial_revolute(points, observations, reference_index, fallback_axis):
    rotations = np.stack([observation.transform[:3, :3]
                          for observation in observations])
    vectors = np.stack([so3_log(rotation) for rotation in rotations])
    useful = vectors[np.linalg.norm(vectors, axis=1) > np.deg2rad(0.25)]
    observable = len(useful) > 0
    if observable:
        _, _, vt = np.linalg.svd(useful, full_matrices=False)
        axis = vt[0]
    else:
        fallback_axis = normalize(fallback_axis)
        seed = (np.array([1.0, 0.0, 0.0])
                if abs(fallback_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0]))
        axis = normalize(np.cross(fallback_axis, seed))
    states = np.unwrap(vectors @ axis)
    states -= states[reference_index]
    axis, states = canonicalize(axis, states)

    reference_valid = np.isfinite(points).all(axis=1)
    center = points[reference_valid].mean(axis=0)
    matrices, targets = [], []
    for observation, state in zip(observations, states):
        if abs(float(state)) < np.deg2rad(0.25):
            continue
        rotation = so3_exp(float(state) * axis)
        target_center = (observation.transform[:3, :3] @ center
                         + observation.transform[:3, 3])
        matrices.append(np.eye(3) - rotation)
        targets.append(target_center - rotation @ center)
    if matrices:
        # Select the unique point on the axis nearest the reference centroid.
        matrices.append(axis[None, :])
        targets.append(np.array([float(axis @ center)]))
        pivot, *_ = np.linalg.lstsq(
            np.vstack(matrices), np.concatenate(targets), rcond=None)
    else:
        pivot = center.copy()
    return axis, pivot, states, observable


def _common_fit_points(points, observations, max_points, rng):
    common = np.isfinite(points).all(axis=1)
    for observation in observations:
        common &= observation.valid
    indices = np.flatnonzero(common)
    if len(indices) < 6:
        raise RuntimeError(
            f"only {len(indices)} tracks are finite in every selected frame; need 6")
    if len(indices) > max_points:
        indices = np.sort(rng.choice(indices, size=max_points, replace=False))
    targets = np.stack([observation.target[indices]
                        for observation in observations])
    return points[indices], targets, indices


def trimmed_rms(distance, quantile):
    distance = np.asarray(distance, dtype=np.float64)
    distance = distance[np.isfinite(distance)]
    if not len(distance):
        return np.inf
    count = max(1, int(np.ceil(quantile * len(distance))))
    kept = np.partition(distance, count - 1)[:count]
    return float(np.sqrt(np.mean(kept ** 2)))


def refine_model(joint_type, points, observations, reference_index,
                 axis_initial, pivot_initial, states_initial, diameter,
                 trim_quantile, max_fit_points, fit_iterations, rng):
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for point-track joint fitting") from exc

    fit_points, fit_targets, _ = _common_fit_points(
        points, observations, max_fit_points, rng)
    state_indices = [index for index in range(len(observations))
                     if index != reference_index]
    if joint_type == "prismatic":
        initial = np.concatenate((axis_initial, states_initial[state_indices]))
    else:
        initial = np.concatenate((
            axis_initial, np.asarray(pivot_initial, dtype=np.float64),
            states_initial[state_indices],
        ))
    center = fit_points.mean(axis=0)

    def unpack(parameters):
        axis = parameters[:3]
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        states = np.zeros(len(observations), dtype=np.float64)
        if joint_type == "prismatic":
            states[state_indices] = parameters[3:]
            return axis, None, states
        raw_pivot = parameters[3:6]
        pivot = raw_pivot - axis * float(axis @ (raw_pivot - center))
        states[state_indices] = parameters[6:]
        return axis, pivot, states

    def residuals(parameters):
        axis, pivot, states = unpack(parameters)
        predicted = []
        if joint_type == "prismatic":
            predicted = (fit_points[None, :, :]
                         + states[:, None, None] * axis[None, None, :])
        else:
            centered = fit_points - pivot
            for state in states:
                rotation = so3_exp(float(state) * axis)
                predicted.append(centered @ rotation.T + pivot)
            predicted = np.stack(predicted)
        return (predicted - fit_targets).reshape(-1)

    nonzero_noise = [observation.rigid_inlier_rms for observation in observations
                     if observation.rigid_inlier_rms > 0.0]
    noise = float(np.median(nonzero_noise)) if nonzero_noise else 0.0
    huber_scale = max(noise, diameter * 1e-4, 1e-7)
    solved = least_squares(
        residuals, initial, loss="huber", f_scale=huber_scale,
        max_nfev=fit_iterations,
    )
    axis, pivot, states = unpack(solved.x)
    if joint_type == "revolute":
        states = np.unwrap(states)
    axis, states = canonicalize(axis, states)

    frame_rms = []
    for observation, state in zip(observations, states):
        valid_points = points[observation.valid]
        if joint_type == "prismatic":
            predicted = valid_points + float(state) * axis
        else:
            rotation = so3_exp(float(state) * axis)
            predicted = (valid_points - pivot) @ rotation.T + pivot
        distance = np.linalg.norm(
            predicted - observation.target[observation.valid], axis=1)
        frame_rms.append(trimmed_rms(distance, trim_quantile))
    frame_rms = np.asarray(frame_rms, dtype=np.float64)
    residual_rms = float(np.sqrt(np.mean(frame_rms ** 2)))
    return ModelFit(
        joint_type=joint_type, axis=axis, pivot=pivot, states=states,
        frame_rms=frame_rms, residual_rms=residual_rms,
        normalized_residual_rms=residual_rms / diameter,
        solver_success=bool(solved.success), solver_nfev=int(solved.nfev),
    )


def observed_rotation_span(observations):
    rotations = [observation.transform[:3, :3] for observation in observations]
    span = 0.0
    for first_index, first in enumerate(rotations):
        for second in rotations[first_index + 1:]:
            span = max(span, rotation_angle(first.T @ second))
    return float(span)


def fit_to_dict(fit, observations):
    return {
        "type": fit.joint_type,
        "axis_direction": fit.axis.tolist(),
        "axis_point": None if fit.pivot is None else fit.pivot.tolist(),
        "state_unit": "radians" if fit.joint_type == "revolute" else "world_units",
        "point_fit_residual_rms": float(fit.residual_rms),
        "normalized_point_fit_residual_rms": float(fit.normalized_residual_rms),
        "solver_success": fit.solver_success,
        "solver_evaluations": fit.solver_nfev,
        "joint_states": [
            {"frame": observation.frame, "q": float(state)}
            for observation, state in zip(observations, fit.states)
        ],
        "frame_residuals": [
            {"frame": observation.frame, "rms": float(error)}
            for observation, error in zip(observations, fit.frame_rms)
        ],
    }


def estimate_joint(points, observations, reference_index, args, rng):
    finite_points = points[np.isfinite(points).all(axis=1)]
    extent = np.ptp(finite_points, axis=0)
    diameter = float(np.linalg.norm(extent))
    if not np.isfinite(diameter) or diameter < 1e-9:
        raise RuntimeError("reference moving points have degenerate extent")

    prismatic_axis, prismatic_states = initial_prismatic(
        points, observations, reference_index)
    revolute_axis, revolute_pivot, revolute_states, rotation_observable = (
        initial_revolute(
            points, observations, reference_index, prismatic_axis))
    prismatic = refine_model(
        "prismatic", points, observations, reference_index,
        prismatic_axis, None, prismatic_states, diameter,
        args.trim_quantile, args.max_fit_points, args.fit_iterations, rng)
    revolute = refine_model(
        "revolute", points, observations, reference_index,
        revolute_axis, revolute_pivot, revolute_states, diameter,
        args.trim_quantile, args.max_fit_points, args.fit_iterations, rng)

    rotation_span = observed_rotation_span(observations)
    rotation_span_degrees = float(np.degrees(rotation_span))
    residual_ratio = (revolute.residual_rms
                      / max(prismatic.residual_rms, 1e-12))
    if args.joint_type == "auto":
        choose_revolute = (
            rotation_span_degrees >= args.min_rotation_deg
            and residual_ratio <= args.revolute_residual_ratio
        )
        selected = revolute if choose_revolute else prismatic
    else:
        selected = revolute if args.joint_type == "revolute" else prismatic
        if args.joint_type == "revolute" and not rotation_observable:
            raise RuntimeError(
                "--joint-type revolute was requested, but tracked poses contain "
                "no observable rotation")

    motion_span = float(max(
        observation.median_flow for observation in observations))
    rigid_noise = float(np.median([
        observation.rigid_inlier_rms for observation in observations
        if observation.frame != observations[reference_index].frame
    ]))
    underconstrained = motion_span <= max(3.0 * rigid_noise, diameter * 1e-4)
    if underconstrained:
        confidence = "low"
    elif selected.joint_type == "revolute":
        confidence = (
            "high" if rotation_span_degrees >= 1.25 * args.min_rotation_deg
            and residual_ratio <= 0.75 * args.revolute_residual_ratio else "low")
    else:
        confidence = (
            "high" if rotation_span_degrees < 0.75 * args.min_rotation_deg
            or residual_ratio > args.revolute_residual_ratio / 0.75 else "low")

    center = finite_points.mean(axis=0)
    selected_dict = fit_to_dict(selected, observations)
    return {
        "type": selected.joint_type,
        "type_confidence": confidence,
        "underconstrained": bool(underconstrained),
        "position": (selected.pivot.tolist()
                     if selected.pivot is not None else center.tolist()),
        "axis_direction": selected.axis.tolist(),
        "axis_point": (None if selected.pivot is None
                       else selected.pivot.tolist()),
        "axis_position_observable": selected.joint_type == "revolute",
        "position_definition": (
            "point_on_axis_nearest_reference_track_centroid"
            if selected.joint_type == "revolute"
            else "reference_track_centroid_display_anchor"
        ),
        "state_unit": selected_dict["state_unit"],
        "reference_frame": int(observations[reference_index].frame),
        "first_frame": int(observations[0].frame),
        "last_frame": int(observations[-1].frame),
        "frames": [observation.frame for observation in observations],
        "joint_states": selected_dict["joint_states"],
        "n_tracks": int(len(points)),
        "n_frames": int(len(observations)),
        "moving_point_diameter": diameter,
        "selection": {
            "requested_joint_type": args.joint_type,
            "observed_rotation_span_degrees": rotation_span_degrees,
            "minimum_revolute_rotation_degrees": args.min_rotation_deg,
            "revolute_to_prismatic_residual_ratio": float(residual_ratio),
            "required_revolute_residual_ratio": args.revolute_residual_ratio,
            "rotation_observable": bool(rotation_observable),
            "motion_span": motion_span,
            "median_rigid_inlier_rms": rigid_noise,
        },
        "candidate_models": {
            "prismatic": fit_to_dict(prismatic, observations),
            "revolute": fit_to_dict(revolute, observations),
        },
        "rigid_track_poses": [
            {
                "frame": observation.frame,
                "transform_from_reference": observation.transform.tolist(),
                "valid_tracks": int(observation.valid.sum()),
                "rigid_inliers": int(observation.inliers.sum()),
                "rigid_inlier_rms": float(observation.rigid_inlier_rms),
                "rigid_all_track_rms": float(observation.rigid_all_rms),
                "median_point_displacement": float(observation.median_flow),
            }
            for observation in observations
        ],
    }


def discover_moving_labels(scene_dir: Path, trackcraft_root: Path, requested):
    available = sorted(
        path.name for path in trackcraft_root.iterdir()
        if path.is_dir() and (path / "pts3d_ref.npy").is_file()
    )
    if not available:
        raise RuntimeError(
            f"no labels containing pts3d_ref.npy found under {trackcraft_root}")
    if requested is not None:
        labels = list(dict.fromkeys(requested))
        if not labels:
            raise RuntimeError("--moving-labels selected no labels")
        missing = [label for label in labels if label not in available]
        if missing:
            raise RuntimeError(
                f"TrackCraft output is missing requested label(s): {missing}; "
                f"available: {available}")
        return labels, "command_line"

    tracking_path = scene_dir / "masks" / "tracking.json"
    if tracking_path.is_file():
        tracking = read_json(tracking_path)
        labels = [label for label, record in tracking.items()
                  if isinstance(record, dict)
                  and record.get("motion") == "moving"
                  and label in available]
        missing = [label for label, record in tracking.items()
                   if isinstance(record, dict)
                   and record.get("motion") == "moving"
                   and label not in available]
        if missing:
            raise RuntimeError(
                f"moving label(s) lack TrackCraft output: {missing}")
        if labels:
            return labels, display_path(tracking_path, scene_dir)
        raise RuntimeError(f"no labels marked moving in {tracking_path}")
    print(f"warning: {tracking_path} is missing; treating every TrackCraft label as moving")
    return available, "all_trackcraft_labels_fallback"


def validate_args(args):
    if not 0.5 <= args.trim_quantile <= 1.0:
        raise RuntimeError("--trim-quantile must be between 0.5 and 1.0")
    if args.rigid_iterations < 1:
        raise RuntimeError("--rigid-iterations must be positive")
    if args.max_fit_points < 6:
        raise RuntimeError("--max-fit-points must be at least 6")
    if args.fit_iterations < 1:
        raise RuntimeError("--fit-iterations must be positive")
    if args.min_rotation_deg < 0.0:
        raise RuntimeError("--min-rotation-deg must be non-negative")
    if args.revolute_residual_ratio <= 0.0:
        raise RuntimeError("--revolute-residual-ratio must be positive")
    if args.min_motion < 0.0:
        raise RuntimeError("--min-motion must be non-negative")


def main(argv=None):
    args = parse_args(argv)
    try:
        validate_args(args)
        scene_dir = args.scene_dir.expanduser().resolve()
        if not scene_dir.is_dir():
            raise RuntimeError(f"scene directory does not exist: {scene_dir}")
        trackcraft_root = scene_dir / "trackcraft"
        if not trackcraft_root.is_dir():
            raise RuntimeError(f"missing {trackcraft_root}; run Stage 08 first")
        config_path = trackcraft_root / "config.json"
        config = read_json(config_path)
        try:
            reference_frame = int(config["ref_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"{config_path} has no valid integer ref_frame") from exc
        labels, label_source = discover_moving_labels(
            scene_dir, trackcraft_root, args.moving_labels)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"scene:           {scene_dir.name}")
    print(f"reference:       {reference_frame:06d}")
    print(f"moving labels:   {', '.join(labels)}")
    print(f"joint selection: {args.joint_type}")
    rng = np.random.default_rng(args.seed)
    results = {}
    skipped_by_label = {}
    for label in labels:
        print(f"\n[{label}]")
        try:
            points, observations, reference_index, skipped = load_label_tracks(
                trackcraft_root / label, reference_frame,
                args.trim_quantile, args.rigid_iterations, args.min_motion)
            result = estimate_joint(
                points, observations, reference_index, args, rng)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            raise SystemExit(f"error processing label {label!r}: {exc}") from exc
        results[label] = result
        skipped_by_label[label] = skipped
        selection = result["selection"]
        print(f"  tracks={result['n_tracks']} frames={result['n_frames']}")
        print(f"  selected={result['type']} confidence={result['type_confidence']}")
        print(f"  rotation_span={selection['observed_rotation_span_degrees']:.4g} deg "
              f"revolute/prismatic={selection['revolute_to_prismatic_residual_ratio']:.4g}")
        print("  axis=" + json.dumps([
            round(value, 7) for value in result["axis_direction"]]))
        if result["axis_point"] is not None:
            print("  pivot=" + json.dumps([
                round(value, 7) for value in result["axis_point"]]))
        if skipped:
            print(f"  skipped {len(skipped)} flow file/frame(s)")

    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir
               else scene_dir / "pointtrack_joint")
    summary_path = out_dir / "joints.json"
    sidecar_paths = {
        label: out_dir / safe_name(label) / "joint.json" for label in labels}
    existing = [path for path in [summary_path, *sidecar_paths.values()]
                if path.exists()]
    if existing and not args.overwrite and not args.dry_run:
        raise SystemExit(
            "error: output(s) already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing))
    for path in existing:
        if not path.is_file():
            raise SystemExit(f"error: refusing to replace non-file output: {path}")

    metadata = {
        "stage": "10_pointrack_to_joint",
        "scene": scene_dir.name,
        "method": "robust_track_rigid_poses_plus_direct_point_joint_fit",
        # Stable coordinate-frame ID from what is now stage 08.
        "coordinate_frame": "DA3_world_from_stage40_trackcraft",
        "trackcraft_config": display_path(config_path, scene_dir),
        "reference_frame": reference_frame,
        "moving_labels": labels,
        "moving_label_source": label_source,
        "joint_type_requested": args.joint_type,
        "minimum_revolute_rotation_degrees": args.min_rotation_deg,
        "required_revolute_residual_ratio": args.revolute_residual_ratio,
        "trim_quantile": args.trim_quantile,
        "rigid_iterations": args.rigid_iterations,
        "max_fit_points": args.max_fit_points,
        "fit_iterations": args.fit_iterations,
        "min_motion": args.min_motion,
        "seed": args.seed,
        "skipped_inputs": skipped_by_label,
        "joints": results,
    }
    if args.dry_run:
        print("\n--dry-run: outputs were not written")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for label, result in results.items():
            sidecar = sidecar_paths[label]
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(result, indent=2) + "\n")
        summary_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"\nwrote {summary_path}")
    return results


if __name__ == "__main__":
    main()
