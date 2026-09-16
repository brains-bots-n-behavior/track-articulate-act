#!/usr/bin/env python
"""Stage 07: estimate simple joints from the PCA motion of registered SegviGen parts.

Stage 06 places every SegviGen piece in one reference coordinate system by
registering the *static* geometry.  This script consumes that result and never
registers, matches, or computes nearest-neighbour correspondences between
moving meshes.  Instead, for every moving label it:

1. computes exact area-weighted surface centroid and PCA moments independently
   for every available moving mesh;
2. resolves PCA sign ambiguity as the smoothest chronological orientation
   sequence (and parallel-transports the observable axis for symmetric parts);
3. fits both a prismatic centroid line and a revolute axis/point model to the
   resulting PCA descriptor poses; and
4. exports the complete Stage-06 scene with a large joint arrow added.

The default outputs are ``<scene>/simple_joint/joint_mesh.glb`` and
``<scene>/simple_joint/joints.json``.  Prismatic motion does not determine a
physical axis position, so its reported ``position`` is a near-part display
anchor and ``axis_point`` remains null for compatibility with later stages.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
import sys
from pathlib import Path

import numpy as np


@dataclass
class PcaObservation:
    """Area-weighted PCA descriptor for one moving mesh."""

    frame: int
    geometry: str
    centroid: np.ndarray
    variances: np.ndarray
    axes: np.ndarray
    diameter: float
    surface_area: float
    source: str | None = None


@dataclass
class JointFit:
    """One fitted one-degree-of-freedom motion hypothesis."""

    joint_type: str
    axis: np.ndarray
    pivot: np.ndarray | None
    states: np.ndarray
    residual: float
    median_residual: float
    frame_errors: np.ndarray


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir", type=Path, required=True,
        help="Scene containing registered_static/{metadata.json,registered_meshes.glb}",
    )
    parser.add_argument(
        "--moving-labels", "--labels", nargs="*", default=None,
        help="Moving-label subset (default: Stage-06 moving_labels)",
    )
    parser.add_argument(
        "--pca-gap-threshold", type=float, default=0.12,
        help=("Normalized eigenvalue gap below which PCA axes are treated as "
              "degenerate and parallel-transported (default: 0.12)"),
    )
    parser.add_argument(
        "--min-rotation-deg", type=float, default=8.0,
        help=("Minimum observable PCA rotation span required to select a "
              "revolute joint (default: 8 degrees)"),
    )
    parser.add_argument(
        "--revolute-residual-ratio", type=float, default=0.80,
        help=("Revolute residual must be at most this fraction of the prismatic "
              "residual (default: 0.80)"),
    )
    parser.add_argument(
        "--arrow-length-scale", type=float, default=1.25,
        help="Arrow half-length in moving-part diagonals (default: 1.25)",
    )
    parser.add_argument(
        "--arrow-scene-scale", type=float, default=0.35,
        help="Minimum arrow half-length in scene diagonals (default: 0.35)",
    )
    parser.add_argument(
        "--arrow-radius-scale", type=float, default=0.025,
        help="Arrow shaft radius in moving-part diagonals (default: 0.025)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: <scene>/simple_joint)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate and print joints without writing GLB or JSON",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace existing joint_mesh.glb and joints.json",
    )
    return parser.parse_args(argv)


def read_json(path: Path):
    try:
        with path.open() as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not read {path}: {exc}") from exc


def safe_name(label: str):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)


def display_path(path: Path, scene_dir: Path):
    try:
        return str(path.relative_to(scene_dir))
    except ValueError:
        return str(path)


def _skew(vector):
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def _project_rotation(rotation):
    """Return the nearest proper rotation matrix."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def rotation_angle(rotation):
    cosine = np.clip(
        (np.trace(np.asarray(rotation, dtype=np.float64)) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.arccos(cosine))


def so3_log(rotation):
    """Stable logarithm of a 3-D proper rotation."""
    rotation = np.asarray(rotation, dtype=np.float64)
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
        norm = float(np.linalg.norm(axis))
        if norm < 1e-12:
            return np.zeros(3)
        return theta * axis / norm
    return theta * vee / (2.0 * np.sin(theta))


def so3_exp(rotation_vector):
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotation_vector))
    if theta < 1e-10:
        matrix = _skew(rotation_vector)
        return np.eye(3) + matrix + 0.5 * matrix @ matrix
    matrix = _skew(rotation_vector / theta)
    return (np.eye(3) + np.sin(theta) * matrix
            + (1.0 - np.cos(theta)) * matrix @ matrix)


def _minimal_vector_rotation(source, target):
    """Shortest proper rotation mapping one unit vector to another."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target /= max(float(np.linalg.norm(target)), 1e-12)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0.0:
            return np.eye(3)
        seed = (np.array([1.0, 0.0, 0.0])
                if abs(source[0]) < 0.8 else np.array([0.0, 1.0, 0.0]))
        axis = np.cross(source, seed)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return so3_exp(np.pi * axis)
    matrix = _skew(cross)
    return np.eye(3) + matrix + matrix @ matrix * ((1.0 - cosine) / sine**2)


def surface_pca(mesh, frame=0, geometry="mesh", source=None):
    """Compute exact first/second moments of a uniform triangle surface.

    Using triangle integrals rather than raw vertices prevents different mesh
    tessellations from biasing PCA.  There is deliberately no other-frame mesh
    or point correspondence anywhere in this calculation.
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise RuntimeError(f"geometry {geometry} has invalid triangles")
    finite = np.isfinite(triangles).all(axis=(1, 2))
    triangles = triangles[finite]
    if not len(triangles):
        raise RuntimeError(f"geometry {geometry} has no finite triangles")

    twice_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    areas = 0.5 * twice_area
    extent = np.ptp(triangles.reshape(-1, 3), axis=0)
    diameter = float(np.linalg.norm(extent))
    area_floor = max(diameter * diameter * 1e-16, np.finfo(float).tiny)
    useful = areas > area_floor
    triangles = triangles[useful]
    areas = areas[useful]
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0 or diameter <= 1e-12:
        raise RuntimeError(f"geometry {geometry} has degenerate surface extent")

    vertex_sum = triangles.sum(axis=1)
    centroid = np.sum(areas[:, None] * vertex_sum / 3.0, axis=0) / total_area

    # For a uniform triangle (a,b,c):
    # E[xx^T] = (sum(v_i v_i^T) + sum(v_i)sum(v_i)^T) / 12.
    vertex_outer_sum = np.einsum(
        "nvi,nvj->nij", triangles, triangles, optimize=True)
    triangle_second = (
        vertex_outer_sum
        + np.einsum("ni,nj->nij", vertex_sum, vertex_sum, optimize=True)
    ) / 12.0
    raw_second = np.sum(
        areas[:, None, None] * triangle_second, axis=0) / total_area
    covariance = raw_second - np.outer(centroid, centroid)
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    if values[0] <= max(diameter * diameter * 1e-14, 1e-20):
        raise RuntimeError(f"geometry {geometry} has degenerate PCA covariance")
    if np.linalg.det(vectors) < 0:
        vectors[:, -1] *= -1.0

    return PcaObservation(
        frame=int(frame),
        geometry=str(geometry),
        centroid=centroid,
        variances=values,
        axes=vectors,
        diameter=diameter,
        surface_area=total_area,
        source=source,
    )


def load_named_geometry(scene, geometry_name, trimesh):
    """Return one named Scene geometry with every instance transform baked in."""
    instances = []
    for node_name in scene.graph.nodes_geometry:
        transform, node_geometry = scene.graph[node_name]
        if node_geometry != geometry_name:
            continue
        mesh = scene.geometry[node_geometry].copy()
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            continue
        mesh.apply_transform(transform)
        instances.append(mesh)
    if not instances and geometry_name in scene.geometry:
        mesh = scene.geometry[geometry_name].copy()
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            instances.append(mesh)
    if not instances:
        raise RuntimeError(f"registered GLB lacks triangle geometry {geometry_name!r}")
    return (instances[0] if len(instances) == 1
            else trimesh.util.concatenate(instances))


def _proper_sign_choices():
    choices = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        matrix = np.diag(signs)
        if np.linalg.det(matrix) > 0:
            choices.append((np.asarray(signs), matrix))
    return choices


def _track_triaxial_bases(observations):
    """Globally resolve the four proper PCA sign choices by temporal DP."""
    if len(observations) == 1:
        return [observations[0].axes.copy()], [[1.0, 1.0, 1.0]], 0.0
    choices = _proper_sign_choices()
    candidates = [
        [observation.axes @ matrix for _, matrix in choices]
        for observation in observations[1:]
    ]

    first = observations[0].axes
    costs = np.array([
        rotation_angle(first.T @ candidate) ** 2
        for candidate in candidates[0]
    ])
    back_pointers = []
    for time_index in range(1, len(candidates)):
        previous_candidates = candidates[time_index - 1]
        current_candidates = candidates[time_index]
        new_costs = np.empty(len(choices), dtype=np.float64)
        pointers = np.empty(len(choices), dtype=np.int64)
        for current_index, current in enumerate(current_candidates):
            transitions = np.array([
                rotation_angle(previous.T @ current) ** 2
                for previous in previous_candidates
            ])
            values = costs + transitions
            pointers[current_index] = int(np.argmin(values))
            new_costs[current_index] = values[pointers[current_index]]
        back_pointers.append(pointers)
        costs = new_costs

    selected = [int(np.argmin(costs))]
    for pointers in reversed(back_pointers):
        selected.append(int(pointers[selected[-1]]))
    selected.reverse()
    bases = [first.copy()]
    signs = [[1.0, 1.0, 1.0]]
    for time_index, choice_index in enumerate(selected):
        bases.append(candidates[time_index][choice_index])
        signs.append(choices[choice_index][0].tolist())
    return bases, signs, float(np.min(costs))


def _track_unique_axis_bases(observations, unique_axis):
    """Track one observable PCA line and parallel-transport its complement."""
    bases = [observations[0].axes.copy()]
    signs = [[1.0, 1.0, 1.0]]
    continuity_cost = 0.0
    for observation in observations[1:]:
        previous = bases[-1]
        target = observation.axes[:, unique_axis].copy()
        sign = 1.0
        if float(previous[:, unique_axis] @ target) < 0.0:
            target *= -1.0
            sign = -1.0
        alignment = _minimal_vector_rotation(previous[:, unique_axis], target)
        current = _project_rotation(alignment @ previous)
        step = rotation_angle(previous.T @ current)
        continuity_cost += step * step
        bases.append(current)
        axis_signs = [1.0, 1.0, 1.0]
        axis_signs[unique_axis] = sign
        signs.append(axis_signs)
    return bases, signs, continuity_cost


def track_pca_bases(observations, gap_threshold=0.12):
    """Create the most continuous observable PCA orientation trajectory."""
    if not observations:
        raise ValueError("at least one PCA observation is required")
    spectra = np.stack([observation.variances for observation in observations])
    gap_01 = (spectra[:, 0] - spectra[:, 1]) / np.maximum(spectra[:, 0], 1e-20)
    gap_12 = (spectra[:, 1] - spectra[:, 2]) / np.maximum(spectra[:, 1], 1e-20)
    median_gap_01 = float(np.median(gap_01))
    median_gap_12 = float(np.median(gap_12))

    if median_gap_01 < gap_threshold and median_gap_12 < gap_threshold:
        mode = "isotropic_orientation_unobservable"
        bases = [observations[0].axes.copy() for _ in observations]
        signs = [[1.0, 1.0, 1.0] for _ in observations]
        cost = 0.0
        unique_axis = None
    elif median_gap_01 < gap_threshold:
        mode = "planar_unique_minor_axis"
        unique_axis = 2
        bases, signs, cost = _track_unique_axis_bases(observations, unique_axis)
    elif median_gap_12 < gap_threshold:
        mode = "axial_unique_major_axis"
        unique_axis = 0
        bases, signs, cost = _track_unique_axis_bases(observations, unique_axis)
    else:
        mode = "triaxial_full_orientation"
        unique_axis = None
        bases, signs, cost = _track_triaxial_bases(observations)

    return bases, {
        "mode": mode,
        "unique_axis_index": unique_axis,
        "median_normalized_eigenvalue_gaps": [median_gap_01, median_gap_12],
        "gap_threshold": float(gap_threshold),
        "selected_axis_signs": signs,
        "continuity_cost_radians_squared": float(cost),
    }


def descriptor_poses(observations, tracked_bases):
    """Return PCA-only rigid descriptor poses relative to the first frame."""
    if len(observations) != len(tracked_bases):
        raise ValueError("observation/basis count mismatch")
    reference_basis = tracked_bases[0]
    reference_centroid = observations[0].centroid
    rotations = []
    transforms = []
    for observation, basis in zip(observations, tracked_bases):
        rotation = _project_rotation(basis @ reference_basis.T)
        translation = observation.centroid - rotation @ reference_centroid
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        rotations.append(rotation)
        transforms.append(transform)
    return np.stack(rotations), np.stack(transforms)


def _canonicalize_toward_last(axis, states):
    axis = np.asarray(axis, dtype=np.float64).copy()
    states = np.asarray(states, dtype=np.float64).copy()
    if states[-1] < states[0]:
        axis *= -1.0
        states *= -1.0
    elif abs(float(states[-1] - states[0])) < 1e-10:
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis *= -1.0
            states *= -1.0
    return axis, states


def _frame_errors(joint_type, rotations, centroids, axis, pivot, states,
                  diameter):
    reference = centroids[0]
    scale = max(float(diameter), 1e-9)
    errors = []
    for observed_rotation, observed_center, state in zip(
            rotations, centroids, states):
        if joint_type == "prismatic":
            predicted_rotation = np.eye(3)
            predicted_center = reference + float(state) * axis
        else:
            predicted_rotation = so3_exp(float(state) * axis)
            predicted_center = predicted_rotation @ (reference - pivot) + pivot
        angle_error = rotation_angle(predicted_rotation.T @ observed_rotation)
        center_error = float(np.linalg.norm(predicted_center - observed_center)) / scale
        errors.append(np.hypot(angle_error, center_error))
    return np.asarray(errors, dtype=np.float64)


def _initial_prismatic(centroids):
    displacements = np.asarray(centroids, dtype=np.float64) - centroids[0]
    useful = displacements[np.linalg.norm(displacements, axis=1) > 1e-10]
    if len(useful):
        _, _, vt = np.linalg.svd(useful, full_matrices=False)
        axis = vt[0]
    else:
        axis = np.array([1.0, 0.0, 0.0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    states = displacements @ axis
    return _canonicalize_toward_last(axis, states)


def _initial_revolute(rotations, centroids):
    rotation_vectors = np.stack([so3_log(rotation) for rotation in rotations])
    useful = rotation_vectors[np.linalg.norm(rotation_vectors, axis=1) > 1e-7]
    if len(useful):
        reference_vector = useful[int(np.argmax(np.linalg.norm(useful, axis=1)))]
        useful = np.stack([
            -vector if float(vector @ reference_vector) < 0.0 else vector
            for vector in useful
        ])
        _, _, vt = np.linalg.svd(useful, full_matrices=False)
        axis = vt[0]
    else:
        axis = np.array([0.0, 0.0, 1.0])
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    states = np.unwrap(rotation_vectors @ axis)
    axis, states = _canonicalize_toward_last(axis, states)

    predicted_rotations = [so3_exp(float(state) * axis) for state in states]
    reference = centroids[0]
    matrices = []
    targets = []
    for rotation, centroid in zip(predicted_rotations, centroids):
        matrices.append(np.eye(3) - rotation)
        targets.append(centroid - rotation @ reference)
    # The final row fixes the physically irrelevant position along the axis by
    # selecting the point on the line nearest the reference moving centroid.
    matrices.append(axis[None, :])
    targets.append(np.array([float(axis @ reference)]))
    pivot, *_ = np.linalg.lstsq(
        np.vstack(matrices), np.concatenate(targets), rcond=None)
    return axis, pivot, states


def fit_joint_model(joint_type, rotations, centroids, diameter):
    """Fit one joint family to PCA orientations and centroids only."""
    rotations = np.asarray(rotations, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    if joint_type == "prismatic":
        axis, states = _initial_prismatic(centroids)
        pivot = None
        initial = np.concatenate([axis, states])
    elif joint_type == "revolute":
        axis, pivot, states = _initial_revolute(rotations, centroids)
        initial = np.concatenate([axis, pivot, states])
    else:
        raise ValueError(f"unsupported joint type: {joint_type}")

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for simple joint fitting") from exc

    count = len(rotations)
    scale = max(float(diameter), 1e-9)
    reference = centroids[0]

    def unpack(parameters):
        fitted_axis = np.asarray(parameters[:3], dtype=np.float64).copy()
        fitted_axis /= max(float(np.linalg.norm(fitted_axis)), 1e-12)
        if joint_type == "prismatic":
            return fitted_axis, None, parameters[3:3 + count]
        raw_pivot = parameters[3:6]
        # Re-anchor the same axis line at the point nearest the first centroid.
        fitted_pivot = (raw_pivot - fitted_axis
                        * float(fitted_axis @ (raw_pivot - reference)))
        return fitted_axis, fitted_pivot, parameters[6:6 + count]

    def residuals(parameters):
        fitted_axis, fitted_pivot, fitted_states = unpack(parameters)
        chunks = []
        for observed_rotation, observed_center, state in zip(
                rotations, centroids, fitted_states):
            if joint_type == "prismatic":
                predicted_rotation = np.eye(3)
                predicted_center = reference + float(state) * fitted_axis
            else:
                predicted_rotation = so3_exp(float(state) * fitted_axis)
                predicted_center = (
                    predicted_rotation @ (reference - fitted_pivot) + fitted_pivot)
            chunks.append(np.concatenate((
                so3_log(predicted_rotation.T @ observed_rotation),
                (predicted_center - observed_center) / scale,
            )))
        return np.concatenate(chunks)

    solved = least_squares(
        residuals, initial, loss="huber", f_scale=0.03, max_nfev=500)
    axis, pivot, states = unpack(solved.x)
    if joint_type == "revolute":
        states = np.unwrap(states)
    axis, states = _canonicalize_toward_last(axis, states)
    frame_errors = _frame_errors(
        joint_type, rotations, centroids, axis, pivot, states, diameter)
    return JointFit(
        joint_type=joint_type,
        axis=axis,
        pivot=pivot,
        states=states,
        residual=float(np.sqrt(np.mean(frame_errors**2))),
        median_residual=float(np.median(frame_errors)),
        frame_errors=frame_errors,
    )


def _rotation_span(rotations):
    span = 0.0
    for first_index, first in enumerate(rotations):
        for second in rotations[first_index + 1:]:
            span = max(span, rotation_angle(first.T @ second))
    return float(span)


def estimate_joint(observations, gap_threshold=0.12, min_rotation_deg=8.0,
                   revolute_residual_ratio=0.80):
    """Estimate and classify one joint from chronological PCA observations."""
    observations = sorted(observations, key=lambda observation: observation.frame)
    if len(observations) < 2:
        raise RuntimeError("joint estimation requires at least two available frames")
    tracked_bases, tracking = track_pca_bases(observations, gap_threshold)
    rotations, transforms = descriptor_poses(observations, tracked_bases)
    centroids = np.stack([observation.centroid for observation in observations])
    diameter = float(np.median([observation.diameter for observation in observations]))
    if not np.isfinite(diameter) or diameter <= 1e-9:
        raise RuntimeError("moving-part diameter is degenerate")

    fits = {
        joint_type: fit_joint_model(
            joint_type, rotations, centroids, diameter)
        for joint_type in ("prismatic", "revolute")
    }
    rotation_span = _rotation_span(rotations)
    rotation_span_degrees = float(np.degrees(rotation_span))
    prism = fits["prismatic"]
    revolute = fits["revolute"]
    ratio = revolute.residual / max(prism.residual, 1e-12)
    orientation_observable = (
        tracking["mode"] != "isotropic_orientation_unobservable")
    choose_revolute = (
        orientation_observable
        and rotation_span_degrees >= min_rotation_deg
        and ratio <= revolute_residual_ratio
    )
    selected = revolute if choose_revolute else prism

    displacement_span = max(
        float(np.linalg.norm(first - second))
        for index, first in enumerate(centroids)
        for second in centroids[index:]
    )
    normalized_displacement_span = displacement_span / diameter
    underconstrained = (
        len(observations) < 3
        or (normalized_displacement_span < 0.01
            and rotation_span_degrees < min_rotation_deg)
    )
    if underconstrained or not orientation_observable:
        confidence = "low"
    elif choose_revolute:
        confidence = "high" if ratio <= 0.5 and rotation_span_degrees >= 15.0 \
            else "medium"
    else:
        inverse_ratio = prism.residual / max(revolute.residual, 1e-12)
        confidence = (
            "high" if rotation_span_degrees < min_rotation_deg
            and normalized_displacement_span >= 0.05
            and inverse_ratio <= 0.8 else "medium")

    if selected.joint_type == "revolute":
        position = selected.pivot.copy()
        axis_point = selected.pivot.tolist()
        position_observable = True
        position_definition = "point_on_axis_nearest_first_surface_centroid"
        state_unit = "radians"
    else:
        anchors = centroids - selected.states[:, None] * selected.axis[None, :]
        position = np.median(anchors, axis=0)
        axis_point = None
        position_observable = False
        position_definition = (
            "display_anchor_only_median_centroid_with_prismatic_state_removed")
        state_unit = "scene_units"

    frame_ids = [observation.frame for observation in observations]

    def fit_payload(fit):
        return {
            "type": fit.joint_type,
            "axis_direction": fit.axis.tolist(),
            "axis_point": None if fit.pivot is None else fit.pivot.tolist(),
            "pose_fit_residual_rms": fit.residual,
            "pose_fit_residual_median": fit.median_residual,
            "frame_errors": fit.frame_errors.tolist(),
            "joint_states": [
                {"frame": int(frame), "q": float(state)}
                for frame, state in zip(frame_ids, fit.states)
            ],
        }

    result = {
        "type": selected.joint_type,
        "position": position.tolist(),
        "axis_direction": selected.axis.tolist(),
        "axis_point": axis_point,
        "axis_position_observable": position_observable,
        "position_definition": position_definition,
        "state_unit": state_unit,
        "first_frame": int(frame_ids[0]),
        "last_frame": int(frame_ids[-1]),
        "frames": frame_ids,
        "joint_states": [
            {"frame": int(frame), "q": float(state)}
            for frame, state in zip(frame_ids, selected.states)
        ],
        "type_confidence": confidence,
        "underconstrained": bool(underconstrained),
        "selection": {
            "rotation_span_degrees": rotation_span_degrees,
            "centroid_displacement_span": displacement_span,
            "normalized_centroid_displacement_span": normalized_displacement_span,
            "revolute_to_prismatic_residual_ratio": float(ratio),
            "required_revolute_residual_ratio": float(revolute_residual_ratio),
            "minimum_revolute_rotation_degrees": float(min_rotation_deg),
            "orientation_observable": bool(orientation_observable),
        },
        "candidate_models": {
            joint_type: fit_payload(fit)
            for joint_type, fit in fits.items()
        },
        "pca_tracking": tracking,
        "pca_descriptor_transforms": transforms.tolist(),
        "pca_frames": [
            {
                "frame": int(observation.frame),
                "geometry": observation.geometry,
                "source": observation.source,
                "surface_centroid": observation.centroid.tolist(),
                "surface_variances": observation.variances.tolist(),
                "raw_pca_axes_columns": observation.axes.tolist(),
                "tracked_pca_axes_columns": basis.tolist(),
                "surface_area": float(observation.surface_area),
                "diameter": float(observation.diameter),
            }
            for observation, basis in zip(observations, tracked_bases)
        ],
        "moving_part_diameter": diameter,
    }
    return result


def _color_mesh(mesh, rgba):
    color = np.asarray(rgba, dtype=np.uint8).reshape(1, 4)
    mesh.visual.face_colors = np.repeat(color, len(mesh.faces), axis=0)
    return mesh


def add_joint_arrow(scene, label, joint, part_diagonal, scene_diagonal, args,
                    trimesh):
    """Append a deliberately large shaft, cone, and center marker."""
    axis = np.asarray(joint["axis_direction"], dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = np.asarray(joint["position"], dtype=np.float64)
    half_length = max(
        args.arrow_length_scale * float(part_diagonal),
        args.arrow_scene_scale * float(scene_diagonal),
    )
    radius = max(
        args.arrow_radius_scale * float(part_diagonal),
        0.006 * float(scene_diagonal),
        1e-4,
    )
    head_height = min(
        0.45 * half_length,
        max(0.16 * half_length, 4.0 * radius),
    )
    start = center - half_length * axis
    tip = center + half_length * axis
    head_base = tip - head_height * axis
    color = ((35, 180, 75, 255) if joint["type"] == "revolute"
             else (50, 110, 230, 255))

    shaft = trimesh.creation.cylinder(
        radius=radius, segment=np.stack((start, head_base)), sections=32)
    head = trimesh.creation.cone(
        radius=2.8 * radius, height=head_height, sections=32)
    alignment = trimesh.geometry.align_vectors(
        np.array([0.0, 0.0, 1.0]), axis)
    head.apply_transform(alignment)
    head.apply_translation(head_base)
    marker = trimesh.creation.icosphere(subdivisions=2, radius=2.2 * radius)
    marker.apply_translation(center)
    shaft = _color_mesh(shaft, color)
    head = _color_mesh(head, color)
    marker = _color_mesh(marker, color)

    prefix = f"joint_{safe_name(label)}"
    geometry_names = {
        "shaft": f"{prefix}_arrow_shaft",
        "head": f"{prefix}_arrow_head",
        "position_marker": f"{prefix}_position",
    }
    for key, mesh in (("shaft", shaft), ("head", head),
                      ("position_marker", marker)):
        name = geometry_names[key]
        scene.add_geometry(mesh, geom_name=name, node_name=name)
    return {
        "geometry": geometry_names,
        "half_length": float(half_length),
        "total_length": float(2.0 * half_length),
        "shaft_radius": float(radius),
        "color_rgba": list(color),
    }


def _resolve_registered_mesh(scene_dir, metadata, metadata_path):
    value = metadata.get("output_mesh", "registered_static/registered_meshes.glb")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    scene_relative = (scene_dir / path).resolve()
    if scene_relative.is_file():
        return scene_relative
    metadata_relative = (metadata_path.parent / path).resolve()
    if metadata_relative.is_file():
        return metadata_relative
    return scene_relative


def _collect_label_records(metadata, label):
    records = []
    for frame_record in metadata.get("frames", []):
        try:
            frame = int(frame_record["frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Stage-06 metadata has an invalid frame record") from exc
        for moving in frame_record.get("moving_meshes", []):
            if moving.get("label") == label:
                geometry = moving.get("geometry")
                if not geometry:
                    raise RuntimeError(
                        f"Stage-06 metadata lacks geometry for {label!r} frame {frame}")
                records.append({
                    "frame": frame,
                    "geometry": geometry,
                    "source": moving.get("source"),
                })
                break
    records.sort(key=lambda record: record["frame"])
    duplicate_frames = [
        records[index]["frame"] for index in range(1, len(records))
        if records[index]["frame"] == records[index - 1]["frame"]
    ]
    if duplicate_frames:
        raise RuntimeError(
            f"duplicate Stage-06 moving records for {label!r}: {duplicate_frames}")
    if len(records) < 2:
        raise RuntimeError(
            f"moving label {label!r} has only {len(records)} available frame(s); need 2")
    return records


def main(argv=None):
    args = parse_args(argv)
    if not (0.0 < args.pca_gap_threshold < 1.0):
        sys.exit("error: --pca-gap-threshold must be between 0 and 1")
    if args.min_rotation_deg < 0.0:
        sys.exit("error: --min-rotation-deg cannot be negative")
    if not (0.0 < args.revolute_residual_ratio <= 1.0):
        sys.exit("error: --revolute-residual-ratio must be in (0, 1]")
    if (args.arrow_length_scale <= 0.0 or args.arrow_scene_scale <= 0.0
            or args.arrow_radius_scale <= 0.0):
        sys.exit("error: arrow scale values must be positive")

    try:
        import trimesh
    except ImportError:
        sys.exit("error: trimesh is required; run this in the trellis2 environment")

    scene_dir = args.scene_dir.expanduser().resolve()
    metadata_path = scene_dir / "registered_static" / "metadata.json"
    try:
        if not scene_dir.is_dir():
            raise RuntimeError(f"scene directory does not exist: {scene_dir}")
        metadata = read_json(metadata_path)
        if metadata.get("mesh_source") not in (None, "segvigen"):
            raise RuntimeError(
                "Stage-06 metadata was produced with --register-sam3d; "
                "moving SegviGen pieces are required")
        registered_mesh_path = _resolve_registered_mesh(
            scene_dir, metadata, metadata_path)
        if not registered_mesh_path.is_file():
            raise RuntimeError(f"missing Stage-06 mesh: {registered_mesh_path}")
        try:
            registered_scene = trimesh.load(
                registered_mesh_path, force="scene", process=False)
        except Exception as exc:
            raise RuntimeError(
                f"could not load Stage-06 mesh {registered_mesh_path}: {exc}") from exc

        available_labels = list(dict.fromkeys(metadata.get("moving_labels", [])))
        if not available_labels:
            available_labels = sorted({
                moving.get("label")
                for frame in metadata.get("frames", [])
                for moving in frame.get("moving_meshes", [])
                if moving.get("label")
            })
        labels = (list(dict.fromkeys(args.moving_labels))
                  if args.moving_labels is not None else available_labels)
        if not labels:
            raise RuntimeError("Stage-06 metadata contains no moving labels")
        unknown = [label for label in labels if label not in available_labels]
        if unknown:
            raise RuntimeError(f"unknown moving label(s): {unknown}")
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")

    original_bounds = np.asarray(registered_scene.bounds, dtype=np.float64)
    scene_diagonal = float(np.linalg.norm(original_bounds[1] - original_bounds[0]))
    if not np.isfinite(scene_diagonal) or scene_diagonal <= 1e-9:
        sys.exit("error: Stage-06 scene has degenerate bounds")

    output_scene = registered_scene.copy()
    joints = {}
    try:
        for label in labels:
            records = _collect_label_records(metadata, label)
            observations = []
            for record in records:
                mesh = load_named_geometry(
                    registered_scene, record["geometry"], trimesh)
                observations.append(surface_pca(
                    mesh,
                    frame=record["frame"],
                    geometry=record["geometry"],
                    source=record["source"],
                ))
            joint = estimate_joint(
                observations,
                gap_threshold=args.pca_gap_threshold,
                min_rotation_deg=args.min_rotation_deg,
                revolute_residual_ratio=args.revolute_residual_ratio,
            )
            joint["label"] = label
            # Stable coordinate-frame ID from what is now stage 06.
            joint["coordinate_frame"] = "stage31_reference_mesh"
            joint["visualization"] = add_joint_arrow(
                output_scene,
                label,
                joint,
                joint["moving_part_diameter"],
                scene_diagonal,
                args,
                trimesh,
            )
            joints[label] = joint
            axis = joint["axis_direction"]
            print(
                f"{label}: type={joint['type']}, "
                f"frames={joint['first_frame']:06d}..{joint['last_frame']:06d}, "
                f"axis=[{axis[0]:+.4f}, {axis[1]:+.4f}, {axis[2]:+.4f}], "
                f"confidence={joint['type_confidence']}",
                flush=True,
            )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        sys.exit(f"error estimating joint: {exc}")

    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir
               else scene_dir / "simple_joint")
    output_mesh = out_dir / "joint_mesh.glb"
    output_json = out_dir / "joints.json"
    if not args.dry_run:
        existing = [path for path in (output_mesh, output_json) if path.exists()]
        if existing and not args.overwrite:
            sys.exit("error: output(s) already exist; pass --overwrite: "
                     + ", ".join(str(path) for path in existing))
        non_files = [path for path in existing if not path.is_file()]
        if non_files:
            sys.exit("error: refusing to replace non-file output(s): "
                     + ", ".join(str(path) for path in non_files))
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            output_scene.export(output_mesh)
        except Exception as exc:
            sys.exit(f"error exporting {output_mesh}: {exc}")

        payload = {
            "stage": "07_simple_joint",
            "scene": scene_dir.name,
            "method": "area_weighted_surface_PCA_no_moving_mesh_registration",
            # Stable coordinate-frame ID from what is now stage 06.
            "coordinate_frame": "stage31_reference_mesh",
            "registered_metadata": display_path(metadata_path, scene_dir),
            "registered_mesh": display_path(registered_mesh_path, scene_dir),
            "output_mesh": display_path(output_mesh, scene_dir),
            "moving_labels": labels,
            "pca_gap_threshold": args.pca_gap_threshold,
            "minimum_revolute_rotation_degrees": args.min_rotation_deg,
            "required_revolute_residual_ratio": args.revolute_residual_ratio,
            "scene_diagonal": scene_diagonal,
            "joints": joints,
        }
        output_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"joint mesh: {output_mesh}")
        print(f"joints: {output_json}")
    else:
        print("--dry-run: skipping GLB and JSON writes")
    return joints


if __name__ == "__main__":
    main()
