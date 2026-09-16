"""Offscreen differentiable mesh rendering and image-space losses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observation:
    frame_id: int
    mask: object
    boundary_distance: object
    camera: object
    features: object | None = None
    depth: object | None = None
    depth_valid: object | None = None


def choose_render_size(image_hw: tuple[int, int], max_side: int,
                       patch_size: int = 14) -> tuple[int, int]:
    h, w = image_hw
    scale = min(1.0, float(max_side) / max(h, w))
    out_h = max(patch_size, int(round(h * scale / patch_size)) * patch_size)
    out_w = max(patch_size, int(round(w * scale / patch_size)) * patch_size)
    return out_h, out_w


def resize_mask_and_boundary(mask: np.ndarray, output_hw: tuple[int, int]):
    import cv2
    import torch

    h, w = output_hw
    resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    kernel = np.ones((3, 3), dtype=np.uint8)
    edge = cv2.morphologyEx(resized.astype(np.uint8), cv2.MORPH_GRADIENT, kernel) > 0
    distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
    distance = distance.astype(np.float32) / max(float(np.hypot(h, w)), 1.0)
    return torch.from_numpy(resized.astype(np.float32)), torch.from_numpy(distance)


class DifferentiableMeshRenderer:
    """Minimal nvdiffrast renderer using positive-Z RDF camera coordinates."""

    def __init__(self, output_hw: tuple[int, int], image_hw: tuple[int, int],
                 device: str = "cuda"):
        import nvdiffrast.torch as dr
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by nvdiffrast; no CUDA device is available")
        self.dr = dr
        self.torch = torch
        self.h, self.w = output_hw
        self.image_h, self.image_w = image_hw
        self.device = torch.device(device)
        self.context = dr.RasterizeCudaContext(device=self.device)

    def camera_tensors(self, camera):
        torch = self.torch
        sx = self.w / float(self.image_w)
        sy = self.h / float(self.image_h)
        K = torch.as_tensor(camera.K, dtype=torch.float32, device=self.device).clone()
        K[0, :] *= sx
        K[1, :] *= sy
        R = torch.as_tensor(camera.R_world_to_camera, dtype=torch.float32, device=self.device)
        t = torch.as_tensor(camera.t_world_to_camera, dtype=torch.float32, device=self.device)
        return K, R, t

    def project(self, vertices_world, camera):
        K, R, t = self.camera_tensors(camera)
        vertices_camera = vertices_world @ R.T + t
        z = vertices_camera[:, 2].clamp_min(1e-5)
        u = K[0, 0] * vertices_camera[:, 0] / z + K[0, 2]
        v = K[1, 1] * vertices_camera[:, 1] / z + K[1, 2]
        return self.torch.stack([u, v], dim=-1), vertices_camera

    def render(self, vertices_world, faces, camera, attributes=None,
               confidence=None, return_raster=False, return_depth=False):
        torch, dr = self.torch, self.dr
        K, R, t = self.camera_tensors(camera)
        vertices_camera = vertices_world @ R.T + t
        x, y, z = vertices_camera.unbind(dim=-1)
        near, far = 0.01, 100.0
        x_clip = (2.0 * K[0, 0] / self.w) * x + (2.0 * K[0, 2] / self.w - 1.0) * z
        # Images have Y down; nvdiffrast NDC has Y up.
        y_clip = (-2.0 * K[1, 1] / self.h) * y + (1.0 - 2.0 * K[1, 2] / self.h) * z
        z_clip = ((far + near) / (far - near)) * z - (2.0 * far * near / (far - near))
        clip = torch.stack([x_clip, y_clip, z_clip, z], dim=-1).unsqueeze(0)
        tri = faces.to(device=self.device, dtype=torch.int32).contiguous()
        rast, _ = dr.rasterize(self.context, clip, tri, resolution=[self.h, self.w])
        hard = (rast[..., 3:4] > 0).to(torch.float32)
        alpha = dr.antialias(hard, rast, clip, tri)[0, ..., 0].clamp(0.0, 1.0)
        result = {"alpha": alpha, "camera_vertices": vertices_camera}
        if return_depth:
            depth_values, _ = dr.interpolate(
                vertices_camera[None, :, 2:3].contiguous(), rast, tri)
            result["depth"] = depth_values[0, ..., 0]
        if attributes is not None:
            attrs = attributes.to(self.device, torch.float32).unsqueeze(0)
            values, _ = dr.interpolate(attrs, rast, tri)
            result["attributes"] = values[0]
        if confidence is not None:
            conf, _ = dr.interpolate(confidence.to(self.device, torch.float32)[None, :, None],
                                     rast, tri)
            result["confidence"] = conf[0, ..., 0].clamp(0.0, 1.0)
        if return_raster:
            result["raster"] = rast[0]
            result["clip"] = clip[0]
        return result


def image_loss(rendered: dict, observation: Observation, weights: dict,
               spill_weight: float = 0.2):
    """Occlusion-tolerant mask/edge/feature objective from the implementation plan."""
    import torch
    import torch.nn.functional as F

    alpha = rendered["alpha"]
    observed = observation.mask
    denom = observed.sum().clamp_min(1.0)
    missing = (observed * (1.0 - alpha)).sum() / denom
    spill = ((1.0 - observed) * alpha).sum() / denom
    mask_term = missing + spill_weight * spill

    # A differentiable rendered boundary weighted by a precomputed observed-boundary DT.
    a = alpha[None, None]
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                           device=alpha.device, dtype=alpha.dtype)[None, None]
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(a, sobel_x, padding=1)
    gy = F.conv2d(a, sobel_y, padding=1)
    edge = torch.sqrt(gx.square() + gy.square() + 1e-8)[0, 0]
    boundary_term = (edge * observation.boundary_distance).sum() / edge.sum().clamp_min(1.0)

    feature_term = alpha.new_zeros(())
    if (observation.features is not None and "attributes" in rendered and
            "confidence" in rendered):
        pred = F.normalize(rendered["attributes"], dim=-1, eps=1e-6)
        target = F.normalize(observation.features, dim=-1, eps=1e-6)
        valid = observed * alpha * rendered["confidence"]
        feature_term = (valid * (1.0 - (pred * target).sum(dim=-1))).sum()
        feature_term = feature_term / valid.sum().clamp_min(1.0)

    # DA3 depth is useful but noisy, so compare relative z only on the eroded
    # observed mask where both the mesh and the depth map are valid. A Huber
    # penalty keeps isolated depth errors and occluders from dominating.
    depth_term = alpha.new_zeros(())
    if (observation.depth is not None and observation.depth_valid is not None
            and "depth" in rendered):
        predicted_depth = rendered["depth"]
        valid_depth = ((observation.depth_valid > 0.5) & (observed > 0.5)
                       & (alpha > 0.5) & torch.isfinite(predicted_depth))
        if bool(valid_depth.any()):
            relative = ((predicted_depth[valid_depth] - observation.depth[valid_depth])
                        / observation.depth[valid_depth].clamp_min(1e-3))
            absolute = relative.abs()
            delta = max(float(weights.get("depth_delta", 0.05)), 1e-6)
            robust = torch.where(
                absolute <= delta,
                0.5 * relative.square() / delta,
                absolute - 0.5 * delta,
            )
            depth_term = robust.mean()

    total = (weights.get("mask", 1.0) * mask_term +
             weights.get("boundary", 0.0) * boundary_term +
             weights.get("feature", 0.0) * feature_term +
             weights.get("depth", 0.0) * depth_term)
    return total, {
        "mask": mask_term.detach(),
        "boundary": boundary_term.detach(),
        "feature": feature_term.detach(),
        "depth": depth_term.detach(),
    }


def sample_vertex_features(renderer: DifferentiableMeshRenderer, vertices, faces,
                           camera, feature_map, observed_mask):
    """Attach reference-frame DINO features only to raster-visible mesh vertices."""
    import torch
    import torch.nn.functional as F

    rendered = renderer.render(vertices, faces, camera, return_raster=True)
    pixels, camera_vertices = renderer.project(vertices, camera)
    gx = 2.0 * pixels[:, 0] / max(renderer.w - 1, 1) - 1.0
    gy = 2.0 * pixels[:, 1] / max(renderer.h - 1, 1) - 1.0
    grid = torch.stack([gx, gy], dim=-1)[None, :, None, :]
    feat = F.grid_sample(feature_map.permute(2, 0, 1)[None], grid,
                         mode="bilinear", padding_mode="zeros", align_corners=True)
    feat = feat[0, :, :, 0].T.contiguous()
    sampled_mask = F.grid_sample(observed_mask[None, None], grid, mode="nearest",
                                 padding_mode="zeros", align_corners=True)[0, 0, :, 0]
    tri_ids = rendered["raster"][..., 3].to(torch.long) - 1
    visible_faces = torch.unique(tri_ids[tri_ids >= 0])
    visible_vertices = torch.zeros(vertices.shape[0], dtype=torch.bool, device=vertices.device)
    if visible_faces.numel():
        visible_vertices[torch.unique(faces[visible_faces].reshape(-1).long())] = True
    in_front = camera_vertices[:, 2] > 1e-4
    in_frame = (gx.abs() <= 1.0) & (gy.abs() <= 1.0)
    valid = visible_vertices & in_front & in_frame & (sampled_mask > 0.5)
    feat = F.normalize(feat, dim=-1, eps=1e-6)
    return feat.detach(), valid.to(torch.float32).detach()
