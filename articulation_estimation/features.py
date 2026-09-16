"""Local DINOv2 extraction and PCA compression for rendered feature matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class DINOFeatures:
    def __init__(self, model_name: str, device: str, output_hw: tuple[int, int],
                 checkpoint: Path | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.output_hw = output_hw
        hub = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if not hub.is_dir():
            raise FileNotFoundError(
                f"local DINOv2 hub checkout missing at {hub}; pre-cache DINOv2 before Stage 09"
            )
        self.model = torch.hub.load(str(hub), model_name, source="local", pretrained=False)
        if checkpoint is None:
            names = {
                "dinov2_vitl14_reg": "dinov2_vitl14_reg4_pretrain.pth",
                "dinov2_vitb14_reg": "dinov2_vitb14_reg4_pretrain.pth",
                "dinov2_vits14_reg": "dinov2_vits14_reg4_pretrain.pth",
                "dinov2_vitg14_reg": "dinov2_vitg14_reg4_pretrain.pth",
            }
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / names.get(
                model_name, f"{model_name}_pretrain.pth")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINO checkpoint missing: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        self.pca_mean = None
        self.pca_basis = None

    def extract_patch_map(self, image_rgb: np.ndarray):
        import torch.nn.functional as F

        torch = self.torch
        # PIL-backed arrays can be read-only; copy before sharing storage with Torch.
        image = torch.as_tensor(np.array(image_rgb, copy=True), dtype=torch.float32,
                                device=self.device)
        image = image.permute(2, 0, 1)[None] / 255.0
        image = F.interpolate(image, size=self.output_hw, mode="bilinear", align_corners=False)
        image = (image - self.mean[None]) / self.std[None]
        with torch.inference_mode():
            tokens = self.model.forward_features(image)["x_norm_patchtokens"][0]
        patch = int(getattr(self.model, "patch_size", 14))
        ph, pw = self.output_hw[0] // patch, self.output_hw[1] // patch
        return tokens.reshape(ph, pw, -1).detach().cpu()

    def fit_pca(self, patch_maps: dict[int, object], dimensions: int):
        torch = self.torch
        samples = torch.cat([value.reshape(-1, value.shape[-1])
                             for value in patch_maps.values()], dim=0).float()
        if samples.shape[0] > 50000:
            generator = torch.Generator().manual_seed(0)
            samples = samples[torch.randperm(samples.shape[0], generator=generator)[:50000]]
        self.pca_mean = samples.mean(dim=0)
        centered = samples - self.pca_mean
        q = min(int(dimensions), centered.shape[0] - 1, centered.shape[1])
        if q < 1:
            raise ValueError("not enough DINO samples for PCA")
        _, _, basis = torch.pca_lowrank(centered, q=q, center=False, niter=4)
        self.pca_basis = basis[:, :q]

    def project(self, patch_map):
        import torch.nn.functional as F

        if self.pca_basis is None:
            raise RuntimeError("fit_pca must run before project")
        compressed = (patch_map.float() - self.pca_mean) @ self.pca_basis
        compressed = compressed.permute(2, 0, 1)[None].to(self.device)
        compressed = F.interpolate(compressed, size=self.output_hw, mode="bilinear",
                                   align_corners=False)[0].permute(1, 2, 0)
        return F.normalize(compressed, dim=-1, eps=1e-6)
