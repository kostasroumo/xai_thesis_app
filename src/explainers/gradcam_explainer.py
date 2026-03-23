from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._forward_handle = None
        self._backward_handle = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            self._activations = output.detach()

        def backward_hook(
            _module: nn.Module,
            _grad_input: tuple[torch.Tensor, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            self._gradients = grad_output[0].detach()

        self._forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def close(self) -> None:
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        score_type: Literal["logit", "prob"] = "logit",
    ) -> np.ndarray:
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError("input_tensor must have shape [1, C, H, W].")
        if score_type not in {"logit", "prob"}:
            raise ValueError("score_type must be either 'logit' or 'prob'.")

        device = next(self.model.parameters()).device
        input_batch = input_tensor.to(device)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        output = self.model(input_batch)
        if target_class is None:
            target_class = int(output.argmax(dim=1).item())

        if score_type == "logit":
            score = output[:, target_class]
        else:
            probabilities = F.softmax(output, dim=1)
            score = probabilities[:, target_class]
        score.backward(retain_graph=False)

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        pooled_gradients = self._gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = pooled_gradients * self._activations

        cam = weighted_activations.sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_batch.shape[-2:], mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().detach().cpu().numpy().astype(np.float32)
        cam_max = float(cam_np.max())
        if cam_max < 1e-8:
            return np.zeros_like(cam_np, dtype=np.float32)
        return cam_np / cam_max

    def __del__(self) -> None:
        self.close()
