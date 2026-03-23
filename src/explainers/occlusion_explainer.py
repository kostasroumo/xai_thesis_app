from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from captum.attr import Occlusion

from src.explainers.common import (
    ScoreType,
    attributions_to_normalized_heatmap,
    build_blur_baseline,
    build_target_score_forward,
)


def generate_occlusion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    target_class: int,
    score_type: ScoreType = "logit",
    patch_size: int = 24,
    stride: int = 12,
    blur_radius: float = 4.0,
) -> np.ndarray:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape [1, C, H, W].")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    model.eval()
    device = next(model.parameters()).device
    input_batch = input_tensor.to(device)
    baseline = build_blur_baseline(image, transform, device=device, blur_radius=blur_radius)

    forward_fn = build_target_score_forward(model, target_class=target_class, score_type=score_type)
    occlusion = Occlusion(forward_fn)
    attributions = occlusion.attribute(
        inputs=input_batch,
        baselines=baseline,
        target=None,
        sliding_window_shapes=(3, int(patch_size), int(patch_size)),
        strides=(3, int(stride), int(stride)),
    )
    return attributions_to_normalized_heatmap(attributions)
