from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from captum.attr import IntegratedGradients

from src.explainers.common import (
    ScoreType,
    attributions_to_normalized_heatmap,
    build_blur_baseline,
    build_target_score_forward,
)


def generate_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    target_class: int,
    score_type: ScoreType = "logit",
    n_steps: int = 50,
    internal_batch_size: int = 16,
    blur_radius: float = 4.0,
) -> np.ndarray:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape [1, C, H, W].")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if internal_batch_size <= 0:
        raise ValueError("internal_batch_size must be positive.")

    model.eval()
    device = next(model.parameters()).device
    input_batch = input_tensor.to(device)
    baseline = build_blur_baseline(image, transform, device=device, blur_radius=blur_radius)

    forward_fn = build_target_score_forward(model, target_class=target_class, score_type=score_type)
    ig = IntegratedGradients(forward_fn)
    attributions = ig.attribute(
        input_batch,
        baselines=baseline,
        target=None,
        n_steps=int(n_steps),
        internal_batch_size=int(internal_batch_size),
    )
    return attributions_to_normalized_heatmap(attributions)
