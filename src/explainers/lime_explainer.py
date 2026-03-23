from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from captum.attr import Lime
from skimage.segmentation import slic

from src.explainers.common import (
    ScoreType,
    attributions_to_normalized_heatmap,
    build_blur_baseline,
    build_target_score_forward,
    seed_everything,
)


def _build_slic_feature_mask(
    image: Image.Image,
    height: int,
    width: int,
    n_segments: int,
    compactness: float,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    resized = image.resize((width, height))
    rgb01 = np.asarray(resized).astype(np.float32) / 255.0
    segments = slic(
        rgb01,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
    ).astype(np.int64)
    return torch.from_numpy(segments)[None, None].to(device)


def generate_lime(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    target_class: int,
    score_type: ScoreType = "logit",
    n_samples: int = 600,
    perturbations_per_eval: int = 64,
    n_segments: int = 70,
    compactness: float = 10.0,
    sigma: float = 1.0,
    blur_radius: float = 2.0,
    random_seed: int = 0,
) -> np.ndarray:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape [1, C, H, W].")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if perturbations_per_eval <= 0:
        raise ValueError("perturbations_per_eval must be positive.")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive.")

    seed_everything(int(random_seed))
    model.eval()
    device = next(model.parameters()).device
    input_batch = input_tensor.to(device)

    _, _, height, width = input_batch.shape
    feature_mask = _build_slic_feature_mask(
        image=image,
        height=height,
        width=width,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        device=device,
    )
    baseline = build_blur_baseline(image, transform, device=device, blur_radius=blur_radius)
    forward_fn = build_target_score_forward(model, target_class=target_class, score_type=score_type)

    lime = Lime(forward_func=forward_fn)
    attributions = lime.attribute(
        inputs=input_batch,
        baselines=baseline,
        target=None,
        feature_mask=feature_mask,
        n_samples=int(n_samples),
        perturbations_per_eval=int(perturbations_per_eval),
        show_progress=False,
    )
    return attributions_to_normalized_heatmap(attributions)
