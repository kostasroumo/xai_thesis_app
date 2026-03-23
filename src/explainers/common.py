from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from src.data.preprocessing import preprocess_pil_image

ScoreType = Literal["logit", "prob"]


def build_target_score_forward(
    model: nn.Module,
    target_class: int,
    score_type: ScoreType,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if score_type not in {"logit", "prob"}:
        raise ValueError("score_type must be either 'logit' or 'prob'.")

    def forward_fn(inputs: torch.Tensor) -> torch.Tensor:
        logits = model(inputs)
        if score_type == "logit":
            return logits[:, int(target_class)]
        probabilities = torch.softmax(logits, dim=1)
        return probabilities[:, int(target_class)]

    return forward_fn


def build_blur_baseline(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
    blur_radius: float,
) -> torch.Tensor:
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return preprocess_pil_image(blurred, transform).to(device)


def attributions_to_normalized_heatmap(attributions: torch.Tensor) -> np.ndarray:
    if attributions.ndim != 4 or attributions.size(0) != 1:
        raise ValueError("Expected attributions with shape [1, C, H, W].")

    attrs_np = attributions.detach().float().cpu().numpy()[0]
    heatmap = np.mean(np.abs(attrs_np), axis=0).astype(np.float32)

    heatmap_min = float(heatmap.min())
    heatmap_max = float(heatmap.max())
    if heatmap_max - heatmap_min < 1e-8:
        return np.zeros_like(heatmap, dtype=np.float32)

    return (heatmap - heatmap_min) / (heatmap_max - heatmap_min)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
