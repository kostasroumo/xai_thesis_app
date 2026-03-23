from __future__ import annotations

from torchvision.models import ResNet50_Weights

from src.utils.config import DEFAULT_WEIGHTS


def get_imagenet_class_names(weights: ResNet50_Weights = DEFAULT_WEIGHTS) -> list[str]:
    categories = weights.meta.get("categories")
    if not categories:
        raise ValueError("ImageNet class names are missing from weights metadata.")
    return [str(name) for name in categories]
