from __future__ import annotations

from torchvision.models import ResNet50_Weights

def get_imagenet_class_names(weights: ResNet50_Weights = ResNet50_Weights.DEFAULT) -> list[str]:
    categories = weights.meta.get("categories")
    if not categories:
        raise ValueError("ImageNet class names are missing from weights metadata.")
    return [str(name) for name in categories]
