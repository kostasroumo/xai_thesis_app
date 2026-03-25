from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.utils.config import DEFAULT_WEIGHTS


def build_model(weights: ResNet50_Weights = DEFAULT_WEIGHTS) -> nn.Module:
    model = resnet50(weights=weights)
    model.eval()
    return model


def load_model(
    device: Optional[torch.device] = None,
    weights: ResNet50_Weights = DEFAULT_WEIGHTS,
) -> tuple[nn.Module, ResNet50_Weights]:
    # Keep CPU threading conservative for cloud stability.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can be raised if called after parallel work has started.
        pass

    model = build_model(weights=weights)
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(resolved_device)
    model.eval()
    return model, weights


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    if not hasattr(model, "layer4"):
        raise ValueError("The provided model does not have layer4.")

    layer4 = getattr(model, "layer4")
    if len(layer4) == 0:
        raise ValueError("layer4 is empty.")

    last_block = layer4[-1]
    if not hasattr(last_block, "conv3"):
        raise ValueError("Expected a Bottleneck block with conv3 in layer4.")

    return last_block.conv3
