from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Union

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights

from src.utils.config import DEFAULT_WEIGHTS

ImageInput = Union[str, Path, bytes, bytearray, BinaryIO]


def load_image(image_input: ImageInput) -> Image.Image:
    try:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input)
        elif isinstance(image_input, (bytes, bytearray)):
            image = Image.open(BytesIO(image_input))
        else:
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            image = Image.open(image_input)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Failed to load image. Please provide a valid image file.") from exc


def get_inference_transform(
    weights: ResNet50_Weights = DEFAULT_WEIGHTS,
) -> Callable[[Image.Image], torch.Tensor]:
    return weights.transforms()


def preprocess_pil_image(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    tensor = transform(image)
    return tensor.unsqueeze(0)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    if tensor.ndim not in (3, 4):
        raise ValueError("tensor must have shape [C, H, W] or [B, C, H, W].")

    input_tensor = tensor.detach().clone()
    was_3d = input_tensor.ndim == 3
    if was_3d:
        input_tensor = input_tensor.unsqueeze(0)

    mean_t = torch.tensor(mean, dtype=input_tensor.dtype, device=input_tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=input_tensor.dtype, device=input_tensor.device).view(1, 3, 1, 1)

    output = input_tensor * std_t + mean_t
    output = output.clamp(0.0, 1.0)

    return output.squeeze(0) if was_3d else output
