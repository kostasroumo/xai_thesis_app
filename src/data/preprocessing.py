from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Callable, Union

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import ResNet50_Weights

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
    weights: ResNet50_Weights = ResNet50_Weights.DEFAULT,
) -> Callable[[Image.Image], torch.Tensor]:
    return weights.transforms()


def preprocess_pil_image(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    tensor = transform(image)
    return tensor.unsqueeze(0)


def _convert_size_arg(size_value: object) -> int | tuple[int, ...]:
    if isinstance(size_value, int):
        return int(size_value)

    if isinstance(size_value, (list, tuple)):
        values = [int(value) for value in size_value]
        if len(values) == 1:
            return values[0]
        return tuple(values)

    return int(size_value)


def preprocess_spatial_pil_image(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> Image.Image:
    resize_size = getattr(transform, "resize_size", None)
    crop_size = getattr(transform, "crop_size", None)
    interpolation = getattr(transform, "interpolation", transforms.InterpolationMode.BILINEAR)
    antialias = getattr(transform, "antialias", True)

    if resize_size is None or crop_size is None:
        return image.convert("RGB")

    spatial_transform = transforms.Compose(
        [
            transforms.Resize(
                _convert_size_arg(resize_size),
                interpolation=interpolation,
                antialias=antialias,
            ),
            transforms.CenterCrop(_convert_size_arg(crop_size)),
        ]
    )
    return spatial_transform(image.convert("RGB"))


def pil_to_model_rgb01(
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
) -> np.ndarray:
    spatial_image = preprocess_spatial_pil_image(image, transform)
    rgb01 = np.asarray(spatial_image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(rgb01)


def rgb01_to_normalized_tensor(
    rgb01: np.ndarray,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device | None = None,
) -> torch.Tensor:
    rgb01 = np.clip(rgb01, 0.0, 1.0).astype(np.float32, copy=False)
    x01 = torch.from_numpy(np.ascontiguousarray(rgb01)).permute(2, 0, 1).unsqueeze(0)
    if device is not None:
        x01 = x01.to(device=device, dtype=torch.float32)
    else:
        x01 = x01.to(dtype=torch.float32)

    mean = getattr(transform, "mean", (0.485, 0.456, 0.406))
    std = getattr(transform, "std", (0.229, 0.224, 0.225))
    mean_t = torch.tensor(mean, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x01.dtype, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean_t) / std_t


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
