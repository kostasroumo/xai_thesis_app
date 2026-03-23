from __future__ import annotations

import cv2
import numpy as np
import torch


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError("Expected batch size 1 for a 4D tensor.")
        tensor = tensor.squeeze(0)

    if tensor.ndim != 3 or tensor.size(0) != 3:
        raise ValueError("Expected tensor shape [3, H, W] or [1, 3, H, W].")

    image = tensor.detach().cpu().float().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    return (image * 255.0).round().astype(np.uint8)


def apply_colormap_to_cam(cam: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D array.")

    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape [H, W, 3].")
    if heatmap_rgb.ndim != 3 or heatmap_rgb.shape[2] != 3:
        raise ValueError("heatmap_rgb must have shape [H, W, 3].")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1].")

    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

    if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
        heatmap_rgb = cv2.resize(
            heatmap_rgb,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0.0)
