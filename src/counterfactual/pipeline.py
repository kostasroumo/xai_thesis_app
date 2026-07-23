from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic

from src.data.preprocessing import pil_to_model_rgb01, rgb01_to_normalized_tensor


@dataclass(frozen=True)
class CounterfactualSettings:
    slic_n_segments: int = 60
    slic_compactness: float = 12.0
    slic_sigma: float = 1.0
    blur_radius: float = 4.0
    max_steps: int = 18
    max_removal_fraction: float = 0.55


def _pil_to_rgb01(
    image: Image.Image,
    height: int,
    width: int,
    transform: Callable[[Image.Image], torch.Tensor],
) -> np.ndarray:
    rgb01 = pil_to_model_rgb01(image, transform)
    if rgb01.shape[:2] != (height, width):
        rgb01 = cv2.resize(
            rgb01.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    return np.ascontiguousarray(rgb01.astype(np.float32, copy=False))


def _slic_segments(
    image_rgb01: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
) -> np.ndarray:
    return slic(
        image_rgb01,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=-1,
        enforce_connectivity=True,
    ).astype(np.int64)


def _aggregate_superpixel_scores(cam: np.ndarray, seg: np.ndarray) -> np.ndarray:
    heat = np.abs(np.asarray(cam, dtype=np.float32))
    n_segments = int(seg.max()) + 1
    scores = np.zeros((n_segments,), dtype=np.float32)
    for sp_id in range(n_segments):
        mask = seg == sp_id
        scores[sp_id] = float(heat[mask].mean()) if np.any(mask) else 0.0
    return scores


def _blur_rgb01(image_rgb01: np.ndarray, radius: float) -> np.ndarray:
    radius = float(radius)
    if radius <= 0.0:
        return image_rgb01.copy()

    kernel = max(3, int(round(2 * radius + 1)))
    if kernel % 2 == 0:
        kernel += 1

    bgr = (np.clip(image_rgb01, 0.0, 1.0)[..., ::-1] * 255.0).astype(np.uint8)
    bgr_blurred = cv2.GaussianBlur(bgr, (kernel, kernel), 0)
    return bgr_blurred[..., ::-1].astype(np.float32) / 255.0


def _apply_baseline_to_superpixels(
    image_rgb01: np.ndarray,
    seg: np.ndarray,
    sp_ids: list[int],
    baseline_rgb01: np.ndarray,
) -> np.ndarray:
    output = image_rgb01.copy()
    if not sp_ids:
        return output
    mask = np.isin(seg, np.asarray(sp_ids, dtype=np.int64))
    output[mask] = baseline_rgb01[mask]
    return output


def _rgb01_to_input_tensor(
    rgb01: np.ndarray,
    transform: Callable[[Image.Image], torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return rgb01_to_normalized_tensor(rgb01, transform, device=device)


def _rgb01_to_uint8(rgb01: np.ndarray) -> np.ndarray:
    return (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)


def _predict_state(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names: Sequence[str],
    reference_class_index: int,
) -> dict[str, object]:
    with torch.no_grad():
        probabilities = F.softmax(model(input_tensor), dim=1)[0].detach().cpu().numpy().astype(np.float64)

    predicted_index = int(np.argmax(probabilities))
    predicted_class = (
        class_names[predicted_index] if predicted_index < len(class_names) else f"class_{predicted_index}"
    )
    reference_probability = float(probabilities[int(reference_class_index)])
    predicted_probability = float(probabilities[predicted_index])
    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "predicted_probability": predicted_probability,
        "reference_probability": reference_probability,
    }


def _build_removed_evidence_image(image_rgb01: np.ndarray, removed_mask: np.ndarray) -> np.ndarray:
    base_rgb = (np.clip(image_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    if not np.any(removed_mask):
        return base_rgb

    output = np.clip(base_rgb.astype(np.float32) * 0.24, 0, 255).astype(np.uint8)
    tinted = output.copy()
    tinted[removed_mask] = base_rgb[removed_mask]
    tint_color = np.asarray([245, 176, 65], dtype=np.float32)
    tinted[removed_mask] = np.clip(0.55 * tinted[removed_mask].astype(np.float32) + 0.45 * tint_color, 0, 255).astype(
        np.uint8
    )

    contour_mask = (removed_mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tinted, contours, -1, (255, 245, 220), 2)
    return tinted


def run_counterfactual_pipeline(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    class_names: Sequence[str],
    cam: np.ndarray,
    method_name: str,
    target_class: int,
    settings: CounterfactualSettings,
) -> dict[str, object]:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape [1, C, H, W].")
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D heatmap.")

    model.eval()
    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else input_tensor.device
    input_batch = input_tensor.to(device)

    original_state = _predict_state(model, input_batch, class_names, reference_class_index=target_class)
    height, width = cam.shape
    rgb01 = _pil_to_rgb01(image=image, height=height, width=width, transform=transform)
    seg = _slic_segments(
        image_rgb01=rgb01,
        n_segments=settings.slic_n_segments,
        compactness=settings.slic_compactness,
        sigma=settings.slic_sigma,
    )
    scores = _aggregate_superpixel_scores(cam=cam, seg=seg)
    ranking = np.argsort(np.abs(scores))[::-1].tolist()
    n_segments = len(ranking)
    max_regions_by_fraction = max(1, int(np.ceil(n_segments * float(settings.max_removal_fraction))))
    max_regions_to_try = min(n_segments, max(int(settings.max_steps), 1), max_regions_by_fraction)
    baseline_rgb01 = _blur_rgb01(rgb01, radius=settings.blur_radius)

    progression_rows: list[dict[str, object]] = []
    step_states: list[dict[str, object]] = []
    final_state = original_state
    final_rgb01 = rgb01.copy()
    final_removed_ids: list[int] = []
    final_removed_mask = np.zeros(seg.shape, dtype=bool)
    flip_found = False
    flip_step: int | None = None

    initial_removed_mask = np.zeros(seg.shape, dtype=bool)
    progression_rows.append(
        {
            "Step": 0,
            "Removed Superpixels": 0,
            "Removed Area (%)": 0.0,
            "Original Class Probability (%)": float(original_state["reference_probability"]) * 100.0,
            "Current Winner": str(original_state["predicted_class"]),
            "Current Winner Probability (%)": float(original_state["predicted_probability"]) * 100.0,
            "Flipped": False,
        }
    )
    step_states.append(
        {
            "Step": 0,
            "Removed Superpixels": 0,
            "Removed Area (%)": 0.0,
            "Current Winner": str(original_state["predicted_class"]),
            "Current Winner Probability": float(original_state["predicted_probability"]),
            "Original Class Probability": float(original_state["reference_probability"]),
            "Flipped": False,
            "counterfactual_rgb": _rgb01_to_uint8(rgb01),
            "removed_evidence_rgb": _build_removed_evidence_image(rgb01, initial_removed_mask),
        }
    )

    for step in range(1, max_regions_to_try + 1):
        removed_ids = ranking[:step]
        perturbed_rgb01 = _apply_baseline_to_superpixels(
            image_rgb01=rgb01,
            seg=seg,
            sp_ids=removed_ids,
            baseline_rgb01=baseline_rgb01,
        )
        perturbed_input = _rgb01_to_input_tensor(perturbed_rgb01, transform=transform, device=device)
        state = _predict_state(model, perturbed_input, class_names, reference_class_index=target_class)
        removed_mask = np.isin(seg, np.asarray(removed_ids, dtype=np.int64))
        removed_area_pct = float(removed_mask.mean() * 100.0) if removed_mask.size else 0.0

        progression_rows.append(
            {
                "Step": step,
                "Removed Superpixels": len(removed_ids),
                "Removed Area (%)": removed_area_pct,
                "Original Class Probability (%)": float(state["reference_probability"]) * 100.0,
                "Current Winner": str(state["predicted_class"]),
                "Current Winner Probability (%)": float(state["predicted_probability"]) * 100.0,
                "Flipped": bool(state["predicted_index"] != int(target_class)),
            }
        )
        step_states.append(
            {
                "Step": step,
                "Removed Superpixels": len(removed_ids),
                "Removed Area (%)": removed_area_pct,
                "Current Winner": str(state["predicted_class"]),
                "Current Winner Probability": float(state["predicted_probability"]),
                "Original Class Probability": float(state["reference_probability"]),
                "Flipped": bool(state["predicted_index"] != int(target_class)),
                "counterfactual_rgb": _rgb01_to_uint8(perturbed_rgb01),
                "removed_evidence_rgb": _build_removed_evidence_image(rgb01, removed_mask),
            }
        )

        final_state = state
        final_rgb01 = perturbed_rgb01
        final_removed_ids = removed_ids
        final_removed_mask = removed_mask

        if state["predicted_index"] != int(target_class):
            flip_found = True
            flip_step = step
            break

    removed_area_pct = float(final_removed_mask.mean() * 100.0) if final_removed_mask.size else 0.0
    original_class_name = str(original_state["predicted_class"])
    final_class_name = str(final_state["predicted_class"])

    if flip_found:
        summary_lines = [
            (
                f"Η θόλωση των {len(final_removed_ids)} σημαντικότερων superpixels από την εξήγηση {method_name} "
                f"άλλαξε την πρόβλεψη από {original_class_name} σε {final_class_name}."
            ),
            f"Τα αφαιρεμένα στοιχεία καλύπτουν το {removed_area_pct:.1f}% της εικόνας.",
            (
                f"Η πιθανότητα της αρχικής κλάσης έπεσε από {float(original_state['reference_probability']) * 100.0:.1f}% "
                f"σε {float(final_state['reference_probability']) * 100.0:.1f}%, ενώ η κλάση {final_class_name} ανέβηκε "
                f"στο {float(final_state['predicted_probability']) * 100.0:.1f}%."
            ),
        ]
    else:
        summary_lines = [
            (
                f"Ακόμη και μετά τη θόλωση των {len(final_removed_ids)} σημαντικότερων superpixels από την εξήγηση {method_name}, "
                f"το μοντέλο κράτησε την πρόβλεψη {original_class_name}."
            ),
            f"Τα αφαιρεμένα στοιχεία καλύπτουν το {removed_area_pct:.1f}% της εικόνας.",
            (
                f"Η πιθανότητα της αρχικής κλάσης μετακινήθηκε από {float(original_state['reference_probability']) * 100.0:.1f}% "
                f"σε {float(final_state['reference_probability']) * 100.0:.1f}% μέσα στο τρέχον όριο αφαίρεσης."
            ),
        ]

    return {
        "flip_found": flip_found,
        "flip_step": flip_step,
        "original_class": original_class_name,
        "original_confidence": float(original_state["reference_probability"]),
        "final_class": final_class_name,
        "final_confidence": float(final_state["predicted_probability"]),
        "final_original_class_probability": float(final_state["reference_probability"]),
        "removed_superpixel_count": len(final_removed_ids),
        "removed_area_pct": removed_area_pct,
        "removed_superpixel_ids": [int(sp_id) for sp_id in final_removed_ids],
        "summary_lines": summary_lines,
        "counterfactual_rgb": (np.clip(final_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8),
        "removed_evidence_rgb": _build_removed_evidence_image(rgb01, final_removed_mask),
        "progression_rows": progression_rows,
        "step_states": step_states,
    }
