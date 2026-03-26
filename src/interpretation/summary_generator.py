from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic


@dataclass(frozen=True)
class RegionAnalysis:
    segmentation: np.ndarray
    normalized_scores: np.ndarray
    top_region_ids: list[int]
    top_mass: float
    concentration_label: str
    border_mass: float
    leakage_flag: bool
    top_region_descriptions: list[str]
    top_region_summary: str


def _image_to_rgb01(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.resize((width, height))
    return np.asarray(resized).astype(np.float32) / 255.0


def _compute_segments(
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
    ).astype(np.int64)


def _region_scores(cam: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    n_regions = int(segmentation.max()) + 1
    scores = np.zeros((n_regions,), dtype=np.float32)
    heat = np.abs(cam.astype(np.float32))
    for region_id in range(n_regions):
        mask = segmentation == region_id
        scores[region_id] = float(heat[mask].mean()) if np.any(mask) else 0.0
    return scores


def _region_centroid(segmentation: np.ndarray, region_id: int) -> tuple[float, float]:
    ys, xs = np.where(segmentation == region_id)
    if len(xs) == 0 or len(ys) == 0:
        return 0.5, 0.5
    return float(xs.mean() / max(segmentation.shape[1] - 1, 1)), float(ys.mean() / max(segmentation.shape[0] - 1, 1))


def _region_description(segmentation: np.ndarray, region_id: int) -> str:
    x_pos, y_pos = _region_centroid(segmentation, region_id)

    if x_pos < 0.33:
        x_label = "left"
    elif x_pos > 0.67:
        x_label = "right"
    else:
        x_label = "center"

    if y_pos < 0.33:
        y_label = "upper"
    elif y_pos > 0.67:
        y_label = "lower"
    else:
        y_label = "central"

    if x_label == "center" and y_label == "central":
        return "the center of the image"
    if y_label == "central":
        return f"the {x_label} side of the image"
    if x_label == "center":
        return f"the {y_label} part of the image"
    return f"the {y_label}-{x_label} area"


def _join_region_descriptions(descriptions: list[str]) -> str:
    unique_descriptions: list[str] = []
    for description in descriptions:
        if description not in unique_descriptions:
            unique_descriptions.append(description)

    if not unique_descriptions:
        return "multiple scattered parts of the image"
    if len(unique_descriptions) == 1:
        return unique_descriptions[0]
    if len(unique_descriptions) == 2:
        return f"{unique_descriptions[0]} and {unique_descriptions[1]}"
    return f"{', '.join(unique_descriptions[:-1])}, and {unique_descriptions[-1]}"


def _border_mass(segmentation: np.ndarray, normalized_scores: np.ndarray, border_ratio: float = 0.12) -> float:
    height, width = segmentation.shape
    border_h = max(1, int(round(height * border_ratio)))
    border_w = max(1, int(round(width * border_ratio)))

    border_mask = np.zeros_like(segmentation, dtype=bool)
    border_mask[:border_h, :] = True
    border_mask[-border_h:, :] = True
    border_mask[:, :border_w] = True
    border_mask[:, -border_w:] = True

    total = 0.0
    for region_id, score in enumerate(normalized_scores):
        region_mask = segmentation == region_id
        if np.any(region_mask & border_mask):
            total += float(score)
    return float(total)


def _concentration_label(top_mass: float) -> str:
    if top_mass >= 0.65:
        return "highly concentrated"
    if top_mass >= 0.45:
        return "moderately concentrated"
    return "diffuse"


def analyze_regions(
    image: Image.Image,
    cam: np.ndarray,
    n_segments: int = 50,
    compactness: float = 10.0,
    sigma: float = 1.0,
    top_k: int = 3,
) -> RegionAnalysis:
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D array.")

    height, width = cam.shape
    image_rgb01 = _image_to_rgb01(image, width=width, height=height)
    segmentation = _compute_segments(
        image_rgb01=image_rgb01,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
    )
    scores = _region_scores(cam=cam, segmentation=segmentation)
    total_score = float(scores.sum())
    normalized_scores = scores / total_score if total_score > 1e-8 else np.zeros_like(scores)

    k = min(max(1, int(top_k)), len(normalized_scores))
    top_region_ids = np.argsort(normalized_scores)[::-1][:k].tolist()
    top_mass = float(normalized_scores[top_region_ids].sum()) if top_region_ids else 0.0
    descriptions = [_region_description(segmentation, region_id) for region_id in top_region_ids]
    border_mass = _border_mass(segmentation=segmentation, normalized_scores=normalized_scores)
    leakage_flag = border_mass >= 0.35

    return RegionAnalysis(
        segmentation=segmentation,
        normalized_scores=normalized_scores,
        top_region_ids=top_region_ids,
        top_mass=top_mass,
        concentration_label=_concentration_label(top_mass),
        border_mass=border_mass,
        leakage_flag=leakage_flag,
        top_region_descriptions=descriptions,
        top_region_summary=_join_region_descriptions(descriptions),
    )


def build_simplified_focus_image(
    image: Image.Image,
    region_analysis: RegionAnalysis,
    dim_factor: float = 0.2,
) -> np.ndarray:
    height, width = region_analysis.segmentation.shape
    base_rgb = np.asarray(image.resize((width, height))).astype(np.uint8)
    dimmed_rgb = np.clip(base_rgb.astype(np.float32) * float(dim_factor), 0, 255).astype(np.uint8)

    mask = np.isin(region_analysis.segmentation, np.asarray(region_analysis.top_region_ids, dtype=np.int64))
    focus_rgb = dimmed_rgb.copy()
    focus_rgb[mask] = base_rgb[mask]

    contour_mask = (mask.astype(np.uint8) * 255).copy()
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(focus_rgb, contours, -1, (255, 210, 90), 2)
    return focus_rgb


def generate_summary_text(
    predicted_class: str,
    confidence: float,
    method_name: str,
    region_analysis: RegionAnalysis,
) -> list[str]:
    confidence_pct = round(float(confidence) * 100.0, 1)
    top_mass_pct = round(float(region_analysis.top_mass) * 100.0, 1)

    leakage_note = ""
    if region_analysis.leakage_flag:
        leakage_note = " Some importance also leaks toward the image borders, so background cues may still influence the decision."

    return [
        f"The model predicted {predicted_class} with {confidence_pct}% confidence.",
        f"The {method_name} explanation is {region_analysis.concentration_label}.",
        f"The top {len(region_analysis.top_region_ids)} regions account for about {top_mass_pct}% of the total importance.",
        f"The explanation focuses mainly on {region_analysis.top_region_summary}.{leakage_note}",
    ]


def compare_method_regions(
    region_analyses: dict[str, RegionAnalysis],
    top_fraction: float = 0.2,
) -> dict[str, object]:
    method_names = list(region_analyses.keys())
    if len(method_names) < 2:
        return {
            "mean_pairwise_iou": 1.0,
            "consistency_label": "single method",
            "summary": "Only one method is currently selected.",
            "pairwise_rows": [],
        }

    top_sets: dict[str, set[int]] = {}
    for method_name, analysis in region_analyses.items():
        scores = analysis.normalized_scores
        k = max(1, int(np.ceil(len(scores) * float(top_fraction))))
        top_sets[method_name] = set(np.argsort(scores)[::-1][:k].tolist())

    pairwise_rows: list[dict[str, object]] = []
    ious: list[float] = []
    for method_a, method_b in combinations(method_names, 2):
        set_a = top_sets[method_a]
        set_b = top_sets[method_b]
        union = len(set_a | set_b)
        iou = float(len(set_a & set_b) / union) if union else 0.0
        ious.append(iou)
        pairwise_rows.append(
            {
                "Methods": f"{method_a} vs {method_b}",
                "Top-region IoU": round(iou, 3),
            }
        )

    mean_pairwise_iou = float(np.mean(ious)) if ious else 0.0
    if mean_pairwise_iou >= 0.55:
        consistency_label = "high consistency"
    elif mean_pairwise_iou >= 0.3:
        consistency_label = "moderate consistency"
    else:
        consistency_label = "low consistency"

    common_region_ids = set.intersection(*top_sets.values()) if top_sets else set()
    common_summary = ""
    if common_region_ids:
        reference_analysis = region_analyses[method_names[0]]
        descriptions = [
            _region_description(reference_analysis.segmentation, region_id)
            for region_id in sorted(common_region_ids)
        ]
        common_summary = _join_region_descriptions(descriptions)

    if common_summary:
        summary = (
            f"The selected methods show {consistency_label}. "
            f"They overlap most clearly around {common_summary}."
        )
    else:
        summary = (
            f"The selected methods show {consistency_label}. "
            "Their most important regions only partially overlap."
        )

    return {
        "mean_pairwise_iou": mean_pairwise_iou,
        "consistency_label": consistency_label,
        "summary": summary,
        "pairwise_rows": pairwise_rows,
    }
