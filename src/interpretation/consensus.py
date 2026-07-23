from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np
from PIL import Image
from skimage.segmentation import slic

from .summary_generator import RegionAnalysis


def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D array.")
    cam = np.clip(cam, 0.0, None)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)


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


def _superpixel_scores(cam: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    n_regions = int(segmentation.max()) + 1
    scores = np.zeros((n_regions,), dtype=np.float32)
    heat = np.abs(np.asarray(cam, dtype=np.float32))
    for region_id in range(n_regions):
        mask = segmentation == region_id
        scores[region_id] = float(heat[mask].mean()) if np.any(mask) else 0.0
    total = float(scores.sum())
    if total > 1e-8:
        scores = scores / total
    return scores.astype(np.float32)


def _scores_to_map(scores: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    output = np.zeros(segmentation.shape, dtype=np.float32)
    for region_id, score in enumerate(np.asarray(scores, dtype=np.float32)):
        output[segmentation == region_id] = float(score)
    return _normalize_map(output)


def _top_region_ids(scores: np.ndarray, top_k: int) -> list[int]:
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0 or float(scores.max()) < 1e-8:
        return []
    k = min(max(1, int(top_k)), len(scores))
    return np.argsort(scores)[::-1][:k].tolist()


def _region_centroid(segmentation: np.ndarray, region_id: int) -> tuple[float, float]:
    ys, xs = np.where(segmentation == region_id)
    if len(xs) == 0 or len(ys) == 0:
        return 0.5, 0.5
    return float(xs.mean() / max(segmentation.shape[1] - 1, 1)), float(ys.mean() / max(segmentation.shape[0] - 1, 1))


def _region_description(segmentation: np.ndarray, region_id: int) -> str:
    x_pos, y_pos = _region_centroid(segmentation, region_id)

    if x_pos < 0.33:
        x_label = "αριστερή"
    elif x_pos > 0.67:
        x_label = "δεξιά"
    else:
        x_label = "κεντρική"

    if y_pos < 0.33:
        y_label = "πάνω"
    elif y_pos > 0.67:
        y_label = "κάτω"
    else:
        y_label = "κεντρική"

    if x_label == "κεντρική" and y_label == "κεντρική":
        return "το κέντρο της εικόνας"
    if y_label == "κεντρική":
        return f"την {x_label} πλευρά της εικόνας"
    if x_label == "κεντρική":
        return f"το {y_label} μέρος της εικόνας"
    return f"την {y_label}-{x_label} περιοχή"


def _join_region_descriptions(descriptions: list[str]) -> str:
    unique_descriptions: list[str] = []
    for description in descriptions:
        if description not in unique_descriptions:
            unique_descriptions.append(description)

    if not unique_descriptions:
        return "διάσπαρτα σημεία της εικόνας"
    if len(unique_descriptions) == 1:
        return unique_descriptions[0]
    if len(unique_descriptions) == 2:
        return f"{unique_descriptions[0]} και {unique_descriptions[1]}"
    return f"{', '.join(unique_descriptions[:-1])}, και {unique_descriptions[-1]}"


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
    for region_id, score in enumerate(np.asarray(normalized_scores, dtype=np.float32)):
        region_mask = segmentation == region_id
        if np.any(region_mask & border_mask):
            total += float(score)
    return float(total)


def _concentration_label(top_mass: float) -> str:
    if top_mass >= 0.65:
        return "υψηλά συγκεντρωμένη"
    if top_mass >= 0.45:
        return "μέτρια συγκεντρωμένη"
    return "διάχυτη"


def _build_region_analysis(segmentation: np.ndarray, normalized_scores: np.ndarray, top_k: int) -> RegionAnalysis:
    top_ids = _top_region_ids(normalized_scores, top_k=top_k)
    top_mass = float(np.asarray(normalized_scores, dtype=np.float32)[top_ids].sum()) if top_ids else 0.0
    descriptions = [_region_description(segmentation, region_id) for region_id in top_ids]
    border_mass = _border_mass(segmentation=segmentation, normalized_scores=normalized_scores)
    return RegionAnalysis(
        segmentation=segmentation,
        normalized_scores=np.asarray(normalized_scores, dtype=np.float32),
        top_region_ids=top_ids,
        top_mass=top_mass,
        concentration_label=_concentration_label(top_mass),
        border_mass=border_mass,
        leakage_flag=border_mass >= 0.35,
        top_region_descriptions=descriptions,
        top_region_summary=_join_region_descriptions(descriptions),
    )


def _top_fraction_ids(scores: np.ndarray, top_fraction: float) -> set[int]:
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0 or float(scores.max()) < 1e-8:
        return set()
    k = max(1, int(np.ceil(len(scores) * float(top_fraction))))
    return set(np.argsort(scores)[::-1][:k].tolist())


def _cosine_similarity(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    flat_a = np.asarray(scores_a, dtype=np.float32).reshape(-1)
    flat_b = np.asarray(scores_b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(flat_a) * np.linalg.norm(flat_b))
    if denom < 1e-8:
        return 1.0 if float(np.linalg.norm(flat_a - flat_b)) < 1e-8 else 0.0
    return float(np.dot(flat_a, flat_b) / denom)


def _iou(ids_a: set[int], ids_b: set[int]) -> float:
    union = len(ids_a | ids_b)
    if union == 0:
        return 0.0
    return float(len(ids_a & ids_b) / union)


def _agreement_label(score: float) -> str:
    if score >= 0.7:
        return "υψηλή συμφωνία"
    if score >= 0.5:
        return "μέτρια συμφωνία"
    return "χαμηλή συμφωνία"


def build_consensus_analysis(
    image: Image.Image,
    method_cams: Mapping[str, np.ndarray],
    *,
    n_segments: int = 50,
    compactness: float = 10.0,
    sigma: float = 1.0,
    top_k: int = 3,
    top_fraction: float = 0.2,
) -> dict[str, object]:
    method_names = list(method_cams.keys())
    if len(method_names) < 2:
        raise ValueError("At least two methods are required to build a consensus analysis.")

    reference_cam = _normalize_map(np.asarray(method_cams[method_names[0]], dtype=np.float32))
    height, width = reference_cam.shape
    image_rgb01 = _image_to_rgb01(image=image, width=width, height=height)
    segmentation = _compute_segments(
        image_rgb01=image_rgb01,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
    )

    method_superpixel_scores: dict[str, np.ndarray] = {}
    for method_name in method_names:
        cam = _normalize_map(np.asarray(method_cams[method_name], dtype=np.float32))
        if cam.shape != (height, width):
            raise ValueError("All method cams must share the same height and width.")
        method_superpixel_scores[method_name] = _superpixel_scores(cam=cam, segmentation=segmentation)

    score_matrix = np.stack([method_superpixel_scores[method_name] for method_name in method_names], axis=0)
    consensus_scores = np.mean(score_matrix, axis=0).astype(np.float32)
    disagreement_scores = np.std(score_matrix, axis=0).astype(np.float32)
    vote_counts = np.zeros(consensus_scores.shape, dtype=np.int32)
    for method_name in method_names:
        for region_id in _top_fraction_ids(method_superpixel_scores[method_name], top_fraction):
            vote_counts[region_id] += 1

    required_votes = max(2, int(np.ceil(len(method_names) * 0.67)))
    shared_evidence_scores = consensus_scores * (vote_counts.astype(np.float32) / float(len(method_names)))
    shared_evidence_scores = np.where(vote_counts >= required_votes, shared_evidence_scores, 0.0).astype(np.float32)

    shared_region = _build_region_analysis(
        segmentation=segmentation,
        normalized_scores=shared_evidence_scores,
        top_k=top_k,
    )
    disagreement_region = _build_region_analysis(
        segmentation=segmentation,
        normalized_scores=disagreement_scores,
        top_k=top_k,
    )

    pairwise_rows: list[dict[str, object]] = []
    pairwise_cosines: list[float] = []
    pairwise_ious: list[float] = []
    for method_a, method_b in combinations(method_names, 2):
        scores_a = method_superpixel_scores[method_a]
        scores_b = method_superpixel_scores[method_b]
        cosine = _cosine_similarity(scores_a, scores_b)
        iou = _iou(_top_fraction_ids(scores_a, top_fraction), _top_fraction_ids(scores_b, top_fraction))
        pairwise_cosines.append(cosine)
        pairwise_ious.append(iou)
        pairwise_rows.append(
            {
                "Methods": f"{method_a} vs {method_b}",
                "Cosine Agreement": cosine,
                "Top-focus IoU": iou,
            }
        )

    method_rows: list[dict[str, object]] = []
    shared_top_ids = _top_fraction_ids(shared_evidence_scores, top_fraction)
    for method_name in method_names:
        scores = method_superpixel_scores[method_name]
        method_rows.append(
            {
                "Method": method_name,
                "Shared Cosine": _cosine_similarity(scores, shared_evidence_scores),
                "Shared IoU": _iou(_top_fraction_ids(scores, top_fraction), shared_top_ids),
            }
        )

    mean_pairwise_cosine = float(np.mean(pairwise_cosines)) if pairwise_cosines else 0.0
    mean_pairwise_iou = float(np.mean(pairwise_ious)) if pairwise_ious else 0.0
    consensus_strength = float((mean_pairwise_cosine + mean_pairwise_iou) / 2.0)
    agreement_label = _agreement_label(consensus_strength)

    if shared_region.top_mass > 0.0:
        shared_line = (
            f"Τα ισχυρότερα κοινά στοιχεία εμφανίζονται γύρω από {shared_region.top_region_summary}."
        )
    else:
        shared_line = "Κανένα ισχυρό κοινό superpixel δεν πέρασε τον τρέχοντα κανόνα συμφωνίας σε αυτή την εκτέλεση."

    summary_lines = [
        f"Η κοινή εστίαση κρατά μόνο τα superpixels που επισημαίνονται από τουλάχιστον {required_votes} από τις {len(method_names)} μεθόδους.",
        shared_line,
        (
            f"Η συμφωνία υπολογίζεται σε κοινό SLIC πλέγμα superpixels: μέσο pairwise cosine {mean_pairwise_cosine:.3f}, "
            f"μέσο top-focus IoU {mean_pairwise_iou:.3f}."
        ),
    ]

    return {
        "method_names": method_names,
        "segmentation": segmentation,
        "required_votes": required_votes,
        "vote_counts": vote_counts,
        "consensus_scores": consensus_scores,
        "shared_evidence_scores": shared_evidence_scores,
        "disagreement_scores": disagreement_scores,
        "shared_evidence_map": _scores_to_map(shared_evidence_scores, segmentation),
        "consensus_map": _scores_to_map(consensus_scores, segmentation),
        "disagreement_map": _scores_to_map(disagreement_scores, segmentation),
        "shared_region": shared_region,
        "disagreement_region": disagreement_region,
        "mean_pairwise_cosine": mean_pairwise_cosine,
        "mean_pairwise_iou": mean_pairwise_iou,
        "consensus_strength": consensus_strength,
        "agreement_label": agreement_label,
        "summary_lines": summary_lines,
        "pairwise_rows": pairwise_rows,
        "method_rows": method_rows,
    }


__all__ = ["build_consensus_analysis"]
