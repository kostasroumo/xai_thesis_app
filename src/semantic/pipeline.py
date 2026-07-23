from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
from PIL import Image
from skimage.segmentation import slic

CONCEPTS_EN: tuple[str, ...] = (
    "animal face",
    "animal ears",
    "animal eyes",
    "animal nose",
    "animal fur",
    "animal body",
    "background",
)

CONCEPTS_GR: dict[str, str] = {
    "animal face": "πρόσωπο",
    "animal ears": "αυτιά",
    "animal eyes": "μάτια",
    "animal nose": "ρύγχος",
    "animal fur": "τρίχωμα",
    "animal body": "σώμα",
    "background": "υπόβαθρο",
}


@dataclass(frozen=True)
class SemanticSettings:
    slic_n_segments: int = 80
    slic_compactness: float = 10.0
    slic_sigma: float = 1.0
    top_k_superpixels: int = 10
    crop_pad: int = 30
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"


@dataclass(frozen=True)
class SemanticRuntime:
    clip_model: Any
    clip_preprocess: Callable[[Image.Image], torch.Tensor]
    device: torch.device
    text_features: torch.Tensor
    concepts_en: tuple[str, ...]
    concepts_gr: Mapping[str, str]


def _load_open_clip() -> Any:
    try:
        import open_clip  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Το σημασιολογικό layer χρειάζεται το `open_clip_torch`. Πρόσθεσέ το στο περιβάλλον για να ενεργοποιηθεί η σημασιολογική ανάλυση."
        ) from exc
    return open_clip


@torch.no_grad()
def build_semantic_runtime(
    settings: SemanticSettings | None = None,
    device: torch.device | None = None,
) -> SemanticRuntime:
    resolved_settings = settings or SemanticSettings()
    resolved_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    open_clip = _load_open_clip()

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        resolved_settings.clip_model_name,
        pretrained=resolved_settings.clip_pretrained,
    )
    clip_model = clip_model.to(resolved_device).eval()

    tokenizer = open_clip.get_tokenizer(resolved_settings.clip_model_name)
    text_tokens = tokenizer(list(CONCEPTS_EN)).to(resolved_device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return SemanticRuntime(
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        device=resolved_device,
        text_features=text_features,
        concepts_en=CONCEPTS_EN,
        concepts_gr=CONCEPTS_GR,
    )


def _pil_to_rgb01(image: Image.Image, width: int, height: int) -> np.ndarray:
    resized = image.resize((width, height))
    return np.asarray(resized).astype(np.float32) / 255.0


def _slic_segments(image_rgb01: np.ndarray, settings: SemanticSettings) -> np.ndarray:
    return slic(
        image_rgb01,
        n_segments=int(settings.slic_n_segments),
        compactness=float(settings.slic_compactness),
        sigma=float(settings.slic_sigma),
        start_label=0,
    ).astype(np.int64)


def _aggregate_superpixel_scores(heat2d: np.ndarray, seg: np.ndarray) -> np.ndarray:
    if heat2d.shape != seg.shape:
        raise ValueError("heat2d and seg must have the same shape.")

    n_segments = int(seg.max()) + 1
    scores = np.zeros((n_segments,), dtype=np.float32)
    heat = np.abs(heat2d.astype(np.float32))
    for sp_id in range(n_segments):
        mask = seg == sp_id
        scores[sp_id] = float(heat[mask].mean()) if np.any(mask) else 0.0
    return scores


def _build_focus_image(
    img_rgb01: np.ndarray,
    seg: np.ndarray,
    sp_scores: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    n_regions = len(sp_scores)
    if n_regions == 0:
        return np.zeros_like(img_rgb01), np.zeros(seg.shape, dtype=bool), []

    k = min(max(1, int(top_k)), n_regions)
    top_ids = np.argsort(np.abs(sp_scores))[::-1][:k].tolist()
    mask = np.isin(seg, np.asarray(top_ids, dtype=np.int64))

    focus_img = img_rgb01.copy()
    focus_img[~mask] = 0.0
    return focus_img, mask, top_ids


@torch.no_grad()
def _classify_crop_with_clip(crop_pil: Image.Image, runtime: SemanticRuntime) -> tuple[str, float]:
    image_input = runtime.clip_preprocess(crop_pil).unsqueeze(0).to(runtime.device)
    image_features = runtime.clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    sim = image_features @ runtime.text_features.T
    probs = sim.softmax(dim=-1)[0].detach().cpu().numpy().astype(np.float32)
    best_idx = int(np.argmax(probs))
    return str(runtime.concepts_en[best_idx]), float(probs[best_idx])


def _crop_superpixel(
    img_rgb01: np.ndarray,
    seg: np.ndarray,
    sp_id: int,
    pad: int,
) -> Image.Image:
    mask = seg == sp_id
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return Image.fromarray((np.clip(img_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8))

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())

    y1 = max(0, y1 - int(pad))
    y2 = min(img_rgb01.shape[0] - 1, y2 + int(pad))
    x1 = max(0, x1 - int(pad))
    x2 = min(img_rgb01.shape[1] - 1, x2 + int(pad))

    crop = img_rgb01[y1 : y2 + 1, x1 : x2 + 1]
    return Image.fromarray((np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8))


def _semantic_from_superpixels(
    img_rgb01: np.ndarray,
    seg: np.ndarray,
    sp_scores: np.ndarray,
    runtime: SemanticRuntime,
    top_superpixel_ids: list[int],
    crop_pad: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    concept_scores = {concept: 0.0 for concept in runtime.concepts_en}
    details: list[dict[str, Any]] = []

    for sp_id in top_superpixel_ids:
        importance = float(abs(sp_scores[int(sp_id)]))
        crop_pil = _crop_superpixel(img_rgb01, seg, int(sp_id), pad=int(crop_pad))
        concept_en, clip_conf = _classify_crop_with_clip(crop_pil, runtime)

        concept_scores[concept_en] += importance
        details.append(
            {
                "Superpixel ID": int(sp_id),
                "Σημασία": importance,
                "Έννοια": str(runtime.concepts_gr[concept_en]),
                "CLIP Βεβαιότητα": float(clip_conf),
            }
        )

    total = float(sum(concept_scores.values())) + 1e-8
    concept_scores_pct = {
        str(runtime.concepts_gr[concept_en]): 100.0 * float(value) / total
        for concept_en, value in concept_scores.items()
        if float(value) > 0.0
    }
    concept_scores_pct = dict(sorted(concept_scores_pct.items(), key=lambda item: item[1], reverse=True))
    return concept_scores_pct, pd.DataFrame(details)


def _top_concepts_text(scores: Mapping[str, float], k: int = 3) -> str:
    names = [str(name) for name, _ in list(scores.items())[:k]]
    if len(names) >= 3:
        return f"{names[0]}, {names[1]} και {names[2]}"
    if len(names) == 2:
        return f"{names[0]} και {names[1]}"
    if len(names) == 1:
        return names[0]
    return "Δεν προέκυψε σταθερή semantic εξήγηση."


def _make_greek_summary(
    predicted_class: str,
    confidence: float,
    concept_scores: Mapping[str, float],
) -> str:
    if not concept_scores:
        return "Δεν προέκυψε σταθερή semantic εξήγηση."

    background_score = float(concept_scores.get("υπόβαθρο", 0.0))
    top_text = _top_concepts_text(concept_scores, k=3)

    if background_score >= 25.0:
        return (
            f"Το μοντέλο ταξινόμησε την εικόνα ως {predicted_class} με πιθανότητα {confidence:.1%}. "
            "Η semantic εξήγηση είναι μέτριας αξιοπιστίας, "
            "καθώς σημαντικό μέρος της απόδοσης σχετίζεται με το υπόβαθρο. "
            f"Τα κυρίαρχα concepts ήταν: {top_text}."
        )

    return (
        f"Το μοντέλο ταξινόμησε την εικόνα ως {predicted_class} "
        f"με πιθανότητα {confidence:.1%}, με κύρια εστίαση σε "
        f"{top_text}."
    )


def _build_score_table(concept_scores: Mapping[str, float]) -> pd.DataFrame:
    rows = [
        {
            "Έννοια": concept_name,
            "Σημασιολογικό Σκορ (%)": float(score),
        }
        for concept_name, score in concept_scores.items()
    ]
    return pd.DataFrame(rows, columns=["Έννοια", "Σημασιολογικό Σκορ (%)"])


def run_semantic_pipeline(
    image: Image.Image,
    cam: np.ndarray,
    predicted_class: str,
    confidence: float,
    runtime: SemanticRuntime,
    settings: SemanticSettings | None = None,
) -> dict[str, Any]:
    resolved_settings = settings or SemanticSettings()
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D heatmap.")

    height, width = cam.shape
    img_rgb01 = _pil_to_rgb01(image, width=width, height=height)
    seg = _slic_segments(img_rgb01, resolved_settings)
    sp_scores = _aggregate_superpixel_scores(cam, seg)
    focus_rgb01, focus_mask, top_superpixel_ids = _build_focus_image(
        img_rgb01=img_rgb01,
        seg=seg,
        sp_scores=sp_scores,
        top_k=resolved_settings.top_k_superpixels,
    )
    concept_scores, details_df = _semantic_from_superpixels(
        img_rgb01=img_rgb01,
        seg=seg,
        sp_scores=sp_scores,
        runtime=runtime,
        top_superpixel_ids=top_superpixel_ids,
        crop_pad=resolved_settings.crop_pad,
    )
    score_table = _build_score_table(concept_scores)
    focus_area_pct = float(focus_mask.mean() * 100.0) if focus_mask.size else 0.0

    return {
        "segmentation": seg,
        "sp_scores": sp_scores,
        "top_superpixel_ids": top_superpixel_ids,
        "focus_rgb": (np.clip(focus_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8),
        "focus_area_pct": focus_area_pct,
        "concept_scores": concept_scores,
        "top_concepts": list(concept_scores.items())[:3],
        "score_table": score_table,
        "details_df": details_df,
        "summary_gr": _make_greek_summary(predicted_class, float(confidence), concept_scores),
        "top_concepts_text": _top_concepts_text(concept_scores, k=3),
    }


def _cosine_sim(values_a: np.ndarray, values_b: np.ndarray) -> float:
    values_a = np.asarray(values_a, dtype=np.float32)
    values_b = np.asarray(values_b, dtype=np.float32)
    return float(np.dot(values_a, values_b) / (norm(values_a) * norm(values_b) + 1e-8))


def build_semantic_agreement(semantic_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if len(semantic_results) < 2:
        return {
            "concept_df": pd.DataFrame(),
            "pairwise_df": pd.DataFrame(),
            "mean_pairwise_cosine": 1.0,
        }

    all_concepts = sorted(
        set().union(
            *[set(result.get("concept_scores", {}).keys()) for result in semantic_results.values()]
        )
    )

    rows: list[dict[str, float | str]] = []
    for method_name, result in semantic_results.items():
        row: dict[str, float | str] = {"Method": method_name}
        concept_scores = result.get("concept_scores", {})
        for concept in all_concepts:
            row[concept] = float(concept_scores.get(concept, 0.0))
        rows.append(row)

    concept_df = pd.DataFrame(rows).set_index("Method")

    pairwise_rows: list[dict[str, float | str]] = []
    similarities: list[float] = []
    method_names = list(concept_df.index)
    for index, method_a in enumerate(method_names):
        for method_b in method_names[index + 1 :]:
            similarity = _cosine_sim(concept_df.loc[method_a].values, concept_df.loc[method_b].values)
            similarities.append(similarity)
            pairwise_rows.append(
                {
                    "Methods": f"{method_a} vs {method_b}",
                    "Semantic Cosine Agreement": similarity,
                }
            )

    return {
        "concept_df": concept_df,
        "pairwise_df": pd.DataFrame(pairwise_rows),
        "mean_pairwise_cosine": float(np.mean(similarities)) if similarities else 0.0,
    }
