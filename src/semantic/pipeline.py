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
            "The semantic layer requires `open_clip_torch`. Add it to the environment to enable the Semantic tab."
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
def _clip_scores(image_pil: Image.Image, runtime: SemanticRuntime) -> dict[str, float]:
    image_input = runtime.clip_preprocess(image_pil).unsqueeze(0).to(runtime.device)
    image_features = runtime.clip_model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    sims = (image_features @ runtime.text_features.T)[0].detach().cpu().numpy().astype(np.float32)
    sims = sims - float(sims.min())
    sims = sims / (float(sims.sum()) + 1e-8)

    scores = {
        str(runtime.concepts_gr[concept_en]): float(score * 100.0)
        for concept_en, score in zip(runtime.concepts_en, sims.tolist(), strict=False)
    }
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def _top_concepts_text(scores: Mapping[str, float], k: int = 3) -> str:
    items = list(scores.items())[:k]
    if not items:
        return "Δεν προέκυψαν σταθερά concepts."
    return ", ".join([f"{name} ({value:.1f}%)" for name, value in items])


def _make_greek_summary(
    predicted_class: str,
    confidence: float,
    concept_scores: Mapping[str, float],
) -> str:
    if not concept_scores:
        return (
            f"Το μοντέλο ταξινόμησε την εικόνα ως {predicted_class} "
            f"με πιθανότητα {confidence:.1%}, αλλά δεν προέκυψε σταθερή semantic ανάγνωση."
        )

    background_score = float(concept_scores.get("υπόβαθρο", 0.0))
    top_text = _top_concepts_text(concept_scores, k=3)

    if background_score >= 20.0:
        return (
            f"Το μοντέλο ταξινόμησε την εικόνα ως {predicted_class} με πιθανότητα {confidence:.1%}. "
            f"Στη focused semantic περιοχή κυριαρχούν τα concepts {top_text}, "
            f"με αισθητή παρουσία υποβάθρου ({background_score:.1f}%)."
        )

    return (
        f"Το μοντέλο ταξινόμησε την εικόνα ως {predicted_class} με πιθανότητα {confidence:.1%}. "
        f"Στη focused semantic περιοχή κυριαρχούν τα concepts {top_text}."
    )


def _build_score_table(concept_scores: Mapping[str, float]) -> pd.DataFrame:
    rows = [
        {
            "Έννοια": concept_name,
            "Semantic Score (%)": float(score),
        }
        for concept_name, score in concept_scores.items()
    ]
    return pd.DataFrame(rows, columns=["Έννοια", "Semantic Score (%)"])


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
    focus_pil = Image.fromarray((np.clip(focus_rgb01, 0.0, 1.0) * 255.0).astype(np.uint8))
    concept_scores = _clip_scores(focus_pil, runtime)
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
