from __future__ import annotations

import gc
import hashlib
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.data.preprocessing import get_inference_transform, load_image, preprocess_pil_image
from src.explainers.gradcam_explainer import GradCAM
from src.explainers.integrated_gradients_explainer import generate_integrated_gradients
from src.explainers.lime_explainer import generate_lime
from src.explainers.occlusion_explainer import generate_occlusion
from src.interpretation.summary_generator import (
    RegionAnalysis,
    analyze_regions,
    build_simplified_focus_image,
    compare_method_regions,
    generate_summary_text,
)
from src.metrics.explanation_metrics import MetricSettings, compute_explanation_metrics
from src.models.class_names import get_imagenet_class_names
from src.models.loader import get_last_conv_layer, load_model
from src.models.predictor import predict
from src.semantic import SemanticSettings, build_semantic_agreement, build_semantic_runtime, run_semantic_pipeline
from src.utils import config as cfg
from src.visualization.heatmaps import apply_colormap_to_cam, overlay_cam_on_image


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


AVAILABLE_METHODS = ["Grad-CAM", "Integrated Gradients", "Occlusion", "LIME"]
ANALYSIS_CACHE_MAX_ENTRIES = int(_cfg("ANALYSIS_CACHE_MAX_ENTRIES", 3))
CAM_OVERLAY_ALPHA = float(_cfg("CAM_OVERLAY_ALPHA", 0.45))
CAM_SCORE_TYPE_DEFAULT = str(_cfg("CAM_SCORE_TYPE_DEFAULT", "logit"))
IG_BASELINE_BLUR_RADIUS_DEFAULT = float(_cfg("IG_BASELINE_BLUR_RADIUS_DEFAULT", 4.0))
IG_INTERNAL_BATCH_SIZE_DEFAULT = int(_cfg("IG_INTERNAL_BATCH_SIZE_DEFAULT", 16))
IG_STEPS_DEFAULT = int(_cfg("IG_STEPS_DEFAULT", 24))
LIME_BASELINE_BLUR_RADIUS_DEFAULT = float(_cfg("LIME_BASELINE_BLUR_RADIUS_DEFAULT", 2.0))
LIME_COMPACTNESS_DEFAULT = float(_cfg("LIME_COMPACTNESS_DEFAULT", 10.0))
LIME_N_SAMPLES_DEFAULT = int(_cfg("LIME_N_SAMPLES_DEFAULT", 120))
LIME_N_SEGMENTS_DEFAULT = int(_cfg("LIME_N_SEGMENTS_DEFAULT", 40))
LIME_PERTURBATIONS_PER_EVAL_DEFAULT = int(_cfg("LIME_PERTURBATIONS_PER_EVAL_DEFAULT", 32))
LIME_RANDOM_SEED_DEFAULT = int(_cfg("LIME_RANDOM_SEED_DEFAULT", 0))
LIME_SIGMA_DEFAULT = float(_cfg("LIME_SIGMA_DEFAULT", 1.0))
MAX_UI_IMAGE_SIDE = int(_cfg("MAX_UI_IMAGE_SIDE", 1024))
METRICS_ENABLED_DEFAULT = bool(_cfg("METRICS_ENABLED_DEFAULT", True))
METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT = float(_cfg("METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT", 4.0))
METRICS_FAITHFULNESS_STEPS_DEFAULT = int(_cfg("METRICS_FAITHFULNESS_STEPS_DEFAULT", 10))
METRICS_RANDOM_SEED_DEFAULT = int(_cfg("METRICS_RANDOM_SEED_DEFAULT", 0))
METRICS_ROBUSTNESS_ENABLED_DEFAULT = bool(_cfg("METRICS_ROBUSTNESS_ENABLED_DEFAULT", False))
METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT = float(_cfg("METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT", 0.05))
METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT = tuple(_cfg("METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT", (0.1, 0.2)))
METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT = float(_cfg("METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT", 4.0))
METRICS_SENSITIVITY_N_RANDOM_DEFAULT = int(_cfg("METRICS_SENSITIVITY_N_RANDOM_DEFAULT", 20))
METRICS_SENSITIVITY_TOP_N_DEFAULT = int(_cfg("METRICS_SENSITIVITY_TOP_N_DEFAULT", 10))
METRICS_SLIC_COMPACTNESS_DEFAULT = float(_cfg("METRICS_SLIC_COMPACTNESS_DEFAULT", 10.0))
METRICS_SLIC_SEGMENTS_DEFAULT = int(_cfg("METRICS_SLIC_SEGMENTS_DEFAULT", 50))
METRICS_SLIC_SIGMA_DEFAULT = float(_cfg("METRICS_SLIC_SIGMA_DEFAULT", 1.0))
OCC_BASELINE_BLUR_RADIUS_DEFAULT = float(_cfg("OCC_BASELINE_BLUR_RADIUS_DEFAULT", 4.0))
OCC_PATCH_SIZE_DEFAULT = int(_cfg("OCC_PATCH_SIZE_DEFAULT", 32))
OCC_STRIDE_DEFAULT = int(_cfg("OCC_STRIDE_DEFAULT", 32))
SEMANTIC_CACHE_MAX_ENTRIES = int(_cfg("SEMANTIC_CACHE_MAX_ENTRIES", 6))
SEMANTIC_CLIP_MODEL_NAME = str(_cfg("SEMANTIC_CLIP_MODEL_NAME", "ViT-B-32"))
SEMANTIC_CLIP_PRETRAINED = str(_cfg("SEMANTIC_CLIP_PRETRAINED", "laion2b_s34b_b79k"))
SEMANTIC_COMPARE_AGREEMENT_DEFAULT = bool(_cfg("SEMANTIC_COMPARE_AGREEMENT_DEFAULT", False))
SEMANTIC_SLIC_COMPACTNESS_DEFAULT = float(_cfg("SEMANTIC_SLIC_COMPACTNESS_DEFAULT", 10.0))
SEMANTIC_SLIC_SEGMENTS_DEFAULT = int(_cfg("SEMANTIC_SLIC_SEGMENTS_DEFAULT", 80))
SEMANTIC_SLIC_SIGMA_DEFAULT = float(_cfg("SEMANTIC_SLIC_SIGMA_DEFAULT", 1.0))
SEMANTIC_TOP_K_SUPERPIXELS_DEFAULT = int(_cfg("SEMANTIC_TOP_K_SUPERPIXELS_DEFAULT", 10))
TOP_K = int(_cfg("TOP_K", 5))

SUMMARY_TOP_K = 3
COMPARISON_LIMIT = 3
SEMANTIC_SETTINGS = SemanticSettings(
    slic_n_segments=SEMANTIC_SLIC_SEGMENTS_DEFAULT,
    slic_compactness=SEMANTIC_SLIC_COMPACTNESS_DEFAULT,
    slic_sigma=SEMANTIC_SLIC_SIGMA_DEFAULT,
    top_k_superpixels=SEMANTIC_TOP_K_SUPERPIXELS_DEFAULT,
    clip_model_name=SEMANTIC_CLIP_MODEL_NAME,
    clip_pretrained=SEMANTIC_CLIP_PRETRAINED,
)

st.set_page_config(page_title="XAI Thesis App", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Manrope:wght@400;500;700;800&display=swap');

    :root {
        --panel-bg: linear-gradient(135deg, #f6f2ea 0%, #fffaf2 100%);
        --hero-bg:
            radial-gradient(circle at top left, rgba(229, 168, 94, 0.20), transparent 34%),
            linear-gradient(135deg, #f6f0e6 0%, #fff8f1 48%, #f0e8dd 100%);
        --panel-line: rgba(106, 73, 41, 0.12);
        --panel-text: #2a221b;
        --accent: #b65f2c;
        --accent-soft: rgba(182, 95, 44, 0.12);
        --muted: #67584b;
        --card-bg: rgba(255, 255, 255, 0.88);
    }

    html, body, [class*="css"] {
        font-family: "Manrope", sans-serif;
    }

    h1, h2, h3 {
        font-family: "Fraunces", serif;
        letter-spacing: -0.02em;
    }

    .block-container {
        padding-top: 1.0rem;
        padding-bottom: 1.6rem;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }

    .xai-hero {
        background: var(--hero-bg);
        border: 1px solid rgba(106, 73, 41, 0.13);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 14px 30px rgba(92, 64, 36, 0.08);
        color: var(--panel-text);
        margin-bottom: 1rem;
    }

    .xai-hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(240px, 0.85fr);
        gap: 1rem;
        align-items: start;
    }

    .xai-hero-kicker {
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.76rem;
        color: #9f5a29;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .xai-hero-title {
        font-family: "Fraunces", serif;
        font-size: 1.95rem;
        line-height: 1.08;
        margin: 0 0 0.45rem 0;
        max-width: 17ch;
    }

    .xai-hero-copy {
        margin: 0;
        max-width: 70ch;
        line-height: 1.62;
        color: #43372b;
        font-size: 1rem;
    }

    .xai-step-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
    }

    .xai-step-card {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid rgba(106, 73, 41, 0.12);
        border-radius: 18px;
        padding: 0.95rem;
        min-height: 120px;
    }

    .xai-step-card h4 {
        margin: 0 0 0.4rem 0;
        color: var(--panel-text);
    }

    .xai-step-card p {
        margin: 0;
        color: #5e4e41;
        line-height: 1.55;
        font-size: 0.95rem;
    }

    .xai-callout {
        background: rgba(255, 250, 243, 0.95);
        border: 1px solid rgba(106, 73, 41, 0.12);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        margin-bottom: 1rem;
        color: var(--panel-text);
    }

    .xai-callout strong {
        color: var(--accent);
    }

    .xai-hero-side {
        display: grid;
        gap: 0.75rem;
    }

    .xai-hero-pill {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(106, 73, 41, 0.11);
        border-radius: 16px;
        padding: 0.85rem 0.9rem;
    }

    .xai-hero-pill strong {
        display: block;
        font-size: 0.92rem;
        margin-bottom: 0.18rem;
        color: var(--panel-text);
    }

    .xai-hero-pill span {
        display: block;
        color: #5b4d3f;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .xai-panel {
        background: var(--panel-bg);
        border: 1px solid var(--panel-line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        color: var(--panel-text);
        box-shadow: 0 12px 28px rgba(92, 64, 36, 0.08);
    }

    .xai-panel h4 {
        margin: 0 0 0.55rem 0;
        color: var(--panel-text);
        letter-spacing: 0.02em;
    }

    .xai-panel p {
        margin: 0.35rem 0;
        line-height: 1.5;
        color: var(--panel-text);
    }

    .xai-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.7rem;
    }

    .xai-chip {
        background: var(--card-bg);
        border: 1px solid var(--panel-line);
        border-radius: 999px;
        padding: 0.4rem 0.75rem;
        font-size: 0.9rem;
        color: var(--panel-text);
    }

    .xai-kpi {
        background: var(--card-bg);
        border: 1px solid var(--panel-line);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        min-height: 110px;
    }

    .xai-kpi-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.35rem;
    }

    .xai-kpi-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--panel-text);
        margin-bottom: 0.25rem;
    }

    .xai-kpi-note {
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.35;
    }

    .xai-compare-card {
        background: rgba(255, 251, 244, 0.9);
        border: 1px solid var(--panel-line);
        border-radius: 18px;
        padding: 0.9rem;
        min-height: 100%;
    }

    .xai-compare-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--panel-text);
        margin-bottom: 0.45rem;
    }

    .xai-section-note {
        color: #6b594b;
        font-size: 0.94rem;
        margin-top: -0.15rem;
        margin-bottom: 0.85rem;
    }

    @media (max-width: 900px) {
        .xai-hero-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_runtime_objects() -> tuple[Any, list[str], Any, Any]:
    model, weights = load_model()
    class_names = get_imagenet_class_names(weights)
    transform = get_inference_transform(weights)
    target_layer = get_last_conv_layer(model)
    return model, class_names, transform, target_layer


@st.cache_resource
def get_semantic_runtime_objects() -> Any:
    return build_semantic_runtime(
        settings=SEMANTIC_SETTINGS,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def get_session_analysis_cache() -> OrderedDict[str, dict[str, Any]]:
    cache = st.session_state.get("analysis_cache")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        st.session_state["analysis_cache"] = cache
    return cache


def get_session_semantic_cache() -> OrderedDict[str, dict[str, Any]]:
    cache = st.session_state.get("semantic_cache")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        st.session_state["semantic_cache"] = cache
    return cache


def trim_analysis_cache(cache: OrderedDict[str, dict[str, Any]]) -> None:
    while len(cache) > ANALYSIS_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def trim_semantic_cache(cache: OrderedDict[str, dict[str, Any]]) -> None:
    while len(cache) > SEMANTIC_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def build_analysis_key(
    image_bytes: bytes,
    explain_method: str,
    score_type: str,
    ig_steps: int,
    ig_internal_batch_size: int,
    ig_blur_radius: float,
    occ_patch_size: int,
    occ_stride: int,
    occ_blur_radius: float,
    lime_n_samples: int,
    lime_perturbations_per_eval: int,
    lime_n_segments: int,
    lime_compactness: float,
    lime_sigma: float,
    lime_blur_radius: float,
    lime_random_seed: int,
    compute_metrics: bool,
    metrics_seed: int,
    metrics_slic_segments: int,
    metrics_slic_compactness: float,
    metrics_slic_sigma: float,
    faithfulness_steps: int,
    faithfulness_blur_radius: float,
    sensitivity_top_n: int,
    sensitivity_n_random: int,
    sensitivity_blur_radius: float,
    compute_robustness: bool,
    robustness_noise_sigma: float,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    params = (
        explain_method,
        score_type,
        ig_steps,
        ig_internal_batch_size,
        round(ig_blur_radius, 4),
        occ_patch_size,
        occ_stride,
        round(occ_blur_radius, 4),
        lime_n_samples,
        lime_perturbations_per_eval,
        lime_n_segments,
        round(lime_compactness, 4),
        round(lime_sigma, 4),
        round(lime_blur_radius, 4),
        lime_random_seed,
        compute_metrics,
        metrics_seed,
        metrics_slic_segments,
        round(metrics_slic_compactness, 4),
        round(metrics_slic_sigma, 4),
        faithfulness_steps,
        round(faithfulness_blur_radius, 4),
        sensitivity_top_n,
        sensitivity_n_random,
        round(sensitivity_blur_radius, 4),
        compute_robustness,
        round(robustness_noise_sigma, 5),
    )
    hasher.update(str(params).encode("utf-8"))
    return hasher.hexdigest()


def build_semantic_key(
    image_bytes: bytes,
    method_name: str,
    cam_uint8: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    hasher.update(method_name.encode("utf-8"))
    hasher.update(np.asarray(cam_uint8, dtype=np.uint8).tobytes())
    params = (
        "focus_region_clip_v2",
        SEMANTIC_SETTINGS.slic_n_segments,
        round(SEMANTIC_SETTINGS.slic_compactness, 4),
        round(SEMANTIC_SETTINGS.slic_sigma, 4),
        SEMANTIC_SETTINGS.top_k_superpixels,
        SEMANTIC_SETTINGS.clip_model_name,
        SEMANTIC_SETTINGS.clip_pretrained,
    )
    hasher.update(str(params).encode("utf-8"))
    return hasher.hexdigest()


def resize_for_display(image: Image.Image, max_side: int = MAX_UI_IMAGE_SIDE) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image

    scale = max_side / float(longest_side)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    resample = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize(new_size, resample=resample)


def build_analysis_request(
    image_bytes: bytes,
    explain_method: str,
    score_type: str,
    ig_steps: int,
    ig_internal_batch_size: int,
    ig_blur_radius: float,
    occ_patch_size: int,
    occ_stride: int,
    occ_blur_radius: float,
    lime_n_samples: int,
    lime_perturbations_per_eval: int,
    lime_n_segments: int,
    lime_compactness: float,
    lime_sigma: float,
    lime_blur_radius: float,
    lime_random_seed: int,
    compute_metrics: bool,
    metrics_seed: int,
    metrics_slic_segments: int,
    metrics_slic_compactness: float,
    metrics_slic_sigma: float,
    faithfulness_steps: int,
    faithfulness_blur_radius: float,
    sensitivity_top_n: int,
    sensitivity_n_random: int,
    sensitivity_blur_radius: float,
    compute_robustness: bool,
    robustness_noise_sigma: float,
) -> dict[str, Any]:
    return {
        "image_bytes": image_bytes,
        "explain_method": explain_method,
        "score_type": score_type,
        "ig_steps": ig_steps,
        "ig_internal_batch_size": ig_internal_batch_size,
        "ig_blur_radius": float(ig_blur_radius),
        "occ_patch_size": occ_patch_size,
        "occ_stride": occ_stride,
        "occ_blur_radius": float(occ_blur_radius),
        "lime_n_samples": lime_n_samples,
        "lime_perturbations_per_eval": lime_perturbations_per_eval,
        "lime_n_segments": lime_n_segments,
        "lime_compactness": float(lime_compactness),
        "lime_sigma": float(lime_sigma),
        "lime_blur_radius": float(lime_blur_radius),
        "lime_random_seed": int(lime_random_seed),
        "compute_metrics": bool(compute_metrics),
        "metrics_seed": int(metrics_seed),
        "metrics_slic_segments": int(metrics_slic_segments),
        "metrics_slic_compactness": float(metrics_slic_compactness),
        "metrics_slic_sigma": float(metrics_slic_sigma),
        "faithfulness_steps": int(faithfulness_steps),
        "faithfulness_blur_radius": float(faithfulness_blur_radius),
        "sensitivity_top_n": int(sensitivity_top_n),
        "sensitivity_n_random": int(sensitivity_n_random),
        "sensitivity_blur_radius": float(sensitivity_blur_radius),
        "compute_robustness": bool(compute_robustness),
        "robustness_noise_sigma": float(robustness_noise_sigma),
    }


def run_analysis(
    image_bytes: bytes,
    explain_method: str,
    score_type: str,
    ig_steps: int,
    ig_internal_batch_size: int,
    ig_blur_radius: float,
    occ_patch_size: int,
    occ_stride: int,
    occ_blur_radius: float,
    lime_n_samples: int,
    lime_perturbations_per_eval: int,
    lime_n_segments: int,
    lime_compactness: float,
    lime_sigma: float,
    lime_blur_radius: float,
    lime_random_seed: int,
    compute_metrics: bool,
    metrics_seed: int,
    metrics_slic_segments: int,
    metrics_slic_compactness: float,
    metrics_slic_sigma: float,
    faithfulness_steps: int,
    faithfulness_blur_radius: float,
    sensitivity_top_n: int,
    sensitivity_n_random: int,
    sensitivity_blur_radius: float,
    compute_robustness: bool,
    robustness_noise_sigma: float,
) -> dict[str, Any]:
    start_total = time.perf_counter()
    pil_image = load_image(image_bytes)
    model, class_names, transform, target_layer = get_runtime_objects()
    input_batch = preprocess_pil_image(pil_image, transform)
    prediction = predict(model, input_batch, class_names, top_k=TOP_K)

    def generate_cam(expl_input_batch: torch.Tensor) -> np.ndarray:
        if explain_method == "Grad-CAM":
            gradcam = GradCAM(model, target_layer)
            try:
                return gradcam.generate(
                    expl_input_batch,
                    target_class=prediction.predicted_index,
                    score_type=score_type,
                )
            finally:
                gradcam.close()
        if explain_method == "Integrated Gradients":
            return generate_integrated_gradients(
                model=model,
                input_tensor=expl_input_batch,
                image=pil_image,
                transform=transform,
                target_class=prediction.predicted_index,
                score_type=score_type,
                n_steps=ig_steps,
                internal_batch_size=ig_internal_batch_size,
                blur_radius=ig_blur_radius,
            )
        if explain_method == "Occlusion":
            return generate_occlusion(
                model=model,
                input_tensor=expl_input_batch,
                image=pil_image,
                transform=transform,
                target_class=prediction.predicted_index,
                score_type=score_type,
                patch_size=occ_patch_size,
                stride=occ_stride,
                blur_radius=occ_blur_radius,
            )
        return generate_lime(
            model=model,
            input_tensor=expl_input_batch,
            image=pil_image,
            transform=transform,
            target_class=prediction.predicted_index,
            score_type=score_type,
            n_samples=lime_n_samples,
            perturbations_per_eval=lime_perturbations_per_eval,
            n_segments=lime_n_segments,
            compactness=lime_compactness,
            sigma=lime_sigma,
            blur_radius=lime_blur_radius,
            random_seed=lime_random_seed,
        )

    explanation_start = time.perf_counter()
    cam = generate_cam(input_batch)
    explanation_runtime_s = float(time.perf_counter() - explanation_start)

    metrics: dict[str, float | int | list[float]] | None = None
    metrics_runtime_s = 0.0
    if compute_metrics:
        metrics_settings = MetricSettings(
            slic_n_segments=int(metrics_slic_segments),
            slic_compactness=float(metrics_slic_compactness),
            slic_sigma=float(metrics_slic_sigma),
            faithfulness_steps=int(faithfulness_steps),
            faithfulness_blur_radius=float(faithfulness_blur_radius),
            sensitivity_top_n=int(sensitivity_top_n),
            sensitivity_n_random=int(sensitivity_n_random),
            sensitivity_blur_radius=float(sensitivity_blur_radius),
            robustness_topk_fracs=tuple(float(v) for v in METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT),
        )

        noisy_cam: np.ndarray | None = None
        if compute_robustness:
            torch.manual_seed(int(metrics_seed))
            noisy_input = input_batch + torch.randn_like(input_batch) * float(robustness_noise_sigma)
            noisy_cam = generate_cam(noisy_input)

        metrics_start = time.perf_counter()
        metrics = compute_explanation_metrics(
            model=model,
            input_tensor=input_batch,
            image=pil_image,
            transform=transform,
            cam=cam,
            target_class=prediction.predicted_index,
            score_type=score_type,
            settings=metrics_settings,
            random_seed=int(metrics_seed),
            noisy_cam=noisy_cam,
        )
        metrics_runtime_s = float(time.perf_counter() - metrics_start)

    cam_uint8 = (np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8)
    topk_rows = [
        {
            "Rank": rank,
            "Class Index": item.class_index,
            "Class Name": item.class_name,
            "Probability (%)": round(item.probability * 100, 4),
        }
        for rank, item in enumerate(prediction.topk, start=1)
    ]

    result = {
        "predicted_index": prediction.predicted_index,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "cam_uint8": cam_uint8,
        "topk_rows": topk_rows,
        "metrics": metrics,
        "explanation_runtime_s": explanation_runtime_s,
        "metrics_runtime_s": metrics_runtime_s,
        "total_runtime_s": float(time.perf_counter() - start_total),
    }

    del input_batch, prediction, cam, cam_uint8
    gc.collect()
    return result


def get_cached_analysis(
    cache: OrderedDict[str, dict[str, Any]],
    request: dict[str, Any],
) -> tuple[str, dict[str, Any] | None]:
    cache_key = build_analysis_key(**request)
    analysis = cache.get(cache_key)

    if analysis is None and not bool(request["compute_metrics"]):
        fallback_request = dict(request)
        fallback_request["compute_metrics"] = True
        fallback_key = build_analysis_key(**fallback_request)
        analysis = cache.get(fallback_key)
        if analysis is not None:
            return fallback_key, analysis

    return cache_key, analysis


def ensure_analysis(
    cache: OrderedDict[str, dict[str, Any]],
    request: dict[str, Any],
    run_if_missing: bool,
) -> dict[str, Any] | None:
    cache_key, analysis = get_cached_analysis(cache, request)
    if analysis is not None or not run_if_missing:
        return analysis

    analysis = run_analysis(**request)
    cache[cache_key] = analysis
    cache.move_to_end(cache_key)
    trim_analysis_cache(cache)
    return analysis


def build_visual_bundle(
    image: Image.Image,
    method_name: str,
    analysis: dict[str, Any],
    overlay_alpha: float,
    region_segments: int,
    region_compactness: float,
    region_sigma: float,
) -> dict[str, Any]:
    cam = np.asarray(analysis["cam_uint8"]).astype(np.float32) / 255.0
    region_analysis = analyze_regions(
        image=image,
        cam=cam,
        n_segments=region_segments,
        compactness=region_compactness,
        sigma=region_sigma,
        top_k=SUMMARY_TOP_K,
    )
    heatmap_rgb = apply_colormap_to_cam(cam)
    overlay_rgb = overlay_cam_on_image(np.asarray(image), heatmap_rgb, alpha=overlay_alpha)
    simplified_rgb = build_simplified_focus_image(image, region_analysis)
    summary_lines = generate_summary_text(
        predicted_class=str(analysis["predicted_class"]),
        confidence=float(analysis["confidence"]),
        method_name=method_name,
        region_analysis=region_analysis,
    )
    return {
        "method_name": method_name,
        "cam": cam,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
        "simplified_rgb": simplified_rgb,
        "region_analysis": region_analysis,
        "summary_lines": summary_lines,
    }


def ensure_semantic_analysis(
    cache: OrderedDict[str, dict[str, Any]],
    image_bytes: bytes,
    image: Image.Image,
    method_name: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    cam_uint8 = np.asarray(analysis["cam_uint8"], dtype=np.uint8)
    cache_key = build_semantic_key(
        image_bytes=image_bytes,
        method_name=method_name,
        cam_uint8=cam_uint8,
    )
    semantic_analysis = cache.get(cache_key)
    if semantic_analysis is not None:
        return semantic_analysis

    runtime = get_semantic_runtime_objects()
    semantic_analysis = run_semantic_pipeline(
        image=image,
        cam=cam_uint8.astype(np.float32) / 255.0,
        predicted_class=str(analysis["predicted_class"]),
        confidence=float(analysis["confidence"]),
        runtime=runtime,
        settings=SEMANTIC_SETTINGS,
    )
    cache[cache_key] = semantic_analysis
    cache.move_to_end(cache_key)
    trim_semantic_cache(cache)
    return semantic_analysis


def render_panel(title: str, body_lines: list[str]) -> None:
    body = "".join(f"<p>{line}</p>" for line in body_lines)
    st.markdown(
        f"""
        <div class="xai-panel">
            <h4>{title}</h4>
            {body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="xai-kpi">
            <div class="xai-kpi-label">{label}</div>
            <div class="xai-kpi-value">{value}</div>
            <div class="xai-kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataframe_compat(
    data: Any,
    *,
    height: int | None = None,
    hide_index: bool = False,
) -> None:
    try:
        st.dataframe(data, width="stretch", hide_index=hide_index, height=height)
        return
    except TypeError:
        pass

    try:
        st.dataframe(data, use_container_width=True, hide_index=hide_index, height=height)
        return
    except TypeError:
        pass

    st.dataframe(data, use_container_width=True, height=height)


def render_line_chart_compat(
    data: Any,
    *,
    height: int | None = None,
) -> None:
    try:
        st.line_chart(data, width="stretch", height=height)
        return
    except TypeError:
        pass

    try:
        st.line_chart(data, use_container_width=True, height=height)
        return
    except TypeError:
        pass

    st.line_chart(data, height=height)


def metric_to_display(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def render_hero() -> None:
    st.markdown(
        """
        <div class="xai-hero">
            <div class="xai-hero-grid">
                <div>
                    <div class="xai-hero-kicker">Single Image Demo</div>
                    <div class="xai-hero-title">Inspect one prediction with explanation, metrics, and comparison.</div>
                    <p class="xai-hero-copy">
                        Upload one image, choose a primary explainer, and inspect the visual output together with the
                        superpixel-based quality metrics. The main area stays focused on the result, while the setup panel
                        keeps the advanced controls tidy and easy to reopen.
                    </p>
                </div>
                <div class="xai-hero-side">
                    <div class="xai-hero-pill">
                        <strong>Primary explainer</strong>
                        <span>Select the method you want to study in detail for the current image.</span>
                    </div>
                    <div class="xai-hero-pill">
                        <strong>Metrics tab</strong>
                        <span>Open the quantitative readout without cluttering the overview screen.</span>
                    </div>
                    <div class="xai-hero-pill">
                        <strong>Compare tab</strong>
                        <span>See where multiple explainers agree or diverge on the same prediction.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_cards() -> None:
    st.markdown(
        """
        <div class="xai-step-grid">
            <div class="xai-step-card">
                <h4>1. Load an image</h4>
                <p>Use the setup panel to upload one image that you want to analyze.</p>
            </div>
            <div class="xai-step-card">
                <h4>2. Choose an explainer</h4>
                <p>Pick a primary method and optionally add comparison methods for side-by-side viewing.</p>
            </div>
            <div class="xai-step-card">
                <h4>3. Run the analysis</h4>
                <p>Inspect the explanation, then open the metrics and comparison tabs for a deeper readout.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_focus_panel(method_name: str, region_analysis: RegionAnalysis) -> None:
    chips = [
        f"Top-{len(region_analysis.top_region_ids)} mass: {region_analysis.top_mass * 100:.1f}%",
        f"Concentration: {region_analysis.concentration_label}",
        f"Focus: {region_analysis.top_region_summary}",
    ]
    if region_analysis.leakage_flag:
        chips.append(f"Border leakage: {region_analysis.border_mass * 100:.1f}%")
    chip_html = "".join(f'<span class="xai-chip">{item}</span>' for item in chips)
    st.markdown(
        f"""
        <div class="xai-panel">
            <h4>Why This Prediction?</h4>
            <p>The most influential regions identified by {method_name} are concentrated around {region_analysis.top_region_summary}.</p>
            <div class="xai-chip-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if region_analysis.leakage_flag:
        st.warning("The explanation places noticeable mass near the image borders, so background cues may still contribute.")


def render_semantic_top_concepts(top_concepts: list[tuple[str, float]]) -> None:
    if not top_concepts:
        st.info("Δεν βρέθηκαν σταθερά semantic concepts για το τρέχον αποτέλεσμα.")
        return

    st.markdown("#### Top Concepts")
    concept_columns = st.columns(len(top_concepts), gap="medium")
    for index, (concept_name, contribution) in enumerate(top_concepts):
        with concept_columns[index]:
            render_kpi_card(
                concept_name,
                f"{float(contribution):.1f}%",
                "CLIP semantic score over the focused explanation area.",
            )


def expand_setup_panel() -> None:
    st.session_state["setup_panel_expanded"] = True


def collapse_setup_panel() -> None:
    st.session_state["setup_panel_expanded"] = False


if "setup_panel_expanded" not in st.session_state:
    st.session_state["setup_panel_expanded"] = True


render_hero()
st.caption(
    "Model context: this dashboard uses the official ImageNet-pretrained ResNet50. "
    "If you analyze Oxford-IIIT Pet images here, the explanations still refer to the ImageNet model's "
    "prediction behavior on those inputs."
)

if not bool(st.session_state.get("setup_panel_expanded", True)):
    setup_action_left, setup_action_right = st.columns([0.78, 0.22], gap="medium")
    with setup_action_left:
        st.markdown(
            '<div class="xai-section-note">The setup panel is collapsed so the analysis stays in focus. Reopen it any time to change the image or settings.</div>',
            unsafe_allow_html=True,
        )
    with setup_action_right:
        st.button("Show setup", on_click=expand_setup_panel)

ig_steps = IG_STEPS_DEFAULT
ig_internal_batch_size = IG_INTERNAL_BATCH_SIZE_DEFAULT
ig_blur_radius = IG_BASELINE_BLUR_RADIUS_DEFAULT

occ_patch_size = OCC_PATCH_SIZE_DEFAULT
occ_stride = OCC_STRIDE_DEFAULT
occ_blur_radius = OCC_BASELINE_BLUR_RADIUS_DEFAULT

lime_n_samples = LIME_N_SAMPLES_DEFAULT
lime_perturbations_per_eval = LIME_PERTURBATIONS_PER_EVAL_DEFAULT
lime_n_segments = LIME_N_SEGMENTS_DEFAULT
lime_compactness = LIME_COMPACTNESS_DEFAULT
lime_sigma = LIME_SIGMA_DEFAULT
lime_blur_radius = LIME_BASELINE_BLUR_RADIUS_DEFAULT
lime_random_seed = LIME_RANDOM_SEED_DEFAULT

compute_metrics = METRICS_ENABLED_DEFAULT
metrics_seed = METRICS_RANDOM_SEED_DEFAULT
metrics_slic_segments = METRICS_SLIC_SEGMENTS_DEFAULT
metrics_slic_compactness = METRICS_SLIC_COMPACTNESS_DEFAULT
metrics_slic_sigma = METRICS_SLIC_SIGMA_DEFAULT
faithfulness_steps = METRICS_FAITHFULNESS_STEPS_DEFAULT
faithfulness_blur_radius = METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT
sensitivity_top_n = METRICS_SENSITIVITY_TOP_N_DEFAULT
sensitivity_n_random = METRICS_SENSITIVITY_N_RANDOM_DEFAULT
sensitivity_blur_radius = METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT
compute_robustness = METRICS_ROBUSTNESS_ENABLED_DEFAULT
robustness_noise_sigma = METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT

with st.expander("Analysis Setup", expanded=bool(st.session_state.get("setup_panel_expanded", True))):
    st.markdown(
        '<div class="xai-section-note">Upload one image and adjust the controls here. After you run the analysis, this panel collapses automatically.</div>',
        unsafe_allow_html=True,
    )

    primary_controls_col, view_controls_col = st.columns([1.06, 0.94], gap="large")
    with primary_controls_col:
        uploaded_file = st.file_uploader(
            "1. Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        explain_method = st.selectbox(
            "2. Primary explainer",
            options=AVAILABLE_METHODS,
            index=0,
        )
        comparison_selection = st.multiselect(
            "3. Comparison methods",
            options=AVAILABLE_METHODS,
            default=[explain_method],
            max_selections=COMPARISON_LIMIT,
            help="Select up to three methods for side-by-side comparison.",
        )
        score_type = st.radio(
            "Score type",
            options=["logit", "prob"],
            index=0 if CAM_SCORE_TYPE_DEFAULT == "logit" else 1,
            help="Use class logit score or softmax probability score for explanation.",
        )

    with view_controls_col:
        st.markdown("#### View Options")
        visual_style = st.radio(
            "Explanation style",
            options=["Raw overlay", "Simplified focus"],
        )
        view_mode = st.radio(
            "Overview layout",
            options=["Tabs", "Side by side"],
        )
        image_size_label = st.select_slider(
            "Image size",
            options=["Small", "Medium", "Large"],
            value="Medium",
        )
        overlay_alpha = st.slider(
            "Overlay opacity",
            min_value=0.1,
            max_value=0.9,
            value=float(CAM_OVERLAY_ALPHA),
            step=0.05,
        )

    with st.expander("Advanced Settings", expanded=False):
        advanced_method_col, advanced_metrics_col = st.columns(2, gap="large")

        with advanced_method_col:
            with st.expander("Method settings", expanded=False):
                if explain_method == "Integrated Gradients":
                    ig_steps = st.slider("IG steps", min_value=10, max_value=300, value=IG_STEPS_DEFAULT, step=10)
                    ig_internal_batch_size = st.slider(
                        "IG internal batch size",
                        min_value=1,
                        max_value=64,
                        value=IG_INTERNAL_BATCH_SIZE_DEFAULT,
                        step=1,
                    )
                    ig_blur_radius = st.slider(
                        "IG baseline blur radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(IG_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                elif explain_method == "Occlusion":
                    occ_patch_size = st.slider(
                        "Occlusion patch size",
                        min_value=4,
                        max_value=64,
                        value=OCC_PATCH_SIZE_DEFAULT,
                        step=2,
                    )
                    occ_stride = st.slider(
                        "Occlusion stride",
                        min_value=1,
                        max_value=32,
                        value=OCC_STRIDE_DEFAULT,
                        step=1,
                    )
                    occ_blur_radius = st.slider(
                        "Occlusion baseline blur radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(OCC_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                elif explain_method == "LIME":
                    lime_n_samples = st.slider(
                        "LIME samples",
                        min_value=100,
                        max_value=2000,
                        value=LIME_N_SAMPLES_DEFAULT,
                        step=50,
                    )
                    lime_perturbations_per_eval = st.slider(
                        "LIME perturbations per eval",
                        min_value=16,
                        max_value=256,
                        value=LIME_PERTURBATIONS_PER_EVAL_DEFAULT,
                        step=16,
                    )
                    lime_n_segments = st.slider(
                        "LIME SLIC segments",
                        min_value=20,
                        max_value=300,
                        value=LIME_N_SEGMENTS_DEFAULT,
                        step=10,
                    )
                    lime_compactness = st.slider(
                        "LIME SLIC compactness",
                        min_value=1.0,
                        max_value=40.0,
                        value=float(LIME_COMPACTNESS_DEFAULT),
                        step=0.5,
                    )
                    lime_sigma = st.slider(
                        "LIME SLIC sigma",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(LIME_SIGMA_DEFAULT),
                        step=0.1,
                    )
                    lime_blur_radius = st.slider(
                        "LIME baseline blur radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(LIME_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                    lime_random_seed = st.number_input(
                        "LIME random seed",
                        min_value=0,
                        max_value=1_000_000,
                        value=LIME_RANDOM_SEED_DEFAULT,
                        step=1,
                    )
                    if lime_n_samples > 1200:
                        st.warning("High LIME sample counts can be slow on CPU.")

        with advanced_metrics_col:
            with st.expander("Metrics settings", expanded=False):
                compute_metrics = st.checkbox("Compute metrics for the primary explainer", value=METRICS_ENABLED_DEFAULT)
                metrics_seed = st.number_input(
                    "Metrics random seed",
                    min_value=0,
                    max_value=1_000_000,
                    value=METRICS_RANDOM_SEED_DEFAULT,
                    step=1,
                )
                metrics_slic_segments = st.slider(
                    "SLIC segments",
                    min_value=20,
                    max_value=200,
                    value=METRICS_SLIC_SEGMENTS_DEFAULT,
                    step=10,
                )
                metrics_slic_compactness = st.slider(
                    "SLIC compactness",
                    min_value=1.0,
                    max_value=40.0,
                    value=float(METRICS_SLIC_COMPACTNESS_DEFAULT),
                    step=0.5,
                )
                metrics_slic_sigma = st.slider(
                    "SLIC sigma",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(METRICS_SLIC_SIGMA_DEFAULT),
                    step=0.1,
                )
                faithfulness_steps = st.slider(
                    "Faithfulness steps",
                    min_value=4,
                    max_value=30,
                    value=METRICS_FAITHFULNESS_STEPS_DEFAULT,
                    step=1,
                )
                faithfulness_blur_radius = st.slider(
                    "Faithfulness blur radius",
                    min_value=0.0,
                    max_value=15.0,
                    value=float(METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT),
                    step=0.5,
                )
                sensitivity_top_n = st.slider(
                    "Sensitivity top-N superpixels",
                    min_value=1,
                    max_value=50,
                    value=METRICS_SENSITIVITY_TOP_N_DEFAULT,
                    step=1,
                )
                sensitivity_n_random = st.slider(
                    "Sensitivity random subsets",
                    min_value=5,
                    max_value=100,
                    value=METRICS_SENSITIVITY_N_RANDOM_DEFAULT,
                    step=5,
                )
                sensitivity_blur_radius = st.slider(
                    "Sensitivity blur radius",
                    min_value=0.0,
                    max_value=15.0,
                    value=float(METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT),
                    step=0.5,
                )
                compute_robustness = st.checkbox(
                    "Compute robustness for the primary explainer",
                    value=METRICS_ROBUSTNESS_ENABLED_DEFAULT,
                )
                if compute_robustness:
                    robustness_noise_sigma = st.slider(
                        "Robustness noise sigma",
                        min_value=0.0,
                        max_value=0.5,
                        value=float(METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT),
                        step=0.01,
                    )

    setup_footer_left, setup_footer_right = st.columns([0.72, 0.28], gap="medium")
    with setup_footer_left:
        st.caption(
            "Only the primary explainer receives per-image metrics. "
            "Comparison methods are visual by default so the demo stays responsive."
        )
    with setup_footer_right:
        run_clicked = st.button("Run analysis", type="primary", on_click=collapse_setup_panel)

comparison_methods: list[str] = []
for method_name in [explain_method, *comparison_selection]:
    if method_name not in comparison_methods:
        comparison_methods.append(method_name)
comparison_methods = comparison_methods[:COMPARISON_LIMIT]

image_width_map = {"Small": 300, "Medium": 420, "Large": 540}
image_width = image_width_map[image_size_label]

if uploaded_file is None:
    render_step_cards()
    st.info("Use the setup panel above to upload an image and start the demo.")
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    source_pil_image = load_image(image_bytes)
    pil_image = resize_for_display(source_pil_image.copy())
except Exception as exc:
    st.error(f"Could not read the image file: {exc}")
    st.stop()

analysis_cache = get_session_analysis_cache()
semantic_cache = get_session_semantic_cache()


def build_request_for_method(method_name: str, metrics_enabled: bool) -> dict[str, Any]:
    return build_analysis_request(
        image_bytes=image_bytes,
        explain_method=method_name,
        score_type=score_type,
        ig_steps=ig_steps,
        ig_internal_batch_size=ig_internal_batch_size,
        ig_blur_radius=float(ig_blur_radius),
        occ_patch_size=occ_patch_size,
        occ_stride=occ_stride,
        occ_blur_radius=float(occ_blur_radius),
        lime_n_samples=lime_n_samples,
        lime_perturbations_per_eval=lime_perturbations_per_eval,
        lime_n_segments=lime_n_segments,
        lime_compactness=float(lime_compactness),
        lime_sigma=float(lime_sigma),
        lime_blur_radius=float(lime_blur_radius),
        lime_random_seed=int(lime_random_seed),
        compute_metrics=metrics_enabled,
        metrics_seed=int(metrics_seed),
        metrics_slic_segments=int(metrics_slic_segments),
        metrics_slic_compactness=float(metrics_slic_compactness),
        metrics_slic_sigma=float(metrics_slic_sigma),
        faithfulness_steps=int(faithfulness_steps),
        faithfulness_blur_radius=float(faithfulness_blur_radius),
        sensitivity_top_n=int(sensitivity_top_n),
        sensitivity_n_random=int(sensitivity_n_random),
        sensitivity_blur_radius=float(sensitivity_blur_radius),
        compute_robustness=bool(compute_robustness),
        robustness_noise_sigma=float(robustness_noise_sigma),
    )


method_analyses: dict[str, dict[str, Any]] = {}
comparison_errors: list[str] = []

methods_to_resolve = comparison_methods or [explain_method]
if run_clicked:
    try:
        with st.spinner("Running inference and assembling the explanation dashboard..."):
            for method_name in methods_to_resolve:
                metrics_enabled = bool(compute_metrics) if method_name == explain_method else False
                request = build_request_for_method(method_name, metrics_enabled)
                try:
                    analysis = ensure_analysis(analysis_cache, request, run_if_missing=True)
                except Exception as exc:
                    if method_name == explain_method:
                        raise
                    comparison_errors.append(f"{method_name}: {exc}")
                    continue
                if analysis is not None:
                    method_analyses[method_name] = analysis
    except Exception as exc:
        st.error("An error occurred during analysis.")
        st.exception(exc)
        st.stop()
else:
    for method_name in methods_to_resolve:
        metrics_enabled = bool(compute_metrics) if method_name == explain_method else False
        request = build_request_for_method(method_name, metrics_enabled)
        analysis = ensure_analysis(analysis_cache, request, run_if_missing=False)
        if analysis is not None:
            method_analyses[method_name] = analysis

if explain_method not in method_analyses:
    preview_col1, preview_col2 = st.columns([0.95, 1.05], gap="large")
    with preview_col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, width=image_width)
    with preview_col2:
        st.markdown(
            f"""
            <div class="xai-callout">
                <strong>{uploaded_file.name}</strong> is ready for analysis. Choose your explainer and settings in
                the setup panel, then click <strong>Run analysis</strong> to populate the overview, metrics, and
                comparison tabs.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_step_cards()
    st.stop()

if comparison_errors:
    st.warning("Some comparison methods could not be generated: " + " | ".join(comparison_errors))
elif not run_clicked:
    st.caption("Showing cached result for the current image and settings.")

selected_analysis = method_analyses[explain_method]
selected_bundle = build_visual_bundle(
    image=pil_image,
    method_name=explain_method,
    analysis=selected_analysis,
    overlay_alpha=overlay_alpha,
    region_segments=metrics_slic_segments,
    region_compactness=metrics_slic_compactness,
    region_sigma=metrics_slic_sigma,
)
selected_semantic: dict[str, Any] | None = None
selected_semantic_error: str | None = None
try:
    selected_semantic = ensure_semantic_analysis(
        cache=semantic_cache,
        image_bytes=image_bytes,
        image=source_pil_image,
        method_name=explain_method,
        analysis=selected_analysis,
    )
except Exception as exc:
    selected_semantic_error = str(exc)

comparison_bundles: OrderedDict[str, dict[str, Any]] = OrderedDict()
for method_name in comparison_methods:
    analysis = method_analyses.get(method_name)
    if analysis is None:
        continue
    comparison_bundles[method_name] = build_visual_bundle(
        image=pil_image,
        method_name=method_name,
        analysis=analysis,
        overlay_alpha=overlay_alpha,
        region_segments=metrics_slic_segments,
        region_compactness=metrics_slic_compactness,
        region_sigma=metrics_slic_sigma,
    )

selected_metrics_raw = selected_analysis.get("metrics")
selected_metrics = selected_metrics_raw if isinstance(selected_metrics_raw, dict) else None
top5_df = pd.DataFrame(selected_analysis["topk_rows"])
display_explanation_image = (
    selected_bundle["overlay_rgb"] if visual_style == "Raw overlay" else selected_bundle["simplified_rgb"]
)

st.markdown(
    f"""
    <div class="xai-callout">
        <strong>Current image:</strong> {uploaded_file.name} &nbsp;|&nbsp;
        <strong>Primary method:</strong> {explain_method} &nbsp;|&nbsp;
        <strong>Comparison set:</strong> {", ".join(comparison_methods)}
    </div>
    """,
    unsafe_allow_html=True,
)

overview_tab, metrics_tab, semantic_tab, compare_tab = st.tabs(["Overview", "Metrics", "Semantic", "Compare"])

with overview_tab:
    st.markdown("### Run Snapshot")
    st.markdown(
        "Core outputs for the current run: the selected prediction, the main explanation view, and a compact interpretation summary."
    )
    snapshot_cols = st.columns(4, gap="medium")
    with snapshot_cols[0]:
        render_kpi_card(
            "Predicted Class",
            str(selected_analysis["predicted_class"]),
            "Top class returned by the ImageNet-pretrained ResNet50.",
        )
    with snapshot_cols[1]:
        render_kpi_card(
            "Confidence",
            f"{float(selected_analysis['confidence']) * 100:.1f}%",
            "Probability/confidence for the selected prediction.",
        )
    with snapshot_cols[2]:
        render_kpi_card(
            "Primary Method",
            explain_method,
            "The explainer driving the overview panels below.",
        )
    with snapshot_cols[3]:
        render_kpi_card(
            "Runtime",
            f"{float(selected_analysis['total_runtime_s']):.2f}s",
            "Total time for prediction, explanation, and any enabled metrics.",
        )

    visual_col, context_col = st.columns([1.14, 0.86], gap="large")
    with visual_col:
        st.subheader("Visual Evidence")
        st.caption("Switch between the interpreted view and raw heatmap while keeping the original image close by.")
        base_col, explanation_col = st.columns([0.92, 1.08], gap="large")
        with base_col:
            st.image(pil_image, width=image_width, caption="Original image")
        with explanation_col:
            if view_mode == "Tabs":
                overlay_tab, heatmap_tab = st.tabs(["Interpreted View", "Heatmap"])
                with overlay_tab:
                    st.image(display_explanation_image, width=image_width)
                with heatmap_tab:
                    st.image(selected_bundle["heatmap_rgb"], width=image_width)
            else:
                side_a, side_b = st.columns(2, gap="large")
                with side_a:
                    st.image(display_explanation_image, width=image_width, caption="Selected explanation view")
                with side_b:
                    st.image(selected_bundle["heatmap_rgb"], width=image_width, caption="Raw heatmap")

    with context_col:
        st.subheader("Interpretation")
        render_panel("Explanation Summary", selected_bundle["summary_lines"])
        render_focus_panel(explain_method, selected_bundle["region_analysis"])
        st.markdown("#### Top-5 Classes")
        render_dataframe_compat(top5_df, hide_index=True, height=235)

with metrics_tab:
    st.markdown("### Metric Readout")
    st.markdown(
        '<div class="xai-section-note">Per-image quality metrics are computed only for the primary explainer so the demo remains responsive.</div>',
        unsafe_allow_html=True,
    )
    if selected_metrics is None:
        st.info("Enable `Compute metrics for the primary explainer` in the setup panel and run the analysis to populate this tab.")
    else:
        quality_cols_1 = st.columns(4, gap="medium")
        with quality_cols_1[0]:
            render_kpi_card(
                "Deletion AUC",
                metric_to_display(selected_metrics.get("deletion_auc")),
                "Faithfulness under progressive removal of important regions.",
            )
        with quality_cols_1[1]:
            render_kpi_card(
                "Insertion AUC",
                metric_to_display(selected_metrics.get("insertion_auc")),
                "Faithfulness when important regions are gradually restored.",
            )
        with quality_cols_1[2]:
            render_kpi_card(
                "Sensitivity",
                metric_to_display(selected_metrics.get("sensitivity")),
                "Drop for top regions relative to random subsets.",
            )
        with quality_cols_1[3]:
            render_kpi_card(
                "Hoyer Sparsity",
                metric_to_display(selected_metrics.get("hoyer_sparsity")),
                "Compactness of the explanation across superpixels.",
            )

        quality_cols_2 = st.columns(4, gap="medium")
        with quality_cols_2[0]:
            render_kpi_card(
                "AOPC Delta",
                metric_to_display(selected_metrics.get("aopc_delta")),
                "Difference between insertion and deletion behavior.",
            )
        with quality_cols_2[1]:
            robustness_value = selected_metrics.get("spearman_rho")
            robustness_note = "Robustness is off for this run."
            if robustness_value is not None:
                robustness_note = "Spearman agreement between the original and noisy explanation."
            render_kpi_card(
                "Robustness",
                metric_to_display(robustness_value) if robustness_value is not None else "-",
                robustness_note,
            )
        with quality_cols_2[2]:
            render_kpi_card(
                "Metrics Runtime",
                metric_to_display(float(selected_analysis["metrics_runtime_s"]), digits=2),
                "Time spent only on the metric computations.",
            )
        with quality_cols_2[3]:
            render_kpi_card(
                "Overlay Opacity",
                f"{overlay_alpha:.2f}",
                "Current visual opacity setting used in the overview images.",
            )

        detail_col1, detail_col2 = st.columns([0.78, 1.22], gap="large")
        with detail_col1:
            metric_details_rows = [
                {"Metric": "Drop Top", "Value": metric_to_display(selected_metrics.get("drop_top"))},
                {"Metric": "Drop Random Mean", "Value": metric_to_display(selected_metrics.get("drop_rand_mean"))},
            ]
            for frac in METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT:
                key = f"iou_top_{int(round(float(frac) * 100.0))}pct"
                if key in selected_metrics:
                    metric_details_rows.append(
                        {
                            "Metric": f"IoU Top-{int(round(float(frac) * 100.0))}%",
                            "Value": metric_to_display(selected_metrics.get(key)),
                        }
                    )
            st.markdown("#### Metric Details")
            render_dataframe_compat(pd.DataFrame(metric_details_rows), hide_index=True, height=245)

        with detail_col2:
            st.markdown("#### Faithfulness Curves")
            if "faithfulness_xs" in selected_metrics:
                curve_df = pd.DataFrame(
                    {
                        "Fraction": [float(value) for value in selected_metrics["faithfulness_xs"]],
                        "Deletion": [float(value) for value in selected_metrics["deletion_curve"]],
                        "Insertion": [float(value) for value in selected_metrics["insertion_curve"]],
                    }
                )
                render_line_chart_compat(curve_df.set_index("Fraction"), height=300)

with semantic_tab:
    st.markdown("### Semantic Layer")
    st.markdown(
        '<div class="xai-section-note">This tab runs the separate focus-region semantic pipeline from `semantic.ipynb`, independently from the metric core.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Semantic defaults: {SEMANTIC_SETTINGS.slic_n_segments} SLIC segments, compactness "
        f"{SEMANTIC_SETTINGS.slic_compactness:.1f}, sigma {SEMANTIC_SETTINGS.slic_sigma:.1f}, "
        f"top-{SEMANTIC_SETTINGS.top_k_superpixels} semantic superpixels."
    )

    if selected_semantic_error is not None:
        st.error(f"Semantic layer is unavailable for this run: {selected_semantic_error}")
    elif selected_semantic is None:
        st.info("Run the analysis to populate the semantic layer.")
    else:
        semantic_text_col, semantic_focus_col = st.columns([1.0, 1.0], gap="large")
        with semantic_text_col:
            render_panel("Greek Semantic Summary", [str(selected_semantic["summary_gr"])])
            render_semantic_top_concepts(list(selected_semantic.get("top_concepts", [])))
        with semantic_focus_col:
            st.markdown("#### Focused Semantic Region")
            st.image(
                np.asarray(selected_semantic["focus_rgb"], dtype=np.uint8),
                width=image_width,
                caption=(
                    f"Top-{len(selected_semantic.get('top_superpixel_ids', []))} semantic superpixels | "
                    f"focus area {float(selected_semantic.get('focus_area_pct', 0.0)):.1f}%"
                ),
            )

        score_table = selected_semantic.get("score_table")
        if isinstance(score_table, pd.DataFrame) and not score_table.empty:
            semantic_table = score_table.copy()
            semantic_table["Semantic Score (%)"] = semantic_table["Semantic Score (%)"].map(
                lambda value: round(float(value), 2)
            )
            st.markdown("#### Concept Score Table")
            render_dataframe_compat(semantic_table, hide_index=True, height=265)
        else:
            st.info("Δεν προέκυψε πίνακας semantic scores για το τρέχον αποτέλεσμα.")

with compare_tab:
    st.markdown("### Multi-Method Comparison")
    st.markdown(
        '<div class="xai-section-note">Use this tab to compare where different explainers focus on the same image.</div>',
        unsafe_allow_html=True,
    )
    if len(comparison_bundles) < 2:
        st.info("Select at least two methods in `Comparison methods` and click `Run analysis` to unlock side-by-side comparison.")
    else:
        comparison_summary = compare_method_regions(
            {method_name: bundle["region_analysis"] for method_name, bundle in comparison_bundles.items()}
        )
        summary_body = [
            str(comparison_summary["summary"]),
            f"Mean pairwise IoU across top regions: {float(comparison_summary['mean_pairwise_iou']):.3f}.",
        ]
        render_panel("Method Agreement", summary_body)

        compare_columns = st.columns(len(comparison_bundles), gap="medium")
        for index, (method_name, bundle) in enumerate(comparison_bundles.items()):
            with compare_columns[index]:
                compare_image = bundle["overlay_rgb"] if visual_style == "Raw overlay" else bundle["simplified_rgb"]
                comparison_runtime = float(method_analyses[method_name]["total_runtime_s"])
                comparison_region: RegionAnalysis = bundle["region_analysis"]
                st.markdown(
                    f'<div class="xai-compare-card"><div class="xai-compare-title">{method_name}</div></div>',
                    unsafe_allow_html=True,
                )
                st.image(compare_image, width=image_width)
                st.caption(
                    f"Top-{len(comparison_region.top_region_ids)} mass: {comparison_region.top_mass * 100:.1f}% | "
                    f"{comparison_region.concentration_label} | {comparison_runtime:.2f}s"
                )
                st.caption(f"Focus: {comparison_region.top_region_summary}")

        pairwise_rows = comparison_summary.get("pairwise_rows", [])
        if pairwise_rows:
            render_dataframe_compat(pd.DataFrame(pairwise_rows), hide_index=True, height=150)

        semantic_compare_enabled = st.checkbox(
            "Include semantic agreement",
            value=SEMANTIC_COMPARE_AGREEMENT_DEFAULT,
            help="Runs the notebook-style focus-region CLIP semantic layer on the selected comparison methods.",
        )
        if semantic_compare_enabled:
            semantic_results: OrderedDict[str, dict[str, Any]] = OrderedDict()
            semantic_errors: list[str] = []

            for method_name in comparison_bundles.keys():
                if method_name == explain_method and selected_semantic is not None:
                    semantic_results[method_name] = selected_semantic
                    continue

                analysis = method_analyses.get(method_name)
                if analysis is None:
                    continue
                try:
                    semantic_results[method_name] = ensure_semantic_analysis(
                        cache=semantic_cache,
                        image_bytes=image_bytes,
                        image=source_pil_image,
                        method_name=method_name,
                        analysis=analysis,
                    )
                except Exception as exc:
                    semantic_errors.append(f"{method_name}: {exc}")

            if semantic_errors:
                st.warning("Some semantic comparison results could not be generated: " + " | ".join(semantic_errors))

            if len(semantic_results) >= 2:
                semantic_agreement = build_semantic_agreement(semantic_results)
                render_panel(
                    "Semantic Agreement",
                    [
                        (
                            "Mean pairwise semantic cosine agreement across the selected methods: "
                            f"{float(semantic_agreement['mean_pairwise_cosine']):.3f}."
                        ),
                        "The score is computed from each explainer's CLIP concept distribution over its focused explanation area.",
                    ],
                )

                semantic_pairwise_df = semantic_agreement["pairwise_df"]
                if isinstance(semantic_pairwise_df, pd.DataFrame) and not semantic_pairwise_df.empty:
                    semantic_pairwise_df = semantic_pairwise_df.copy()
                    semantic_pairwise_df["Semantic Cosine Agreement"] = semantic_pairwise_df[
                        "Semantic Cosine Agreement"
                    ].map(lambda value: round(float(value), 3))
                    render_dataframe_compat(semantic_pairwise_df, hide_index=True, height=150)

                semantic_concept_df = semantic_agreement["concept_df"]
                if isinstance(semantic_concept_df, pd.DataFrame) and not semantic_concept_df.empty:
                    st.markdown("#### Semantic Concept Distribution by Method")
                    render_dataframe_compat(semantic_concept_df.round(2), height=225)
