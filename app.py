from __future__ import annotations

import gc
import hashlib
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.counterfactual import CounterfactualSettings, run_counterfactual_pipeline
from src.data.preprocessing import get_inference_transform, load_image, preprocess_pil_image, preprocess_spatial_pil_image
from src.explainers.gradcam_explainer import GradCAM
from src.explainers.integrated_gradients_explainer import generate_integrated_gradients
from src.explainers.lime_explainer import generate_lime
from src.explainers.occlusion_explainer import generate_occlusion
from src.interpretation.consensus import build_consensus_analysis
from src.interpretation.summary_generator import (
    RegionAnalysis,
    analyze_regions,
    build_simplified_focus_image,
    generate_summary_text,
)
from src.metrics.explanation_metrics import MetricSettings, compute_explanation_metrics
from src.models.class_names import get_imagenet_class_names
from src.models.loader import get_last_conv_layer, load_model
from src.models.predictor import predict
from src.reporting import build_pdf_report
from src.semantic import SemanticSettings, build_semantic_agreement, build_semantic_runtime, run_semantic_pipeline
from src.utils import config as cfg
from src.visualization.heatmaps import apply_colormap_to_cam, overlay_cam_on_image


def _cfg(name: str, default: Any) -> Any:
    return getattr(cfg, name, default)


AVAILABLE_METHODS = ["Grad-CAM", "Integrated Gradients", "Occlusion", "LIME"]
ANALYSIS_CACHE_MAX_ENTRIES = int(_cfg("ANALYSIS_CACHE_MAX_ENTRIES", 3))
CAM_OVERLAY_ALPHA = float(_cfg("CAM_OVERLAY_ALPHA", 0.45))
CAM_SCORE_TYPE_DEFAULT = str(_cfg("CAM_SCORE_TYPE_DEFAULT", "logit"))
COUNTERFACTUAL_BLUR_RADIUS_DEFAULT = float(_cfg("COUNTERFACTUAL_BLUR_RADIUS_DEFAULT", 4.0))
COUNTERFACTUAL_CACHE_MAX_ENTRIES = int(_cfg("COUNTERFACTUAL_CACHE_MAX_ENTRIES", 6))
COUNTERFACTUAL_MAX_REMOVAL_FRACTION_DEFAULT = float(_cfg("COUNTERFACTUAL_MAX_REMOVAL_FRACTION_DEFAULT", 0.55))
COUNTERFACTUAL_MAX_STEPS_DEFAULT = int(_cfg("COUNTERFACTUAL_MAX_STEPS_DEFAULT", 18))
COUNTERFACTUAL_SLIC_COMPACTNESS_DEFAULT = float(_cfg("COUNTERFACTUAL_SLIC_COMPACTNESS_DEFAULT", 12.0))
COUNTERFACTUAL_SLIC_SEGMENTS_DEFAULT = int(_cfg("COUNTERFACTUAL_SLIC_SEGMENTS_DEFAULT", 60))
COUNTERFACTUAL_SLIC_SIGMA_DEFAULT = float(_cfg("COUNTERFACTUAL_SLIC_SIGMA_DEFAULT", 1.0))
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


def get_session_counterfactual_cache() -> OrderedDict[str, dict[str, Any]]:
    cache = st.session_state.get("counterfactual_cache")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        st.session_state["counterfactual_cache"] = cache
    return cache


def trim_analysis_cache(cache: OrderedDict[str, dict[str, Any]]) -> None:
    while len(cache) > ANALYSIS_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def trim_semantic_cache(cache: OrderedDict[str, dict[str, Any]]) -> None:
    while len(cache) > SEMANTIC_CACHE_MAX_ENTRIES:
        cache.popitem(last=False)


def trim_counterfactual_cache(cache: OrderedDict[str, dict[str, Any]]) -> None:
    while len(cache) > COUNTERFACTUAL_CACHE_MAX_ENTRIES:
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


def build_counterfactual_key(
    image_bytes: bytes,
    method_name: str,
    cam_uint8: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(image_bytes)
    hasher.update(method_name.encode("utf-8"))
    hasher.update(np.asarray(cam_uint8, dtype=np.uint8).tobytes())
    params = (
        "counterfactual_v2",
        COUNTERFACTUAL_SLIC_SEGMENTS_DEFAULT,
        round(COUNTERFACTUAL_SLIC_COMPACTNESS_DEFAULT, 4),
        round(COUNTERFACTUAL_SLIC_SIGMA_DEFAULT, 4),
        round(COUNTERFACTUAL_BLUR_RADIUS_DEFAULT, 4),
        COUNTERFACTUAL_MAX_STEPS_DEFAULT,
        round(COUNTERFACTUAL_MAX_REMOVAL_FRACTION_DEFAULT, 4),
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


def ensure_counterfactual_analysis(
    cache: OrderedDict[str, dict[str, Any]],
    image_bytes: bytes,
    image: Image.Image,
    method_name: str,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    cam_uint8 = np.asarray(analysis["cam_uint8"], dtype=np.uint8)
    cache_key = build_counterfactual_key(
        image_bytes=image_bytes,
        method_name=method_name,
        cam_uint8=cam_uint8,
    )
    counterfactual_analysis = cache.get(cache_key)
    if counterfactual_analysis is not None:
        return counterfactual_analysis

    model, class_names, transform, _ = get_runtime_objects()
    input_batch = preprocess_pil_image(image, transform)
    counterfactual_analysis = run_counterfactual_pipeline(
        model=model,
        input_tensor=input_batch,
        image=image,
        transform=transform,
        class_names=class_names,
        cam=cam_uint8.astype(np.float32) / 255.0,
        method_name=method_name,
        target_class=int(analysis["predicted_index"]),
        settings=CounterfactualSettings(
            slic_n_segments=COUNTERFACTUAL_SLIC_SEGMENTS_DEFAULT,
            slic_compactness=COUNTERFACTUAL_SLIC_COMPACTNESS_DEFAULT,
            slic_sigma=COUNTERFACTUAL_SLIC_SIGMA_DEFAULT,
            blur_radius=COUNTERFACTUAL_BLUR_RADIUS_DEFAULT,
            max_steps=COUNTERFACTUAL_MAX_STEPS_DEFAULT,
            max_removal_fraction=COUNTERFACTUAL_MAX_REMOVAL_FRACTION_DEFAULT,
        ),
    )
    cache[cache_key] = counterfactual_analysis
    cache.move_to_end(cache_key)
    trim_counterfactual_cache(cache)
    return counterfactual_analysis


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
                    <div class="xai-hero-kicker">Single Image Analysis</div>
                    <div class="xai-hero-title">Εξέτασε μία πρόβλεψη με εξήγηση, μετρικές, σημασιολογική ανάλυση, αντιπαραδείγματα και κοινή εστίαση.</div>
                    <p class="xai-hero-copy">
                        Ανέβασε μία εικόνα, επίλεξε τη βασική μέθοδο εξήγησης και εξέτασε το οπτικό αποτέλεσμα μαζί με
                        τις superpixel-based μετρικές ποιότητας. Η κύρια περιοχή μένει εστιασμένη στο αποτέλεσμα,
                        ενώ το panel ρυθμίσεων κρατά τα προχωρημένα controls τακτοποιημένα και εύκολα προσβάσιμα.
                    </p>
                </div>
                <div class="xai-hero-side">
                    <div class="xai-hero-pill">
                        <strong>Primary Method</strong>
                        <span>Επίλεξε τη μέθοδο που θέλεις να μελετήσεις σε βάθος για την τρέχουσα εικόνα.</span>
                    </div>
                    <div class="xai-hero-pill">
                        <strong>Μετρικές</strong>
                        <span>Άνοιξε τη ποσοτική αποτίμηση χωρίς να γεμίζει η βασική οθόνη.</span>
                    </div>
                    <div class="xai-hero-pill">
                        <strong>Shared Focus</strong>
                        <span>Δες ποιες περιοχές της εικόνας παραμένουν σημαντικές σε πολλές μεθόδους, με πιο προχωρημένες λεπτομέρειες μόνο όταν χρειάζονται.</span>
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
                <h4>1. Upload Image</h4>
                <p>Χρησιμοποίησε το panel ρυθμίσεων για να ανεβάσεις την εικόνα που θέλεις να αναλύσεις.</p>
            </div>
            <div class="xai-step-card">
                <h4>2. Choose Explainer</h4>
                <p>Διάλεξε μία βασική μέθοδο και προαιρετικά πρόσθεσε κι άλλες για να δεις την κοινή τους εστίαση.</p>
            </div>
            <div class="xai-step-card">
                <h4>3. Run Analysis</h4>
                <p>Δες την εξήγηση και μετά άνοιξε τις μετρικές, το αντιπαράδειγμα και την κοινή εστίαση για βαθύτερη ανάγνωση.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_focus_panel(method_name: str, region_analysis: RegionAnalysis) -> None:
    chips = [
        f"Top-{len(region_analysis.top_region_ids)} μάζα: {region_analysis.top_mass * 100:.1f}%",
        f"Συγκέντρωση: {region_analysis.concentration_label}",
        f"Εστίαση: {region_analysis.top_region_summary}",
    ]
    if region_analysis.leakage_flag:
        chips.append(f"Διαρροή στα όρια: {region_analysis.border_mass * 100:.1f}%")
    chip_html = "".join(f'<span class="xai-chip">{item}</span>' for item in chips)
    st.markdown(
        f"""
        <div class="xai-panel">
            <h4>Γιατί Αυτή η Πρόβλεψη;</h4>
            <p>Οι πιο επιδραστικές περιοχές που εντόπισε το {method_name} συγκεντρώνονται γύρω από {region_analysis.top_region_summary}.</p>
            <div class="xai-chip-row">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if region_analysis.leakage_flag:
        st.warning("Η εξήγηση τοποθετεί αξιοσημείωτη μάζα κοντά στα όρια της εικόνας, άρα το υπόβαθρο μπορεί ακόμη να συμβάλλει.")


def render_semantic_top_concepts(top_concepts: list[tuple[str, float]]) -> None:
    if not top_concepts:
        st.info("Δεν βρέθηκαν σταθερές σημασιολογικές έννοιες για το τρέχον αποτέλεσμα.")
        return

    st.markdown("#### Top Concepts")
    concept_columns = st.columns(len(top_concepts), gap="medium")
    for index, (concept_name, contribution) in enumerate(top_concepts):
        with concept_columns[index]:
            render_kpi_card(
                concept_name,
                f"{float(contribution):.1f}%",
                "CLIP σκορ πάνω στη σημασιολογική περιοχή εστίασης.",
            )


def expand_setup_panel() -> None:
    st.session_state["setup_panel_expanded"] = True


def collapse_setup_panel() -> None:
    st.session_state["setup_panel_expanded"] = False


if "setup_panel_expanded" not in st.session_state:
    st.session_state["setup_panel_expanded"] = True


render_hero()
st.caption(
    "Πλαίσιο μοντέλου: αυτό το dashboard χρησιμοποιεί το επίσημο ImageNet-pretrained ResNet50. "
    "Αν αναλύσεις εικόνες Oxford-IIIT Pet εδώ, οι εξηγήσεις εξακολουθούν να αναφέρονται "
    "στη συμπεριφορά πρόβλεψης του ImageNet μοντέλου πάνω σε αυτά τα inputs."
)

if not bool(st.session_state.get("setup_panel_expanded", True)):
    setup_action_left, setup_action_right = st.columns([0.78, 0.22], gap="medium")
    with setup_action_left:
        st.markdown(
            '<div class="xai-section-note">Το panel ρυθμίσεων είναι κλειστό ώστε να μένει η ανάλυση στο επίκεντρο. Άνοιξέ το ξανά οποιαδήποτε στιγμή για να αλλάξεις εικόνα ή ρυθμίσεις.</div>',
            unsafe_allow_html=True,
        )
    with setup_action_right:
        st.button("Show Settings", on_click=expand_setup_panel)

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

with st.expander("Analysis Settings", expanded=bool(st.session_state.get("setup_panel_expanded", True))):
    st.markdown(
        '<div class="xai-section-note">Ανέβασε μία εικόνα και ρύθμισε εδώ τα controls. Μόλις τρέξει η ανάλυση, αυτό το panel κλείνει αυτόματα.</div>',
        unsafe_allow_html=True,
    )

    primary_controls_col, view_controls_col = st.columns([1.06, 0.94], gap="large")
    with primary_controls_col:
        uploaded_file = st.file_uploader(
            "1. Upload Image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        explain_method = st.selectbox(
            "2. Primary Explainer",
            options=AVAILABLE_METHODS,
            index=0,
        )
        comparison_selection = st.multiselect(
            "3. Comparison Methods",
            options=AVAILABLE_METHODS,
            default=[explain_method],
            max_selections=COMPARISON_LIMIT,
            help="Επίλεξε έως τρεις μεθόδους για οπτική σύγκριση.",
        )
        score_type = st.radio(
            "Score Type",
            options=["logit", "prob"],
            index=0 if CAM_SCORE_TYPE_DEFAULT == "logit" else 1,
            help="Χρησιμοποίησε score logit της κλάσης ή πιθανότητα softmax για την εξήγηση.",
        )

    with view_controls_col:
        st.markdown("#### View Options")
        visual_style = st.radio(
            "Explanation Style",
            options=["Heatmap Overlay", "Simplified Focus"],
        )
        view_mode = st.radio(
            "Overview Layout",
            options=["Tabs", "Side by Side"],
        )
        image_size_label = st.select_slider(
            "Image Size",
            options=["Small", "Medium", "Large"],
            value="Medium",
        )
        overlay_alpha = st.slider(
            "Overlay Opacity",
            min_value=0.1,
            max_value=0.9,
            value=float(CAM_OVERLAY_ALPHA),
            step=0.05,
        )

    with st.expander("Advanced Settings", expanded=False):
        advanced_method_col, advanced_metrics_col = st.columns(2, gap="large")

        with advanced_method_col:
            with st.expander("Method Settings", expanded=False):
                if explain_method == "Integrated Gradients":
                    ig_steps = st.slider("IG Steps", min_value=10, max_value=300, value=IG_STEPS_DEFAULT, step=10)
                    ig_internal_batch_size = st.slider(
                        "IG Internal Batch Size",
                        min_value=1,
                        max_value=64,
                        value=IG_INTERNAL_BATCH_SIZE_DEFAULT,
                        step=1,
                    )
                    ig_blur_radius = st.slider(
                        "IG Baseline Blur Radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(IG_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                elif explain_method == "Occlusion":
                    occ_patch_size = st.slider(
                        "Occlusion Patch Size",
                        min_value=4,
                        max_value=64,
                        value=OCC_PATCH_SIZE_DEFAULT,
                        step=2,
                    )
                    occ_stride = st.slider(
                        "Occlusion Stride",
                        min_value=1,
                        max_value=32,
                        value=OCC_STRIDE_DEFAULT,
                        step=1,
                    )
                    occ_blur_radius = st.slider(
                        "Occlusion Baseline Blur Radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(OCC_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                elif explain_method == "LIME":
                    lime_n_samples = st.slider(
                        "LIME Samples",
                        min_value=100,
                        max_value=2000,
                        value=LIME_N_SAMPLES_DEFAULT,
                        step=50,
                    )
                    lime_perturbations_per_eval = st.slider(
                        "LIME Perturbations per Eval",
                        min_value=16,
                        max_value=256,
                        value=LIME_PERTURBATIONS_PER_EVAL_DEFAULT,
                        step=16,
                    )
                    lime_n_segments = st.slider(
                        "LIME SLIC Segments",
                        min_value=20,
                        max_value=300,
                        value=LIME_N_SEGMENTS_DEFAULT,
                        step=10,
                    )
                    lime_compactness = st.slider(
                        "LIME SLIC Compactness",
                        min_value=1.0,
                        max_value=40.0,
                        value=float(LIME_COMPACTNESS_DEFAULT),
                        step=0.5,
                    )
                    lime_sigma = st.slider(
                        "LIME SLIC Sigma",
                        min_value=0.0,
                        max_value=5.0,
                        value=float(LIME_SIGMA_DEFAULT),
                        step=0.1,
                    )
                    lime_blur_radius = st.slider(
                        "LIME Baseline Blur Radius",
                        min_value=0.0,
                        max_value=15.0,
                        value=float(LIME_BASELINE_BLUR_RADIUS_DEFAULT),
                        step=0.5,
                    )
                    lime_random_seed = st.number_input(
                        "LIME Random Seed",
                        min_value=0,
                        max_value=1_000_000,
                        value=LIME_RANDOM_SEED_DEFAULT,
                        step=1,
                    )
                    if lime_n_samples > 1200:
                        st.warning("Μεγάλος αριθμός δειγμάτων LIME μπορεί να είναι αργός σε CPU.")

        with advanced_metrics_col:
            with st.expander("Metric Settings", expanded=False):
                compute_metrics = st.checkbox("Compute Metrics for Primary Method", value=METRICS_ENABLED_DEFAULT)
                metrics_seed = st.number_input(
                    "Metrics Random Seed",
                    min_value=0,
                    max_value=1_000_000,
                    value=METRICS_RANDOM_SEED_DEFAULT,
                    step=1,
                )
                metrics_slic_segments = st.slider(
                    "SLIC Segments",
                    min_value=20,
                    max_value=200,
                    value=METRICS_SLIC_SEGMENTS_DEFAULT,
                    step=10,
                )
                metrics_slic_compactness = st.slider(
                    "SLIC Compactness",
                    min_value=1.0,
                    max_value=40.0,
                    value=float(METRICS_SLIC_COMPACTNESS_DEFAULT),
                    step=0.5,
                )
                metrics_slic_sigma = st.slider(
                    "SLIC Sigma",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(METRICS_SLIC_SIGMA_DEFAULT),
                    step=0.1,
                )
                faithfulness_steps = st.slider(
                    "Faithfulness Steps",
                    min_value=4,
                    max_value=30,
                    value=METRICS_FAITHFULNESS_STEPS_DEFAULT,
                    step=1,
                )
                faithfulness_blur_radius = st.slider(
                    "Faithfulness Blur Radius",
                    min_value=0.0,
                    max_value=15.0,
                    value=float(METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT),
                    step=0.5,
                )
                sensitivity_top_n = st.slider(
                    "Sensitivity Top-N Superpixels",
                    min_value=1,
                    max_value=50,
                    value=METRICS_SENSITIVITY_TOP_N_DEFAULT,
                    step=1,
                )
                sensitivity_n_random = st.slider(
                    "Sensitivity Random Subsets",
                    min_value=5,
                    max_value=100,
                    value=METRICS_SENSITIVITY_N_RANDOM_DEFAULT,
                    step=5,
                )
                sensitivity_blur_radius = st.slider(
                    "Sensitivity Blur Radius",
                    min_value=0.0,
                    max_value=15.0,
                    value=float(METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT),
                    step=0.5,
                )
                compute_robustness = st.checkbox(
                    "Compute Robustness for Primary Method",
                    value=METRICS_ROBUSTNESS_ENABLED_DEFAULT,
                )
                if compute_robustness:
                    robustness_noise_sigma = st.slider(
                        "Robustness Noise Sigma",
                        min_value=0.0,
                        max_value=0.5,
                        value=float(METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT),
                        step=0.01,
                    )

    setup_footer_left, setup_footer_right = st.columns([0.72, 0.28], gap="medium")
    with setup_footer_left:
        st.caption(
            "Μόνο η βασική μέθοδος λαμβάνει μετρικές ανά εικόνα. "
            "Οι μέθοδοι σύγκρισης παραμένουν οπτικές από προεπιλογή ώστε το demo να μένει γρήγορο."
        )
    with setup_footer_right:
        run_clicked = st.button("Run Analysis", type="primary", on_click=collapse_setup_panel)

comparison_methods: list[str] = []
for method_name in [explain_method, *comparison_selection]:
    if method_name not in comparison_methods:
        comparison_methods.append(method_name)
comparison_methods = comparison_methods[:COMPARISON_LIMIT]

image_width_map = {"Small": 300, "Medium": 420, "Large": 540}
image_width = image_width_map[image_size_label]

if uploaded_file is None:
    render_step_cards()
    st.info("Χρησιμοποίησε το panel ρυθμίσεων πιο πάνω για να ανεβάσεις εικόνα και να ξεκινήσεις το demo.")
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    source_pil_image = load_image(image_bytes)
    pil_image = resize_for_display(source_pil_image.copy())
except Exception as exc:
    st.error(f"Δεν ήταν δυνατή η ανάγνωση του αρχείου εικόνας: {exc}")
    st.stop()

analysis_cache = get_session_analysis_cache()
semantic_cache = get_session_semantic_cache()
counterfactual_cache = get_session_counterfactual_cache()


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
        with st.spinner("Τρέχει η πρόβλεψη και συναρμολογείται το dashboard εξήγησης..."):
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
        st.error("Παρουσιάστηκε σφάλμα κατά την ανάλυση.")
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
                Το αρχείο <strong>{uploaded_file.name}</strong> είναι έτοιμο για ανάλυση. Επίλεξε μέθοδο και ρυθμίσεις
                από το panel και πάτησε <strong>Run Analysis</strong> για να εμφανιστούν η επισκόπηση, οι μετρικές,
                το αντιπαράδειγμα και η κοινή εστίαση.
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_step_cards()
    st.stop()

if comparison_errors:
    st.warning("Δεν ήταν δυνατό να παραχθούν ορισμένες μέθοδοι σύγκρισης: " + " | ".join(comparison_errors))
elif not run_clicked:
    st.caption("Εμφανίζεται αποθηκευμένο αποτέλεσμα για την τρέχουσα εικόνα και τις τρέχουσες ρυθμίσεις.")

selected_analysis = method_analyses[explain_method]
_, _, display_transform, _ = get_runtime_objects()
model_view_image = preprocess_spatial_pil_image(source_pil_image, display_transform)
selected_bundle = build_visual_bundle(
    image=model_view_image,
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
        image=model_view_image,
        method_name=explain_method,
        analysis=selected_analysis,
    )
except Exception as exc:
    selected_semantic_error = str(exc)

selected_counterfactual: dict[str, Any] | None = None
selected_counterfactual_error: str | None = None
try:
    selected_counterfactual = ensure_counterfactual_analysis(
        cache=counterfactual_cache,
        image_bytes=image_bytes,
        image=source_pil_image,
        method_name=explain_method,
        analysis=selected_analysis,
    )
except Exception as exc:
    selected_counterfactual_error = str(exc)

comparison_bundles: OrderedDict[str, dict[str, Any]] = OrderedDict()
for method_name in comparison_methods:
    analysis = method_analyses.get(method_name)
    if analysis is None:
        continue
    comparison_bundles[method_name] = build_visual_bundle(
        image=model_view_image,
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
top5_display_df = top5_df.rename(
    columns={
        "Rank": "Θέση",
        "Class Index": "Δείκτης Κλάσης",
        "Class Name": "Όνομα Κλάσης",
        "Probability (%)": "Πιθανότητα (%)",
    }
)
display_explanation_image = (
    selected_bundle["overlay_rgb"] if visual_style == "Heatmap Overlay" else selected_bundle["simplified_rgb"]
)

shared_focus_export: dict[str, Any] | None = None
if len(comparison_bundles) >= 2:
    consensus_analysis = build_consensus_analysis(
        image=model_view_image,
        method_cams={method_name: np.asarray(bundle["cam"], dtype=np.float32) for method_name, bundle in comparison_bundles.items()},
        n_segments=metrics_slic_segments,
        compactness=metrics_slic_compactness,
        sigma=metrics_slic_sigma,
        top_k=SUMMARY_TOP_K,
    )
    shared_evidence_map = np.asarray(
        consensus_analysis.get("shared_evidence_map", consensus_analysis.get("consensus_map")),
        dtype=np.float32,
    )
    disagreement_map = np.asarray(consensus_analysis["disagreement_map"], dtype=np.float32)
    shared_region: RegionAnalysis = consensus_analysis.get(
        "shared_region",
        consensus_analysis.get("consensus_region"),
    )
    disagreement_region: RegionAnalysis = consensus_analysis["disagreement_region"]
    required_votes = int(
        consensus_analysis.get(
            "required_votes",
            max(2, int(np.ceil(len(comparison_bundles) * 0.67))),
        )
    )

    shared_evidence_heatmap_rgb = apply_colormap_to_cam(shared_evidence_map)
    disagreement_heatmap_rgb = apply_colormap_to_cam(disagreement_map)
    shared_evidence_overlay_rgb = overlay_cam_on_image(
        np.asarray(model_view_image),
        shared_evidence_heatmap_rgb,
        alpha=overlay_alpha,
    )
    disagreement_overlay_rgb = overlay_cam_on_image(np.asarray(model_view_image), disagreement_heatmap_rgb, alpha=overlay_alpha)
    shared_evidence_focus_rgb = build_simplified_focus_image(model_view_image, shared_region)
    disagreement_focus_rgb = build_simplified_focus_image(model_view_image, disagreement_region)
    shared_evidence_display_rgb = (
        shared_evidence_overlay_rgb if visual_style == "Heatmap Overlay" else shared_evidence_focus_rgb
    )
    disagreement_display_rgb = (
        disagreement_overlay_rgb if visual_style == "Heatmap Overlay" else disagreement_focus_rgb
    )

    pairwise_report_df = pd.DataFrame(consensus_analysis["pairwise_rows"])
    if not pairwise_report_df.empty:
        pairwise_report_df = pairwise_report_df.copy()
        pairwise_report_df["Cosine Agreement"] = pairwise_report_df["Cosine Agreement"].map(
            lambda value: round(float(value), 3)
        )
        pairwise_report_df["Top-focus IoU"] = pairwise_report_df["Top-focus IoU"].map(
            lambda value: round(float(value), 3)
        )

    method_report_df = pd.DataFrame(consensus_analysis["method_rows"])
    if not method_report_df.empty:
        method_report_df = method_report_df.copy()
        if "Shared Cosine" not in method_report_df.columns and "Consensus Cosine" in method_report_df.columns:
            method_report_df = method_report_df.rename(
                columns={
                    "Consensus Cosine": "Shared Cosine",
                    "Consensus IoU": "Shared IoU",
                }
            )
        method_report_df["Shared Cosine"] = method_report_df["Shared Cosine"].map(lambda value: round(float(value), 3))
        method_report_df["Shared IoU"] = method_report_df["Shared IoU"].map(lambda value: round(float(value), 3))

    shared_focus_export = {
        "available": True,
        "consensus_analysis": consensus_analysis,
        "shared_region": shared_region,
        "disagreement_region": disagreement_region,
        "required_votes": required_votes,
        "shared_evidence_display_rgb": shared_evidence_display_rgb,
        "disagreement_display_rgb": disagreement_display_rgb,
        "pairwise_report_df": pairwise_report_df,
        "method_report_df": method_report_df,
        "shared_caption": (
            f"Η κοινή εστίαση είναι {shared_region.concentration_label} | "
            f"top-{len(shared_region.top_region_ids)} μάζα {shared_region.top_mass * 100:.1f}%"
        ),
        "disagreement_caption": (
            f"Η διαφωνία στα κοινά superpixels είναι {disagreement_region.concentration_label} | "
            f"top-{len(disagreement_region.top_region_ids)} μάζα {disagreement_region.top_mass * 100:.1f}%"
        ),
    }

metric_details_df: pd.DataFrame | None = None
metric_curve_df: pd.DataFrame | None = None
if selected_metrics is not None:
    metric_details_rows = [
        {"Metric": "Drop Top", "Value": metric_to_display(selected_metrics.get("drop_top"))},
        {"Metric": "Mean Drop Random", "Value": metric_to_display(selected_metrics.get("drop_rand_mean"))},
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
    metric_details_df = pd.DataFrame(metric_details_rows)
    if "faithfulness_xs" in selected_metrics:
        metric_curve_df = pd.DataFrame(
            {
                "Deletion": [float(value) for value in selected_metrics["deletion_curve"]],
                "Insertion": [float(value) for value in selected_metrics["insertion_curve"]],
            },
            index=[float(value) for value in selected_metrics["faithfulness_xs"]],
        )
        metric_curve_df.index.name = "Fraction"

counterfactual_progression_df: pd.DataFrame | None = None
if selected_counterfactual is not None:
    progression_rows = selected_counterfactual.get("progression_rows", [])
    if progression_rows:
        counterfactual_progression_df = pd.DataFrame(progression_rows)

report_payload = {
    "title": "XAI Analysis Report",
    "meta": {
        "image_name": uploaded_file.name,
        "primary_method": explain_method,
        "method_set": ", ".join(comparison_methods),
        "predicted_class": str(selected_analysis["predicted_class"]),
        "confidence_pct": f"{float(selected_analysis['confidence']) * 100:.1f}%",
        "runtime_s": f"{float(selected_analysis['total_runtime_s']):.2f}s",
    },
    "overview": {
        "original_image": np.asarray(model_view_image, dtype=np.uint8),
        "explained_image": np.asarray(display_explanation_image, dtype=np.uint8),
        "heatmap_image": np.asarray(selected_bundle["heatmap_rgb"], dtype=np.uint8),
        "top5_df": top5_df.copy(),
    },
}
if selected_semantic is not None and selected_semantic_error is None:
    semantic_table = selected_semantic.get("score_table")
    if isinstance(semantic_table, pd.DataFrame) and not semantic_table.empty:
        semantic_table = semantic_table.copy()
        semantic_table["Σημασιολογικό Σκορ (%)"] = semantic_table["Σημασιολογικό Σκορ (%)"].map(
            lambda value: round(float(value), 2)
        )
    report_payload["semantic"] = {
        "summary": str(selected_semantic.get("summary_gr", "")),
        "top_concepts_text": str(selected_semantic.get("top_concepts_text", "")),
        "top_concepts": list(selected_semantic.get("top_concepts", [])),
        "focus_image": np.asarray(selected_semantic["focus_rgb"], dtype=np.uint8),
        "focus_caption": (
            f"Top-{len(selected_semantic.get('top_superpixel_ids', []))} σημασιολογικά superpixels | "
            f"περιοχή εστίασης {float(selected_semantic.get('focus_area_pct', 0.0)):.1f}%"
        ),
        "score_table": semantic_table if isinstance(semantic_table, pd.DataFrame) else pd.DataFrame(),
    }
if selected_metrics is not None:
    report_payload["metrics"] = {
        "deletion_auc": metric_to_display(selected_metrics.get("deletion_auc")),
        "insertion_auc": metric_to_display(selected_metrics.get("insertion_auc")),
        "sensitivity": metric_to_display(selected_metrics.get("sensitivity")),
        "hoyer_sparsity": metric_to_display(selected_metrics.get("hoyer_sparsity")),
        "aopc_delta": metric_to_display(selected_metrics.get("aopc_delta")),
        "robustness": metric_to_display(selected_metrics.get("spearman_rho")) if selected_metrics.get("spearman_rho") is not None else "-",
        "curve_df": metric_curve_df if isinstance(metric_curve_df, pd.DataFrame) else pd.DataFrame(),
        "details_df": metric_details_df if isinstance(metric_details_df, pd.DataFrame) else pd.DataFrame(),
    }
if selected_counterfactual is not None and selected_counterfactual_error is None:
    report_payload["counterfactual"] = {
        "summary_lines": [str(line) for line in selected_counterfactual.get("summary_lines", [])],
        "original_image": np.asarray(model_view_image, dtype=np.uint8),
        "removed_image": np.asarray(selected_counterfactual["removed_evidence_rgb"], dtype=np.uint8),
        "counterfactual_image": np.asarray(selected_counterfactual["counterfactual_rgb"], dtype=np.uint8),
        "progression_df": counterfactual_progression_df if isinstance(counterfactual_progression_df, pd.DataFrame) else pd.DataFrame(),
    }
if shared_focus_export is not None:
    report_payload["shared_focus"] = {
        "available": True,
        "summary_lines": [str(line) for line in shared_focus_export["consensus_analysis"]["summary_lines"]],
        "original_image": np.asarray(model_view_image, dtype=np.uint8),
        "shared_focus_image": np.asarray(shared_focus_export["shared_evidence_display_rgb"], dtype=np.uint8),
        "disagreement_image": np.asarray(shared_focus_export["disagreement_display_rgb"], dtype=np.uint8),
        "shared_caption": str(shared_focus_export["shared_caption"]),
        "disagreement_caption": str(shared_focus_export["disagreement_caption"]),
        "pairwise_df": shared_focus_export["pairwise_report_df"],
        "method_df": shared_focus_export["method_report_df"],
    }

pdf_report_bytes = build_pdf_report(report_payload)
pdf_report_filename = f"{Path(uploaded_file.name).stem}_xai_report.pdf"

st.markdown(
    f"""
    <div class="xai-callout">
        <strong>Current image:</strong> {uploaded_file.name} &nbsp;|&nbsp;
        <strong>Primary method:</strong> {explain_method} &nbsp;|&nbsp;
        <strong>Method set:</strong> {", ".join(comparison_methods)}
    </div>
    """,
    unsafe_allow_html=True,
)

export_col_left, export_col_right = st.columns([0.74, 0.26], gap="medium")
with export_col_left:
    st.caption("Κατέβασε την τρέχουσα ανάλυση σε πολυσέλιδο PDF report με εικόνες, πίνακες και βασικά συμπεράσματα.")
with export_col_right:
    st.download_button(
        "Export PDF Report",
        data=pdf_report_bytes,
        file_name=pdf_report_filename,
        mime="application/pdf",
        use_container_width=True,
    )

overview_tab, metrics_tab, counterfactual_tab, shared_evidence_tab = st.tabs(
    ["Overview", "Metrics", "Counterfactual", "Shared Focus"]
)

with overview_tab:
    st.markdown("### Run Overview")
    st.markdown(
        "Τα βασικά αποτελέσματα της τρέχουσας εκτέλεσης: η επιλεγμένη πρόβλεψη, η κύρια οπτική εξήγηση και η σημασιολογική της ανάγνωση."
    )
    snapshot_cols = st.columns(4, gap="medium")
    with snapshot_cols[0]:
        render_kpi_card(
            "Predicted Class",
            str(selected_analysis["predicted_class"]),
            "Η κορυφαία κλάση που επέστρεψε το ImageNet-pretrained ResNet50.",
        )
    with snapshot_cols[1]:
        render_kpi_card(
            "Confidence",
            f"{float(selected_analysis['confidence']) * 100:.1f}%",
            "Πιθανότητα/βεβαιότητα για την επιλεγμένη πρόβλεψη.",
        )
    with snapshot_cols[2]:
        render_kpi_card(
            "Primary Method",
            explain_method,
            "Η μέθοδος που καθοδηγεί τις παρακάτω ενότητες.",
        )
    with snapshot_cols[3]:
        render_kpi_card(
            "Runtime",
            f"{float(selected_analysis['total_runtime_s']):.2f}s",
            "Συνολικός χρόνος για πρόβλεψη, εξήγηση και τυχόν ενεργές μετρικές.",
        )

    visual_col, context_col = st.columns([1.14, 0.86], gap="large")
    with visual_col:
        st.subheader("Visual Evidence")
        st.caption("Εναλλαγή ανάμεσα στην ερμηνευμένη προβολή και τον ακατέργαστο χάρτη θερμότητας, κρατώντας δίπλα την αρχική εικόνα.")
        base_col, explanation_col = st.columns([0.92, 1.08], gap="large")
        with base_col:
            st.image(model_view_image, width=image_width, caption="Model input view")
        with explanation_col:
            if view_mode == "Tabs":
                overlay_tab, heatmap_tab = st.tabs(["Explained View", "Heatmap"])
                with overlay_tab:
                    st.image(display_explanation_image, width=image_width)
                with heatmap_tab:
                    st.image(selected_bundle["heatmap_rgb"], width=image_width)
            else:
                side_a, side_b = st.columns(2, gap="large")
                with side_a:
                    st.image(display_explanation_image, width=image_width, caption="Επιλεγμένη προβολή εξήγησης")
                with side_b:
                    st.image(selected_bundle["heatmap_rgb"], width=image_width, caption="Ακατέργαστος χάρτης θερμότητας")

    with context_col:
        st.subheader("Semantic Interpretation")
        if selected_semantic_error is not None:
            st.error(f"Η σημασιολογική ανάλυση δεν είναι διαθέσιμη σε αυτή την εκτέλεση: {selected_semantic_error}")
        elif selected_semantic is None:
            st.info("Τρέξε την ανάλυση για να εμφανιστεί η σημασιολογική ανάγνωση.")
        else:
            render_panel("Semantic Summary", [str(selected_semantic["summary_gr"])])
            render_semantic_top_concepts(list(selected_semantic.get("top_concepts", [])))
        st.markdown("#### Top-5 Predictions")
        top5_display_df = top5_df.rename(
            columns={
                "Rank": "Θέση",
                "Class Index": "Δείκτης Κλάσης",
                "Class Name": "Όνομα Κλάσης",
                "Probability (%)": "Πιθανότητα (%)",
            }
        )
        render_dataframe_compat(top5_display_df, hide_index=True, height=235)

    if selected_semantic is not None and selected_semantic_error is None:
        semantic_overview_cols = st.columns([1.0, 1.0], gap="large")
        with semantic_overview_cols[0]:
            st.markdown("#### Semantic Focus Region")
            st.image(
                np.asarray(selected_semantic["focus_rgb"], dtype=np.uint8),
                width=image_width,
                caption=(
                    f"Top-{len(selected_semantic.get('top_superpixel_ids', []))} σημασιολογικά superpixels | "
                    f"περιοχή εστίασης {float(selected_semantic.get('focus_area_pct', 0.0)):.1f}%"
                ),
            )
        with semantic_overview_cols[1]:
            score_table = selected_semantic.get("score_table")
            if isinstance(score_table, pd.DataFrame) and not score_table.empty:
                semantic_table = score_table.copy()
                semantic_table["Σημασιολογικό Σκορ (%)"] = semantic_table["Σημασιολογικό Σκορ (%)"].map(
                    lambda value: round(float(value), 2)
                )
                st.markdown("#### Semantic Concept Table")
                render_dataframe_compat(semantic_table, hide_index=True, height=265)
            else:
                st.info("Δεν προέκυψε πίνακας σημασιολογικών scores για το τρέχον αποτέλεσμα.")

with metrics_tab:
    st.markdown("### Metrics")
    st.markdown(
        '<div class="xai-section-note">Οι μετρικές ποιότητας ανά εικόνα υπολογίζονται μόνο για τη βασική μέθοδο ώστε το demo να παραμένει γρήγορο.</div>',
        unsafe_allow_html=True,
    )
    if selected_metrics is None:
        st.info("Ενεργοποίησε τον υπολογισμό μετρικών για τη βασική μέθοδο και ξανατρέξε την ανάλυση για να εμφανιστεί αυτή η καρτέλα.")
    else:
        quality_cols_1 = st.columns(4, gap="medium")
        with quality_cols_1[0]:
            render_kpi_card(
                "Deletion AUC",
                metric_to_display(selected_metrics.get("deletion_auc")),
                "Faithfulness κατά την προοδευτική αφαίρεση σημαντικών περιοχών.",
            )
        with quality_cols_1[1]:
            render_kpi_card(
                "Insertion AUC",
                metric_to_display(selected_metrics.get("insertion_auc")),
                "Faithfulness όταν οι σημαντικές περιοχές επανέρχονται σταδιακά.",
            )
        with quality_cols_1[2]:
            render_kpi_card(
                "Sensitivity",
                metric_to_display(selected_metrics.get("sensitivity")),
                "Πτώση για τις κορυφαίες περιοχές σε σχέση με τυχαία υποσύνολα.",
            )
        with quality_cols_1[3]:
            render_kpi_card(
                "Hoyer Sparsity",
                metric_to_display(selected_metrics.get("hoyer_sparsity")),
                "Συμπαγότητα της εξήγησης πάνω στα superpixels.",
            )

        quality_cols_2 = st.columns(4, gap="medium")
        with quality_cols_2[0]:
            render_kpi_card(
                "AOPC Delta",
                metric_to_display(selected_metrics.get("aopc_delta")),
                "Διαφορά ανάμεσα στη συμπεριφορά insertion και deletion.",
            )
        with quality_cols_2[1]:
            robustness_value = selected_metrics.get("spearman_rho")
            robustness_note = "Το robustness είναι απενεργοποιημένο σε αυτή την εκτέλεση."
            if robustness_value is not None:
                robustness_note = "Συμφωνία Spearman ανάμεσα στην αρχική και τη θορυβώδη εξήγηση."
            render_kpi_card(
                "Robustness",
                metric_to_display(robustness_value) if robustness_value is not None else "-",
                robustness_note,
            )
        with quality_cols_2[2]:
            render_kpi_card(
                "Metrics Runtime",
                metric_to_display(float(selected_analysis["metrics_runtime_s"]), digits=2),
                "Χρόνος που καταναλώθηκε μόνο για τους υπολογισμούς των μετρικών.",
            )
        with quality_cols_2[3]:
            render_kpi_card(
                "Overlay Opacity",
                f"{overlay_alpha:.2f}",
                "Η τρέχουσα ρύθμιση αδιαφάνειας που χρησιμοποιείται στις εικόνες της επισκόπησης.",
            )

        detail_col1, detail_col2 = st.columns([0.78, 1.22], gap="large")
        with detail_col1:
            metric_details_rows = [
                {"Metric": "Drop Top", "Value": metric_to_display(selected_metrics.get("drop_top"))},
                {"Metric": "Mean Drop Random", "Value": metric_to_display(selected_metrics.get("drop_rand_mean"))},
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

with counterfactual_tab:
    st.markdown("### Counterfactual")
    st.markdown(
        '<div class="xai-section-note">Αυτή η καρτέλα θολώνει προοδευτικά τα πιο επιδραστικά superpixels από τη βασική μέθοδο μέχρι το μοντέλο να αλλάξει πρόβλεψη ή να φτάσει το τρέχον όριο αφαίρεσης.</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Προεπιλεγμένες ρυθμίσεις αντιπαραδείγματος: {COUNTERFACTUAL_SLIC_SEGMENTS_DEFAULT} τμήματα SLIC, compactness "
        f"{COUNTERFACTUAL_SLIC_COMPACTNESS_DEFAULT:.1f}, sigma {COUNTERFACTUAL_SLIC_SIGMA_DEFAULT:.1f}, "
        f"ακτίνα θόλωσης {COUNTERFACTUAL_BLUR_RADIUS_DEFAULT:.1f}, έως {COUNTERFACTUAL_MAX_STEPS_DEFAULT} βήματα "
        f"ή {COUNTERFACTUAL_MAX_REMOVAL_FRACTION_DEFAULT * 100.0:.0f}% αφαίρεση στοιχείων."
    )

    if selected_counterfactual_error is not None:
        st.error(f"Η αντιπαραδειγματική ανάλυση δεν είναι διαθέσιμη σε αυτή την εκτέλεση: {selected_counterfactual_error}")
    elif selected_counterfactual is None:
        st.info("Τρέξε την ανάλυση για να εμφανιστεί το αντιπαράδειγμα.")
    else:
        counterfactual_kpis = st.columns(4, gap="medium")
        with counterfactual_kpis[0]:
            render_kpi_card(
                "Original Class",
                str(selected_counterfactual["original_class"]),
                f"{float(selected_counterfactual['original_confidence']) * 100.0:.1f}% βεβαιότητα.",
            )
        with counterfactual_kpis[1]:
            render_kpi_card(
                "Final Top Class",
                str(selected_counterfactual["final_class"]),
                f"{float(selected_counterfactual['final_confidence']) * 100.0:.1f}% βεβαιότητα μετά την αφαίρεση.",
            )
        with counterfactual_kpis[2]:
            render_kpi_card(
                "Removed Evidence",
                f"{int(selected_counterfactual['removed_superpixel_count'])} SP",
                f"{float(selected_counterfactual['removed_area_pct']):.1f}% της εικόνας.",
            )
        with counterfactual_kpis[3]:
            outcome_label = "Αλλαγή" if bool(selected_counterfactual["flip_found"]) else "Χωρίς Αλλαγή"
            outcome_note = "Η πρόβλεψη άλλαξε μέσα στο τρέχον όριο αφαίρεσης."
            if not bool(selected_counterfactual["flip_found"]):
                outcome_note = "Η πρόβλεψη παρέμεινε σταθερή μέσα στο τρέχον όριο αφαίρεσης."
            render_kpi_card("Outcome", outcome_label, outcome_note)

        render_panel("Counterfactual Summary", [str(line) for line in selected_counterfactual["summary_lines"]])

        step_states = selected_counterfactual.get("step_states", [])
        if isinstance(step_states, list) and step_states:
            st.markdown("#### Interactive Counterfactual Scrubber")
            st.caption(
                "Σύρε το slider για να θολώσεις προοδευτικά τα κορυφαία superpixels και να δεις live πότε αλλάζει η πρόβλεψη και η confidence."
            )

            default_step = len(step_states) - 1
            flip_step_value = selected_counterfactual.get("flip_step")
            if isinstance(flip_step_value, int):
                default_step = min(max(int(flip_step_value), 0), len(step_states) - 1)

            scrubber_key = f"counterfactual_scrubber_{uploaded_file.name}_{explain_method}"
            if scrubber_key not in st.session_state:
                st.session_state[scrubber_key] = default_step
            st.session_state[scrubber_key] = min(max(int(st.session_state[scrubber_key]), 0), len(step_states) - 1)

            selected_step_index = st.slider(
                "Counterfactual Step",
                min_value=0,
                max_value=len(step_states) - 1,
                key=scrubber_key,
                help="Κάθε βήμα θολώνει ένα επιπλέον κορυφαίο superpixel από τη βασική εξήγηση.",
            )
            selected_step_state = step_states[int(selected_step_index)]

            scrubber_kpis = st.columns(5, gap="medium")
            with scrubber_kpis[0]:
                render_kpi_card(
                    "Step",
                    f"{int(selected_step_state['Step'])}/{len(step_states) - 1}",
                    "Το τρέχον βήμα της προοδευτικής αφαίρεσης evidence.",
                )
            with scrubber_kpis[1]:
                render_kpi_card(
                    "Current Top Class",
                    str(selected_step_state["Current Winner"]),
                    "Η κορυφαία κλάση του μοντέλου στο συγκεκριμένο βήμα.",
                )
            with scrubber_kpis[2]:
                render_kpi_card(
                    "Current Confidence",
                    f"{float(selected_step_state['Current Winner Probability']) * 100.0:.1f}%",
                    "Confidence της τρέχουσας κορυφαίας κλάσης.",
                )
            with scrubber_kpis[3]:
                render_kpi_card(
                    "Original-Class Confidence",
                    f"{float(selected_step_state['Original Class Probability']) * 100.0:.1f}%",
                    "Η confidence της αρχικής κλάσης καθώς αφαιρείται evidence.",
                )
            with scrubber_kpis[4]:
                render_kpi_card(
                    "Removed Evidence",
                    f"{int(selected_step_state['Removed Superpixels'])} SP",
                    f"{float(selected_step_state['Removed Area (%)']):.1f}% της εικόνας.",
                )

            if int(selected_step_state["Step"]) == 0:
                st.info("Βήμα 0: η αρχική εικόνα πριν αφαιρεθεί οποιοδήποτε evidence.")
            elif bool(selected_step_state["Flipped"]):
                st.success(
                    f"Σε αυτό το βήμα η πρόβλεψη έχει ήδη αλλάξει σε {selected_step_state['Current Winner']}."
                )
            else:
                st.info("Σε αυτό το βήμα το μοντέλο δεν έχει αλλάξει ακόμη την κορυφαία πρόβλεψή του.")

            scrubber_visual_cols = st.columns(3, gap="large")
            with scrubber_visual_cols[0]:
                st.markdown("#### Original Image")
                st.image(model_view_image, width=image_width, caption="Model input view")
            with scrubber_visual_cols[1]:
                removed_caption = "Δεν έχει αφαιρεθεί evidence ακόμη."
                if int(selected_step_state["Step"]) > 0:
                    removed_caption = "Τα στοιχεία που έχουν αφαιρεθεί μέχρι το τρέχον βήμα."
                st.markdown("#### Removed Evidence")
                st.image(
                    np.asarray(selected_step_state["removed_evidence_rgb"], dtype=np.uint8),
                    width=image_width,
                    caption=removed_caption,
                )
            with scrubber_visual_cols[2]:
                state_caption = "Η αρχική εικόνα στο βήμα 0."
                if int(selected_step_state["Step"]) > 0:
                    state_caption = "Η εικόνα μετά την προοδευτική θόλωση των κορυφαίων superpixels."
                st.markdown("#### Current Counterfactual State")
                st.image(
                    np.asarray(selected_step_state["counterfactual_rgb"], dtype=np.uint8),
                    width=image_width,
                    caption=state_caption,
                )
        else:
            cf_visual_cols = st.columns(3, gap="large")
            with cf_visual_cols[0]:
                st.markdown("#### Original Image")
                st.image(model_view_image, width=image_width, caption="Model input view")
            with cf_visual_cols[1]:
                st.markdown("#### Removed Evidence")
                st.image(
                    np.asarray(selected_counterfactual["removed_evidence_rgb"], dtype=np.uint8),
                    width=image_width,
                    caption="Τα στοιχεία που αφαιρέθηκαν από την αρχική εικόνα.",
                )
            with cf_visual_cols[2]:
                st.markdown("#### Counterfactual Image")
                st.image(
                    np.asarray(selected_counterfactual["counterfactual_rgb"], dtype=np.uint8),
                    width=image_width,
                    caption="Η εικόνα αφού θολωθούν τα πιο επιδραστικά στοιχεία.",
                )

        progression_rows = selected_counterfactual.get("progression_rows", [])
        if progression_rows:
            progression_df = pd.DataFrame(progression_rows)
            curve_df = progression_df[
                [
                    "Removed Area (%)",
                    "Original Class Probability (%)",
                    "Current Winner Probability (%)",
                ]
            ].copy()
            render_cols = st.columns([1.16, 0.84], gap="large")
            with render_cols[0]:
                st.markdown("#### Confidence Trajectory")
                curve_df = curve_df.rename(
                    columns={
                        "Removed Area (%)": "Αφαιρεμένη Περιοχή (%)",
                        "Original Class Probability (%)": "Πιθανότητα Αρχικής Κλάσης (%)",
                        "Current Winner Probability (%)": "Πιθανότητα Τρέχουσας Κορυφαίας Κλάσης (%)",
                    }
                )
                render_line_chart_compat(curve_df.set_index("Αφαιρεμένη Περιοχή (%)"), height=300)
            with render_cols[1]:
                st.markdown("#### Step Table")
                table_df = progression_df.copy()
                table_df["Removed Area (%)"] = table_df["Removed Area (%)"].map(lambda value: round(float(value), 1))
                table_df["Original Class Probability (%)"] = table_df["Original Class Probability (%)"].map(
                    lambda value: round(float(value), 2)
                )
                table_df["Current Winner Probability (%)"] = table_df["Current Winner Probability (%)"].map(
                    lambda value: round(float(value), 2)
                )
                table_df = table_df.rename(
                    columns={
                        "Step": "Βήμα",
                        "Removed Superpixels": "Αφαιρεμένα Superpixels",
                        "Removed Area (%)": "Αφαιρεμένη Περιοχή (%)",
                        "Original Class Probability (%)": "Πιθανότητα Αρχικής Κλάσης (%)",
                        "Current Winner": "Τρέχουσα Κορυφαία Κλάση",
                        "Current Winner Probability (%)": "Πιθανότητα Τρέχουσας Κλάσης (%)",
                        "Flipped": "Άλλαξε;",
                    }
                )
                render_dataframe_compat(table_df, hide_index=True, height=300)

with shared_evidence_tab:
    st.markdown("### Shared Focus")
    st.markdown(
        '<div class="xai-section-note">Αυτή η καρτέλα κρατά μόνο τις περιοχές της εικόνας που παραμένουν σημαντικές σε πολλές μεθόδους πάνω στο ίδιο SLIC πλέγμα superpixels.</div>',
        unsafe_allow_html=True,
    )
    if len(comparison_bundles) < 2:
        st.info("Επίλεξε τουλάχιστον δύο μεθόδους στις `Comparison Methods` και πάτησε `Run Analysis` για να εμφανιστεί η κοινή εστίαση.")
    else:
        consensus_analysis = build_consensus_analysis(
            image=model_view_image,
            method_cams={method_name: np.asarray(bundle["cam"], dtype=np.float32) for method_name, bundle in comparison_bundles.items()},
            n_segments=metrics_slic_segments,
            compactness=metrics_slic_compactness,
            sigma=metrics_slic_sigma,
            top_k=SUMMARY_TOP_K,
        )
        shared_evidence_map = np.asarray(
            consensus_analysis.get("shared_evidence_map", consensus_analysis.get("consensus_map")),
            dtype=np.float32,
        )
        disagreement_map = np.asarray(consensus_analysis["disagreement_map"], dtype=np.float32)
        shared_region: RegionAnalysis = consensus_analysis.get(
            "shared_region",
            consensus_analysis.get("consensus_region"),
        )
        disagreement_region: RegionAnalysis = consensus_analysis["disagreement_region"]
        required_votes = int(
            consensus_analysis.get(
                "required_votes",
                max(2, int(np.ceil(len(comparison_bundles) * 0.67))),
            )
        )

        shared_evidence_heatmap_rgb = apply_colormap_to_cam(
            shared_evidence_map
        )
        disagreement_heatmap_rgb = apply_colormap_to_cam(disagreement_map)
        shared_evidence_overlay_rgb = overlay_cam_on_image(
            np.asarray(model_view_image),
            shared_evidence_heatmap_rgb,
            alpha=overlay_alpha,
        )
        disagreement_overlay_rgb = overlay_cam_on_image(np.asarray(model_view_image), disagreement_heatmap_rgb, alpha=overlay_alpha)
        shared_evidence_focus_rgb = build_simplified_focus_image(model_view_image, shared_region)
        disagreement_focus_rgb = build_simplified_focus_image(model_view_image, disagreement_region)
        shared_evidence_display_rgb = (
            shared_evidence_overlay_rgb if visual_style == "Heatmap Overlay" else shared_evidence_focus_rgb
        )
        disagreement_display_rgb = (
            disagreement_overlay_rgb if visual_style == "Heatmap Overlay" else disagreement_focus_rgb
        )

        kpi_cols = st.columns(4, gap="medium")
        with kpi_cols[0]:
            render_kpi_card(
                "Methods Compared",
                str(len(comparison_bundles)),
                ", ".join(comparison_bundles.keys()),
            )
        with kpi_cols[1]:
            render_kpi_card(
                "Agreement Rule",
                f"{required_votes}/{len(comparison_bundles)}",
                "Ένα superpixel παραμένει ορατό μόνο αν το στηρίζουν τόσες μέθοδοι.",
            )
        with kpi_cols[2]:
            render_kpi_card(
                "Agreement Strength",
                metric_to_display(float(consensus_analysis["consensus_strength"])),
                str(consensus_analysis["agreement_label"]).capitalize(),
            )
        with kpi_cols[3]:
            render_kpi_card(
                "Shared IoU",
                metric_to_display(float(consensus_analysis["mean_pairwise_iou"])),
                "Επικάλυψη στις ισχυρότερες κοινές περιοχές.",
            )

        render_panel("Shared Focus Summary", [str(line) for line in consensus_analysis["summary_lines"]])

        shared_visual_cols = st.columns([0.92, 1.08], gap="large")
        with shared_visual_cols[0]:
            st.markdown("#### Original Image")
            st.image(model_view_image, width=image_width, caption="Model input view")
        with shared_visual_cols[1]:
            st.markdown("#### Shared Focus Map")
            if float(shared_region.top_mass) <= 0.0:
                st.info(
                    "Καμία περιοχή δεν πέρασε τον τρέχοντα κανόνα συμφωνίας. Δοκίμασε άλλο σύνολο μεθόδων ή δες τις προχωρημένες λεπτομέρειες πιο κάτω."
                )
                st.image(model_view_image, width=image_width, caption="Δεν βρέθηκαν ισχυρά κοινά στοιχεία σε αυτή την εκτέλεση")
            else:
                st.image(
                    shared_evidence_display_rgb,
                    width=image_width,
                    caption=(
                        f"Η κοινή εστίαση είναι {shared_region.concentration_label} | "
                        f"top-{len(shared_region.top_region_ids)} μάζα {shared_region.top_mass * 100:.1f}%"
                    ),
                )
                st.caption(f"Κοινή περιοχή στο πλέγμα κοινής εστίασης: {shared_region.top_region_summary}")

        with st.expander("Advanced Agreement Details", expanded=False):
            table_col_a, table_col_b = st.columns([1.0, 1.0], gap="large")
            with table_col_a:
                st.markdown("#### Pairwise Agreement")
                pairwise_df = pd.DataFrame(consensus_analysis["pairwise_rows"])
                if not pairwise_df.empty:
                    pairwise_df["Cosine Agreement"] = pairwise_df["Cosine Agreement"].map(lambda value: round(float(value), 3))
                    pairwise_df["Top-focus IoU"] = pairwise_df["Top-focus IoU"].map(lambda value: round(float(value), 3))
                    pairwise_df = pairwise_df.rename(
                        columns={
                            "Methods": "Μέθοδοι",
                            "Cosine Agreement": "Συμφωνία Cosine",
                            "Top-focus IoU": "IoU Κορυφαίας Εστίασης",
                        }
                    )
                    render_dataframe_compat(pairwise_df, hide_index=True, height=185)
            with table_col_b:
                st.markdown("#### Method Alignment to Shared Focus")
                method_df = pd.DataFrame(consensus_analysis["method_rows"])
                if not method_df.empty:
                    if "Shared Cosine" not in method_df.columns and "Consensus Cosine" in method_df.columns:
                        method_df = method_df.rename(
                            columns={
                                "Consensus Cosine": "Shared Cosine",
                                "Consensus IoU": "Shared IoU",
                            }
                        )
                    method_df["Shared Cosine"] = method_df["Shared Cosine"].map(lambda value: round(float(value), 3))
                    method_df["Shared IoU"] = method_df["Shared IoU"].map(lambda value: round(float(value), 3))
                    method_df = method_df.rename(
                        columns={
                            "Method": "Μέθοδος",
                            "Shared Cosine": "Κοινό Cosine",
                            "Shared IoU": "Κοινό IoU",
                        }
                    )
                    render_dataframe_compat(method_df, hide_index=True, height=185)

            advanced_visual_col, advanced_text_col = st.columns([1.0, 1.0], gap="large")
            with advanced_visual_col:
                st.markdown("#### Disagreement Map")
                st.image(
                    disagreement_display_rgb,
                    width=image_width,
                    caption=(
                        f"Η διαφωνία στα κοινά superpixels είναι {disagreement_region.concentration_label} | "
                        f"top-{len(disagreement_region.top_region_ids)} μάζα {disagreement_region.top_mass * 100:.1f}%"
                    ),
                )
                st.caption(f"Περιοχή διαφωνίας στο κοινό πλέγμα: {disagreement_region.top_region_summary}")
            with advanced_text_col:
                st.markdown("#### Method Snapshots")
                compare_columns = st.columns(len(comparison_bundles), gap="medium")
                for index, (method_name, bundle) in enumerate(comparison_bundles.items()):
                    with compare_columns[index]:
                        compare_image = (
                            bundle["overlay_rgb"] if visual_style == "Heatmap Overlay" else bundle["simplified_rgb"]
                        )
                        comparison_region: RegionAnalysis = bundle["region_analysis"]
                        st.markdown(
                            f'<div class="xai-compare-card"><div class="xai-compare-title">{method_name}</div></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(compare_image, width=image_width)
                        st.caption(
                            f"Top-{len(comparison_region.top_region_ids)} μάζα: {comparison_region.top_mass * 100:.1f}%"
                        )
                        st.caption(f"Εστίαση: {comparison_region.top_region_summary}")

            semantic_compare_enabled = st.checkbox(
                "Include Semantic Agreement",
                value=SEMANTIC_COMPARE_AGREEMENT_DEFAULT,
                help="Τρέχει το σημασιολογικό layer CLIP από το notebook στις επιλεγμένες μεθόδους μέσα στην προβολή κοινής εστίασης.",
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
                            image=model_view_image,
                            method_name=method_name,
                            analysis=analysis,
                        )
                    except Exception as exc:
                        semantic_errors.append(f"{method_name}: {exc}")

                if semantic_errors:
                    st.warning("Ορισμένα σημασιολογικά αποτελέσματα σύγκρισης δεν ήταν δυνατό να παραχθούν: " + " | ".join(semantic_errors))

                if len(semantic_results) >= 2:
                    semantic_agreement = build_semantic_agreement(semantic_results)
                    render_panel(
                        "Semantic Agreement",
                        [
                            (
                                "Μέση ζευγαρωτή σημασιολογική συμφωνία cosine στις επιλεγμένες μεθόδους: "
                                f"{float(semantic_agreement['mean_pairwise_cosine']):.3f}."
                            ),
                            "Το σκορ υπολογίζεται από την κατανομή CLIP εννοιών κάθε μεθόδου πάνω στη δική της περιοχή εστίασης.",
                        ],
                    )

                    semantic_pairwise_df = semantic_agreement["pairwise_df"]
                    if isinstance(semantic_pairwise_df, pd.DataFrame) and not semantic_pairwise_df.empty:
                        semantic_pairwise_df = semantic_pairwise_df.copy()
                        semantic_pairwise_df["Σημασιολογική Συμφωνία Cosine"] = semantic_pairwise_df[
                            "Semantic Cosine Agreement"
                        ].map(lambda value: round(float(value), 3))
                        semantic_pairwise_df = semantic_pairwise_df.drop(columns=["Semantic Cosine Agreement"]).rename(
                            columns={"Methods": "Μέθοδοι"}
                        )
                        render_dataframe_compat(semantic_pairwise_df, hide_index=True, height=150)

                    semantic_concept_df = semantic_agreement["concept_df"]
                    if isinstance(semantic_concept_df, pd.DataFrame) and not semantic_concept_df.empty:
                        st.markdown("#### Semantic Concept Distribution by Method")
                        render_dataframe_compat(semantic_concept_df.round(2), height=225)
