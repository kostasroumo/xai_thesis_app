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

from src.data.preprocessing import (
    get_inference_transform,
    load_image,
    preprocess_pil_image,
)
from src.explainers.gradcam_explainer import GradCAM
from src.explainers.integrated_gradients_explainer import generate_integrated_gradients
from src.explainers.lime_explainer import generate_lime
from src.explainers.occlusion_explainer import generate_occlusion
from src.metrics.explanation_metrics import MetricSettings, compute_explanation_metrics
from src.models.class_names import get_imagenet_class_names
from src.models.loader import get_last_conv_layer, load_model
from src.models.predictor import predict
from src.utils.config import (
    ANALYSIS_CACHE_MAX_ENTRIES,
    CAM_OVERLAY_ALPHA,
    CAM_SCORE_TYPE_DEFAULT,
    IG_BASELINE_BLUR_RADIUS_DEFAULT,
    IG_INTERNAL_BATCH_SIZE_DEFAULT,
    IG_STEPS_DEFAULT,
    LIME_BASELINE_BLUR_RADIUS_DEFAULT,
    LIME_COMPACTNESS_DEFAULT,
    LIME_N_SAMPLES_DEFAULT,
    LIME_N_SEGMENTS_DEFAULT,
    LIME_PERTURBATIONS_PER_EVAL_DEFAULT,
    LIME_RANDOM_SEED_DEFAULT,
    LIME_SIGMA_DEFAULT,
    METRICS_ENABLED_DEFAULT,
    METRICS_FAITHFULNESS_BLUR_RADIUS_DEFAULT,
    METRICS_FAITHFULNESS_STEPS_DEFAULT,
    METRICS_RANDOM_SEED_DEFAULT,
    METRICS_ROBUSTNESS_ENABLED_DEFAULT,
    METRICS_ROBUSTNESS_NOISE_SIGMA_DEFAULT,
    METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT,
    METRICS_SENSITIVITY_BLUR_RADIUS_DEFAULT,
    METRICS_SENSITIVITY_N_RANDOM_DEFAULT,
    METRICS_SENSITIVITY_TOP_N_DEFAULT,
    METRICS_SLIC_COMPACTNESS_DEFAULT,
    METRICS_SLIC_SEGMENTS_DEFAULT,
    METRICS_SLIC_SIGMA_DEFAULT,
    OCC_BASELINE_BLUR_RADIUS_DEFAULT,
    OCC_PATCH_SIZE_DEFAULT,
    OCC_STRIDE_DEFAULT,
    MAX_UI_IMAGE_SIDE,
    TOP_K,
)
from src.visualization.heatmaps import (
    apply_colormap_to_cam,
    overlay_cam_on_image,
)

st.set_page_config(page_title="Single Image Analysis", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Single Image Analysis")
st.write("Upload one image to run classification and generate an explanation map.")

control_col1, control_col2, control_col3, control_col4 = st.columns([1.1, 1.2, 1, 1], gap="medium")
with control_col1:
    explain_method = st.selectbox(
        "Explainability method",
        options=["Grad-CAM", "Integrated Gradients", "Occlusion", "LIME"],
        index=0,
    )
with control_col2:
    score_type = st.radio(
        "Score type",
        options=["logit", "prob"],
        index=0 if CAM_SCORE_TYPE_DEFAULT == "logit" else 1,
        horizontal=True,
        help="Use class logit score (recommended) or softmax probability score for explanation.",
    )
with control_col3:
    view_mode = st.radio(
        "Visualization layout",
        options=["Tabs", "Side by side"],
        index=0,
        horizontal=True,
    )
with control_col4:
    image_size_label = st.select_slider(
        "Image size",
        options=["Small", "Medium", "Large"],
        value="Medium",
    )

image_width_map = {"Small": 300, "Medium": 420, "Large": 540}
image_width = image_width_map[image_size_label]

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

with st.expander("Metrics settings", expanded=False):
    compute_metrics = st.checkbox("Compute metrics dashboard", value=METRICS_ENABLED_DEFAULT)
    metrics_seed = st.number_input(
        "Metrics random seed",
        min_value=0,
        max_value=1_000_000,
        value=METRICS_RANDOM_SEED_DEFAULT,
        step=1,
    )

    metrics_slic_segments = st.slider(
        "SLIC segments (for metrics)",
        min_value=20,
        max_value=200,
        value=METRICS_SLIC_SEGMENTS_DEFAULT,
        step=10,
    )
    metrics_slic_compactness = st.slider(
        "SLIC compactness (for metrics)",
        min_value=1.0,
        max_value=40.0,
        value=float(METRICS_SLIC_COMPACTNESS_DEFAULT),
        step=0.5,
    )
    metrics_slic_sigma = st.slider(
        "SLIC sigma (for metrics)",
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
        "Compute robustness (slower: re-runs explanation on noisy input)",
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


@st.cache_resource
def get_runtime_objects():
    model, weights = load_model()
    class_names = get_imagenet_class_names(weights)
    transform = get_inference_transform(weights)
    target_layer = get_last_conv_layer(model)
    return model, class_names, transform, target_layer


def get_session_analysis_cache() -> OrderedDict[str, dict[str, Any]]:
    cache = st.session_state.get("analysis_cache")
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        st.session_state["analysis_cache"] = cache
    return cache


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


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)
run_clicked = st.button("Run analysis", type="primary")

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    pil_image = resize_for_display(load_image(image_bytes))
except Exception as exc:
    st.error(f"Could not read the image file: {exc}")
    st.stop()

analysis_key = build_analysis_key(
    image_bytes=image_bytes,
    explain_method=explain_method,
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
    compute_metrics=bool(compute_metrics),
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
analysis_cache = get_session_analysis_cache()
analysis = analysis_cache.get(analysis_key)

if run_clicked:
    try:
        with st.spinner("Running inference and generating explanations..."):
            analysis = run_analysis(
                image_bytes=image_bytes,
                explain_method=explain_method,
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
                compute_metrics=bool(compute_metrics),
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
    except Exception as exc:
        st.error("An error occurred during analysis.")
        st.exception(exc)
        st.stop()

    analysis_cache[analysis_key] = analysis
    analysis_cache.move_to_end(analysis_key)
    while len(analysis_cache) > ANALYSIS_CACHE_MAX_ENTRIES:
        analysis_cache.popitem(last=False)

if analysis is None:
    st.subheader("Uploaded Image")
    st.image(pil_image, width=image_width)
    st.info("Choose method/settings and click `Run analysis`.")
    st.stop()
elif not run_clicked:
    st.caption("Showing cached result for current image and settings.")

predicted_class = str(analysis["predicted_class"])
confidence = float(analysis["confidence"])
cam = np.array(analysis["cam_uint8"]).astype(np.float32) / 255.0
analysis_metrics_raw = analysis.get("metrics")
analysis_metrics = analysis_metrics_raw if isinstance(analysis_metrics_raw, dict) else None
explanation_runtime_s = float(analysis.get("explanation_runtime_s", 0.0))
metrics_runtime_s = float(analysis.get("metrics_runtime_s", 0.0))
total_runtime_s = float(analysis.get("total_runtime_s", 0.0))
heatmap_rgb = apply_colormap_to_cam(cam)
overlay_rgb = overlay_cam_on_image(
    np.array(pil_image),
    heatmap_rgb,
    alpha=CAM_OVERLAY_ALPHA,
)
top5_df = pd.DataFrame(analysis["topk_rows"])

summary_col, preview_col = st.columns([1.25, 1], gap="large")

with summary_col:
    st.subheader("Prediction")
    metric_col1, metric_col2 = st.columns([2.2, 1], gap="medium")
    with metric_col1:
        st.success(f"Predicted class: {predicted_class}")
    with metric_col2:
        st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.caption(f"Method: `{explain_method}` | Score type: `{score_type}`")
    runtime_col1, runtime_col2, runtime_col3 = st.columns(3, gap="small")
    with runtime_col1:
        st.metric("Explain time", f"{explanation_runtime_s:.2f}s")
    with runtime_col2:
        st.metric("Metrics time", f"{metrics_runtime_s:.2f}s")
    with runtime_col3:
        st.metric("Total time", f"{total_runtime_s:.2f}s")
    st.subheader("Top-5 Classes")
    st.dataframe(top5_df, width="stretch", hide_index=True, height=215)

with preview_col:
    st.subheader("Uploaded Image")
    st.image(pil_image, width=image_width)

if analysis_metrics is not None:
    st.subheader("Metrics Dashboard")
    dashboard_df = pd.DataFrame(
        [
            {"Metric": "Deletion AUC", "Value": float(analysis_metrics["deletion_auc"])},
            {"Metric": "Insertion AUC", "Value": float(analysis_metrics["insertion_auc"])},
            {"Metric": "AOPC Delta", "Value": float(analysis_metrics["aopc_delta"])},
            {"Metric": "Sensitivity", "Value": float(analysis_metrics["sensitivity"])},
            {"Metric": "Drop Top", "Value": float(analysis_metrics["drop_top"])},
            {"Metric": "Drop Random Mean", "Value": float(analysis_metrics["drop_rand_mean"])},
            {"Metric": "Hoyer Sparsity", "Value": float(analysis_metrics["hoyer_sparsity"])},
            {"Metric": "Superpixels", "Value": float(analysis_metrics["n_superpixels"])},
        ]
    )
    if "spearman_rho" in analysis_metrics:
        dashboard_df = pd.concat(
            [
                dashboard_df,
                pd.DataFrame(
                    [
                        {"Metric": "Robustness Spearman", "Value": float(analysis_metrics["spearman_rho"])},
                    ]
                ),
            ],
            ignore_index=True,
        )
    for frac in METRICS_ROBUSTNESS_TOPK_FRACS_DEFAULT:
        frac_key = f"iou_top_{int(round(float(frac) * 100.0))}pct"
        if frac_key in analysis_metrics:
            dashboard_df = pd.concat(
                [
                    dashboard_df,
                    pd.DataFrame(
                        [
                            {
                                "Metric": f"Robustness IoU Top-{int(round(float(frac) * 100.0))}%",
                                "Value": float(analysis_metrics[frac_key]),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    st.dataframe(dashboard_df, width="stretch", hide_index=True, height=325)

    if "faithfulness_xs" in analysis_metrics:
        curve_df = pd.DataFrame(
            {
                "Fraction": [float(v) for v in analysis_metrics["faithfulness_xs"]],
                "Deletion": [float(v) for v in analysis_metrics["deletion_curve"]],
                "Insertion": [float(v) for v in analysis_metrics["insertion_curve"]],
            }
        )
        st.line_chart(curve_df.set_index("Fraction"), width="stretch", height=260)

st.subheader("Explanation Visualizations")
if view_mode == "Tabs":
    overlay_tab, heatmap_tab = st.tabs(["Overlay", "Heatmap"])
    with overlay_tab:
        st.image(overlay_rgb, width=image_width)
    with heatmap_tab:
        st.image(heatmap_rgb, width=image_width)
else:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(heatmap_rgb, width=image_width)
    with col2:
        st.image(overlay_rgb, width=image_width)
