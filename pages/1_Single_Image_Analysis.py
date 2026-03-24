from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.data.preprocessing import (
    get_inference_transform,
    load_image,
    preprocess_pil_image,
)
from src.explainers.gradcam_explainer import GradCAM
from src.explainers.integrated_gradients_explainer import generate_integrated_gradients
from src.explainers.lime_explainer import generate_lime
from src.explainers.occlusion_explainer import generate_occlusion
from src.models.class_names import get_imagenet_class_names
from src.models.loader import get_last_conv_layer, load_model
from src.models.predictor import predict
from src.utils.config import (
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
    OCC_BASELINE_BLUR_RADIUS_DEFAULT,
    OCC_PATCH_SIZE_DEFAULT,
    OCC_STRIDE_DEFAULT,
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


@st.cache_resource
def get_runtime_objects():
    model, weights = load_model()
    class_names = get_imagenet_class_names(weights)
    transform = get_inference_transform(weights)
    target_layer = get_last_conv_layer(model)
    return model, class_names, transform, target_layer


@st.cache_data(show_spinner=False)
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
) -> dict[str, Any]:
    pil_image = load_image(image_bytes)
    model, class_names, transform, target_layer = get_runtime_objects()
    input_batch = preprocess_pil_image(pil_image, transform)
    prediction = predict(model, input_batch, class_names, top_k=TOP_K)

    if explain_method == "Grad-CAM":
        gradcam = GradCAM(model, target_layer)
        try:
            cam = gradcam.generate(
                input_batch,
                target_class=prediction.predicted_index,
                score_type=score_type,
            )
        finally:
            gradcam.close()
    elif explain_method == "Integrated Gradients":
        cam = generate_integrated_gradients(
            model=model,
            input_tensor=input_batch,
            image=pil_image,
            transform=transform,
            target_class=prediction.predicted_index,
            score_type=score_type,
            n_steps=ig_steps,
            internal_batch_size=ig_internal_batch_size,
            blur_radius=ig_blur_radius,
        )
    elif explain_method == "Occlusion":
        cam = generate_occlusion(
            model=model,
            input_tensor=input_batch,
            image=pil_image,
            transform=transform,
            target_class=prediction.predicted_index,
            score_type=score_type,
            patch_size=occ_patch_size,
            stride=occ_stride,
            blur_radius=occ_blur_radius,
        )
    else:
        cam = generate_lime(
            model=model,
            input_tensor=input_batch,
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

    heatmap_rgb = apply_colormap_to_cam(cam)
    original_rgb = np.array(pil_image)
    overlay_rgb = overlay_cam_on_image(
        original_rgb,
        heatmap_rgb,
        alpha=CAM_OVERLAY_ALPHA,
    )

    topk_rows = [
        {
            "Rank": rank,
            "Class Index": item.class_index,
            "Class Name": item.class_name,
            "Probability (%)": round(item.probability * 100, 4),
        }
        for rank, item in enumerate(prediction.topk, start=1)
    ]

    return {
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "heatmap_rgb": heatmap_rgb,
        "overlay_rgb": overlay_rgb,
        "topk_rows": topk_rows,
    }


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

try:
    image_bytes = uploaded_file.getvalue()
    pil_image = load_image(image_bytes)
except Exception as exc:
    st.error(f"Could not read the image file: {exc}")
    st.stop()

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
        )
except Exception as exc:
    st.error("An error occurred during analysis.")
    st.exception(exc)
    st.stop()

predicted_class = str(analysis["predicted_class"])
confidence = float(analysis["confidence"])
heatmap_rgb = np.array(analysis["heatmap_rgb"])
overlay_rgb = np.array(analysis["overlay_rgb"])
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
    st.subheader("Top-5 Classes")
    st.dataframe(top5_df, use_container_width=True, hide_index=True, height=215)

with preview_col:
    st.subheader("Uploaded Image")
    st.image(pil_image, width=image_width)

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
