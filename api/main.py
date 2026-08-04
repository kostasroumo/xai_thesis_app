from __future__ import annotations

import base64
import time
from functools import lru_cache
from io import BytesIO
from typing import Literal

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image

from src.data.preprocessing import load_image, preprocess_pil_image, preprocess_spatial_pil_image
from src.explainers.gradcam_explainer import GradCAM
from src.explainers.integrated_gradients_explainer import generate_integrated_gradients
from src.explainers.lime_explainer import generate_lime
from src.explainers.occlusion_explainer import generate_occlusion
from src.models.class_names import get_imagenet_class_names
from src.models.loader import get_last_conv_layer, load_model
from src.models.predictor import PredictionResult, predict
from src.visualization.heatmaps import apply_colormap_to_cam, overlay_cam_on_image

MethodName = Literal["Grad-CAM", "Integrated Gradients", "Occlusion", "LIME"]
ScoreType = Literal["logit", "prob"]

METHOD_HEATMAP_COLORMAPS = {
    "LIME": cv2.COLORMAP_VIRIDIS,
}

app = FastAPI(
    title="XAI Thesis API",
    version="0.1.0",
    description=(
        "Thin FastAPI layer around the same ResNet50, explainers and preprocessing core "
        "used by the Streamlit thesis app."
    ),
)


@lru_cache(maxsize=1)
def get_runtime() -> tuple[torch.nn.Module, list[str], object, torch.nn.Module]:
    model, weights = load_model()
    transform = weights.transforms()
    class_names = get_imagenet_class_names(weights)
    target_layer = get_last_conv_layer(model)
    return model, class_names, transform, target_layer


def encode_png_base64(image: np.ndarray) -> str:
    rgb = np.clip(np.asarray(image), 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def prediction_to_payload(result: PredictionResult) -> dict[str, object]:
    return {
        "predicted_index": result.predicted_index,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "confidence_pct": round(result.confidence * 100.0, 4),
        "top5": [
            {
                "rank": rank,
                "class_index": item.class_index,
                "class_name": item.class_name,
                "probability": item.probability,
                "probability_pct": round(item.probability * 100.0, 4),
            }
            for rank, item in enumerate(result.topk, start=1)
        ],
    }


async def read_uploaded_image(file: UploadFile) -> tuple[bytes, Image.Image]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image bytes were uploaded.")
    try:
        return image_bytes, load_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def generate_explanation(
    method: MethodName,
    image: Image.Image,
    input_tensor: torch.Tensor,
    target_class: int,
    score_type: ScoreType,
    *,
    ig_steps: int,
    lime_samples: int,
) -> np.ndarray:
    model, _, transform, target_layer = get_runtime()
    if method == "Grad-CAM":
        gradcam = GradCAM(model, target_layer)
        try:
            return gradcam.generate(input_tensor, target_class=target_class, score_type=score_type)
        finally:
            gradcam.close()
    if method == "Integrated Gradients":
        return generate_integrated_gradients(
            model,
            input_tensor,
            image,
            transform,
            target_class=target_class,
            score_type=score_type,
            n_steps=int(ig_steps),
        )
    if method == "Occlusion":
        return generate_occlusion(
            model,
            input_tensor,
            image,
            transform,
            target_class=target_class,
            score_type=score_type,
        )
    if method == "LIME":
        return generate_lime(
            model,
            input_tensor,
            image,
            transform,
            target_class=target_class,
            score_type=score_type,
            n_samples=int(lime_samples),
        )
    raise HTTPException(status_code=400, detail=f"Unsupported explanation method: {method}")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/runtime")
def runtime() -> dict[str, object]:
    model, class_names, _, _ = get_runtime()
    device = str(next(model.parameters()).device)
    return {
        "model": "torchvision ResNet50",
        "weights": "ResNet50_Weights.DEFAULT",
        "device": device,
        "classes": len(class_names),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> dict[str, object]:
    _, image = await read_uploaded_image(file)
    start = time.perf_counter()
    model, class_names, transform, _ = get_runtime()
    input_tensor = preprocess_pil_image(image, transform)
    result = predict(model, input_tensor, class_names, top_k=5)
    payload = prediction_to_payload(result)
    payload["runtime_s"] = round(time.perf_counter() - start, 4)
    return payload


@app.post("/explain")
async def explain_image(
    file: UploadFile = File(...),
    method: MethodName = Query(default="Grad-CAM"),
    score_type: ScoreType = Query(default="logit"),
    target_class: int | None = Query(default=None, ge=0, le=999),
    overlay_alpha: float = Query(default=0.45, ge=0.1, le=0.9),
    ig_steps: int = Query(default=50, ge=10, le=300),
    lime_samples: int = Query(default=300, ge=50, le=2000),
) -> dict[str, object]:
    _, image = await read_uploaded_image(file)
    start = time.perf_counter()

    model, class_names, transform, _ = get_runtime()
    input_tensor = preprocess_pil_image(image, transform)
    prediction = predict(model, input_tensor, class_names, top_k=5)
    resolved_target = int(target_class) if target_class is not None else int(prediction.predicted_index)

    cam = generate_explanation(
        method,
        image,
        input_tensor,
        resolved_target,
        score_type,
        ig_steps=ig_steps,
        lime_samples=lime_samples,
    )

    spatial_image = preprocess_spatial_pil_image(image, transform)
    original_rgb = np.asarray(spatial_image, dtype=np.uint8)
    heatmap_rgb = apply_colormap_to_cam(cam, colormap=METHOD_HEATMAP_COLORMAPS.get(method, cv2.COLORMAP_JET))
    overlay_rgb = overlay_cam_on_image(original_rgb, heatmap_rgb, alpha=float(overlay_alpha))

    return {
        "method": method,
        "score_type": score_type,
        "target_class": resolved_target,
        "target_class_name": class_names[resolved_target] if resolved_target < len(class_names) else f"class_{resolved_target}",
        "prediction": prediction_to_payload(prediction),
        "images": {
            "original_png_base64": encode_png_base64(original_rgb),
            "heatmap_png_base64": encode_png_base64(heatmap_rgb),
            "overlay_png_base64": encode_png_base64(overlay_rgb),
        },
        "runtime_s": round(time.perf_counter() - start, 4),
    }
