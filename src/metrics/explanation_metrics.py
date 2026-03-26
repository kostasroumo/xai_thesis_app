from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic

ScoreType = Literal["logit", "prob"]


@dataclass(frozen=True)
class MetricSettings:
    slic_n_segments: int = 50
    slic_compactness: float = 10.0
    slic_sigma: float = 1.0
    faithfulness_steps: int = 10
    faithfulness_blur_radius: float = 4.0
    sensitivity_top_n: int = 10
    sensitivity_n_random: int = 20
    sensitivity_blur_radius: float = 4.0
    robustness_topk_fracs: tuple[float, ...] = (0.1, 0.2)


def _target_score(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    score_type: ScoreType,
) -> float:
    if score_type not in {"logit", "prob"}:
        raise ValueError("score_type must be either 'logit' or 'prob'.")

    with torch.no_grad():
        logits = model(input_tensor)
        if score_type == "logit":
            return float(logits[0, int(target_class)].item())
        probs = F.softmax(logits, dim=1)
        return float(probs[0, int(target_class)].item())


def _pil_to_rgb01(image: Image.Image, height: int, width: int) -> np.ndarray:
    resized = image.resize((width, height))
    return np.asarray(resized).astype(np.float32) / 255.0


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
    ).astype(np.int64)


def _aggregate_superpixel_scores(cam: np.ndarray, seg: np.ndarray) -> np.ndarray:
    if cam.shape != seg.shape:
        raise ValueError("cam and seg must have the same height and width.")

    heat = np.abs(cam.astype(np.float32))
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
    pil = Image.fromarray((np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8))
    return transform(pil).unsqueeze(0).to(device)


def _hoyer_sparsity(values: np.ndarray) -> float:
    v = np.abs(np.asarray(values, dtype=np.float64))
    n = len(v)
    if n == 0:
        return float("nan")

    l1 = float(np.sum(v))
    l2 = float(np.sqrt(np.sum(v**2)) + 1e-12)
    if l1 <= 1e-12:
        return 0.0

    return float((np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0 + 1e-12))


def _spearman_rho(values_a: np.ndarray, values_b: np.ndarray) -> float:
    rank_a = pd.Series(np.asarray(values_a)).rank(method="average").to_numpy(dtype=np.float64)
    rank_b = pd.Series(np.asarray(values_b)).rank(method="average").to_numpy(dtype=np.float64)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denom = np.sqrt(np.sum(rank_a**2)) * np.sqrt(np.sum(rank_b**2)) + 1e-12
    return float(np.sum(rank_a * rank_b) / denom)


def _iou_topk_abs(scores_a: np.ndarray, scores_b: np.ndarray, frac: float) -> float:
    if frac <= 0.0:
        raise ValueError("frac must be > 0.")

    a = np.asarray(scores_a)
    b = np.asarray(scores_b)
    if len(a) != len(b):
        raise ValueError("scores_a and scores_b must have same length.")

    k = max(1, int(np.ceil(len(a) * float(frac))))
    top_a = set(np.argsort(np.abs(a))[-k:].tolist())
    top_b = set(np.argsort(np.abs(b))[-k:].tolist())

    union = len(top_a | top_b)
    if union == 0:
        return 0.0
    return float(len(top_a & top_b) / union)


def compute_explanation_metrics(
    model: nn.Module,
    input_tensor: torch.Tensor,
    image: Image.Image,
    transform: Callable[[Image.Image], torch.Tensor],
    cam: np.ndarray,
    target_class: int,
    score_type: ScoreType,
    settings: MetricSettings,
    random_seed: int = 0,
    noisy_cam: np.ndarray | None = None,
) -> dict[str, float | int | list[float]]:
    if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
        raise ValueError("input_tensor must have shape [1, C, H, W].")
    if cam.ndim != 2:
        raise ValueError("cam must be a 2D heatmap.")
    if settings.faithfulness_steps <= 0:
        raise ValueError("faithfulness_steps must be positive.")
    if settings.sensitivity_n_random <= 0:
        raise ValueError("sensitivity_n_random must be positive.")

    model.eval()
    device = next(model.parameters()).device
    input_batch = input_tensor.to(device)

    height, width = cam.shape
    rgb01 = _pil_to_rgb01(image=image, height=height, width=width)
    seg = _slic_segments(
        image_rgb01=rgb01,
        n_segments=settings.slic_n_segments,
        compactness=settings.slic_compactness,
        sigma=settings.slic_sigma,
    )
    scores = _aggregate_superpixel_scores(cam=cam, seg=seg)
    ranking = np.argsort(np.abs(scores))[::-1].tolist()
    n_segments = len(scores)

    xs = np.linspace(0.0, 1.0, settings.faithfulness_steps + 1, dtype=np.float64)
    baseline_faith = _blur_rgb01(rgb01, radius=settings.faithfulness_blur_radius)

    deletion_scores: list[float] = []
    insertion_scores: list[float] = []
    for frac in xs:
        n_select = int(round(float(frac) * n_segments))
        selected = ranking[:n_select]

        rgb_del = _apply_baseline_to_superpixels(
            image_rgb01=rgb01,
            seg=seg,
            sp_ids=selected,
            baseline_rgb01=baseline_faith,
        )
        x_del = _rgb01_to_input_tensor(rgb_del, transform=transform, device=device)
        deletion_scores.append(
            _target_score(model=model, input_tensor=x_del, target_class=target_class, score_type=score_type)
        )

        rgb_ins = baseline_faith.copy()
        if selected:
            mask = np.isin(seg, np.asarray(selected, dtype=np.int64))
            rgb_ins[mask] = rgb01[mask]
        x_ins = _rgb01_to_input_tensor(rgb_ins, transform=transform, device=device)
        insertion_scores.append(
            _target_score(model=model, input_tensor=x_ins, target_class=target_class, score_type=score_type)
        )

    deletion_auc = float(np.trapz(np.asarray(deletion_scores, dtype=np.float64), xs))
    insertion_auc = float(np.trapz(np.asarray(insertion_scores, dtype=np.float64), xs))
    aopc_delta = float(insertion_auc - deletion_auc)

    n_top = min(int(settings.sensitivity_top_n), n_segments)
    top_ids = ranking[:n_top]

    baseline_sens = _blur_rgb01(rgb01, radius=settings.sensitivity_blur_radius)
    base_score = _target_score(model=model, input_tensor=input_batch, target_class=target_class, score_type=score_type)

    rgb_top = _apply_baseline_to_superpixels(
        image_rgb01=rgb01,
        seg=seg,
        sp_ids=top_ids,
        baseline_rgb01=baseline_sens,
    )
    x_top = _rgb01_to_input_tensor(rgb_top, transform=transform, device=device)
    drop_top = float(
        base_score
        - _target_score(model=model, input_tensor=x_top, target_class=target_class, score_type=score_type)
    )

    rng = np.random.default_rng(int(random_seed))
    all_ids = np.arange(n_segments)
    random_drops: list[float] = []
    for _ in range(int(settings.sensitivity_n_random)):
        random_ids = rng.choice(all_ids, size=n_top, replace=False).tolist()
        rgb_rand = _apply_baseline_to_superpixels(
            image_rgb01=rgb01,
            seg=seg,
            sp_ids=random_ids,
            baseline_rgb01=baseline_sens,
        )
        x_rand = _rgb01_to_input_tensor(rgb_rand, transform=transform, device=device)
        random_drops.append(
            base_score
            - _target_score(model=model, input_tensor=x_rand, target_class=target_class, score_type=score_type)
        )

    drop_rand_mean = float(np.mean(random_drops)) if random_drops else 0.0
    sensitivity = float(drop_top - drop_rand_mean)

    result: dict[str, float | int | list[float]] = {
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
        "aopc_delta": aopc_delta,
        "drop_top": drop_top,
        "drop_rand_mean": drop_rand_mean,
        "sensitivity": sensitivity,
        "hoyer_sparsity": _hoyer_sparsity(scores),
        "n_superpixels": int(n_segments),
        "faithfulness_xs": xs.astype(np.float64).tolist(),
        "deletion_curve": [float(v) for v in deletion_scores],
        "insertion_curve": [float(v) for v in insertion_scores],
    }

    if noisy_cam is not None:
        if noisy_cam.ndim != 2:
            raise ValueError("noisy_cam must be a 2D heatmap.")
        if noisy_cam.shape != cam.shape:
            noisy_cam = cv2.resize(
                noisy_cam.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        noisy_scores = _aggregate_superpixel_scores(cam=noisy_cam, seg=seg)
        result["spearman_rho"] = _spearman_rho(scores, noisy_scores)
        for frac in settings.robustness_topk_fracs:
            key = f"iou_top_{int(round(float(frac) * 100.0))}pct"
            result[key] = _iou_topk_abs(scores, noisy_scores, float(frac))

    return result
