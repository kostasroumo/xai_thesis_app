from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TopPrediction:
    class_index: int
    class_name: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    predicted_index: int
    predicted_class: str
    confidence: float
    topk: list[TopPrediction]


def predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_names: Sequence[str],
    top_k: int = 5,
) -> PredictionResult:
    if input_tensor.ndim != 4:
        raise ValueError("input_tensor must have shape [B, C, H, W].")
    if input_tensor.size(0) != 1:
        raise ValueError("This predictor expects a batch size of 1.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probabilities = F.softmax(logits, dim=1)

    k = min(top_k, probabilities.size(1))
    top_probs, top_indices = torch.topk(probabilities, k=k, dim=1)

    top_predictions: list[TopPrediction] = []
    for prob, idx in zip(top_probs[0].detach().cpu().tolist(), top_indices[0].detach().cpu().tolist()):
        class_name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        top_predictions.append(
            TopPrediction(
                class_index=int(idx),
                class_name=class_name,
                probability=float(prob),
            )
        )

    best = top_predictions[0]
    return PredictionResult(
        predicted_index=best.class_index,
        predicted_class=best.class_name,
        confidence=best.probability,
        topk=top_predictions,
    )
