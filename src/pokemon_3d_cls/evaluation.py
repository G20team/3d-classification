"""分類モデルの評価処理。"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from tqdm import tqdm

from pokemon_3d_cls.models import MVCNN


@dataclass(frozen=True)
class EvaluationResult:
    """評価結果。"""

    accuracy: float
    confusion_matrix: torch.Tensor

    def to_metrics(self) -> dict[str, object]:
        """JSON保存しやすいdictへ変換する。"""

        return {
            "accuracy": self.accuracy,
            "confusion_matrix": self.confusion_matrix.cpu().tolist(),
        }


def evaluate_model(
    model: MVCNN,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    num_classes: int,
    progress_desc: str | None = None,
) -> EvaluationResult:
    """Top-1精度と混同行列を計算する。"""

    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    batches = tqdm(loader, desc=progress_desc, unit="batch", leave=False) if progress_desc else loader

    with torch.no_grad():
        for views, labels in batches:
            views = views.to(device)
            labels = labels.to(device)
            logits = model(views)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
            for true_label, predicted_label in zip(labels.view(-1), predictions.view(-1), strict=True):
                confusion[int(true_label.item()), int(predicted_label.item())] += 1

    accuracy = correct / total if total > 0 else 0.0
    return EvaluationResult(accuracy=accuracy, confusion_matrix=confusion)


def invert_label_map(label_map: dict[str, int]) -> dict[int, str]:
    """ラベルmapを index -> 個体ID へ反転する。"""

    return {index: individual_id for individual_id, index in label_map.items()}


def topk_from_logits(
    logits: torch.Tensor,
    *,
    label_map: dict[str, int],
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """logitsからTop-Kラベルと確信度を取り出す。"""

    inverse_map = invert_label_map(label_map)
    probabilities = torch.softmax(logits, dim=1)
    values, indices = probabilities[0].topk(min(top_k, probabilities.shape[1]))
    return [(inverse_map[int(index.item())], float(value.item())) for index, value in zip(indices, values, strict=True)]


def summarize_epoch_metrics(rows: Sequence[dict[str, float]]) -> dict[str, object]:
    """epochごとのmetrics一覧をJSON保存用にまとめる。"""

    return {"epochs": list(rows)}
