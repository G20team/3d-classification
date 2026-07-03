"""分類metricsと可視化。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compute_classification_metrics(
    *,
    labels: list[int],
    probabilities: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    """Top-1/Top-5/Macro-F1とper-class metricsを返す。"""

    predictions = probabilities.argmax(axis=1)
    top1 = float((predictions == np.asarray(labels)).mean()) if labels else 0.0
    top_k = min(5, probabilities.shape[1])
    top5 = (
        float(np.mean([label in np.argsort(row)[-top_k:] for label, row in zip(labels, probabilities, strict=True)]))
        if labels
        else 0.0
    )
    class_indices = list(range(len(class_names)))
    macro_f1 = (
        float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,  # pyright: ignore[reportArgumentType]
            )
        )
        if labels
        else 0.0
    )
    report = classification_report(
        labels,
        predictions,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,  # pyright: ignore[reportArgumentType]
    )
    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "per_class": report,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=class_indices).tolist(),
    }


def save_confusion_matrix_png(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    """混同行列をPNG保存する。"""

    fig, axis = plt.subplots(figsize=(max(6, len(class_names) * 0.25), max(5, len(class_names) * 0.25)))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    if len(class_names) <= 40:
        axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
        axis.set_yticks(range(len(class_names)), class_names, fontsize=6)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
