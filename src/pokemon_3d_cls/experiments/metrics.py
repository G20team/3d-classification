"""Classification metrics and visualization."""

from __future__ import annotations

import csv
from collections import defaultdict
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
    """Return Top-1/Top-5/Macro-F1 and per-class metrics."""

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


def compute_pose_metrics(
    *,
    labels: list[int],
    probabilities: np.ndarray,
    pose_offsets: list[tuple[float, float]],
) -> list[dict[str, float | int]]:
    """Return sample count and Top-k accuracy for every yaw/elevation condition."""

    if len(labels) != len(pose_offsets) or probabilities.shape[0] != len(labels):
        msg = "labels, probabilities, and pose_offsets must have the same sample count."
        raise ValueError(msg)
    grouped_indices: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, pose in enumerate(pose_offsets):
        grouped_indices[pose].append(index)

    rows: list[dict[str, float | int]] = []
    for (yaw_offset, elevation_offset), indices in sorted(grouped_indices.items()):
        group_probabilities = probabilities[indices]
        group_labels = np.asarray([labels[index] for index in indices])
        predictions = group_probabilities.argmax(axis=1)
        top_k = min(5, group_probabilities.shape[1])
        top5 = np.mean(
            [
                label in np.argsort(row)[-top_k:]
                for label, row in zip(group_labels, group_probabilities, strict=True)
            ]
        )
        rows.append(
            {
                "yaw_offset": yaw_offset,
                "elevation_offset": elevation_offset,
                "sample_count": len(indices),
                "top1_accuracy": float((predictions == group_labels).mean()),
                "top5_accuracy": float(top5),
            }
        )
    return rows


def save_pose_metrics_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    """Save per-pose evaluation metrics as CSV."""

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["yaw_offset", "elevation_offset", "sample_count", "top1_accuracy", "top5_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix_png(matrix: list[list[int]], class_names: list[str], output_path: Path) -> None:
    """Save a confusion matrix as PNG."""

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
