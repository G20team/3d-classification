from __future__ import annotations

import warnings

import numpy as np

from pokemon_3d_cls.experiments.metrics import compute_classification_metrics


def test_compute_classification_metrics_handles_unpredicted_classes_without_warning() -> None:
    labels = [0, 1, 2]
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
        ],
        dtype=np.float32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metrics = compute_classification_metrics(
            labels=labels,
            probabilities=probabilities,
            class_names=["a", "b", "c"],
        )

    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    macro_f1 = metrics["macro_f1"]
    assert isinstance(macro_f1, float)
    assert macro_f1 >= 0.0
