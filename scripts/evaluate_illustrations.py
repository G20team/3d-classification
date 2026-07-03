"""学習済みシルエット分類器を公式イラスト由来シルエットで評価するCLI。"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import _bootstrap  # noqa: F401
import numpy as np
import torch
from PIL import Image

from pokemon_3d_cls.experiments.metrics import compute_classification_metrics
from pokemon_3d_cls.io import read_jsonl
from pokemon_3d_cls.models import MVCNN
from pokemon_3d_cls.paths import find_project_root, resolve_project_path
from pokemon_3d_cls.training import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="公式イラスト由来シルエットで学習済みモデルを評価します。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="data/manifests/selected_regular.jsonl")
    parser.add_argument("--illustration-silhouette-dir", default="data/illustrations_silhouette")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)
    manifest_path = resolve_project_path(args.manifest, project_root)
    illustration_dir = resolve_project_path(args.illustration_silhouette_dir, project_root)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = _mapping(checkpoint.get("config"), "checkpoint.config")
    model_config = _mapping(config_dict.get("model"), "checkpoint.config.model")
    data_config = _mapping(config_dict.get("data"), "checkpoint.config.data")
    rendering_config = _mapping(config_dict.get("rendering", {}), "checkpoint.config.rendering")
    experiment_config = _mapping(config_dict.get("experiment"), "checkpoint.config.experiment")

    num_views = _required_int(model_config, "num_views")
    image_size = _optional_int(rendering_config, "image_size", 224)
    rows = _class_rows(manifest_path, _optional_int_or_none(data_config, "class_limit"))
    class_names = [_required_str(row, "pokemon_name") for row in rows]

    device = resolve_device(args.device)
    model = MVCNN(
        num_classes=len(class_names),
        backbone=_required_str(model_config, "backbone"),
        input_channels=1,
        feature_dim=_required_int(model_config, "feature_dim"),
        pretrained=False,
        dropout=_required_float(model_config, "dropout"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels: list[int] = []
    probabilities: list[np.ndarray] = []
    missing = 0
    with torch.inference_mode():
        for label_index, row in enumerate(rows):
            pokemon_id = _required_int(row, "pokemon_id")
            pokemon_name = _required_str(row, "pokemon_name")
            image_path = illustration_dir / f"{pokemon_id:04d}_{pokemon_name}.png"
            if not image_path.is_file():
                missing += 1
                continue
            with Image.open(image_path) as image:
                array = np.array(image.convert("L").resize((image_size, image_size)))
            tensor = torch.from_numpy(array).to(dtype=torch.float32).unsqueeze(0) / 255.0
            views = tensor.unsqueeze(0).repeat(1, num_views, 1, 1, 1).to(device)
            logits = model(views)
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.append(label_index)

    probs = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, len(class_names)))
    metrics = compute_classification_metrics(labels=labels, probabilities=probs, class_names=class_names)
    condition_id = _required_str(experiment_config, "condition_id")
    print(f"checkpoint condition_id={condition_id}, evaluated={len(labels)}, missing={missing}")
    print(
        f"top1_accuracy={metrics['top1_accuracy']:.4f}, "
        f"top5_accuracy={metrics['top5_accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def _class_rows(manifest_path: Path, class_limit: int | None) -> list[dict[str, object]]:
    rows = read_jsonl(manifest_path)
    rows = [row for row in rows if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None]
    rows.sort(key=lambda row: _required_int(row, "pokemon_id"))
    if class_limit is not None:
        rows = rows[:class_limit]
    return rows


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} はmappingである必要があります。"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} は整数に変換できる値である必要があります。"
        raise ValueError(msg)
    return int(value)


def _optional_int(row: Mapping[str, object], key: str, default: int) -> int:
    value = row.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} は整数に変換できる値である必要があります。"
        raise ValueError(msg)
    return int(value)


def _optional_int_or_none(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} は整数またはnullである必要があります。"
        raise ValueError(msg)
    return int(value)


def _required_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"{key} は数値に変換できる値である必要があります。"
        raise ValueError(msg)
    return float(value)


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} は空でない文字列である必要があります。"
        raise ValueError(msg)
    return value


if __name__ == "__main__":
    main()
