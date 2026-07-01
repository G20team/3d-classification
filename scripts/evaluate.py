"""保存checkpointを指定splitで評価するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from pokemon_3d_cls.config import parse_mesh_experiment_config
from pokemon_3d_cls.experiments.dataset import MeshPoseDataset, collate_mesh_samples
from pokemon_3d_cls.experiments.metrics import save_confusion_matrix_png
from pokemon_3d_cls.experiments.training import evaluate_loader
from pokemon_3d_cls.io import write_json
from pokemon_3d_cls.models import CircularViewPredictor, MVCNNClassifier
from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path
from pokemon_3d_cls.rendering.pytorch3d_renderer import PyTorch3DRenderer, RendererSettings
from pokemon_3d_cls.training import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="mesh実験checkpointを評価します。")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = parse_mesh_experiment_config(cast("dict[str, object]", checkpoint["config"]))
    device = resolve_device(config.training.device)

    dataset = MeshPoseDataset(
        manifest_path=resolve_project_path(config.data.manifest_path, project_root),
        mesh_cache_root=resolve_project_path(config.data.mesh_cache_root, project_root),
        splits_path=resolve_project_path(config.data.splits_path, project_root),
        split=args.split,
        class_limit=config.data.class_limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_mesh_samples,
    )
    model = MVCNNClassifier(
        num_classes=len(dataset.class_names),
        backbone=config.model.backbone,
        input_channels=3,
        feature_dim=config.model.feature_dim,
        pretrained=False,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    mvtn = None
    if checkpoint.get("mvtn_state_dict") is not None:
        mvtn = CircularViewPredictor(
            num_views=config.model.mvtn.num_views,
            point_samples=config.model.mvtn.point_samples,
            hidden_dim=config.model.mvtn.hidden_dim,
            max_azimuth_offset_deg=config.model.mvtn.max_azimuth_offset_deg,
            max_elevation_offset_deg=config.model.mvtn.max_elevation_offset_deg,
        ).to(device)
        mvtn.load_state_dict(checkpoint["mvtn_state_dict"])
    renderer = PyTorch3DRenderer(
        RendererSettings(
            image_size=config.rendering.image_size,
            camera_distance=config.rendering.camera_distance,
            background_color=config.rendering.background_color,
            mesh_color=config.rendering.mesh_color,
        ),
        device=device,
    )
    print(f"evaluation started: checkpoint={checkpoint_path}")
    print(f"device={device}, split={args.split}, samples={len(dataset)}, batches={len(loader)}")
    metrics = evaluate_loader(
        model=model,
        mvtn=mvtn,
        renderer=renderer,
        loader=loader,
        config=config,
        device=device,
        class_names=dataset.class_names,
        progress_desc=f"evaluate {args.split}",
    )
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / f"eval_{args.split}"
    output_dir = ensure_directory(
        output_dir if output_dir.is_absolute() else resolve_project_path(output_dir, project_root)
    )
    write_json(output_dir / "metrics.json", metrics)
    save_confusion_matrix_png(
        cast("list[list[int]]", metrics["confusion_matrix"]),
        dataset.class_names,
        output_dir / "confusion_matrix.png",
    )
    print(f"evaluation finished: {output_dir}")


if __name__ == "__main__":
    main()
