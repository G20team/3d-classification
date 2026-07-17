"""CLI for evaluating a saved checkpoint on a selected split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import _bootstrap  # noqa: F401
import torch
from torch.utils.data import DataLoader

from pokemon_3d_cls.config import parse_mesh_experiment_config
from pokemon_3d_cls.experiments.dataset import (
    MeshPoseDataset,
    RGBRenderPoseDataset,
    SilhouettePoseDataset,
    collate_cached_image_samples,
    collate_mesh_samples,
)
from pokemon_3d_cls.experiments.metrics import save_confusion_matrix_png, save_pose_metrics_csv
from pokemon_3d_cls.experiments.rgb_render_cache import rgb_render_cache_identity
from pokemon_3d_cls.experiments.training import evaluate_cached_image_loader, evaluate_loader
from pokemon_3d_cls.io import write_json
from pokemon_3d_cls.models import CircularViewPredictor, build_experiment_classifier
from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path
from pokemon_3d_cls.rendering.pytorch3d_renderer import PyTorch3DRenderer, RendererSettings
from pokemon_3d_cls.training import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a mesh experiment checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="Evaluation batch size. Defaults to 1 to reduce memory use on WSL.",
    )
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    checkpoint_path = resolve_project_path(args.checkpoint, project_root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = parse_mesh_experiment_config(cast("dict[str, object]", checkpoint["config"]))
    device = resolve_device(config.training.device)

    manifest_path = resolve_project_path(config.data.manifest_path, project_root)
    splits_path = resolve_project_path(config.data.splits_path, project_root)
    if config.data.input_source == "mesh":
        dataset = MeshPoseDataset(
            manifest_path=manifest_path,
            mesh_cache_root=resolve_project_path(config.data.mesh_cache_root, project_root),
            splits_path=splits_path,
            split=args.split,
            class_limit=config.data.class_limit,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_mesh_samples,
        )
        input_channels = 3
    elif config.data.input_source == "silhouette_cache":
        dataset = SilhouettePoseDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split=args.split,
            render_cache_root=(
                resolve_project_path(config.data.render_cache_root, project_root) / config.experiment.condition_id
            ),
            num_views=config.model.num_views,
            class_limit=config.data.class_limit,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_cached_image_samples,
            persistent_workers=config.data.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        input_channels = 1
    else:
        identity = rgb_render_cache_identity(config, project_root)
        dataset = RGBRenderPoseDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split=args.split,
            render_cache_dir=identity.cache_dir,
            num_views=config.model.num_views,
            image_size=config.rendering.image_size,
            class_limit=config.data.class_limit,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collate_cached_image_samples,
            persistent_workers=config.data.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        input_channels = 3
    transformer = config.model.transformer
    model = build_experiment_classifier(
        experiment_kind=config.model.experiment_kind,
        num_classes=len(dataset.class_names),
        num_views=config.model.num_views,
        backbone=config.model.backbone,
        input_channels=input_channels,
        feature_dim=config.model.feature_dim,
        pretrained=False,
        dropout=config.model.dropout,
        transformer_num_layers=transformer.num_layers,
        transformer_num_heads=transformer.num_heads,
        transformer_mlp_dim=transformer.mlp_dim,
        transformer_dropout=transformer.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"evaluation started: checkpoint={checkpoint_path}")
    print(f"device={device}, split={args.split}, samples={len(dataset)}, batches={len(loader)}")
    print(f"eval_batch_size={args.batch_size} (training config batch_size={config.training.batch_size})")
    if config.data.input_source == "mesh":
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
        metrics = evaluate_loader(
            model=model,
            mvtn=mvtn,
            renderer=renderer,
            loader=loader,
            config=config,
            device=device,
            class_names=dataset.class_names,
            progress_desc=f"evaluate {args.split}",
            cleanup_interval=1,
        )
    else:
        metrics = evaluate_cached_image_loader(
            model=model,
            loader=loader,
            device=device,
            class_names=dataset.class_names,
            progress_desc=f"evaluate {args.split}",
        )
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent.parent / f"eval_{args.split}"
    output_dir = ensure_directory(
        output_dir if output_dir.is_absolute() else resolve_project_path(output_dir, project_root)
    )
    write_json(output_dir / "metrics.json", metrics)
    save_pose_metrics_csv(cast("list[dict[str, object]]", metrics["per_pose"]), output_dir / "pose_metrics.csv")
    save_confusion_matrix_png(
        cast("list[list[int]]", metrics["confusion_matrix"]),
        dataset.class_names,
        output_dir / "confusion_matrix.png",
    )
    print(f"evaluation finished: {output_dir}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "Specify a positive integer."
        raise argparse.ArgumentTypeError(msg)
    return parsed


if __name__ == "__main__":
    main()
