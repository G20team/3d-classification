"""PyTorch3D renderを使うSingle/Fixed Ring-4/MVTN実験の学習。"""

from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from pokemon_3d_cls.config import MeshExperimentConfig
from pokemon_3d_cls.environment import collect_environment_report
from pokemon_3d_cls.experiments.dataset import MeshPoseDataset, collate_mesh_samples
from pokemon_3d_cls.experiments.metrics import compute_classification_metrics, save_confusion_matrix_png
from pokemon_3d_cls.io import write_json, write_yaml
from pokemon_3d_cls.models import (
    CircularViewPredictor,
    MVCNNClassifier,
    camera_statistics,
    detect_view_collapse,
)
from pokemon_3d_cls.models.camera import fixed_camera_angles
from pokemon_3d_cls.models.mvtn import pack_vertices_for_mvtn
from pokemon_3d_cls.paths import ensure_directory, resolve_project_path
from pokemon_3d_cls.rendering.pytorch3d_renderer import PyTorch3DRenderer, RendererSettings
from pokemon_3d_cls.training import resolve_device, set_seed


def train_mesh_experiment(config: MeshExperimentConfig, project_root: Path) -> Path:
    """設定に従ってSingle/Fixed Ring-4/MVTN実験を学習する。"""

    set_seed(config.experiment.seed)
    device = resolve_device(config.training.device)
    run_dir = _prepare_run_dir(config, project_root)
    write_yaml(run_dir / "config.yaml", config.to_dict())
    environment_report = collect_environment_report(run_dir / "environment_report.json")
    write_json(run_dir / "metadata.json", _metadata(config, environment_report))

    train_dataset = _dataset(config, project_root, config.data.train_split)
    validation_dataset = _dataset(config, project_root, config.data.validation_split)
    train_loader = _loader(train_dataset, config, shuffle=True)
    validation_loader = _loader(validation_dataset, config, shuffle=False)

    model = MVCNNClassifier(
        num_classes=len(train_dataset.class_names),
        backbone=config.model.backbone,
        input_channels=3,
        feature_dim=config.model.feature_dim,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)
    mvtn = _build_mvtn(config, device)
    parameters = list(model.parameters()) + (list(mvtn.parameters()) if mvtn is not None else [])
    optimizer = AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    renderer = PyTorch3DRenderer(
        RendererSettings(
            image_size=config.rendering.image_size,
            camera_distance=config.rendering.camera_distance,
            background_color=config.rendering.background_color,
            mesh_color=config.rendering.mesh_color,
        ),
        device=device,
    )
    writer = SummaryWriter(log_dir=str(run_dir / "logs"))
    checkpoint_dir = ensure_directory(run_dir / "checkpoints")
    camera_history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    print(f"training started: {run_dir}")
    print(
        f"device={device}, train={len(train_dataset)}, "
        f"validation={len(validation_dataset)}, epochs={config.training.epochs}"
    )

    try:
        epoch_progress = tqdm(range(1, config.training.epochs + 1), desc="epochs", unit="epoch")
        for epoch in epoch_progress:
            train_loss, train_camera_stats = _run_epoch(
                model=model,
                mvtn=mvtn,
                renderer=renderer,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                config=config,
                device=device,
                progress_desc=f"train {epoch}/{config.training.epochs}",
            )
            validation = evaluate_loader(
                model=model,
                mvtn=mvtn,
                renderer=renderer,
                loader=validation_loader,
                config=config,
                device=device,
                class_names=validation_dataset.class_names,
                progress_desc=f"validation {epoch}/{config.training.epochs}",
            )
            writer.add_scalar("loss/train", train_loss, epoch)
            validation_macro_f1 = _metric_float(validation, "macro_f1")
            writer.add_scalar("macro_f1/validation", validation_macro_f1, epoch)
            if train_camera_stats is not None:
                camera_history.append({"epoch": epoch, **train_camera_stats})
            if validation_macro_f1 >= best_macro_f1:
                best_macro_f1 = validation_macro_f1
                _save_checkpoint(checkpoint_dir / "best.ckpt", model, mvtn, config, epoch, best_macro_f1)
            epoch_progress.set_postfix(loss=f"{train_loss:.4f}", macro_f1=f"{validation_macro_f1:.4f}")
            tqdm.write(f"epoch {epoch}: train_loss={train_loss:.4f}, validation_macro_f1={validation_macro_f1:.4f}")
    finally:
        writer.close()

    write_json(run_dir / "camera_positions.json", {"epochs": camera_history})
    final_metrics = evaluate_loader(
        model=model,
        mvtn=mvtn,
        renderer=renderer,
        loader=validation_loader,
        config=config,
        device=device,
        class_names=validation_dataset.class_names,
        progress_desc="final validation",
    )
    write_json(run_dir / "metrics.json", final_metrics)
    _write_per_class_csv(run_dir / "per_class_metrics.csv", cast("dict[str, object]", final_metrics["per_class"]))
    save_confusion_matrix_png(
        cast("list[list[int]]", final_metrics["confusion_matrix"]),
        validation_dataset.class_names,
        run_dir / "confusion_matrix.png",
    )
    _save_camera_visualization(run_dir / "learned_camera_visualization.png", camera_history)
    ensure_directory(run_dir / "rendered_examples")
    return run_dir


def evaluate_loader(
    *,
    model: MVCNNClassifier,
    mvtn: CircularViewPredictor | None,
    renderer: PyTorch3DRenderer,
    loader: DataLoader,
    config: MeshExperimentConfig,
    device: torch.device,
    class_names: list[str],
    progress_desc: str | None = None,
) -> dict[str, object]:
    """DataLoaderを評価してmetrics dictを返す。"""

    model.eval()
    if mvtn is not None:
        mvtn.eval()
    labels_all: list[int] = []
    probabilities_all: list[np.ndarray] = []
    batches = tqdm(loader, desc=progress_desc, unit="batch", leave=False) if progress_desc else loader
    with torch.no_grad():
        for batch in batches:
            labels = cast("torch.Tensor", batch["labels"]).to(device)
            azimuths, elevations, _stats = _camera_angles(batch, config, device, mvtn)
            images = renderer.render_batch_views(
                cast("list[torch.Tensor]", batch["vertices"]),
                cast("list[torch.Tensor]", batch["faces"]),
                azimuths,
                elevations,
            )
            logits = model(images)
            probabilities_all.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels_all.extend(int(label) for label in labels.cpu().tolist())
    probabilities = np.concatenate(probabilities_all, axis=0) if probabilities_all else np.zeros((0, len(class_names)))
    return compute_classification_metrics(labels=labels_all, probabilities=probabilities, class_names=class_names)


def _run_epoch(
    *,
    model: MVCNNClassifier,
    mvtn: CircularViewPredictor | None,
    renderer: PyTorch3DRenderer,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    config: MeshExperimentConfig,
    device: torch.device,
    progress_desc: str | None = None,
) -> tuple[float, dict[str, object] | None]:
    model.train()
    if mvtn is not None:
        mvtn.train()
    running_loss = 0.0
    seen = 0
    last_camera_stats: dict[str, object] | None = None
    batches = tqdm(loader, desc=progress_desc, unit="batch", leave=False) if progress_desc else loader
    for batch in batches:
        labels = cast("torch.Tensor", batch["labels"]).to(device)
        optimizer.zero_grad()
        azimuths, elevations, camera_stats = _camera_angles(batch, config, device, mvtn)
        images = renderer.render_batch_views(
            cast("list[torch.Tensor]", batch["vertices"]),
            cast("list[torch.Tensor]", batch["faces"]),
            azimuths,
            elevations,
        )
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * int(labels.size(0))
        seen += int(labels.size(0))
        last_camera_stats = camera_stats
    return running_loss / max(seen, 1), last_camera_stats


def _camera_angles(
    batch: dict[str, object],
    config: MeshExperimentConfig,
    device: torch.device,
    mvtn: CircularViewPredictor | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object] | None]:
    base_azimuths, base_elevations = fixed_camera_angles(config.model.experiment_kind, device=device)
    yaw_offsets = cast("torch.Tensor", batch["yaw_offsets"]).to(device).unsqueeze(1)
    elevation_offsets = cast("torch.Tensor", batch["elevation_offsets"]).to(device).unsqueeze(1)
    base_azimuths = base_azimuths.unsqueeze(0) + yaw_offsets
    base_elevations = base_elevations.unsqueeze(0) + elevation_offsets
    if mvtn is None:
        return base_azimuths, base_elevations, None
    vertices = pack_vertices_for_mvtn(
        cast("list[torch.Tensor]", batch["vertices"]),
        point_samples=config.model.mvtn.point_samples,
        device=device,
    )
    azimuths, elevations, offsets = mvtn(vertices, base_azimuths, base_elevations)
    stats: dict[str, object] = {
        **camera_statistics(azimuths, elevations),
        "view_collapse": detect_view_collapse(
            azimuths,
            elevations,
            threshold_deg=config.model.mvtn.collapse_threshold_deg,
        ),
        "offset_abs_mean": float(offsets.detach().abs().mean().item()),
    }
    return azimuths, elevations, stats


def _build_mvtn(config: MeshExperimentConfig, device: torch.device) -> CircularViewPredictor | None:
    if config.model.experiment_kind != "mvtn_circular4":
        return None
    return CircularViewPredictor(
        num_views=config.model.mvtn.num_views,
        point_samples=config.model.mvtn.point_samples,
        hidden_dim=config.model.mvtn.hidden_dim,
        max_azimuth_offset_deg=config.model.mvtn.max_azimuth_offset_deg,
        max_elevation_offset_deg=config.model.mvtn.max_elevation_offset_deg,
    ).to(device)


def _dataset(config: MeshExperimentConfig, project_root: Path, split: str) -> MeshPoseDataset:
    return MeshPoseDataset(
        manifest_path=resolve_project_path(config.data.manifest_path, project_root),
        mesh_cache_root=resolve_project_path(config.data.mesh_cache_root, project_root),
        splits_path=resolve_project_path(config.data.splits_path, project_root),
        split=split,
        class_limit=config.data.class_limit,
    )


def _loader(dataset: MeshPoseDataset, config: MeshExperimentConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.experiment.seed)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
        collate_fn=collate_mesh_samples,
        generator=generator if shuffle else None,
    )


def _prepare_run_dir(config: MeshExperimentConfig, project_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = config.experiment.run_id or f"{timestamp}_seed{config.experiment.seed}"
    return ensure_directory(
        resolve_project_path(config.output.runs_root, project_root) / config.experiment.condition_id / run_id
    )


def _save_checkpoint(
    path: Path,
    model: MVCNNClassifier,
    mvtn: CircularViewPredictor | None,
    config: MeshExperimentConfig,
    epoch: int,
    macro_f1: float,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mvtn_state_dict": mvtn.state_dict() if mvtn is not None else None,
            "config": config.to_dict(),
            "epoch": epoch,
            "macro_f1": macro_f1,
        },
        path,
    )


def _write_per_class_csv(path: Path, per_class: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_name", "precision", "recall", "f1_score", "support"])
        for class_name, values in per_class.items():
            if not isinstance(values, dict):
                continue
            writer.writerow(
                [
                    class_name,
                    values.get("precision", 0.0),
                    values.get("recall", 0.0),
                    values.get("f1-score", 0.0),
                    values.get("support", 0),
                ]
            )


def _save_camera_visualization(path: Path, camera_history: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 4))
    if camera_history:
        epochs = [_object_to_int(row["epoch"]) for row in camera_history]
        distances = [_object_to_float(row["pairwise_distance_mean"]) for row in camera_history]
        axis.plot(epochs, distances, marker="o")
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean pairwise angular distance")
    axis.set_title("Learned camera distance")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _metadata(config: MeshExperimentConfig, environment_report: dict[str, object]) -> dict[str, object]:
    return {
        "condition_id": config.experiment.condition_id,
        "seed": config.experiment.seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": environment_report,
    }


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def _metric_float(metrics: dict[str, object], key: str) -> float:
    return _object_to_float(metrics[key])


def _object_to_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = "floatへ変換できない値です。"
        raise ValueError(msg)
    return float(value)


def _object_to_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = "intへ変換できない値です。"
        raise ValueError(msg)
    return int(value)
