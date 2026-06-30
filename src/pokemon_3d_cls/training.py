"""MVCNN学習パイプライン。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from pokemon_3d_cls.config import TrainRunConfig
from pokemon_3d_cls.data import MultiViewDataset, make_holdout_indices
from pokemon_3d_cls.evaluation import evaluate_model
from pokemon_3d_cls.io import write_json, write_yaml
from pokemon_3d_cls.models import MVCNN, build_model
from pokemon_3d_cls.paths import ensure_directory, make_run_id, resolve_project_path


def set_seed(seed: int) -> None:
    """再現性のために主要な乱数seedを設定する。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """設定文字列からtorch deviceを決定する。"""

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def prepare_run(config: TrainRunConfig, project_root: Path) -> tuple[str, Path]:
    """run_idとrun_dirを確定する。"""

    run_id = config.experiment.run_id or make_run_id(config.experiment.condition_id, config.experiment.seed)
    runs_root = resolve_project_path(config.output.runs_root, project_root)
    run_dir = ensure_directory(runs_root / config.experiment.condition_id / run_id)
    return run_id, run_dir


def train_config(config: TrainRunConfig, project_root: Path) -> Path:
    """設定に従ってMVCNNを学習し、成果物をrun_dirへ保存する。"""

    set_seed(config.experiment.seed)
    run_id, run_dir = prepare_run(config, project_root)
    write_yaml(run_dir / "config.yaml", config.to_dict())

    dataset_root = resolve_project_path(config.data.dataset_root, project_root)
    holdout_indices = make_holdout_indices(config.data.num_views, config.data.holdout_stride)
    train_dataset = MultiViewDataset(
        dataset_root,
        split="train",
        holdout_indices=holdout_indices,
        image_size=config.data.image_size,
        expected_num_views=config.data.num_views,
    )
    test_dataset = MultiViewDataset(
        dataset_root,
        split="test",
        holdout_indices=holdout_indices,
        image_size=config.data.image_size,
        expected_num_views=config.data.num_views,
    )

    label_map = train_dataset.get_label_map()
    write_json(run_dir / "label_map.json", label_map)

    device = resolve_device(config.training.device)
    generator = torch.Generator().manual_seed(config.experiment.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        num_classes=len(label_map),
        backbone=config.model.backbone,
        input_channels=config.model.input_channels,
        feature_dim=config.model.feature_dim,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    checkpoint_dir = ensure_directory(run_dir / "checkpoints")
    best_model_path = checkpoint_dir / "best_model.pt"

    epoch_rows: list[dict[str, float]] = []
    best_accuracy = -1.0
    try:
        for epoch in range(1, config.training.epochs + 1):
            train_loss = _train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                dataset_size=len(train_dataset),
            )
            evaluation = evaluate_model(model, test_loader, device=device, num_classes=model.num_classes)
            row = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "holdout_accuracy": evaluation.accuracy,
            }
            epoch_rows.append(row)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("accuracy/holdout", evaluation.accuracy, epoch)

            if evaluation.accuracy >= best_accuracy:
                best_accuracy = evaluation.accuracy
                torch.save(model.state_dict(), best_model_path)
    finally:
        writer.close()

    state_dict = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    final_evaluation = evaluate_model(model, test_loader, device=device, num_classes=model.num_classes)
    torch.save(final_evaluation.confusion_matrix.cpu(), run_dir / "confusion_matrix.pt")

    write_json(
        run_dir / "metrics.json",
        {
            "condition_id": config.experiment.condition_id,
            "condition_name": config.experiment.condition_name,
            "run_id": run_id,
            "seed": config.experiment.seed,
            "dataset_root": str(dataset_root),
            "num_classes": len(label_map),
            "holdout_indices": holdout_indices,
            "best_accuracy": best_accuracy,
            "final_accuracy": final_evaluation.accuracy,
            "epochs": epoch_rows,
            "checkpoint": str(best_model_path),
            "confusion_matrix": final_evaluation.confusion_matrix.cpu().tolist(),
        },
    )
    return run_dir


def _train_one_epoch(
    *,
    model: MVCNN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    dataset_size: int,
) -> float:
    model.train()
    running_loss = 0.0
    for views, labels in loader:
        views = views.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(views)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * int(labels.size(0))
    return running_loss / dataset_size
