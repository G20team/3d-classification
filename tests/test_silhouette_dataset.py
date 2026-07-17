from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from pokemon_3d_cls.experiments.dataset import SilhouettePoseDataset, collate_silhouette_samples
from pokemon_3d_cls.io import write_json, write_jsonl


def test_silhouette_pose_dataset_loads_cached_views(tmp_path: Path) -> None:
    manifest_path, splits_path = _write_fixture(tmp_path)
    render_cache_root = tmp_path / "render_cache" / "silhouette_fixed_ring4"
    split_dir = render_cache_root / "train"
    split_dir.mkdir(parents=True)
    for view_index in range(4):
        array = np.full((8, 8), fill_value=view_index * 10, dtype=np.uint8)
        Image.fromarray(array).save(split_dir / f"000000_v{view_index}.png")

    dataset = SilhouettePoseDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="train",
        render_cache_root=render_cache_root,
        num_views=4,
    )

    assert len(dataset) == 1
    sample = dataset[0]
    images = sample["images"]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4, 1, 8, 8)
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_collate_silhouette_samples_stacks_batch() -> None:
    rows = [
        {
            "images": torch.zeros(4, 1, 8, 8),
            "label": 0,
            "pokemon_id": 1,
            "pokemon_name": "bulbasaur",
            "yaw_offset": 0.0,
            "elevation_offset": 0.0,
        }
    ]

    batch = collate_silhouette_samples(rows)

    images = batch["images"]
    labels = cast("torch.Tensor", batch["labels"])
    yaw_offsets = cast("torch.Tensor", batch["yaw_offsets"])
    elevation_offsets = cast("torch.Tensor", batch["elevation_offsets"])
    assert isinstance(images, torch.Tensor)
    assert images.shape == (1, 4, 1, 8, 8)
    assert labels.shape == (1,)
    assert yaw_offsets.shape == (1,)
    assert elevation_offsets.shape == (1,)


def test_silhouette_dataset_resolves_sample_level_split_and_class_limit(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest_path,
        [
            {"pokemon_id": 1, "pokemon_name": "bulbasaur"},
            {"pokemon_id": 2, "pokemon_name": "ivysaur"},
        ],
    )
    splits_path = tmp_path / "pose_splits.json"
    write_json(
        splits_path,
        {
            "schema_version": 2,
            "splits": {
                "train": [
                    {"pokemon_id": 2, "yaw_offset": 20.0, "elevation_offset": 10.0},
                    {"pokemon_id": 1, "yaw_offset": -20.0, "elevation_offset": -10.0},
                ]
            },
        },
    )

    dataset = SilhouettePoseDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="train",
        render_cache_root=tmp_path / "render_cache",
        num_views=4,
        class_limit=1,
    )

    assert len(dataset) == 1
    assert dataset.samples[0].pokemon_id == 1
    assert dataset.samples[0].yaw_offset == -20.0


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(manifest_path, [{"pokemon_id": 1, "pokemon_name": "bulbasaur"}])

    splits_path = tmp_path / "pose_splits.json"
    splits_path.write_text(
        json.dumps({"train": [{"yaw_offset": 0.0, "elevation_offset": 0.0}]}),
        encoding="utf-8",
    )
    return manifest_path, splits_path
