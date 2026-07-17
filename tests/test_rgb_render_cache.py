from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pokemon_3d_cls.config import parse_mesh_experiment_config
from pokemon_3d_cls.experiments.dataset import RGBRenderPoseDataset, collate_cached_image_samples
from pokemon_3d_cls.experiments.rgb_render_cache import _save_rgb_view_strip, rgb_render_cache_identity
from pokemon_3d_cls.io import write_json, write_jsonl


def test_rgb_render_dataset_loads_view_strip_and_preserves_order(tmp_path: Path) -> None:
    manifest_path, splits_path = _write_source_fixture(tmp_path)
    cache_dir = tmp_path / "render_cache" / "rgb" / "fixture"
    split_dir = cache_dir / "train"
    split_dir.mkdir(parents=True)
    strip = np.zeros((8, 32, 3), dtype=np.uint8)
    for view_index in range(4):
        strip[:, view_index * 8 : (view_index + 1) * 8] = view_index * 40
    Image.fromarray(strip).save(split_dir / "000000.png")
    write_json(
        cache_dir / "cache_manifest.json",
        {
            "schema_version": 1,
            "cache_type": "pytorch3d_rgb8_png_strip",
            "completed_splits": {"train": {"sample_count": 1, "num_views": 4, "image_size": 8}},
        },
    )

    dataset = RGBRenderPoseDataset(
        manifest_path=manifest_path,
        splits_path=splits_path,
        split="train",
        render_cache_dir=cache_dir,
        num_views=4,
        image_size=8,
    )

    sample = dataset[0]
    images = sample["images"]
    assert isinstance(images, torch.Tensor)
    assert images.shape == (4, 3, 8, 8)
    assert torch.allclose(images[:, 0, 0, 0], torch.tensor([0.0, 40.0, 80.0, 120.0]) / 255.0)
    batch = collate_cached_image_samples([sample])
    assert isinstance(batch["images"], torch.Tensor)
    assert batch["images"].shape == (1, 4, 3, 8, 8)


def test_rgb_render_dataset_rejects_incomplete_split(tmp_path: Path) -> None:
    manifest_path, splits_path = _write_source_fixture(tmp_path)
    cache_dir = tmp_path / "render_cache" / "rgb" / "fixture"
    write_json(
        cache_dir / "cache_manifest.json",
        {
            "schema_version": 1,
            "cache_type": "pytorch3d_rgb8_png_strip",
            "completed_splits": {"train": {"sample_count": 1, "num_views": 1, "image_size": 8}},
        },
    )

    with pytest.raises(FileNotFoundError, match="incomplete"):
        RGBRenderPoseDataset(
            manifest_path=manifest_path,
            splits_path=splits_path,
            split="train",
            render_cache_dir=cache_dir,
            num_views=1,
            image_size=8,
        )


def test_rgb_cache_identity_is_shared_by_fixed_ring_and_transformer(tmp_path: Path) -> None:
    manifest_path, splits_path = _write_source_fixture(tmp_path)
    common_data = {
        "manifest_path": str(manifest_path),
        "splits_path": str(splits_path),
        "render_cache_root": str(tmp_path / "render_cache"),
    }
    fixed = parse_mesh_experiment_config(
        {"data": common_data, "model": {"experiment_kind": "fixed_ring4"}, "rendering": {"image_size": 8}}
    )
    transformer = parse_mesh_experiment_config(
        {"data": common_data, "model": {"experiment_kind": "view_transformer4"}, "rendering": {"image_size": 8}}
    )
    changed_size = parse_mesh_experiment_config(
        {"data": common_data, "model": {"experiment_kind": "fixed_ring4"}, "rendering": {"image_size": 16}}
    )

    fixed_identity = rgb_render_cache_identity(fixed, tmp_path)
    transformer_identity = rgb_render_cache_identity(transformer, tmp_path)
    changed_identity = rgb_render_cache_identity(changed_size, tmp_path)

    assert fixed_identity.cache_key == transformer_identity.cache_key
    assert fixed_identity.cache_dir == transformer_identity.cache_dir
    assert fixed_identity.cache_key != changed_identity.cache_key


def test_rgb_view_strip_round_trip_has_at_most_one_over_255_error(tmp_path: Path) -> None:
    views = torch.linspace(0.0, 1.0, steps=4 * 3 * 4 * 4).reshape(4, 3, 4, 4)
    path = tmp_path / "000000.png"
    _save_rgb_view_strip(path, views)
    with Image.open(path) as image:
        array = torch.from_numpy(np.array(image, copy=True)).reshape(4, 4, 4, 3).permute(1, 3, 0, 2)
    restored = array.to(dtype=torch.float32) / 255.0

    assert torch.max(torch.abs(restored - views)) <= 1.0 / 255.0


def _write_source_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest_path = tmp_path / "manifest.jsonl"
    write_jsonl(manifest_path, [{"pokemon_id": 1, "pokemon_name": "bulbasaur"}])
    splits_path = tmp_path / "pose_splits.json"
    splits_path.write_text(
        json.dumps({"train": [{"yaw_offset": 0.0, "elevation_offset": 0.0}]}),
        encoding="utf-8",
    )
    return manifest_path, splits_path
