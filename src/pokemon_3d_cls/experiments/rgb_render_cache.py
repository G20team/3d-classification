"""PyTorch3Dの固定視点RGB描画を再利用可能なPNGキャッシュへ保存する。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pokemon_3d_cls.config import MeshExperimentConfig
from pokemon_3d_cls.experiments.dataset import MeshPoseDataset, collate_mesh_samples
from pokemon_3d_cls.io import read_json, read_jsonl, write_json
from pokemon_3d_cls.models.camera import fixed_camera_angles
from pokemon_3d_cls.paths import ensure_directory, resolve_project_path
from pokemon_3d_cls.rendering.pytorch3d_renderer import PyTorch3DRenderer, RendererSettings
from pokemon_3d_cls.training import resolve_device

RGB_CACHE_SCHEMA_VERSION = 1
RGB_CACHE_TYPE = "pytorch3d_rgb8_png_strip"


@dataclass(frozen=True)
class RGBRenderCacheIdentity:
    """描画条件から決まるRGBキャッシュの識別情報。"""

    cache_key: str
    cache_dir: Path
    specification: dict[str, object]


def rgb_render_cache_identity(
    config: MeshExperimentConfig,
    project_root: Path,
) -> RGBRenderCacheIdentity:
    """入力・split・固定カメラ・renderer設定から安定したキャッシュキーを作る。"""

    if config.model.experiment_kind == "mvtn_circular4":
        msg = "MVTNの学習可能なカメラ角度は固定RGBキャッシュへ保存できません。"
        raise ValueError(msg)

    manifest_path = resolve_project_path(config.data.manifest_path, project_root)
    splits_path = resolve_project_path(config.data.splits_path, project_root)
    pokemon_ids = _selected_pokemon_ids(manifest_path, config.data.class_limit)
    azimuths, elevations = fixed_camera_angles(config.model.experiment_kind)
    if len(azimuths) != config.model.num_views:
        msg = (
            f"model.num_views={config.model.num_views} does not match fixed camera views={len(azimuths)} "
            f"for {config.model.experiment_kind}."
        )
        raise ValueError(msg)

    specification: dict[str, object] = {
        "schema_version": RGB_CACHE_SCHEMA_VERSION,
        "cache_type": RGB_CACHE_TYPE,
        "source_manifest": {
            "path": config.data.manifest_path,
            "sha256": _file_sha256(manifest_path),
        },
        "source_splits": {
            "path": config.data.splits_path,
            "sha256": _file_sha256(splits_path),
        },
        "class_limit": config.data.class_limit,
        "pokemon_ids": pokemon_ids,
        "camera": {
            "azimuths": [float(value) for value in azimuths.tolist()],
            "elevations": [float(value) for value in elevations.tolist()],
        },
        "rendering": {
            "image_size": config.rendering.image_size,
            "camera_distance": config.rendering.camera_distance,
            "background_color": list(config.rendering.background_color),
            "mesh_color": list(config.rendering.mesh_color),
            "raster_bin_size": 0,
        },
    }
    serialized = json.dumps(specification, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    view_label = f"{config.model.num_views}view"
    cache_root = resolve_project_path(config.data.render_cache_root, project_root) / "rgb"
    cache_dir = cache_root / f"{view_label}_{cache_key[:16]}"
    return RGBRenderCacheIdentity(cache_key=cache_key, cache_dir=cache_dir, specification=specification)


def build_rgb_render_cache(
    config: MeshExperimentConfig,
    project_root: Path,
    *,
    splits: Sequence[str],
    batch_size: int | None = None,
    force: bool = False,
) -> Path:
    """固定視点RGB画像をsampleごとの横連結PNGとして生成する。"""

    identity = rgb_render_cache_identity(config, project_root)
    cache_dir = ensure_directory(identity.cache_dir)
    manifest_path = cache_dir / "cache_manifest.json"
    cache_manifest = _load_or_initialize_manifest(manifest_path, identity)
    completed_splits = _completed_splits(cache_manifest)
    device = resolve_device(config.training.device)
    renderer: PyTorch3DRenderer | None = None
    loader_batch_size = batch_size or config.training.batch_size
    if loader_batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)

    for split in splits:
        dataset = MeshPoseDataset(
            manifest_path=resolve_project_path(config.data.manifest_path, project_root),
            mesh_cache_root=resolve_project_path(config.data.mesh_cache_root, project_root),
            splits_path=resolve_project_path(config.data.splits_path, project_root),
            split=split,
            class_limit=config.data.class_limit,
        )
        split_dir = ensure_directory(cache_dir / split)
        if not force and _is_completed_split(completed_splits, split, len(dataset), split_dir):
            print(f"RGB render cache already complete: split={split}, samples={len(dataset)}")
            continue
        if renderer is None:
            renderer = PyTorch3DRenderer(_renderer_settings(config), device=device)
        loader = DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_mesh_samples,
        )
        _render_split(config, renderer, loader, split_dir, split, device)
        completed_splits[split] = {
            "sample_count": len(dataset),
            "num_views": config.model.num_views,
            "image_size": config.rendering.image_size,
        }
        cache_manifest["completed_splits"] = completed_splits
        write_json(manifest_path, cache_manifest)

    return cache_dir


def _render_split(
    config: MeshExperimentConfig,
    renderer: PyTorch3DRenderer,
    loader: DataLoader,
    split_dir: Path,
    split: str,
    device: torch.device,
) -> None:
    base_azimuths, base_elevations = fixed_camera_angles(config.model.experiment_kind, device=device)
    sample_index = 0
    batches = tqdm(loader, desc=f"rgb_render_cache[{split}]", unit="batch")
    with torch.inference_mode():
        for batch in batches:
            yaw_offsets = cast("torch.Tensor", batch["yaw_offsets"]).to(device).unsqueeze(1)
            elevation_offsets = cast("torch.Tensor", batch["elevation_offsets"]).to(device).unsqueeze(1)
            azimuths = base_azimuths.unsqueeze(0) + yaw_offsets
            elevations = base_elevations.unsqueeze(0) + elevation_offsets
            images = renderer.render_batch_views(
                cast("list[torch.Tensor]", batch["vertices"]),
                cast("list[torch.Tensor]", batch["faces"]),
                azimuths,
                elevations,
            )
            for batch_index in range(images.shape[0]):
                _save_rgb_view_strip(split_dir / f"{sample_index:06d}.png", images[batch_index])
                sample_index += 1


def _save_rgb_view_strip(path: Path, views: torch.Tensor) -> None:
    """(V,3,H,W)を8-bit RGBの横連結PNGとしてatomicに保存する。"""

    uint8_views = views.detach().clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
    arrays = uint8_views.permute(0, 2, 3, 1).cpu().numpy()
    strip = np.concatenate([arrays[index] for index in range(arrays.shape[0])], axis=1)
    temporary_path = path.with_suffix(".png.tmp")
    Image.fromarray(strip, mode="RGB").save(temporary_path, format="PNG", compress_level=1)
    temporary_path.replace(path)


def _renderer_settings(config: MeshExperimentConfig) -> RendererSettings:
    return RendererSettings(
        image_size=config.rendering.image_size,
        camera_distance=config.rendering.camera_distance,
        background_color=config.rendering.background_color,
        mesh_color=config.rendering.mesh_color,
    )


def _selected_pokemon_ids(manifest_path: Path, class_limit: int | None) -> list[int]:
    rows = read_jsonl(manifest_path)
    pokemon_ids = sorted(
        _required_int(row, "pokemon_id")
        for row in rows
        if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None
    )
    if class_limit is not None:
        pokemon_ids = pokemon_ids[:class_limit]
    if not pokemon_ids:
        msg = f"Manifest has no selected classes: {manifest_path}"
        raise ValueError(msg)
    return pokemon_ids


def _load_or_initialize_manifest(
    manifest_path: Path,
    identity: RGBRenderCacheIdentity,
) -> dict[str, object]:
    if not manifest_path.is_file():
        return {
            "schema_version": RGB_CACHE_SCHEMA_VERSION,
            "cache_type": RGB_CACHE_TYPE,
            "cache_key": identity.cache_key,
            "specification": identity.specification,
            "completed_splits": {},
        }
    manifest = read_json(manifest_path)
    if manifest.get("cache_key") != identity.cache_key or manifest.get("specification") != identity.specification:
        msg = f"RGB render cache manifest does not match its computed identity: {manifest_path}"
        raise ValueError(msg)
    return manifest


def _completed_splits(manifest: Mapping[str, object]) -> dict[str, object]:
    value = manifest.get("completed_splits", {})
    if not isinstance(value, dict):
        msg = "RGB cache completed_splits must be a mapping."
        raise ValueError(msg)
    return cast("dict[str, object]", value)


def _is_completed_split(
    completed_splits: Mapping[str, object],
    split: str,
    sample_count: int,
    split_dir: Path,
) -> bool:
    value = completed_splits.get(split)
    if not isinstance(value, Mapping) or value.get("sample_count") != sample_count:
        return False
    return sum(1 for _path in split_dir.glob("*.png")) == sample_count


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} must be convertible to an integer."
        raise ValueError(msg)
    return int(value)
