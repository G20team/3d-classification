"""Datasets built from mesh caches and pose splits."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pokemon_3d_cls.io import read_json, read_jsonl
from pokemon_3d_cls.splits import resolve_pose_samples


@dataclass(frozen=True)
class MeshSample:
    """One mesh plus pose-condition sample."""

    mesh_cache_path: Path
    label: int
    pokemon_id: int
    pokemon_name: str
    yaw_offset: float
    elevation_offset: float


@dataclass(frozen=True)
class SilhouetteSample:
    """One Pokemon plus pose-condition silhouette sample."""

    label: int
    pokemon_id: int
    pokemon_name: str
    yaw_offset: float
    elevation_offset: float


@dataclass(frozen=True)
class RGBRenderSample:
    """One Pokemon plus pose-condition cached RGB render sample."""

    label: int
    pokemon_id: int
    pokemon_name: str
    yaw_offset: float
    elevation_offset: float


class MeshPoseDataset(Dataset):
    """Dataset where Pokemon IDs are shared across splits and only pose conditions are split."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        mesh_cache_root: Path,
        splits_path: Path,
        split: str,
        class_limit: int | None = None,
    ) -> None:
        rows = read_jsonl(manifest_path)
        rows = [row for row in rows if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None]
        rows.sort(key=lambda row: _required_int(row, "pokemon_id"))
        if class_limit is not None:
            rows = rows[:class_limit]
        if not rows:
            msg = f"Manifest has no selected classes: {manifest_path}"
            raise ValueError(msg)

        self.label_map = {_required_int(row, "pokemon_id"): index for index, row in enumerate(rows)}
        self.class_names = [str(row["pokemon_name"]) for row in rows]
        rows_by_id = {_required_int(row, "pokemon_id"): row for row in rows}
        assignments = resolve_pose_samples(splits_path, split=split, pokemon_ids=list(self.label_map))

        self.samples: list[MeshSample] = []
        for assignment in assignments:
            pokemon_id = assignment.pokemon_id
            row = rows_by_id[pokemon_id]
            pokemon_name = _required_str(row, "pokemon_name")
            mesh_cache_path = _mesh_cache_path(row, mesh_cache_root)
            if not mesh_cache_path.is_file():
                msg = f"Mesh cache was not found: {mesh_cache_path}"
                raise FileNotFoundError(msg)
            self.samples.append(
                MeshSample(
                    mesh_cache_path=mesh_cache_path,
                    label=self.label_map[pokemon_id],
                    pokemon_id=pokemon_id,
                    pokemon_name=pokemon_name,
                    yaw_offset=assignment.yaw_offset,
                    elevation_offset=assignment.elevation_offset,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        cached = torch.load(sample.mesh_cache_path, map_location="cpu", weights_only=False)
        return {
            "vertices": cached["vertices"].to(dtype=torch.float32),
            "faces": cached["faces"].to(dtype=torch.int64),
            "label": sample.label,
            "pokemon_id": sample.pokemon_id,
            "pokemon_name": sample.pokemon_name,
            "yaw_offset": sample.yaw_offset,
            "elevation_offset": sample.elevation_offset,
        }


class SilhouettePoseDataset(Dataset):
    """Dataset built from cached filled-silhouette PNG renders."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        splits_path: Path,
        split: str,
        render_cache_root: Path,
        num_views: int,
        class_limit: int | None = None,
    ) -> None:
        self.render_cache_split_dir = render_cache_root / split
        self.num_views = num_views

        rows = read_jsonl(manifest_path)
        rows = [row for row in rows if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None]
        rows.sort(key=lambda row: _required_int(row, "pokemon_id"))
        if class_limit is not None:
            rows = rows[:class_limit]
        if not rows:
            msg = f"Manifest has no selected classes: {manifest_path}"
            raise ValueError(msg)

        self.label_map = {_required_int(row, "pokemon_id"): index for index, row in enumerate(rows)}
        self.class_names = [str(row["pokemon_name"]) for row in rows]
        rows_by_id = {_required_int(row, "pokemon_id"): row for row in rows}
        assignments = resolve_pose_samples(splits_path, split=split, pokemon_ids=list(self.label_map))

        self.samples: list[SilhouetteSample] = []
        for assignment in assignments:
            pokemon_id = assignment.pokemon_id
            row = rows_by_id[pokemon_id]
            pokemon_name = _required_str(row, "pokemon_name")
            self.samples.append(
                SilhouetteSample(
                    label=self.label_map[pokemon_id],
                    pokemon_id=pokemon_id,
                    pokemon_name=pokemon_name,
                    yaw_offset=assignment.yaw_offset,
                    elevation_offset=assignment.elevation_offset,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "images": _load_cached_views(self.render_cache_split_dir, index, self.num_views),
            "label": sample.label,
            "pokemon_id": sample.pokemon_id,
            "pokemon_name": sample.pokemon_name,
            "yaw_offset": sample.yaw_offset,
            "elevation_offset": sample.elevation_offset,
        }


class RGBRenderPoseDataset(Dataset):
    """Dataset built from cached PyTorch3D RGB view-strip PNG renders."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        splits_path: Path,
        split: str,
        render_cache_dir: Path,
        num_views: int,
        image_size: int,
        class_limit: int | None = None,
    ) -> None:
        self.render_cache_split_dir = render_cache_dir / split
        self.num_views = num_views
        self.image_size = image_size

        cache_manifest_path = render_cache_dir / "cache_manifest.json"
        if not cache_manifest_path.is_file():
            msg = (
                f"RGB render cache is not prepared: {cache_manifest_path}. "
                "Run `uv run python scripts/prepare_rgb_render_cache.py --config <config>`."
            )
            raise FileNotFoundError(msg)
        cache_manifest = read_json(cache_manifest_path)
        if cache_manifest.get("schema_version") != 1 or cache_manifest.get("cache_type") != "pytorch3d_rgb8_png_strip":
            msg = f"Unsupported RGB render cache manifest: {cache_manifest_path}"
            raise ValueError(msg)
        completed_splits = _required_mapping(cache_manifest, "completed_splits")
        split_info = _required_mapping(completed_splits, split)
        if _required_int(split_info, "num_views") != num_views:
            msg = f"RGB cache num_views does not match config: {cache_manifest_path}"
            raise ValueError(msg)
        if _required_int(split_info, "image_size") != image_size:
            msg = f"RGB cache image_size does not match config: {cache_manifest_path}"
            raise ValueError(msg)

        rows = read_jsonl(manifest_path)
        rows = [row for row in rows if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None]
        rows.sort(key=lambda row: _required_int(row, "pokemon_id"))
        if class_limit is not None:
            rows = rows[:class_limit]
        if not rows:
            msg = f"Manifest has no selected classes: {manifest_path}"
            raise ValueError(msg)

        self.label_map = {_required_int(row, "pokemon_id"): index for index, row in enumerate(rows)}
        self.class_names = [str(row["pokemon_name"]) for row in rows]
        rows_by_id = {_required_int(row, "pokemon_id"): row for row in rows}
        assignments = resolve_pose_samples(splits_path, split=split, pokemon_ids=list(self.label_map))

        self.samples: list[RGBRenderSample] = []
        for assignment in assignments:
            pokemon_id = assignment.pokemon_id
            row = rows_by_id[pokemon_id]
            self.samples.append(
                RGBRenderSample(
                    label=self.label_map[pokemon_id],
                    pokemon_id=pokemon_id,
                    pokemon_name=_required_str(row, "pokemon_name"),
                    yaw_offset=assignment.yaw_offset,
                    elevation_offset=assignment.elevation_offset,
                )
            )
        expected_count = _required_int(split_info, "sample_count")
        if expected_count != len(self.samples):
            msg = f"RGB cache sample_count={expected_count} does not match resolved assignments={len(self.samples)}."
            raise ValueError(msg)
        cached_file_count = sum(1 for _path in self.render_cache_split_dir.glob("*.png"))
        if cached_file_count != len(self.samples):
            msg = (
                f"RGB cache split is incomplete: split={split}, expected={len(self.samples)}, "
                f"found={cached_file_count}, directory={self.render_cache_split_dir}"
            )
            raise FileNotFoundError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        return {
            "images": _load_rgb_view_strip(
                self.render_cache_split_dir / f"{index:06d}.png",
                num_views=self.num_views,
                image_size=self.image_size,
            ),
            "label": sample.label,
            "pokemon_id": sample.pokemon_id,
            "pokemon_name": sample.pokemon_name,
            "yaw_offset": sample.yaw_offset,
            "elevation_offset": sample.elevation_offset,
        }


def collate_mesh_samples(rows: list[dict[str, object]]) -> dict[str, object]:
    """collate_fn that keeps variable-length meshes as lists."""

    return {
        "vertices": [cast("torch.Tensor", row["vertices"]) for row in rows],
        "faces": [cast("torch.Tensor", row["faces"]) for row in rows],
        "labels": torch.tensor([_required_int(row, "label") for row in rows], dtype=torch.long),
        "pokemon_ids": [_required_int(row, "pokemon_id") for row in rows],
        "pokemon_names": [_required_str(row, "pokemon_name") for row in rows],
        "yaw_offsets": torch.tensor([_required_float(row, "yaw_offset") for row in rows], dtype=torch.float32),
        "elevation_offsets": torch.tensor(
            [_required_float(row, "elevation_offset") for row in rows],
            dtype=torch.float32,
        ),
    }


def collate_silhouette_samples(rows: list[dict[str, object]]) -> dict[str, object]:
    """collate_fn that stacks silhouette images along the batch dimension."""

    return collate_cached_image_samples(rows)


def collate_cached_image_samples(rows: list[dict[str, object]]) -> dict[str, object]:
    """collate_fn that stacks cached multi-view images along the batch dimension."""

    return {
        "images": torch.stack([cast("torch.Tensor", row["images"]) for row in rows], dim=0),
        "labels": torch.tensor([_required_int(row, "label") for row in rows], dtype=torch.long),
        "pokemon_ids": [_required_int(row, "pokemon_id") for row in rows],
        "pokemon_names": [_required_str(row, "pokemon_name") for row in rows],
        "yaw_offsets": torch.tensor([_required_float(row, "yaw_offset") for row in rows], dtype=torch.float32),
        "elevation_offsets": torch.tensor(
            [_required_float(row, "elevation_offset") for row in rows],
            dtype=torch.float32,
        ),
    }


def _load_cached_views(split_dir: Path, index: int, num_views: int) -> torch.Tensor:
    """Read cached PNG renders as float32 [0,1] tensors shaped (V,1,H,W)."""

    views = []
    for view_index in range(num_views):
        image_path = split_dir / f"{index:06d}_v{view_index}.png"
        with Image.open(image_path) as image:
            array = torch.from_numpy(np.array(image.convert("L")))
        views.append(array.unsqueeze(0).to(dtype=torch.float32) / 255.0)
    return torch.stack(views, dim=0)


def _load_rgb_view_strip(path: Path, *, num_views: int, image_size: int) -> torch.Tensor:
    """Read a horizontal RGB view strip as a (V,3,H,W) float32 tensor."""

    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        expected_size = (image_size * num_views, image_size)
        if rgb_image.size != expected_size:
            msg = f"RGB cache image has size={rgb_image.size}, expected={expected_size}: {path}"
            raise ValueError(msg)
        array = np.array(rgb_image, copy=True)
    tensor = torch.from_numpy(array).reshape(image_size, num_views, image_size, 3)
    return tensor.permute(1, 3, 0, 2).contiguous().to(dtype=torch.float32) / 255.0


def _mesh_cache_path(row: dict[str, object], mesh_cache_root: Path) -> Path:
    cache_path = row.get("mesh_cache_path")
    if isinstance(cache_path, str) and cache_path:
        path = Path(cache_path)
        return path if path.is_absolute() else path
    pokemon_id = _required_int(row, "pokemon_id")
    pokemon_name = _required_str(row, "pokemon_name")
    return mesh_cache_root / f"{pokemon_id:04d}_{pokemon_name}.pt"


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} must be convertible to an integer."
        raise ValueError(msg)
    return int(value)


def _required_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"{key} must be convertible to a number."
        raise ValueError(msg)
    return float(value)


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value


def _required_mapping(row: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = row.get(key)
    if not isinstance(value, Mapping):
        msg = f"{key} must be a mapping."
        raise ValueError(msg)
    return value
