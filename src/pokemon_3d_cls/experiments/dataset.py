"""mesh cacheと姿勢splitから作るDataset。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pokemon_3d_cls.io import read_jsonl
from pokemon_3d_cls.splits import load_pose_splits


@dataclass(frozen=True)
class MeshSample:
    """1つのmesh + 姿勢条件サンプル。"""

    mesh_cache_path: Path
    label: int
    pokemon_id: int
    pokemon_name: str
    yaw_offset: float
    elevation_offset: float


@dataclass(frozen=True)
class SilhouetteSample:
    """1つのポケモン + 姿勢条件シルエットサンプル。"""

    label: int
    pokemon_id: int
    pokemon_name: str
    yaw_offset: float
    elevation_offset: float


class MeshPoseDataset(Dataset):
    """ポケモンIDは全split共通、姿勢条件だけをsplitするDataset。"""

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
            msg = f"manifestに採用クラスがありません: {manifest_path}"
            raise ValueError(msg)

        self.label_map = {_required_int(row, "pokemon_id"): index for index, row in enumerate(rows)}
        self.class_names = [str(row["pokemon_name"]) for row in rows]
        pose_splits = load_pose_splits(splits_path)
        if split not in pose_splits:
            msg = f"splitが見つかりません: {split}"
            raise ValueError(msg)
        conditions = pose_splits[split]

        self.samples: list[MeshSample] = []
        for row in rows:
            pokemon_id = _required_int(row, "pokemon_id")
            pokemon_name = _required_str(row, "pokemon_name")
            mesh_cache_path = _mesh_cache_path(row, mesh_cache_root)
            if not mesh_cache_path.is_file():
                msg = f"mesh cacheが見つかりません: {mesh_cache_path}"
                raise FileNotFoundError(msg)
            for condition in conditions:
                self.samples.append(
                    MeshSample(
                        mesh_cache_path=mesh_cache_path,
                        label=self.label_map[pokemon_id],
                        pokemon_id=pokemon_id,
                        pokemon_name=pokemon_name,
                        yaw_offset=_required_float(condition, "yaw_offset"),
                        elevation_offset=_required_float(condition, "elevation_offset"),
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
    """render cache済み黒塗りシルエットPNGから作るDataset。"""

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
            msg = f"manifestに採用クラスがありません: {manifest_path}"
            raise ValueError(msg)

        self.label_map = {_required_int(row, "pokemon_id"): index for index, row in enumerate(rows)}
        self.class_names = [str(row["pokemon_name"]) for row in rows]
        pose_splits = load_pose_splits(splits_path)
        if split not in pose_splits:
            msg = f"splitが見つかりません: {split}"
            raise ValueError(msg)
        conditions = pose_splits[split]

        self.samples: list[SilhouetteSample] = []
        for row in rows:
            pokemon_id = _required_int(row, "pokemon_id")
            pokemon_name = _required_str(row, "pokemon_name")
            for condition in conditions:
                self.samples.append(
                    SilhouetteSample(
                        label=self.label_map[pokemon_id],
                        pokemon_id=pokemon_id,
                        pokemon_name=pokemon_name,
                        yaw_offset=_required_float(condition, "yaw_offset"),
                        elevation_offset=_required_float(condition, "elevation_offset"),
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


def collate_mesh_samples(rows: list[dict[str, object]]) -> dict[str, object]:
    """可変長meshをlistのまま保つcollate_fn。"""

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
    """batch方向にシルエット画像をstackするcollate_fn。"""

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
    """render cache済みPNGを (V,1,H,W) のfloat32 [0,1] tensorとして読む。"""

    views = []
    for view_index in range(num_views):
        image_path = split_dir / f"{index:06d}_v{view_index}.png"
        with Image.open(image_path) as image:
            array = torch.from_numpy(np.array(image.convert("L")))
        views.append(array.unsqueeze(0).to(dtype=torch.float32) / 255.0)
    return torch.stack(views, dim=0)


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
        msg = f"{key} は整数に変換できる値である必要があります。"
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
