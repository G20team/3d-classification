"""mesh cacheから黒塗りシルエットのPNG render cacheを作る。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from pokemon_3d_cls.config import MeshExperimentConfig
from pokemon_3d_cls.experiments.silhouette_rendering import render_silhouette
from pokemon_3d_cls.io import read_jsonl
from pokemon_3d_cls.models.camera import fixed_camera_angles
from pokemon_3d_cls.paths import ensure_directory, resolve_project_path
from pokemon_3d_cls.splits import load_pose_splits


def build_render_cache(config: MeshExperimentConfig, project_root: Path, *, splits: list[str]) -> Path:
    """指定splitについて、mesh cacheから1回だけレンダリングしてPNGへ保存する。"""

    manifest_path = resolve_project_path(config.data.manifest_path, project_root)
    mesh_cache_root = resolve_project_path(config.data.mesh_cache_root, project_root)
    splits_path = resolve_project_path(config.data.splits_path, project_root)
    cache_root = resolve_project_path(config.data.render_cache_root, project_root) / config.experiment.condition_id

    rows = read_jsonl(manifest_path)
    rows = [row for row in rows if row.get("pokemon_id") is not None and row.get("pokemon_name") is not None]
    rows.sort(key=lambda row: _required_int(row, "pokemon_id"))
    if config.data.class_limit is not None:
        rows = rows[: config.data.class_limit]
    if not rows:
        msg = f"manifestに採用クラスがありません: {manifest_path}"
        raise ValueError(msg)

    pose_splits = load_pose_splits(splits_path)
    azimuths_tensor, elevations_tensor = fixed_camera_angles(config.model.experiment_kind)
    azimuths = [float(value) for value in azimuths_tensor.tolist()]
    elevations = [float(value) for value in elevations_tensor.tolist()]

    for split in splits:
        if split not in pose_splits:
            msg = f"splitが見つかりません: {split}"
            raise ValueError(msg)
        conditions = pose_splits[split]
        split_dir = ensure_directory(cache_root / split)
        index = 0
        for row in tqdm(rows, desc=f"render_cache[{split}]", unit="pokemon"):
            mesh_cache_path = _mesh_cache_path(row, mesh_cache_root)
            cached = torch.load(mesh_cache_path, map_location="cpu", weights_only=False)
            vertices = cached["vertices"].numpy()
            faces = cached["faces"].numpy()
            for condition in conditions:
                yaw_offset = _required_float(condition, "yaw_offset")
                elevation_offset = _required_float(condition, "elevation_offset")
                for view_index, (base_azimuth, base_elevation) in enumerate(zip(azimuths, elevations, strict=True)):
                    image = render_silhouette(
                        vertices,
                        faces,
                        base_azimuth + yaw_offset,
                        base_elevation + elevation_offset,
                        resolution=config.rendering.image_size,
                        supersample=config.rendering.supersample,
                        line_width=config.rendering.line_width,
                    )
                    Image.fromarray(image).save(split_dir / f"{index:06d}_v{view_index}.png")
                index += 1

    return cache_root


def _mesh_cache_path(row: Mapping[str, object], mesh_cache_root: Path) -> Path:
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
