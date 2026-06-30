"""正規化済みmesh cacheの作成。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from tqdm import tqdm

from pokemon_3d_cls.assets.audit import AssetAuditRecord, load_trimesh_mesh
from pokemon_3d_cls.io import read_jsonl, write_jsonl
from pokemon_3d_cls.mesh.normalize import normalize_trimesh
from pokemon_3d_cls.paths import ensure_directory


@dataclass(frozen=True)
class MeshCacheRecord:
    """mesh cacheのmanifest行。"""

    asset_path: str
    pokemon_id: int
    pokemon_name: str
    mesh_cache_path: str
    num_vertices: int
    num_faces: int
    center: tuple[float, float, float]
    scale: float

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", asdict(self))


def prepare_mesh_cache(*, manifest_path: Path, output_root: Path) -> list[MeshCacheRecord]:
    """selected_regular.jsonlから正規化済みmesh cacheを作成する。"""

    rows = read_jsonl(manifest_path)
    ensure_directory(output_root)
    records: list[MeshCacheRecord] = []
    for row in tqdm(rows, desc="mesh cache", unit="mesh"):
        audit = _audit_record_from_row(row)
        if audit.pokemon_id is None or audit.pokemon_name is None:
            continue
        mesh = load_trimesh_mesh(Path(audit.asset_path))
        normalized = normalize_trimesh(mesh)
        cache_path = output_root / f"{audit.pokemon_id:04d}_{audit.pokemon_name}.pt"
        torch.save(
            {
                "vertices": normalized.vertices,
                "faces": normalized.faces,
                "metadata": {
                    "asset_path": audit.asset_path,
                    "pokemon_id": audit.pokemon_id,
                    "pokemon_name": audit.pokemon_name,
                    "center": normalized.center,
                    "scale": normalized.scale,
                },
            },
            cache_path,
        )
        records.append(
            MeshCacheRecord(
                asset_path=audit.asset_path,
                pokemon_id=audit.pokemon_id,
                pokemon_name=audit.pokemon_name,
                mesh_cache_path=str(cache_path),
                num_vertices=int(normalized.vertices.shape[0]),
                num_faces=int(normalized.faces.shape[0]),
                center=normalized.center,
                scale=normalized.scale,
            )
        )
    write_jsonl(output_root / "mesh_cache_manifest.jsonl", [record.to_dict() for record in records])
    return records


def _audit_record_from_row(row: dict[str, object]) -> AssetAuditRecord:
    return AssetAuditRecord(
        asset_path=str(row["asset_path"]),
        asset_filename=str(row["asset_filename"]),
        category=str(row["category"]),
        pokemon_id=_optional_int(row, "pokemon_id"),
        pokemon_name=str(row["pokemon_name"]) if row.get("pokemon_name") is not None else None,
        num_vertices=_required_int(row, "num_vertices"),
        num_faces=_required_int(row, "num_faces"),
        has_texture=bool(row["has_texture"]),
        is_valid=bool(row["is_valid"]),
        exclude_reason=str(row["exclude_reason"]) if row.get("exclude_reason") is not None else None,
        mesh_cache_path=str(row["mesh_cache_path"]) if row.get("mesh_cache_path") is not None else None,
    )


def _optional_int(row: dict[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} は整数に変換できる値である必要があります。"
        raise ValueError(msg)
    return int(value)


def _required_int(row: dict[str, object], key: str) -> int:
    value = _optional_int(row, key)
    if value is None:
        msg = f"{key} は必須です。"
        raise ValueError(msg)
    return value
