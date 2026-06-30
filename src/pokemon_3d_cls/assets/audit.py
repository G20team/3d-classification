"""Pokemon 3D GLBアセットの監査。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import trimesh

from pokemon_3d_cls.assets.glb import glb_has_texture, load_draco_glb_mesh
from pokemon_3d_cls.assets.pokeapi import load_or_fetch_pokemon_names
from pokemon_3d_cls.io import write_json, write_jsonl
from pokemon_3d_cls.paths import ensure_directory

NON_REGULAR_MARKERS = {
    "shiny",
    "mega",
    "gigantamax",
    "gmax",
    "alola",
    "galar",
    "hisui",
    "paldea",
    "female",
    "male",
    "origin",
    "therian",
    "sky",
    "primal",
    "totem",
    "battle-bond",
}


@dataclass(frozen=True)
class AssetAuditRecord:
    """1つのGLBアセットの監査結果。"""

    asset_path: str
    asset_filename: str
    category: str
    pokemon_id: int | None
    pokemon_name: str | None
    num_vertices: int
    num_faces: int
    has_texture: bool
    is_valid: bool
    exclude_reason: str | None
    mesh_cache_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return cast("dict[str, object]", asdict(self))


def audit_assets(
    *,
    asset_root: Path,
    output_path: Path,
    selected_output_path: Path,
    summary_json_path: Path,
    summary_markdown_path: Path,
    pokeapi_cache_path: Path,
    allow_pokeapi_fetch: bool = True,
) -> tuple[list[AssetAuditRecord], list[AssetAuditRecord]]:
    """GLBを再帰走査し、監査JSONLと採用manifestを保存する。"""

    pokemon_names = load_or_fetch_pokemon_names(pokeapi_cache_path, allow_fetch=allow_pokeapi_fetch)
    glb_files = sorted(asset_root.rglob("*.glb"))
    records = [_audit_one(path, asset_root, pokemon_names) for path in glb_files]
    records = _mark_duplicate_ids(records)
    selected = [record for record in records if record.is_valid and record.exclude_reason is None]

    write_jsonl(output_path, [record.to_dict() for record in records])
    write_jsonl(selected_output_path, [record.to_dict() for record in selected])
    summary = summarize_audit(records, selected)
    write_json(summary_json_path, summary)
    _write_summary_markdown(summary_markdown_path, summary)
    return records, selected


def summarize_audit(records: list[AssetAuditRecord], selected: list[AssetAuditRecord]) -> dict[str, object]:
    """監査件数を集計する。"""

    reasons: dict[str, int] = {}
    for record in records:
        reason = record.exclude_reason or "selected"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "total_assets": len(records),
        "selected_assets": len(selected),
        "excluded_assets": len(records) - len(selected),
        "exclude_reasons": dict(sorted(reasons.items())),
    }


def extract_pokemon_id_from_path(path: str | Path) -> int | None:
    """ファイル名や親ディレクトリからNational Dex IDを推定する。"""

    for part in reversed(Path(path).parts):
        candidates = re.findall(r"(?<!\d)(\d{1,4})(?!\d)", part)
        for candidate in candidates:
            value = int(candidate)
            if 1 <= value <= 2000:
                return value
    return None


def infer_category(path: Path, asset_root: Path) -> str:
    """asset rootから見たpath segmentからカテゴリを推定する。"""

    try:
        relative_parts = path.relative_to(asset_root).parts
    except ValueError:
        relative_parts = path.parts
    lowered = [part.lower() for part in relative_parts]
    for part in lowered:
        if part in {"regular", "shiny", "mega", "gmax", "gigantamax", "forms"}:
            return part
    return lowered[-2] if len(lowered) >= 2 else "unknown"


def is_regular_candidate(path: str | Path, category: str) -> bool:
    """初期実験に使うregular通常形候補かを判定する。"""

    text = Path(path).as_posix().lower()
    if any(marker in text for marker in NON_REGULAR_MARKERS):
        return False
    if category.lower() == "regular":
        return True
    # 構成決め打ちは避けるが、regularが明示されないassetsは初期実験から外す。
    return "/regular/" in f"/{text}/"


def load_trimesh_mesh(path: Path) -> trimesh.Trimesh:
    """GLBを読み込み、scene内geometryを結合したTrimeshを返す。"""

    try:
        mesh = _load_trimesh_mesh_raw(path)
        if _mesh_exclude_reason(np.asarray(mesh.vertices), np.asarray(mesh.faces)) is None:
            return mesh
    except Exception:
        if path.suffix.lower() != ".glb":
            raise

    if path.suffix.lower() == ".glb":
        return load_draco_glb_mesh(path)

    msg = f"未対応または正規化不能なmeshです: {path}"
    raise ValueError(msg)


def _load_trimesh_mesh_raw(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not geometries:
            msg = "sceneにTrimesh geometryがありません。"
            raise ValueError(msg)
        return trimesh.util.concatenate(tuple(geometries))
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    msg = f"未対応のtrimesh読み込み結果です: {type(loaded).__name__}"
    raise ValueError(msg)


def _audit_one(path: Path, asset_root: Path, pokemon_names: dict[int, str]) -> AssetAuditRecord:
    category = infer_category(path, asset_root)
    pokemon_id = extract_pokemon_id_from_path(_asset_relative_path(path, asset_root))
    pokemon_name = pokemon_names.get(pokemon_id) if pokemon_id is not None else None
    try:
        mesh = load_trimesh_mesh(path)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        exclude_reason = _mesh_exclude_reason(vertices, faces)
        if pokemon_id is None:
            exclude_reason = exclude_reason or "pokemon_id_not_found"
        elif pokemon_name is None:
            exclude_reason = exclude_reason or "pokemon_name_not_found"
        if not is_regular_candidate(path, category):
            exclude_reason = exclude_reason or "non_regular_form"
        return AssetAuditRecord(
            asset_path=str(path),
            asset_filename=path.name,
            category=category,
            pokemon_id=pokemon_id,
            pokemon_name=pokemon_name,
            num_vertices=int(len(vertices)),
            num_faces=int(len(faces)),
            has_texture=_has_texture(mesh) or glb_has_texture(path),
            is_valid=exclude_reason is None,
            exclude_reason=exclude_reason,
        )
    except Exception as exc:
        return AssetAuditRecord(
            asset_path=str(path),
            asset_filename=path.name,
            category=category,
            pokemon_id=pokemon_id,
            pokemon_name=pokemon_name,
            num_vertices=0,
            num_faces=0,
            has_texture=False,
            is_valid=False,
            exclude_reason=f"load_failed: {exc}",
        )


def _mesh_exclude_reason(vertices: np.ndarray, faces: np.ndarray) -> str | None:
    if vertices.size == 0:
        return "empty_mesh"
    if faces.size == 0:
        return "mesh_has_no_faces"
    if not np.isfinite(vertices).all():
        return "mesh_has_nan_or_inf"
    if _non_degenerate_face_count(vertices, faces) == 0:
        return "mesh_has_no_non_degenerate_faces"
    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    if radius <= 0:
        return "zero_scale_mesh"
    return None


def _asset_relative_path(path: Path, asset_root: Path) -> Path:
    try:
        return path.relative_to(asset_root)
    except ValueError:
        return path


def _non_degenerate_face_count(vertices: np.ndarray, faces: np.ndarray) -> int:
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    return int((np.isfinite(areas) & (areas > 1e-12)).sum())


def _has_texture(mesh: trimesh.Trimesh) -> bool:
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    if material is None:
        return False
    return bool(getattr(material, "image", None) is not None or getattr(material, "baseColorTexture", None) is not None)


def _mark_duplicate_ids(records: list[AssetAuditRecord]) -> list[AssetAuditRecord]:
    seen: set[int] = set()
    updated: list[AssetAuditRecord] = []
    for record in records:
        if not record.is_valid or record.pokemon_id is None:
            updated.append(record)
            continue
        if record.pokemon_id in seen:
            updated.append(
                AssetAuditRecord(
                    **{
                        **record.to_dict(),
                        "is_valid": False,
                        "exclude_reason": "duplicate_pokemon_id",
                    }
                )
            )
            continue
        seen.add(record.pokemon_id)
        updated.append(record)
    return updated


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    ensure_directory(path.parent)
    reasons = cast("dict[str, int]", summary["exclude_reasons"])
    lines = [
        "# Asset Audit Summary",
        "",
        f"- total_assets: {summary['total_assets']}",
        f"- selected_assets: {summary['selected_assets']}",
        f"- excluded_assets: {summary['excluded_assets']}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in reasons.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
