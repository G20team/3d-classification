"""GLBからシルエットデータセットを生成するパイプライン。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from pokemon_3d_cls.config import GenerationConfig
from pokemon_3d_cls.data import natural_sort_key
from pokemon_3d_cls.io import write_csv_rows, write_yaml
from pokemon_3d_cls.paths import ensure_directory, resolve_project_path
from pokemon_3d_cls.rendering.glb import load_mesh, make_label, normalize_vertices, render_silhouette, viewpoints


@dataclass(frozen=True)
class GenerationSummary:
    """シルエット生成結果の要約。"""

    output_dir: Path
    manifest_path: Path
    models_ok: int
    models_failed: int
    images_written: int
    failures: tuple[str, ...]


def build_silhouette_dataset(config: GenerationConfig, project_root: Path) -> GenerationSummary:
    """設定に従ってGLBからシルエット画像データセットを生成する。"""

    input_path = resolve_project_path(config.input.path, project_root)
    output_dir = ensure_directory(resolve_project_path(config.output.dataset_root, project_root))
    manifest_path = output_dir / config.output.manifest_name
    write_yaml(output_dir / "generation_config.yaml", config.to_dict())

    glb_files = find_glb_files(input_path)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    models_ok = 0
    images_written = 0

    for glb_file in tqdm(glb_files, desc="GLB", unit="model"):
        label = make_label(glb_file, config.labels.mode)
        try:
            vertices, faces = load_mesh(glb_file)
            vertices = normalize_vertices(vertices, up=config.rendering.up)
        except Exception as exc:
            failures.append(f"{glb_file}: {exc}")
            continue

        label_dir = ensure_directory(output_dir / label)
        view_points = viewpoints(
            config.rendering.mode,
            config.rendering.views,
            base_azimuth=config.rendering.base_azimuth,
            rng=rng,
        )
        for view_index, (azimuth, elevation) in enumerate(view_points):
            image = render_silhouette(
                vertices,
                faces,
                azimuth,
                elevation,
                resolution=config.rendering.resolution,
                supersample=config.rendering.supersample,
                invert=config.rendering.invert,
                pad=config.rendering.pad,
            )
            image_path = label_dir / f"{label}_{view_index:03d}.png"
            Image.fromarray(image).save(image_path)
            rows.append(
                {
                    "filepath": str(image_path.relative_to(output_dir)),
                    "label": label,
                    "source_glb": _display_path(glb_file, project_root),
                    "azimuth": round(float(azimuth), 2),
                    "elevation": round(float(elevation), 2),
                }
            )
            images_written += 1
        models_ok += 1

    if models_ok == 0:
        detail = "\n".join(failures[:5])
        msg = f"シルエット生成に成功したGLBがありません。\n{detail}"
        raise RuntimeError(msg)

    write_csv_rows(
        manifest_path,
        fieldnames=["filepath", "label", "source_glb", "azimuth", "elevation"],
        rows=rows,
    )
    return GenerationSummary(
        output_dir=output_dir,
        manifest_path=manifest_path,
        models_ok=models_ok,
        models_failed=len(failures),
        images_written=images_written,
        failures=tuple(failures),
    )


def find_glb_files(input_path: Path) -> list[Path]:
    """GLBファイルまたはGLBを含むディレクトリから入力一覧を作る。"""

    if input_path.is_file() and input_path.suffix.lower() == ".glb":
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.rglob("*.glb"), key=natural_sort_key)
        if files:
            return files
    msg = f".glb が見つかりません: {input_path}"
    raise FileNotFoundError(msg)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)
