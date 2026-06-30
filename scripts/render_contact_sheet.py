"""採用アセットのturntable contact sheetを作るCLI。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from pokemon_3d_cls.assets.audit import load_trimesh_mesh
from pokemon_3d_cls.io import read_jsonl
from pokemon_3d_cls.mesh.normalize import normalize_trimesh
from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path
from pokemon_3d_cls.rendering.glb import render_silhouette


def main() -> None:
    parser = argparse.ArgumentParser(description="selected_regular manifestからcontact sheetを生成します。")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    rows = read_jsonl(resolve_project_path(args.manifest, project_root))
    rng = random.Random(args.seed)
    sample_rows = rng.sample(rows, min(args.num_samples, len(rows)))
    tiles: list[Image.Image] = []
    azimuths = np.linspace(0, 360, num=args.views, endpoint=False)
    for row in sample_rows:
        mesh = load_trimesh_mesh(Path(str(row["asset_path"])))
        normalized = normalize_trimesh(mesh)
        view_images = []
        vertices = normalized.vertices.numpy()
        faces = normalized.faces.numpy()
        for azimuth in azimuths:
            image = render_silhouette(vertices, faces, float(azimuth), 0.0, resolution=args.image_size)
            view_images.append(Image.fromarray(image).convert("RGB"))
        tiles.append(_hstack(view_images))
    sheet = _grid(tiles, columns=5)
    output_path = resolve_project_path(args.output, project_root)
    ensure_directory(output_path.parent)
    sheet.save(output_path)
    print(f"contact sheet saved: {output_path}")


def _hstack(images: list[Image.Image]) -> Image.Image:
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def _grid(images: list[Image.Image], *, columns: int) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), color=(255, 255, 255))
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), color=(255, 255, 255))
    for index, image in enumerate(images):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        canvas.paste(image, (x, y))
    return canvas


if __name__ == "__main__":
    main()
