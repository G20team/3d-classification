"""公式イラストのalphaチャンネルから黒塗りシルエットを作るCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from PIL import Image

from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path


def illustration_to_silhouette(image: Image.Image, *, resolution: int, alpha_threshold: int = 32) -> np.ndarray:
    """透過PNGのalphaチャンネルを二値化し、白背景+黒塗りのシルエット配列を返す。"""

    rgba = image.convert("RGBA")
    alpha = np.array(rgba)[:, :, 3]
    silhouette = np.where(alpha > alpha_threshold, 0, 255).astype(np.uint8)
    return np.array(Image.fromarray(silhouette).resize((resolution, resolution), Image.Resampling.LANCZOS))


def main() -> None:
    parser = argparse.ArgumentParser(description="公式イラストを黒塗りシルエットへ変換します。")
    parser.add_argument("--input", default="data/illustrations")
    parser.add_argument("--output", default="data/illustrations_silhouette")
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--alpha-threshold", type=int, default=32)
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    input_dir = resolve_project_path(args.input, project_root)
    output_dir = ensure_directory(resolve_project_path(args.output, project_root))

    paths = sorted(input_dir.glob("*.png"))
    for path in paths:
        with Image.open(path) as image:
            silhouette = illustration_to_silhouette(
                image,
                resolution=args.resolution,
                alpha_threshold=args.alpha_threshold,
            )
        Image.fromarray(silhouette).save(output_dir / path.name)

    print(f"converted: {len(paths)} -> {output_dir}")


if __name__ == "__main__":
    main()
