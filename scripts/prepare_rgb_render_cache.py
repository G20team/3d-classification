"""固定視点PyTorch3D RGB画像キャッシュを生成するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_mesh_experiment_config
from pokemon_3d_cls.experiments.rgb_render_cache import build_rgb_render_cache
from pokemon_3d_cls.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed-view PyTorch3D RGB PNG cache.")
    parser.add_argument("--config", required=True, help="Single/Fixed Ring-4/View Transformer config YAML.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--force", action="store_true", help="Render completed splits again and overwrite PNG files.")
    args = parser.parse_args()

    config_path = Path(args.config)
    project_root = find_project_root(config_path.resolve().parent)
    config = load_mesh_experiment_config(config_path)
    cache_dir = build_rgb_render_cache(
        config,
        project_root,
        splits=args.splits,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(f"RGB render cache saved: {cache_dir}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        msg = "Specify a positive integer."
        raise argparse.ArgumentTypeError(msg)
    return parsed


if __name__ == "__main__":
    main()
