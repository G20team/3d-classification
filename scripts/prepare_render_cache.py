"""CLI for creating a PNG render cache of filled silhouettes."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_mesh_experiment_config
from pokemon_3d_cls.experiments.render_cache import build_render_cache
from pokemon_3d_cls.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a filled-silhouette render cache for single_view/fixed_ring4.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    args = parser.parse_args()

    config_path = Path(args.config)
    project_root = find_project_root(config_path.resolve().parent)
    config = load_mesh_experiment_config(config_path)
    cache_root = build_render_cache(config, project_root, splits=args.splits)
    print(f"render cache saved: {cache_root}")


if __name__ == "__main__":
    main()
