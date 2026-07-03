"""CLI for generating silhouette datasets from GLB files based on a YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_generation_config
from pokemon_3d_cls.generation import build_silhouette_dataset
from pokemon_3d_cls.paths import find_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a silhouette image dataset from GLB files.")
    parser.add_argument("--config", required=True, help="Generation config YAML.")
    args = parser.parse_args()

    config_path = Path(args.config)
    project_root = find_project_root(config_path.resolve().parent)
    config = load_generation_config(config_path)
    summary = build_silhouette_dataset(config, project_root)
    print(
        "generation finished: "
        f"models_ok={summary.models_ok}, "
        f"models_failed={summary.models_failed}, "
        f"images={summary.images_written}, "
        f"output={summary.output_dir}, "
        f"manifest={summary.manifest_path}"
    )
    if summary.failures:
        print("failed models:")
        for failure in summary.failures[:10]:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
