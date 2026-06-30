"""YAML設定に基づいてMVCNNを学習するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_training_config
from pokemon_3d_cls.paths import find_project_root
from pokemon_3d_cls.training import train_config


def main() -> None:
    parser = argparse.ArgumentParser(description="MVCNNの学習を実行します。")
    parser.add_argument("--config", required=True, help="学習設定YAML")
    args = parser.parse_args()

    config_path = Path(args.config)
    project_root = find_project_root(config_path.resolve().parent)
    config = load_training_config(config_path)
    run_dir = train_config(config, project_root)
    print(f"training finished: {run_dir}")


if __name__ == "__main__":
    main()
