"""姿勢split JSONを生成するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_split_config
from pokemon_3d_cls.paths import find_project_root, resolve_project_path
from pokemon_3d_cls.splits import save_pose_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="closed-set cross-orientation用の姿勢splitを作成します。")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    project_root = find_project_root(Path(args.config).resolve().parent)
    config = load_split_config(args.config)
    output_path = resolve_project_path(config.output_path, project_root)
    splits = save_pose_splits(config, output_path)
    print(f"pose splits saved: {output_path} ({', '.join(splits)})")


if __name__ == "__main__":
    main()
