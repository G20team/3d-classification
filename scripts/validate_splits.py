"""CLI for validating pose split overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_split_config
from pokemon_3d_cls.paths import find_project_root, resolve_project_path
from pokemon_3d_cls.splits import build_pose_splits, load_pose_splits, validate_pose_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/validation/test overlap in pose splits.")
    parser.add_argument("--config", help="Split generation config.")
    parser.add_argument("--splits", help="Saved pose_splits.json.")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    if args.config:
        config = load_split_config(args.config)
        splits = build_pose_splits(config)
    elif args.splits:
        splits = load_pose_splits(resolve_project_path(args.splits, project_root))
    else:
        raise SystemExit("--config or --splits is required.")
    errors = validate_pose_splits(splits)
    if errors:
        raise SystemExit("\n".join(errors))
    print("pose split validation passed")


if __name__ == "__main__":
    main()
