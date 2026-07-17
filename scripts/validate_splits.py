"""CLI for validating pose split overlap."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.config import load_split_config, parse_split_config
from pokemon_3d_cls.io import read_json, read_jsonl
from pokemon_3d_cls.paths import find_project_root, resolve_project_path
from pokemon_3d_cls.splits import (
    build_pose_splits,
    load_pose_splits,
    manifest_sha256,
    pose_conditions,
    validate_pose_splits,
    validate_stratified_pose_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate train/validation/test overlap in pose splits.")
    parser.add_argument("--config", help="Split generation config.")
    parser.add_argument("--splits", help="Saved pose_splits.json.")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    if args.config:
        config = load_split_config(args.config)
        manifest_path = (
            resolve_project_path(config.manifest_path, project_root) if config.manifest_path is not None else None
        )
        pokemon_ids = _manifest_ids(manifest_path) if manifest_path is not None else None
        splits = build_pose_splits(config, pokemon_ids=pokemon_ids)
    elif args.splits:
        splits_path = resolve_project_path(args.splits, project_root)
        document = read_json(splits_path)
        splits = load_pose_splits(splits_path)
        raw_config = document.get("config")
        config = parse_split_config(raw_config) if isinstance(raw_config, dict) else None
        manifest_path = None
        pokemon_ids = None
        if config is not None and config.manifest_path is not None:
            manifest_path = resolve_project_path(config.manifest_path, project_root)
            pokemon_ids = _manifest_ids(manifest_path)
            source = document.get("source_manifest")
            if isinstance(source, dict) and source.get("sha256") != manifest_sha256(manifest_path):
                raise SystemExit("Source manifest SHA-256 does not match the saved split document.")
    else:
        raise SystemExit("--config or --splits is required.")
    if config is not None and config.strategy == "stratified_samples":
        if pokemon_ids is None:
            raise SystemExit("A source manifest is required to validate stratified_samples.")
        errors = validate_stratified_pose_splits(
            splits,
            pokemon_ids=pokemon_ids,
            conditions=pose_conditions(config),
            split_counts=config.split_counts,
        )
    else:
        errors = validate_pose_splits(splits)
    if errors:
        raise SystemExit("\n".join(errors))
    print("pose split validation passed")


def _manifest_ids(path: Path) -> list[int]:
    ids = []
    for row in read_jsonl(path):
        value = row.get("pokemon_id")
        if isinstance(value, bool) or not isinstance(value, int | str):
            continue
        ids.append(int(value))
    return sorted(ids)


if __name__ == "__main__":
    main()
