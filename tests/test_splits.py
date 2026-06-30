from __future__ import annotations

from pokemon_3d_cls.config import parse_split_config
from pokemon_3d_cls.splits import build_pose_splits, validate_pose_splits


def test_default_pose_splits_do_not_overlap() -> None:
    config = parse_split_config({})
    splits = build_pose_splits(config)

    assert validate_pose_splits(splits) == []


def test_validate_pose_splits_detects_overlap() -> None:
    splits = {
        "train": [{"yaw_offset": 0.0, "elevation_offset": 0.0}],
        "validation": [{"yaw_offset": 0.0, "elevation_offset": 0.0}],
        "test": [{"yaw_offset": 45.0, "elevation_offset": 25.0}],
    }

    assert validate_pose_splits(splits)
