from __future__ import annotations

from collections import Counter

from pokemon_3d_cls.config import SplitConfig, parse_split_config
from pokemon_3d_cls.splits import (
    build_pose_splits,
    pose_conditions,
    validate_pose_splits,
    validate_stratified_pose_splits,
)


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


def test_stratified_pose_splits_are_reproducible_and_balanced() -> None:
    config = _stratified_config(seed=7)
    pokemon_ids = list(range(1, 972))

    first = build_pose_splits(config, pokemon_ids=pokemon_ids)
    second = build_pose_splits(config, pokemon_ids=pokemon_ids)
    other_seed = build_pose_splits(_stratified_config(seed=8), pokemon_ids=pokemon_ids)

    assert first == second
    assert first != other_seed
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 8_739,
        "validation": 3_884,
        "test": 3_884,
    }
    assert sum(len(rows) for rows in first.values()) == 16_507
    assert (
        validate_stratified_pose_splits(
            first,
            pokemon_ids=pokemon_ids,
            conditions=pose_conditions(config),
            split_counts=config.split_counts,
        )
        == []
    )

    for pokemon_id in pokemon_ids:
        counts = Counter(
            split_name
            for split_name, rows in first.items()
            for row in rows
            if row["pokemon_id"] == pokemon_id
        )
        assert counts == {"train": 9, "validation": 4, "test": 4}


def _stratified_config(*, seed: int) -> SplitConfig:
    return parse_split_config(
        {
            "strategy": "stratified_samples",
            "manifest_path": "manifest.jsonl",
            "seed": seed,
            "split_counts": {"train": 9, "validation": 4, "test": 4},
            "pose_groups": [
                {"yaw_offsets": [-20, 0, 20], "elevation_offsets": [-10, 0, 10]},
                {"yaw_offsets": [-30, 30], "elevation_offsets": [-15, 15]},
                {"yaw_offsets": [-45, 45], "elevation_offsets": [-25, 25]},
            ],
        }
    )
