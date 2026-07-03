"""Pose splits for closed-set cross-orientation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pokemon_3d_cls.config import PoseSplitValues, SplitConfig
from pokemon_3d_cls.io import read_json, write_json


@dataclass(frozen=True)
class PoseCondition:
    """Global pose condition."""

    yaw_offset: float
    elevation_offset: float

    def key(self) -> tuple[float, float]:
        return (self.yaw_offset, self.elevation_offset)


def build_pose_splits(config: SplitConfig) -> dict[str, list[dict[str, float]]]:
    """Build JSON-serializable pose splits from SplitConfig."""

    return {
        "train": [_condition_to_dict(condition) for condition in _conditions(config.train)],
        "validation": [_condition_to_dict(condition) for condition in _conditions(config.validation)],
        "test": [_condition_to_dict(condition) for condition in _conditions(config.test)],
    }


def validate_pose_splits(splits: dict[str, list[dict[str, float]]]) -> list[str]:
    """Validate pose-condition overlap and return error messages."""

    errors: list[str] = []
    seen: dict[tuple[float, float], str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            condition = (float(row["yaw_offset"]), float(row["elevation_offset"]))
            previous = seen.get(condition)
            if previous is not None and previous != split_name:
                errors.append(f"{condition} overlaps between {previous} and {split_name}.")
            seen[condition] = split_name
    return errors


def save_pose_splits(config: SplitConfig, output_path: Path) -> dict[str, list[dict[str, float]]]:
    """Save pose splits and raise an exception if overlap exists."""

    splits = build_pose_splits(config)
    errors = validate_pose_splits(splits)
    if errors:
        msg = "Pose splits overlap:\n" + "\n".join(errors)
        raise ValueError(msg)
    write_json(output_path, {"splits": splits, "config": cast("dict[str, object]", asdict(config))})
    return splits


def load_pose_splits(path: str | Path) -> dict[str, list[dict[str, float]]]:
    """Load a saved split JSON file."""

    raw = read_json(path)
    splits = raw.get("splits", raw)
    if not isinstance(splits, dict):
        msg = f"splits is not an object: {path}"
        raise ValueError(msg)
    return cast("dict[str, list[dict[str, float]]]", splits)


def _conditions(values: PoseSplitValues) -> list[PoseCondition]:
    return [
        PoseCondition(yaw_offset=float(yaw), elevation_offset=float(elevation))
        for yaw in values.yaw_offsets
        for elevation in values.elevation_offsets
    ]


def _condition_to_dict(condition: PoseCondition) -> dict[str, float]:
    return {
        "yaw_offset": condition.yaw_offset,
        "elevation_offset": condition.elevation_offset,
    }
