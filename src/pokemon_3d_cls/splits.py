"""closed-set cross-orientation用の姿勢split。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from pokemon_3d_cls.config import PoseSplitValues, SplitConfig
from pokemon_3d_cls.io import read_json, write_json


@dataclass(frozen=True)
class PoseCondition:
    """グローバル姿勢条件。"""

    yaw_offset: float
    elevation_offset: float

    def key(self) -> tuple[float, float]:
        return (self.yaw_offset, self.elevation_offset)


def build_pose_splits(config: SplitConfig) -> dict[str, list[dict[str, float]]]:
    """SplitConfigからJSON保存可能な姿勢splitを作る。"""

    return {
        "train": [_condition_to_dict(condition) for condition in _conditions(config.train)],
        "validation": [_condition_to_dict(condition) for condition in _conditions(config.validation)],
        "test": [_condition_to_dict(condition) for condition in _conditions(config.test)],
    }


def validate_pose_splits(splits: dict[str, list[dict[str, float]]]) -> list[str]:
    """姿勢条件の重複を検証し、エラー一覧を返す。"""

    errors: list[str] = []
    seen: dict[tuple[float, float], str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            condition = (float(row["yaw_offset"]), float(row["elevation_offset"]))
            previous = seen.get(condition)
            if previous is not None and previous != split_name:
                errors.append(f"{condition} が {previous} と {split_name} で重複しています。")
            seen[condition] = split_name
    return errors


def save_pose_splits(config: SplitConfig, output_path: Path) -> dict[str, list[dict[str, float]]]:
    """姿勢splitを保存し、重複があれば例外を投げる。"""

    splits = build_pose_splits(config)
    errors = validate_pose_splits(splits)
    if errors:
        msg = "姿勢splitが重複しています:\n" + "\n".join(errors)
        raise ValueError(msg)
    write_json(output_path, {"splits": splits, "config": cast("dict[str, object]", asdict(config))})
    return splits


def load_pose_splits(path: str | Path) -> dict[str, list[dict[str, float]]]:
    """保存済みsplit JSONを読み込む。"""

    raw = read_json(path)
    splits = raw.get("splits", raw)
    if not isinstance(splits, dict):
        msg = f"splitsがobjectではありません: {path}"
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
