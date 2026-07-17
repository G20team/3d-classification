"""Pose split generation and sample-level split resolution."""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, cast

from pokemon_3d_cls.config import PoseSplitValues, SplitConfig, SplitCounts
from pokemon_3d_cls.io import read_json, read_jsonl, write_json

SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class PoseCondition:
    """Global pose condition."""

    yaw_offset: float
    elevation_offset: float

    def key(self) -> tuple[float, float]:
        return (self.yaw_offset, self.elevation_offset)


@dataclass(frozen=True)
class ResolvedPoseSample:
    """One Pokemon and pose assignment resolved for a dataset split."""

    pokemon_id: int
    yaw_offset: float
    elevation_offset: float


def build_pose_splits(
    config: SplitConfig,
    *,
    pokemon_ids: list[int] | None = None,
) -> dict[str, list[dict[str, float | int]]]:
    """Build JSON-serializable legacy or sample-level pose splits."""

    if config.strategy == "stratified_samples":
        if pokemon_ids is None:
            msg = "pokemon_ids are required for stratified_samples."
            raise ValueError(msg)
        return _build_stratified_splits(config, pokemon_ids)
    return {
        "train": [_condition_to_dict(condition) for condition in _conditions(config.train)],
        "validation": [_condition_to_dict(condition) for condition in _conditions(config.validation)],
        "test": [_condition_to_dict(condition) for condition in _conditions(config.test)],
    }


def build_split_document(config: SplitConfig, *, manifest_path: Path | None = None) -> dict[str, object]:
    """Build a complete split document including provenance and distribution statistics."""

    if config.strategy == "exclusive_conditions":
        splits = build_pose_splits(config)
        return {
            "schema_version": 1,
            "strategy": config.strategy,
            "splits": splits,
            "config": cast("dict[str, object]", asdict(config)),
        }

    if manifest_path is None:
        msg = "manifest_path is required for stratified_samples."
        raise ValueError(msg)
    pokemon_ids = _manifest_pokemon_ids(manifest_path)
    splits = build_pose_splits(config, pokemon_ids=pokemon_ids)
    conditions = pose_conditions(config)
    return {
        "schema_version": 2,
        "strategy": config.strategy,
        "seed": config.seed,
        "source_manifest": {
            "path": config.manifest_path,
            "sha256": _sha256(manifest_path),
            "class_count": len(pokemon_ids),
        },
        "split_counts": cast("dict[str, object]", asdict(config.split_counts)),
        "pose_conditions": [_condition_to_dict(condition) for condition in conditions],
        "splits": splits,
        "statistics": _split_statistics(splits, conditions),
        "config": cast("dict[str, object]", asdict(config)),
    }


def validate_pose_splits(splits: Mapping[str, list[dict[str, float | int]]]) -> list[str]:
    """Validate condition or sample overlap and return error messages."""

    errors: list[str] = []
    seen: dict[tuple[int | None, float, float], str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            pokemon_id = _optional_row_int(row, "pokemon_id")
            condition = (
                pokemon_id,
                _row_float(row, "yaw_offset"),
                _row_float(row, "elevation_offset"),
            )
            previous = seen.get(condition)
            if previous is not None:
                errors.append(f"{condition} is duplicated in {previous} and {split_name}.")
            seen[condition] = split_name
    return errors


def validate_stratified_pose_splits(
    splits: Mapping[str, list[dict[str, float | int]]],
    *,
    pokemon_ids: list[int],
    conditions: list[PoseCondition],
    split_counts: SplitCounts,
    max_pose_imbalance: int = 2,
) -> list[str]:
    """Validate class quotas, pose coverage, balance, and sample uniqueness."""

    errors = validate_pose_splits(splits)
    expected_ids = set(pokemon_ids)
    expected_conditions = {condition.key() for condition in conditions}
    expected_counts = cast(
        "dict[str, int]",
        {"train": split_counts.train, "validation": split_counts.validation, "test": split_counts.test},
    )
    per_class: dict[int, Counter[str]] = {pokemon_id: Counter() for pokemon_id in pokemon_ids}

    for split_name in SPLIT_NAMES:
        rows = splits.get(split_name)
        if rows is None:
            errors.append(f"Missing split: {split_name}.")
            continue
        pose_counts: Counter[tuple[float, float]] = Counter()
        observed_ids: set[int] = set()
        for row in rows:
            pokemon_id = _optional_row_int(row, "pokemon_id")
            if pokemon_id is None:
                errors.append(f"{split_name} contains a row without pokemon_id.")
                continue
            condition = (_row_float(row, "yaw_offset"), _row_float(row, "elevation_offset"))
            if pokemon_id not in expected_ids:
                errors.append(f"Unknown pokemon_id in {split_name}: {pokemon_id}.")
            if condition not in expected_conditions:
                errors.append(f"Unknown pose condition in {split_name}: {condition}.")
            observed_ids.add(pokemon_id)
            pose_counts[condition] += 1
            if pokemon_id in per_class:
                per_class[pokemon_id][split_name] += 1
        missing_ids = expected_ids - observed_ids
        if missing_ids:
            errors.append(f"{split_name} is missing {len(missing_ids)} Pokemon classes.")
        missing_conditions = expected_conditions - set(pose_counts)
        if missing_conditions:
            errors.append(f"{split_name} is missing {len(missing_conditions)} pose conditions.")
        if pose_counts and max(pose_counts.values()) - min(pose_counts.values()) > max_pose_imbalance:
            errors.append(f"{split_name} pose distribution differs by more than {max_pose_imbalance} samples.")

    for pokemon_id, counts in per_class.items():
        for split_name, expected in expected_counts.items():
            if counts[split_name] != expected:
                errors.append(
                    f"pokemon_id={pokemon_id} has {counts[split_name]} rows in {split_name}; expected {expected}."
                )
    return errors


def save_pose_splits(
    config: SplitConfig,
    output_path: Path,
    *,
    manifest_path: Path | None = None,
) -> dict[str, list[dict[str, float | int]]]:
    """Save pose splits with validation and provenance metadata."""

    document = build_split_document(config, manifest_path=manifest_path)
    splits = cast("dict[str, list[dict[str, float | int]]]", document["splits"])
    if config.strategy == "stratified_samples":
        if manifest_path is None:
            msg = "manifest_path is required for stratified_samples."
            raise ValueError(msg)
        errors = validate_stratified_pose_splits(
            splits,
            pokemon_ids=_manifest_pokemon_ids(manifest_path),
            conditions=pose_conditions(config),
            split_counts=config.split_counts,
        )
    else:
        errors = validate_pose_splits(splits)
    if errors:
        msg = "Pose split validation failed:\n" + "\n".join(errors)
        raise ValueError(msg)
    write_json(output_path, document)
    return splits


def load_pose_splits(path: str | Path) -> dict[str, list[dict[str, float | int]]]:
    """Load the splits mapping from either a legacy or versioned JSON document."""

    raw = read_json(path)
    splits = raw.get("splits", raw)
    if not isinstance(splits, dict):
        msg = f"splits is not an object: {path}"
        raise ValueError(msg)
    return cast("dict[str, list[dict[str, float | int]]]", splits)


def resolve_pose_samples(
    path: str | Path,
    *,
    split: str,
    pokemon_ids: list[int],
) -> list[ResolvedPoseSample]:
    """Resolve either split schema to a stable manifest-order sample sequence."""

    splits = load_pose_splits(path)
    if split not in splits:
        msg = f"Split was not found: {split}"
        raise ValueError(msg)
    rows = splits[split]
    if not rows:
        return []
    sample_level = any("pokemon_id" in row for row in rows)
    if not sample_level:
        return [
            ResolvedPoseSample(
                pokemon_id=pokemon_id,
                yaw_offset=_row_float(row, "yaw_offset"),
                elevation_offset=_row_float(row, "elevation_offset"),
            )
            for pokemon_id in pokemon_ids
            for row in rows
        ]

    allowed_ids = set(pokemon_ids)
    by_id: dict[int, list[ResolvedPoseSample]] = defaultdict(list)
    for row in rows:
        pokemon_id = _optional_row_int(row, "pokemon_id")
        if pokemon_id is None:
            msg = f"Sample-level split row does not have pokemon_id: {row}"
            raise ValueError(msg)
        if pokemon_id not in allowed_ids:
            continue
        by_id[pokemon_id].append(
            ResolvedPoseSample(
                pokemon_id=pokemon_id,
                yaw_offset=_row_float(row, "yaw_offset"),
                elevation_offset=_row_float(row, "elevation_offset"),
            )
        )
    missing_ids = allowed_ids - set(by_id)
    if missing_ids:
        msg = f"Sample-level split {split} is missing {len(missing_ids)} requested Pokemon IDs."
        raise ValueError(msg)
    return [sample for pokemon_id in pokemon_ids for sample in by_id[pokemon_id]]


def pose_conditions(config: SplitConfig) -> list[PoseCondition]:
    """Return the deduplicated condition pool for stratified splitting."""

    values = config.pose_groups or (config.train, config.validation, config.test)
    conditions: list[PoseCondition] = []
    seen: set[tuple[float, float]] = set()
    for group in values:
        for condition in _conditions(group):
            if condition.key() in seen:
                msg = f"Duplicate pose condition in configuration: {condition.key()}"
                raise ValueError(msg)
            seen.add(condition.key())
            conditions.append(condition)
    return conditions


def manifest_sha256(path: Path) -> str:
    """Return the SHA-256 digest used in split provenance validation."""

    return _sha256(path)


def _build_stratified_splits(
    config: SplitConfig,
    pokemon_ids: list[int],
) -> dict[str, list[dict[str, float | int]]]:
    if len(set(pokemon_ids)) != len(pokemon_ids):
        msg = "pokemon_ids must be unique."
        raise ValueError(msg)
    conditions = pose_conditions(config)
    counts = config.split_counts
    split_slots = ["train"] * counts.train + ["validation"] * counts.validation + ["test"] * counts.test
    if len(split_slots) != len(conditions):
        msg = "The split quota total must match the number of pose conditions."
        raise ValueError(msg)

    rng = random.Random(config.seed)
    shuffled_ids = list(pokemon_ids)
    shuffled_conditions = list(conditions)
    rng.shuffle(shuffled_ids)
    rng.shuffle(shuffled_conditions)
    assignments: dict[str, list[dict[str, float | int]]] = {name: [] for name in SPLIT_NAMES}
    for class_rank, pokemon_id in enumerate(shuffled_ids):
        for condition_rank, condition in enumerate(shuffled_conditions):
            split_name = split_slots[(condition_rank + class_rank) % len(split_slots)]
            assignments[split_name].append(
                {
                    "pokemon_id": pokemon_id,
                    "yaw_offset": condition.yaw_offset,
                    "elevation_offset": condition.elevation_offset,
                }
            )
    return assignments


def _split_statistics(
    splits: Mapping[str, list[dict[str, float | int]]],
    conditions: list[PoseCondition],
) -> dict[str, object]:
    statistics: dict[str, object] = {}
    for split_name in SPLIT_NAMES:
        rows = splits[split_name]
        pose_counts = Counter((_row_float(row, "yaw_offset"), _row_float(row, "elevation_offset")) for row in rows)
        statistics[split_name] = {
            "sample_count": len(rows),
            "class_count": len({_optional_row_int(row, "pokemon_id") for row in rows}),
            "pose_counts": [
                {**_condition_to_dict(condition), "count": pose_counts[condition.key()]} for condition in conditions
            ],
        }
    return statistics


def _manifest_pokemon_ids(path: Path) -> list[int]:
    ids: list[int] = []
    for row in read_jsonl(path):
        value = row.get("pokemon_id")
        if isinstance(value, bool) or not isinstance(value, int | str):
            continue
        ids.append(int(value))
    ids.sort()
    if not ids:
        msg = f"Manifest has no Pokemon IDs: {path}"
        raise ValueError(msg)
    if len(ids) != len(set(ids)):
        msg = f"Manifest contains duplicate Pokemon IDs: {path}"
        raise ValueError(msg)
    return ids


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


def _optional_row_int(row: Mapping[str, object], key: str) -> int | None:
    value = row.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} must be convertible to an integer."
        raise ValueError(msg)
    return int(value)


def _row_float(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        msg = f"{key} must be convertible to a number."
        raise ValueError(msg)
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
