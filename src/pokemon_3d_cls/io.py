"""JSON/YAML/CSV I/O."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, cast

from pokemon_3d_cls.paths import ensure_directory


def read_yaml_mapping(path: str | Path) -> dict[str, object]:
    """Load YAML and validate that the root is a mapping."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        msg = "PyYAML is required. Run again after `uv sync`."
        raise RuntimeError(msg) from exc

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        msg = f"YAML root must be a mapping: {yaml_path}"
        raise ValueError(msg)
    return raw


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    """Save YAML as UTF-8."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        msg = "PyYAML is required. Run again after `uv sync`."
        raise RuntimeError(msg) from exc

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    """Save JSON as UTF-8."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def read_json(path: str | Path) -> dict[str, object]:
    """Load JSON and validate that the root is an object."""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        msg = f"JSON root must be an object: {json_path}"
        raise ValueError(msg)
    return cast("dict[str, object]", raw)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Save JSONL as UTF-8."""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """Load JSONL."""

    jsonl_path = Path(path)
    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                msg = f"JSONL row must be an object: {jsonl_path}:{line_number}"
                raise ValueError(msg)
            rows.append(cast("dict[str, object]", raw))
    return rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    """Save CSV as UTF-8."""

    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
