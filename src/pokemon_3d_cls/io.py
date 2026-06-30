"""JSON/YAML/CSV入出力。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, cast

from pokemon_3d_cls.paths import ensure_directory


def read_yaml_mapping(path: str | Path) -> dict[str, object]:
    """YAMLを読み込み、rootがmappingであることを検証する。"""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        msg = "PyYAML が必要です。`uv sync` 後に再実行してください。"
        raise RuntimeError(msg) from exc

    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        msg = f"YAML rootはmappingである必要があります: {yaml_path}"
        raise ValueError(msg)
    return raw


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    """YAMLをUTF-8で保存する。"""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        msg = "PyYAML が必要です。`uv sync` 後に再実行してください。"
        raise RuntimeError(msg) from exc

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)


def write_json(path: Path, data: Mapping[str, object]) -> None:
    """JSONをUTF-8で保存する。"""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def read_json(path: str | Path) -> dict[str, object]:
    """JSONを読み込み、rootがobjectであることを検証する。"""

    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        msg = f"JSON rootはobjectである必要があります: {json_path}"
        raise ValueError(msg)
    return cast("dict[str, object]", raw)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """JSONLをUTF-8で保存する。"""

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False, sort_keys=True)
            file.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    """JSONLを読み込む。"""

    jsonl_path = Path(path)
    rows: list[dict[str, object]] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                msg = f"JSONL rowはobjectである必要があります: {jsonl_path}:{line_number}"
                raise ValueError(msg)
            rows.append(cast("dict[str, object]", raw))
    return rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, object]]) -> None:
    """CSVをUTF-8で保存する。"""

    ensure_directory(path.parent)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
