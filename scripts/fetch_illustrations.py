"""PokeAPI公式イラスト(official-artwork)を取得するCLI。"""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

import _bootstrap  # noqa: F401
import requests

from pokemon_3d_cls.io import read_jsonl
from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path

ARTWORK_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/{id}.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="manifestに載っているポケモンの公式イラストを取得します。")
    parser.add_argument("--manifest", default="data/manifests/selected_regular.jsonl")
    parser.add_argument("--output", default="data/illustrations")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    manifest_path = resolve_project_path(args.manifest, project_root)
    output_dir = ensure_directory(resolve_project_path(args.output, project_root))

    rows = read_jsonl(manifest_path)
    ok = 0
    failed: list[str] = []
    for row in rows:
        pokemon_id = _required_int(row, "pokemon_id")
        pokemon_name = _required_str(row, "pokemon_name")
        output_path = output_dir / f"{pokemon_id:04d}_{pokemon_name}.png"
        if output_path.is_file():
            ok += 1
            continue
        url = ARTWORK_URL_TEMPLATE.format(id=pokemon_id)
        for attempt in range(args.retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    output_path.write_bytes(response.content)
                    ok += 1
                    break
                failed.append(f"{pokemon_name}(id={pokemon_id}): status={response.status_code}")
                break
            except requests.RequestException as exc:
                if attempt == args.retries - 1:
                    failed.append(f"{pokemon_name}(id={pokemon_id}): {exc}")
                else:
                    time.sleep(1.0)

    print(f"fetched: {ok}/{len(rows)}")
    if failed:
        print(f"failed: {len(failed)}")
        for item in failed[:20]:
            print(f"  - {item}")


def _required_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | str):
        msg = f"{key} は整数に変換できる値である必要があります。"
        raise ValueError(msg)
    return int(value)


def _required_str(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{key} は空でない文字列である必要があります。"
        raise ValueError(msg)
    return value


if __name__ == "__main__":
    main()
