"""PokeAPIのNational Dex ID -> 英語名対応キャッシュ。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pokemon_3d_cls.io import read_json, write_json


def load_or_fetch_pokemon_names(cache_path: Path, *, allow_fetch: bool = True) -> dict[int, str]:
    """PokeAPIからID/英語名対応を取得し、ローカルcacheへ保存する。"""

    if cache_path.is_file():
        raw = read_json(cache_path)
        return {int(key): str(value) for key, value in raw.items()}
    if not allow_fetch:
        return {}

    try:
        import requests
    except ModuleNotFoundError:
        return {}

    names: dict[int, str] = {}
    # National Dexは増え続けるため、まず大きめlimitで一覧を取得する。
    response = requests.get("https://pokeapi.co/api/v2/pokemon-species?limit=2000", timeout=30)
    response.raise_for_status()
    payload = response.json()
    for item in cast("list[dict[str, object]]", payload.get("results", [])):
        name = str(item.get("name", ""))
        url = str(item.get("url", ""))
        pokemon_id = _id_from_species_url(url)
        if pokemon_id is not None and name:
            names[pokemon_id] = name
    if names:
        write_json(cache_path, {str(key): value for key, value in sorted(names.items())})
    return names


def _id_from_species_url(url: str) -> int | None:
    parts = [part for part in url.rstrip("/").split("/") if part]
    if not parts:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None
