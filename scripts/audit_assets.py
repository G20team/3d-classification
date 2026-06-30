"""GLBアセットを監査してmanifestを生成するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.assets.audit import audit_assets
from pokemon_3d_cls.paths import find_project_root, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon 3D GLB assetsを監査します。")
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output", default="data/manifests/asset_audit.jsonl")
    parser.add_argument("--selected-output", default="data/manifests/selected_regular.jsonl")
    parser.add_argument("--summary-json", default="data/manifests/asset_audit_summary.json")
    parser.add_argument("--summary-md", default="data/manifests/asset_audit_summary.md")
    parser.add_argument("--pokeapi-cache", default="data/manifests/pokeapi_cache.json")
    parser.add_argument("--offline", action="store_true", help="PokeAPIを取得せずcacheのみ使う")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    records, selected = audit_assets(
        asset_root=resolve_project_path(args.asset_root, project_root),
        output_path=resolve_project_path(args.output, project_root),
        selected_output_path=resolve_project_path(args.selected_output, project_root),
        summary_json_path=resolve_project_path(args.summary_json, project_root),
        summary_markdown_path=resolve_project_path(args.summary_md, project_root),
        pokeapi_cache_path=resolve_project_path(args.pokeapi_cache, project_root),
        allow_pokeapi_fetch=not args.offline,
    )
    print(f"audit finished: total={len(records)} selected={len(selected)}")


if __name__ == "__main__":
    main()
