"""CLI for creating normalized mesh caches from selected GLB files."""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.mesh.cache import prepare_mesh_cache
from pokemon_3d_cls.paths import find_project_root, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create mesh caches from the selected_regular manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="data/mesh_cache")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    records = prepare_mesh_cache(
        manifest_path=resolve_project_path(args.manifest, project_root),
        output_root=resolve_project_path(args.output_root, project_root),
    )
    print(f"mesh cache prepared: {len(records)} meshes")


if __name__ == "__main__":
    main()
