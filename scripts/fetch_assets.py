"""Optional helper CLI for fetching Pokemon-3D-api/assets into data/raw_assets."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.paths import find_project_root, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Pokemon 3D assets repository with a shallow clone.")
    parser.add_argument("--output", default="data/raw_assets")
    parser.add_argument("--repo", default="https://github.com/Pokemon-3D-api/assets.git")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    output = resolve_project_path(args.output, project_root)
    if output.exists():
        print(f"already exists: {output}")
        return
    subprocess.run(["git", "clone", "--depth", "1", args.repo, str(output)], check=True)
    print(f"assets cloned: {output}")


if __name__ == "__main__":
    main()
