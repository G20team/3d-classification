"""Pokemon-3D-api/assetsをdata/raw_assetsへ取得する任意補助CLI。"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.paths import find_project_root, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pokemon 3D assets repositoryを浅いcloneで取得します。")
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
