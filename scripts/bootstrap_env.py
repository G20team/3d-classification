"""環境診断レポートを保存するCLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from pokemon_3d_cls.environment import collect_environment_report
from pokemon_3d_cls.paths import find_project_root, resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch/PyTorch3D/CUDA環境を診断します。")
    parser.add_argument("--output", default="outputs/environment_report.json")
    args = parser.parse_args()
    project_root = find_project_root(Path.cwd())
    output_path = resolve_project_path(args.output, project_root)
    report = collect_environment_report(output_path)
    print(f"environment report saved: {output_path}")
    print(report)


if __name__ == "__main__":
    main()
