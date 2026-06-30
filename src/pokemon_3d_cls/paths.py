"""プロジェクト内のpathを一元管理する補助関数。"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """`pyproject.toml` を手がかりにプロジェクトrootを見つける。"""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    msg = f"pyproject.toml が見つかりません: start={current}"
    raise FileNotFoundError(msg)


def resolve_project_path(path_text: str | Path, project_root: Path) -> Path:
    """相対pathをproject root基準の絶対pathへ変換する。"""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_directory(path: Path) -> Path:
    """ディレクトリを作成してpathを返す。"""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_slug(value: str) -> str:
    """run_idやファイル名に使いやすいASCII slugへ変換する。"""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "run"


def make_run_id(condition_id: str, seed: int, now: datetime | None = None) -> str:
    """条件ID、seed、UTC時刻からrun_idを作る。"""

    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return sanitize_slug(f"{condition_id}_seed{seed}_{timestamp}")
