"""Helpers for centralized project path handling."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the project root using `pyproject.toml` as the marker."""

    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    msg = f"pyproject.toml was not found: start={current}"
    raise FileNotFoundError(msg)


def resolve_project_path(path_text: str | Path, project_root: Path) -> Path:
    """Convert a relative path to an absolute path under the project root."""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_directory(path: Path) -> Path:
    """Create a directory and return its path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_slug(value: str) -> str:
    """Convert text to an ASCII slug suitable for run IDs and file names."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "run"


def make_run_id(condition_id: str, seed: int, now: datetime | None = None) -> str:
    """Build a run ID from the condition ID, seed, and UTC time."""

    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return sanitize_slug(f"{condition_id}_seed{seed}_{timestamp}")
