from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pokemon_3d_cls.paths import make_run_id, resolve_project_path, sanitize_slug


def test_resolve_project_path_uses_project_root(tmp_path: Path) -> None:
    resolved = resolve_project_path("data/dataset", tmp_path)

    assert resolved == (tmp_path / "data/dataset").resolve()


def test_make_run_id_is_stable_with_given_time() -> None:
    run_id = make_run_id("baseline condition", 7, now=datetime(2026, 6, 30, 1, 2, 3, tzinfo=UTC))

    assert run_id == "baseline-condition_seed7_20260630T010203Z"


def test_sanitize_slug_fallback() -> None:
    assert sanitize_slug("///") == "run"
