from __future__ import annotations

from pokemon_3d_cls.assets.audit import (
    AssetAuditRecord,
    extract_pokemon_id_from_path,
    is_regular_candidate,
    summarize_audit,
)


def test_extract_pokemon_id_from_path() -> None:
    assert extract_pokemon_id_from_path("regular/0025_pikachu/model.glb") == 25
    assert extract_pokemon_id_from_path("regular/pikachu.glb") is None


def test_is_regular_candidate_rejects_special_forms() -> None:
    assert is_regular_candidate("regular/0025_pikachu/model.glb", "regular")
    assert not is_regular_candidate("regular/0006_charizard_mega/model.glb", "regular")
    assert not is_regular_candidate("shiny/0025_pikachu/model.glb", "shiny")


def test_summarize_audit_counts_reasons() -> None:
    records = [
        AssetAuditRecord("a.glb", "a.glb", "regular", 1, "bulbasaur", 10, 8, False, True, None),
        AssetAuditRecord("b.glb", "b.glb", "regular", None, None, 0, 0, False, False, "pokemon_id_not_found"),
    ]
    summary = summarize_audit(records, [records[0]])

    assert summary["total_assets"] == 2
    assert summary["selected_assets"] == 1
    assert summary["exclude_reasons"] == {"pokemon_id_not_found": 1, "selected": 1}
