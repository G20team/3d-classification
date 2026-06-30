from __future__ import annotations

import numpy as np

from pokemon_3d_cls.rendering.glb import make_label, normalize_vertices, viewpoints


def test_make_label_supports_stem_and_species() -> None:
    assert make_label("128-1.glb", "stem") == "128-1"
    assert make_label("128-1.glb", "species") == "128"


def test_viewpoints_are_deterministic_with_rng() -> None:
    rng = np.random.default_rng(0)
    points = viewpoints("quiz", 3, base_azimuth=5.0, rng=rng)

    assert len(points) == 3
    assert points[0] == (pytest_approx(98.287), pytest_approx(4.856))


def test_normalize_vertices_centers_and_scales() -> None:
    vertices = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]])
    normalized = normalize_vertices(vertices)

    assert np.allclose(normalized.mean(axis=0), [0.0, 0.0, 0.0])
    assert np.isclose(np.abs(normalized).max(), 1.0)


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, abs=1e-3)
