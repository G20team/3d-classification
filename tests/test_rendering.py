from __future__ import annotations

import numpy as np

from pokemon_3d_cls.experiments.silhouette_rendering import render_silhouette
from pokemon_3d_cls.rendering.glb import make_label, normalize_vertices, viewpoints
from pokemon_3d_cls.rendering.pytorch3d_renderer import RendererSettings


def test_renderer_settings_use_naive_rasterization_by_default() -> None:
    settings = RendererSettings()

    assert settings.raster_bin_size == 0


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


def test_render_silhouette_returns_grayscale_uint8_image() -> None:
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])
    faces = np.array([[0, 1, 2]])

    image = render_silhouette(vertices, faces, azimuth=0.0, elevation=0.0, resolution=16, supersample=1)

    assert image.shape == (16, 16)
    assert image.dtype == np.uint8


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value, abs=1e-3)
