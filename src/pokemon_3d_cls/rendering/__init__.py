"""Utilities for generating silhouette images from GLB files."""

from __future__ import annotations

from pokemon_3d_cls.rendering.glb import (
    load_mesh,
    make_label,
    normalize_vertices,
    render_silhouette,
    viewpoints,
)
from pokemon_3d_cls.rendering.pytorch3d_renderer import (
    PyTorch3DRenderer,
    RendererSettings,
    is_pytorch3d_available,
)

__all__ = [
    "PyTorch3DRenderer",
    "RendererSettings",
    "is_pytorch3d_available",
    "load_mesh",
    "make_label",
    "normalize_vertices",
    "render_silhouette",
    "viewpoints",
]
