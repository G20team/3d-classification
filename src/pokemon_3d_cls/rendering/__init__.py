"""GLBからシルエット画像を生成する処理。"""

from __future__ import annotations

from pokemon_3d_cls.rendering.glb import (
    load_mesh,
    make_label,
    normalize_vertices,
    render_silhouette,
    viewpoints,
)

__all__ = [
    "load_mesh",
    "make_label",
    "normalize_vertices",
    "render_silhouette",
    "viewpoints",
]
