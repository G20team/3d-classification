"""PyTorch3D互換mesh cacheのための正規化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import trimesh


@dataclass(frozen=True)
class NormalizedMesh:
    """正規化済みmesh。"""

    vertices: torch.Tensor
    faces: torch.Tensor
    center: tuple[float, float, float]
    scale: float


def normalize_trimesh(mesh: trimesh.Trimesh) -> NormalizedMesh:
    """bbox中心化とunit sphere相当のscale正規化を行う。"""

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(_non_degenerate_face_mask(mesh))
    mesh.remove_unreferenced_vertices()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        msg = "正規化対象meshが空です。"
        raise ValueError(msg)
    if not np.isfinite(vertices).all():
        msg = "mesh頂点にNaN/Infが含まれています。"
        raise ValueError(msg)

    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    center = (min_bounds + max_bounds) / 2.0
    centered = vertices - center
    radius = float(np.linalg.norm(centered, axis=1).max())
    if radius <= 0:
        msg = "mesh scaleが0です。"
        raise ValueError(msg)
    normalized = centered / radius
    return NormalizedMesh(
        vertices=torch.from_numpy(normalized.astype(np.float32)),
        faces=torch.from_numpy(faces.astype(np.int64)),
        center=(float(center[0]), float(center[1]), float(center[2])),
        scale=radius,
    )


def _non_degenerate_face_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    faces = np.asarray(mesh.faces)
    if faces.size == 0:
        return np.zeros((0,), dtype=bool)
    vertices = np.asarray(mesh.vertices)
    tri = vertices[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) * 0.5
    return np.isfinite(areas) & (areas > 1e-12)
