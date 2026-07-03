"""Mesh normalization and caching."""

from __future__ import annotations

from pokemon_3d_cls.mesh.cache import MeshCacheRecord, prepare_mesh_cache
from pokemon_3d_cls.mesh.normalize import NormalizedMesh, normalize_trimesh

__all__ = ["MeshCacheRecord", "NormalizedMesh", "normalize_trimesh", "prepare_mesh_cache"]
