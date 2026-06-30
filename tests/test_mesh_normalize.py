from __future__ import annotations

import numpy as np
import trimesh

from pokemon_3d_cls.mesh.normalize import normalize_trimesh


def test_normalize_trimesh_centers_and_scales() -> None:
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=int),
        process=False,
    )

    normalized = normalize_trimesh(mesh)

    assert normalized.vertices.shape[1] == 3
    assert normalized.faces.shape[1] == 3
    assert float(normalized.vertices.norm(dim=1).max()) <= 1.00001
