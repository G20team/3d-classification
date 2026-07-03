"""Helpers for loading GLB assets."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import cast

import DracoPy
import numpy as np
import trimesh

JSON_CHUNK_TYPE = 0x4E4F534A
BIN_CHUNK_TYPE = 0x004E4942
DRACO_EXTENSION = "KHR_draco_mesh_compression"


def load_draco_glb_mesh(path: Path) -> trimesh.Trimesh:
    """Decode a GLB with KHR_draco_mesh_compression into a Trimesh."""

    gltf, bin_chunk = _read_glb(path)
    primitives: list[tuple[np.ndarray, np.ndarray]] = []
    for transform, mesh_index in _iter_scene_mesh_nodes(gltf):
        meshes = cast("list[dict[str, object]]", gltf.get("meshes", []))
        mesh = meshes[mesh_index]
        for primitive in cast("list[dict[str, object]]", mesh.get("primitives", [])):
            decoded = _decode_draco_primitive(primitive, gltf, bin_chunk)
            if decoded is None:
                continue
            vertices, faces = decoded
            vertices = _apply_transform(vertices, transform)
            primitives.append((vertices, faces))
    if not primitives:
        msg = "No Draco-compressed primitive was found."
        raise ValueError(msg)

    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    offset = 0
    for vertices, faces in primitives:
        vertices_parts.append(vertices)
        faces_parts.append(faces + offset)
        offset += int(vertices.shape[0])
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices_parts, axis=0),
        faces=np.concatenate(faces_parts, axis=0),
        process=False,
    )


def glb_has_texture(path: Path) -> bool:
    """Check whether GLB JSON contains texture references."""

    try:
        gltf, _ = _read_glb(path)
    except Exception:
        return False
    if gltf.get("textures") or gltf.get("images"):
        return True
    for material in cast("list[dict[str, object]]", gltf.get("materials", [])):
        if _contains_texture_info(material):
            return True
    return False


def _read_glb(path: Path) -> tuple[dict[str, object], bytes]:
    data = path.read_bytes()
    if len(data) < 20:
        msg = "GLB file is too short."
        raise ValueError(msg)
    magic, version, declared_length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        msg = "GLB is not version 2."
        raise ValueError(msg)
    if declared_length > len(data):
        msg = "Declared GLB size exceeds the actual file size."
        raise ValueError(msg)

    offset = 12
    gltf: dict[str, object] | None = None
    bin_chunk: bytes | None = None
    while offset + 8 <= declared_length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK_TYPE:
            gltf = cast("dict[str, object]", json.loads(chunk.decode("utf-8")))
        elif chunk_type == BIN_CHUNK_TYPE:
            bin_chunk = chunk
    if gltf is None or bin_chunk is None:
        msg = "GLB does not contain a JSON chunk or BIN chunk."
        raise ValueError(msg)
    return gltf, bin_chunk


def _decode_draco_primitive(
    primitive: dict[str, object],
    gltf: dict[str, object],
    bin_chunk: bytes,
) -> tuple[np.ndarray, np.ndarray] | None:
    extensions = cast("dict[str, object]", primitive.get("extensions", {}))
    draco = cast("dict[str, object] | None", extensions.get(DRACO_EXTENSION))
    if draco is None:
        return None

    buffer_view_index = _as_int(draco["bufferView"], "bufferView")
    buffer_views = cast("list[dict[str, object]]", gltf["bufferViews"])
    buffer_view = buffer_views[buffer_view_index]
    byte_offset = _as_int(buffer_view.get("byteOffset", 0), "byteOffset")
    byte_length = _as_int(buffer_view["byteLength"], "byteLength")
    decoded = DracoPy.decode(bin_chunk[byte_offset : byte_offset + byte_length])
    if not hasattr(decoded, "faces"):
        msg = "Decoded a point cloud instead of a Draco mesh."
        raise ValueError(msg)

    attributes = cast("dict[str, int]", draco["attributes"])
    position_attribute = decoded.get_attribute_by_unique_id(int(attributes["POSITION"]))
    vertices = np.asarray(position_attribute["data"], dtype=np.float32)
    faces = np.asarray(decoded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        msg = "POSITION attribute is not 3D vertices."
        raise ValueError(msg)
    if faces.ndim != 2 or faces.shape[1] != 3:
        msg = "Draco mesh faces are not triangles."
        raise ValueError(msg)
    return vertices, faces


def _iter_scene_mesh_nodes(gltf: dict[str, object]) -> list[tuple[np.ndarray, int]]:
    nodes = cast("list[dict[str, object]]", gltf.get("nodes", []))
    if not nodes:
        return []

    scene_roots = _scene_roots(gltf)
    results: list[tuple[np.ndarray, int]] = []
    visited: set[int] = set()
    for node_index in scene_roots:
        _collect_mesh_nodes(nodes, node_index, np.eye(4, dtype=np.float32), visited, results)
    return results


def _scene_roots(gltf: dict[str, object]) -> list[int]:
    scenes = cast("list[dict[str, object]]", gltf.get("scenes", []))
    if scenes:
        scene_index = _as_int(gltf.get("scene", 0), "scene")
        scene = scenes[scene_index]
        return [int(index) for index in cast("list[int]", scene.get("nodes", []))]
    nodes = cast("list[dict[str, object]]", gltf.get("nodes", []))
    child_indices = {
        int(child)
        for node in nodes
        for child in cast("list[int]", cast("dict[str, object]", node).get("children", []))
    }
    return [index for index in range(len(nodes)) if index not in child_indices]


def _collect_mesh_nodes(
    nodes: list[dict[str, object]],
    node_index: int,
    parent_transform: np.ndarray,
    visited: set[int],
    results: list[tuple[np.ndarray, int]],
) -> None:
    if node_index in visited:
        return
    visited.add(node_index)
    node = nodes[node_index]
    transform = parent_transform @ _node_transform(node)
    if "mesh" in node:
        results.append((transform, _as_int(node["mesh"], "mesh")))
    for child_index in cast("list[int]", node.get("children", [])):
        _collect_mesh_nodes(nodes, int(child_index), transform, visited, results)


def _node_transform(node: dict[str, object]) -> np.ndarray:
    if "matrix" in node:
        matrix = np.asarray(cast("list[float]", node["matrix"]), dtype=np.float32).reshape((4, 4)).T
        return matrix

    translation = np.asarray(cast("list[float]", node.get("translation", [0.0, 0.0, 0.0])), dtype=np.float32)
    rotation = np.asarray(cast("list[float]", node.get("rotation", [0.0, 0.0, 0.0, 1.0])), dtype=np.float32)
    scale = np.asarray(cast("list[float]", node.get("scale", [1.0, 1.0, 1.0])), dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = translation
    transform[:3, :3] = _quaternion_to_matrix(rotation) @ np.diag(scale)
    return transform


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(value) for value in quaternion]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _apply_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [vertices.astype(np.float32), np.ones((vertices.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    transformed = homogeneous @ transform.T
    return transformed[:, :3].astype(np.float32)


def _contains_texture_info(value: object) -> bool:
    if isinstance(value, dict):
        return any(key.endswith("Texture") or _contains_texture_info(child) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_texture_info(item) for item in value)
    return False


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{name} must be an integer."
        raise ValueError(msg)
    return value
