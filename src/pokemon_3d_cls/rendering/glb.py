"""GLB parsing and silhouette rendering."""

from __future__ import annotations

import importlib
import json
import os
import struct
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pokemon_3d_cls_matplotlib"))

import matplotlib
import numpy as np
from PIL import Image

from pokemon_3d_cls.config import LabelMode, UpAxis, ViewpointMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

JsonDict = dict[str, object]


class _DecodedDraco(Protocol):
    points: object
    faces: object


class _DracoDecoder(Protocol):
    def decode(self, encoded: bytes) -> _DecodedDraco:
        """Minimal Protocol matching the shape of DracoPy.decode."""
        ...


class _RgbaCanvas(Protocol):
    def draw(self) -> object:
        """Draw the matplotlib canvas."""
        ...

    def buffer_rgba(self) -> memoryview:
        """Return the RGBA buffer."""
        ...


_COMPONENT_DTYPE: dict[int, type[np.generic]] = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

try:
    _DRACO_MODULE: ModuleType | None = importlib.import_module("DracoPy")
except ModuleNotFoundError:
    _DRACO_MODULE = None


def parse_glb(path: str | Path) -> tuple[JsonDict, bytes]:
    """Split a GLB binary into a JSON dictionary and BIN blob."""

    glb_path = Path(path)
    data = glb_path.read_bytes()
    if len(data) < 12:
        msg = f"{glb_path}: GLB header is too short."
        raise ValueError(msg)

    magic, _version, length = struct.unpack("<III", data[:12])
    if magic != 0x46546C67:
        msg = f"{glb_path}: File is not a glTF binary."
        raise ValueError(msg)
    if length > len(data):
        msg = f"{glb_path}: GLB length exceeds the file size."
        raise ValueError(msg)

    offset = 12
    gltf: JsonDict | None = None
    bin_blob = b""
    while offset + 8 <= length:
        chunk_length, chunk_type = struct.unpack("<II", data[offset : offset + 8])
        chunk = data[offset + 8 : offset + 8 + chunk_length]
        if chunk_type == 0x4E4F534A:
            raw_json = json.loads(chunk.decode("utf-8"))
            if not isinstance(raw_json, dict):
                msg = f"{glb_path}: JSON chunk root is not an object."
                raise ValueError(msg)
            gltf = cast("JsonDict", raw_json)
        elif chunk_type == 0x004E4942:
            bin_blob = chunk
        offset += 8 + chunk_length

    if gltf is None:
        msg = f"{glb_path}: JSON chunk is missing."
        raise ValueError(msg)
    return gltf, bin_blob


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Combine mesh instances in a GLB and return vertices V and triangle faces F."""

    gltf, bin_blob = parse_glb(path)
    meshes = _list_field(gltf, "meshes")
    instances = _mesh_instances(gltf)
    if not instances:
        instances = [(index, np.eye(4)) for index in range(len(meshes))]

    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_index, matrix in instances:
        mesh = _dict_at(meshes, mesh_index, "mesh")
        for primitive in _list_field(mesh, "primitives"):
            primitive_mapping = _as_mapping(primitive, "primitive")
            vertices, faces = _decode_primitive(gltf, bin_blob, primitive_mapping)
            homogeneous = np.c_[vertices, np.ones(len(vertices))] @ matrix.T
            transformed_vertices = homogeneous[:, :3]
            all_vertices.append(transformed_vertices)
            all_faces.append(faces + vertex_offset)
            vertex_offset += len(transformed_vertices)

    if not all_vertices:
        msg = f"{path}: No mesh was found."
        raise ValueError(msg)
    return np.vstack(all_vertices), np.vstack(all_faces)


def normalize_vertices(vertices: np.ndarray, up: UpAxis = "y") -> np.ndarray:
    """Center vertices at the origin and normalize by maximum radius 1."""

    normalized = vertices.copy()
    if up == "z":
        normalized = normalized[:, [0, 2, 1]]
        normalized[:, 2] *= -1
    center = (normalized.max(0) + normalized.min(0)) / 2.0
    normalized = normalized - center
    radius = float(np.abs(normalized).max())
    if radius > 0:
        normalized = normalized / radius
    return normalized


def viewpoints(
    mode: ViewpointMode,
    count: int,
    *,
    base_azimuth: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Return a list of (azimuth_deg, elevation_deg) pairs."""

    if count <= 0:
        msg = "viewpoint count must be at least 1."
        raise ValueError(msg)
    if rng is None:
        rng = np.random.default_rng(0)
    points: list[tuple[float, float]] = []
    if mode == "turntable":
        for index in range(count):
            points.append((base_azimuth + 360.0 * index / count, 10.0))
    elif mode == "sphere":
        golden_angle = np.pi * (3 - np.sqrt(5))
        for index in range(count):
            y = 1 - 2 * (index + 0.5) / count
            elevation = np.degrees(np.arcsin(y))
            azimuth = np.degrees((index * golden_angle) % (2 * np.pi))
            points.append((base_azimuth + float(azimuth), float(elevation)))
    else:
        anchors = [90, -90, 45, -45, 135, -135, 60, -60]
        for index in range(count):
            anchor = anchors[index % len(anchors)]
            azimuth = base_azimuth + anchor + float(rng.uniform(-12, 12))
            elevation = float(rng.uniform(0, 18))
            points.append((azimuth, elevation))
    return points


def render_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth: float,
    elevation: float,
    *,
    resolution: int = 256,
    supersample: int = 2,
    invert: bool = False,
    pad: float = 1.25,
    line_width: float = 0.4,
) -> np.ndarray:
    """Return a one-view silhouette image as a uint8 grayscale array."""

    rotation = _rotation_x(np.radians(elevation)) @ _rotation_y(np.radians(azimuth))
    projected = vertices @ rotation.T
    triangles = projected[:, [0, 1]][faces]

    foreground, background = ("white", "black") if invert else ("black", "white")
    canvas_size = resolution * supersample
    fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100), dpi=100)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(-pad, pad)
    axis.set_ylim(-pad, pad)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor(background)
    collection = PolyCollection(triangles.tolist(), facecolors=foreground, edgecolors=foreground, linewidths=line_width)
    axis.add_collection(collection)
    canvas = cast("_RgbaCanvas", fig.canvas)
    canvas.draw()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buffer = buffer.reshape(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), 4)
    plt.close(fig)

    image = Image.fromarray(buffer[:, :, :3]).convert("L")
    if supersample != 1:
        image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.asarray(image)


def make_label(path: str | Path, mode: LabelMode) -> str:
    """Create a label from the GLB file name."""

    stem = Path(path).stem
    if mode == "stem":
        return stem
    if mode == "species":
        return stem.split("-")[0].split("_")[0]
    msg = f"Unknown label mode: {mode}"
    raise ValueError(msg)


def _decode_primitive(
    gltf: Mapping[str, object],
    bin_blob: bytes,
    primitive: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    draco = _draco_extension(primitive)
    if draco is not None:
        encoded = _buffer_view_bytes(gltf, bin_blob, _int_field(draco, "bufferView"))
        return _decode_draco(encoded)

    attributes = _mapping_field(primitive, "attributes")
    vertices = _read_accessor(gltf, bin_blob, _int_field(attributes, "POSITION")).astype(np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        msg = "POSITION accessor must have shape (N, 3)."
        raise ValueError(msg)

    if "indices" in primitive:
        faces = _read_accessor(gltf, bin_blob, _int_field(primitive, "indices")).astype(np.int64).reshape(-1, 3)
    else:
        if len(vertices) % 3 != 0:
            msg = "Primitive without indices has a vertex count that is not divisible by 3."
            raise ValueError(msg)
        faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _decode_draco(encoded: bytes) -> tuple[np.ndarray, np.ndarray]:
    if _DRACO_MODULE is None:
        msg = "This is a Draco-compressed model, but DracoPy is unavailable. Run again after `uv sync`."
        raise RuntimeError(msg)
    decoder = cast("_DracoDecoder", _DRACO_MODULE)
    decoded = decoder.decode(encoded)
    vertices = np.asarray(decoded.points, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(decoded.faces, dtype=np.int64).reshape(-1, 3)
    return vertices, faces


def _read_accessor(gltf: Mapping[str, object], bin_blob: bytes, accessor_index: int) -> np.ndarray:
    accessor = _dict_at(_list_field(gltf, "accessors"), accessor_index, "accessor")
    if "bufferView" not in accessor:
        msg = f"Accessors without bufferView are not supported: index={accessor_index}"
        raise ValueError(msg)
    buffer_view = _dict_at(_list_field(gltf, "bufferViews"), _int_field(accessor, "bufferView"), "bufferView")
    component_type = _int_field(accessor, "componentType")
    if component_type not in _COMPONENT_DTYPE:
        msg = f"Unsupported componentType: {component_type}"
        raise ValueError(msg)
    accessor_type = _str_field(accessor, "type")
    if accessor_type not in _TYPE_NCOMP:
        msg = f"Unsupported accessor type: {accessor_type}"
        raise ValueError(msg)

    dtype = np.dtype(_COMPONENT_DTYPE[component_type])
    component_count = _TYPE_NCOMP[accessor_type]
    count = _int_field(accessor, "count")
    start = _optional_int(buffer_view, "byteOffset", 0) + _optional_int(accessor, "byteOffset", 0)
    item_size = dtype.itemsize * component_count
    stride = _optional_int(buffer_view, "byteStride", item_size)

    if stride == item_size:
        array = np.frombuffer(bin_blob, dtype=dtype, count=count * component_count, offset=start)
        shaped = array.reshape(count, component_count)
    else:
        shaped = np.ndarray(
            shape=(count, component_count),
            dtype=dtype,
            buffer=bin_blob,
            offset=start,
            strides=(stride, dtype.itemsize),
        ).copy()

    if component_count == 1:
        return shaped.reshape(count)
    return shaped


def _buffer_view_bytes(gltf: Mapping[str, object], bin_blob: bytes, buffer_view_index: int) -> bytes:
    buffer_view = _dict_at(_list_field(gltf, "bufferViews"), buffer_view_index, "bufferView")
    start = _optional_int(buffer_view, "byteOffset", 0)
    length = _int_field(buffer_view, "byteLength")
    return bin_blob[start : start + length]


def _mesh_instances(gltf: Mapping[str, object]) -> list[tuple[int, np.ndarray]]:
    nodes = _list_field(gltf, "nodes")
    if not nodes:
        return []
    roots = _scene_roots(gltf, len(nodes))
    instances: list[tuple[int, np.ndarray]] = []

    def visit(node_index: int, parent: np.ndarray) -> None:
        node = _dict_at(nodes, node_index, "node")
        matrix = parent @ _node_local_matrix(node)
        mesh_value = node.get("mesh")
        if isinstance(mesh_value, int):
            instances.append((mesh_value, matrix))
        for child_index in _optional_int_sequence(node, "children"):
            visit(child_index, matrix)

    for root_index in roots:
        visit(root_index, np.eye(4))
    return instances


def _scene_roots(gltf: Mapping[str, object], node_count: int) -> list[int]:
    scenes = _list_field(gltf, "scenes", required=False)
    if not scenes:
        return list(range(node_count))
    scene_index = _optional_int(gltf, "scene", 0)
    scene = _dict_at(scenes, scene_index, "scene")
    roots = _optional_int_sequence(scene, "nodes")
    return roots or list(range(node_count))


def _node_local_matrix(node: Mapping[str, object]) -> np.ndarray:
    matrix_value = node.get("matrix")
    if matrix_value is not None:
        matrix = _number_sequence(matrix_value, "matrix", expected_length=16)
        return np.asarray(matrix, dtype=np.float64).reshape(4, 4).T

    matrix = np.eye(4)
    scale_value = node.get("scale")
    if scale_value is not None:
        scale = _number_sequence(scale_value, "scale", expected_length=3)
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] = np.diag(scale)
        matrix = matrix @ scale_matrix

    rotation_value = node.get("rotation")
    if rotation_value is not None:
        x, y, z, w = _number_sequence(rotation_value, "rotation", expected_length=4)
        rotation_matrix = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        matrix = matrix @ transform

    translation_value = node.get("translation")
    if translation_value is not None:
        translation = _number_sequence(translation_value, "translation", expected_length=3)
        transform = np.eye(4)
        transform[:3, 3] = translation
        matrix = transform @ matrix
    return matrix


def _draco_extension(primitive: Mapping[str, object]) -> Mapping[str, object] | None:
    extensions = primitive.get("extensions")
    if not isinstance(extensions, Mapping):
        return None
    draco = extensions.get("KHR_draco_mesh_compression")
    if draco is None:
        return None
    return _as_mapping(draco, "KHR_draco_mesh_compression")


def _rotation_y(angle: float) -> np.ndarray:
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    return np.array([[cos_value, 0, sin_value], [0, 1, 0], [-sin_value, 0, cos_value]])


def _rotation_x(angle: float) -> np.ndarray:
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, cos_value, -sin_value], [0, sin_value, cos_value]])


def _list_field(mapping: Mapping[str, object], key: str, *, required: bool = True) -> list[object]:
    value = mapping.get(key)
    if value is None and not required:
        return []
    if not isinstance(value, list):
        msg = f"{key} must be a list."
        raise ValueError(msg)
    return value


def _dict_at(items: Sequence[object], index: int, name: str) -> Mapping[str, object]:
    if index < 0 or index >= len(items):
        msg = f"{name}  index is out of range: {index}"
        raise ValueError(msg)
    return _as_mapping(items[index], name)


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping."
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _mapping_field(mapping: Mapping[str, object], key: str) -> Mapping[str, object]:
    return _as_mapping(mapping.get(key), key)


def _int_field(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer."
        raise ValueError(msg)
    return value


def _optional_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{key} must be an integer."
        raise ValueError(msg)
    return value


def _str_field(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        msg = f"{key} must be a string."
        raise ValueError(msg)
    return value


def _optional_int_sequence(mapping: Mapping[str, object], key: str) -> list[int]:
    value = mapping.get(key, [])
    if not isinstance(value, Sequence) or isinstance(value, str):
        msg = f"{key} must be a list of integers."
        raise ValueError(msg)
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            msg = f"{key} must be a list of integers."
            raise ValueError(msg)
        result.append(item)
    return result


def _number_sequence(value: object, name: str, *, expected_length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != expected_length:
        msg = f"{name} must be a numeric list of length {expected_length}."
        raise ValueError(msg)
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            msg = f"{name} must be a list of numbers."
            raise ValueError(msg)
        result.append(float(item))
    return result
