"""Filled-silhouette rendering from mesh caches."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pokemon_3d_cls_matplotlib"))

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402


class _RgbaCanvas(Protocol):
    def draw(self) -> object:
        """Draw the matplotlib canvas."""
        ...

    def buffer_rgba(self) -> memoryview:
        """Return the RGBA buffer."""
        ...


def render_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    azimuth: float,
    elevation: float,
    *,
    resolution: int = 224,
    supersample: int = 2,
    pad: float = 1.25,
    line_width: float = 0.4,
) -> np.ndarray:
    """Return a one-view filled silhouette image as a uint8 grayscale array."""

    rotation = _rotation_x(np.radians(elevation)) @ _rotation_y(np.radians(azimuth))
    projected = vertices @ rotation.T
    triangles = projected[:, [0, 1]][faces]

    canvas_size = resolution * supersample
    fig = plt.figure(figsize=(canvas_size / 100, canvas_size / 100), dpi=100)
    axis = fig.add_axes((0, 0, 1, 1))
    axis.set_xlim(-pad, pad)
    axis.set_ylim(-pad, pad)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor("white")
    collection = PolyCollection(triangles.tolist(), facecolors="black", edgecolors="black", linewidths=line_width)
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


def _rotation_y(angle: float) -> np.ndarray:
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    return np.array([[cos_value, 0, sin_value], [0, 1, 0], [-sin_value, 0, cos_value]])


def _rotation_x(angle: float) -> np.ndarray:
    cos_value, sin_value = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, cos_value, -sin_value], [0, sin_value, cos_value]])
