"""環境診断レポート。"""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path

import torch
import torchvision
from torch import version as torch_version

from pokemon_3d_cls.io import write_json
from pokemon_3d_cls.paths import ensure_directory
from pokemon_3d_cls.rendering.pytorch3d_renderer import (
    PyTorch3DRenderer,
    RendererSettings,
    build_tetrahedron,
    is_pytorch3d_available,
)


def collect_environment_report(output_path: Path | None = None) -> dict[str, object]:
    """Python/PyTorch/CUDA/PyTorch3D診断情報を収集する。"""

    report: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "pytorch3d_available": is_pytorch3d_available(),
        "pytorch3d_version": _version_or_none("pytorch3d"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime_version": torch_version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
    }
    render_report = _pytorch3d_smoke()
    report.update(render_report)
    if output_path is not None:
        ensure_directory(output_path.parent)
        write_json(output_path, report)
    return report


def _pytorch3d_smoke() -> dict[str, object]:
    if not is_pytorch3d_available():
        return {
            "pytorch3d_forward_render_ok": False,
            "pytorch3d_camera_gradient_ok": False,
            "pytorch3d_error": "pytorch3d is not installed",
        }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        renderer = PyTorch3DRenderer(RendererSettings(image_size=32), device=device)
        vertices, faces = build_tetrahedron()
        azimuth = torch.tensor([[0.0]], device=device, requires_grad=True)
        elevation = torch.tensor([[0.0]], device=device, requires_grad=True)
        images = renderer.render_batch_views([vertices.to(device)], [faces.to(device)], azimuth, elevation)
        loss = images.mean()
        loss.backward()
        grad_ok = azimuth.grad is not None and torch.isfinite(azimuth.grad).all().item()
        return {
            "pytorch3d_forward_render_ok": True,
            "pytorch3d_camera_gradient_ok": bool(grad_ok),
            "pytorch3d_smoke_image_shape": list(images.shape),
            "pytorch3d_error": None,
        }
    except Exception as exc:
        return {
            "pytorch3d_forward_render_ok": False,
            "pytorch3d_camera_gradient_ok": False,
            "pytorch3d_error": str(exc),
        }


def _version_or_none(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None
