"""Fixed and differentiable rendering with PyTorch3D."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RendererSettings:
    """PyTorch3D renderer settings."""

    image_size: int = 224
    camera_distance: float = 2.7
    background_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    mesh_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    raster_bin_size: int = 0


def is_pytorch3d_available() -> bool:
    """Return whether PyTorch3D can be imported."""

    return importlib.util.find_spec("pytorch3d") is not None


class PyTorch3DRenderer:
    """Thin wrapper around PyTorch3D MeshRenderer."""

    def __init__(self, settings: RendererSettings, *, device: torch.device) -> None:
        if not is_pytorch3d_available():
            msg = "PyTorch3D was not found. Install it with a compatible wheel or source build."
            raise RuntimeError(msg)
        self.settings = settings
        self.device = device
        self.renderer_mod = importlib.import_module("pytorch3d.renderer")
        self.structures_mod = importlib.import_module("pytorch3d.structures")
        self._renderer = self._build_renderer()

    def render_batch_views(
        self,
        vertices_list: list[torch.Tensor],
        faces_list: list[torch.Tensor],
        azimuths: torch.Tensor,
        elevations: torch.Tensor,
    ) -> torch.Tensor:
        """Render a variable-length mesh batch into RGB images shaped (B,V,3,H,W)."""

        if azimuths.ndim == 1:
            azimuths = azimuths.unsqueeze(0).expand(len(vertices_list), -1)
        if elevations.ndim == 1:
            elevations = elevations.unsqueeze(0).expand(len(vertices_list), -1)
        batch_size, num_views = azimuths.shape
        if batch_size != len(vertices_list) or batch_size != len(faces_list):
            msg = "The number of meshes does not match the camera-angle batch size."
            raise ValueError(msg)

        verts_flat: list[torch.Tensor] = []
        faces_flat: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            for _view_index in range(num_views):
                verts_flat.append(vertices_list[batch_index].to(self.device, dtype=torch.float32))
                faces_flat.append(faces_list[batch_index].to(self.device, dtype=torch.int64))

        meshes = self._make_meshes(verts_flat, faces_flat)
        azimuth_flat = azimuths.reshape(-1).to(self.device, dtype=torch.float32)
        elevation_flat = elevations.reshape(-1).to(self.device, dtype=torch.float32)
        cameras = self._make_cameras(azimuth_flat, elevation_flat)
        images = self._renderer(meshes, cameras=cameras)
        rgb = images[..., :3].permute(0, 3, 1, 2).contiguous()
        return rgb.reshape(batch_size, num_views, 3, self.settings.image_size, self.settings.image_size)

    def _build_renderer(self) -> Any:
        RasterizationSettings = self.renderer_mod.RasterizationSettings
        MeshRasterizer = self.renderer_mod.MeshRasterizer
        MeshRenderer = self.renderer_mod.MeshRenderer
        SoftPhongShader = self.renderer_mod.SoftPhongShader
        PointLights = self.renderer_mod.PointLights
        raster_settings = RasterizationSettings(
            image_size=self.settings.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=self.settings.raster_bin_size,
        )
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 3.0]])
        shader = SoftPhongShader(
            device=self.device,
            lights=lights,
            blend_params=self.renderer_mod.BlendParams(background_color=self.settings.background_color),
        )
        return MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=shader,
        )

    def _make_cameras(self, azimuths: torch.Tensor, elevations: torch.Tensor) -> Any:
        look_at_view_transform = self.renderer_mod.look_at_view_transform
        FoVPerspectiveCameras = self.renderer_mod.FoVPerspectiveCameras
        rotation, translation = look_at_view_transform(
            dist=self.settings.camera_distance,
            elev=elevations,
            azim=azimuths,
            device=self.device,
        )
        return FoVPerspectiveCameras(device=self.device, R=rotation, T=translation)

    def _make_meshes(self, vertices_list: list[torch.Tensor], faces_list: list[torch.Tensor]) -> Any:
        Meshes = self.structures_mod.Meshes
        TexturesVertex = self.renderer_mod.TexturesVertex
        color = torch.tensor(self.settings.mesh_color, device=self.device, dtype=torch.float32)
        textures = [color.expand(vertices.shape[0], 3) for vertices in vertices_list]
        return Meshes(
            verts=vertices_list,
            faces=faces_list,
            textures=TexturesVertex(verts_features=textures),
        )


def build_tetrahedron() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a minimal mesh for environment diagnostics."""

    vertices = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2],
        ],
        dtype=torch.int64,
    )
    return vertices, faces
