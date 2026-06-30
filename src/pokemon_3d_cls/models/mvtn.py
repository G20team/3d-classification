"""Learned Circular MVTNの視点予測補助。"""

from __future__ import annotations

import torch
import torch.nn as nn


class CircularViewPredictor(nn.Module):
    """mesh頂点から円環カメラの角度補正を予測する小さなPointNet風MLP。"""

    def __init__(
        self,
        *,
        num_views: int,
        point_samples: int,
        hidden_dim: int = 128,
        max_azimuth_offset_deg: float = 45.0,
        max_elevation_offset_deg: float = 25.0,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            msg = "num_views は1以上である必要があります。"
            raise ValueError(msg)
        if point_samples <= 0:
            msg = "point_samples は1以上である必要があります。"
            raise ValueError(msg)
        if hidden_dim <= 0:
            msg = "hidden_dim は1以上である必要があります。"
            raise ValueError(msg)

        self.num_views = num_views
        self.point_samples = point_samples
        self.max_azimuth_offset_deg = max_azimuth_offset_deg
        self.max_elevation_offset_deg = max_elevation_offset_deg
        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_views * 2),
        )
        self._reset_offset_head()

    def forward(
        self,
        vertices: torch.Tensor,
        base_azimuths: torch.Tensor,
        base_elevations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """base角度に学習オフセットを足したazimuth/elevationを返す。"""

        if vertices.ndim != 3 or vertices.shape[-1] != 3:
            msg = "vertices は (B,N,3) のTensorである必要があります。"
            raise ValueError(msg)
        batch_size = vertices.shape[0]
        base_azimuths = _expand_base_angles(base_azimuths, batch_size=batch_size, num_views=self.num_views)
        base_elevations = _expand_base_angles(base_elevations, batch_size=batch_size, num_views=self.num_views)

        point_features = self.point_encoder(vertices)
        global_features = point_features.max(dim=1).values
        raw_offsets = self.offset_head(global_features).reshape(batch_size, self.num_views, 2)
        azimuth_offsets = torch.tanh(raw_offsets[..., 0]) * self.max_azimuth_offset_deg
        elevation_offsets = torch.tanh(raw_offsets[..., 1]) * self.max_elevation_offset_deg
        offsets = torch.stack((azimuth_offsets, elevation_offsets), dim=-1)
        azimuths = base_azimuths + azimuth_offsets
        elevations = torch.clamp(base_elevations + elevation_offsets, min=-89.0, max=89.0)
        return azimuths, elevations, offsets

    def _reset_offset_head(self) -> None:
        final_layer = self.offset_head[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.zeros_(final_layer.weight)
            nn.init.zeros_(final_layer.bias)


def pack_vertices_for_mvtn(
    vertices_list: list[torch.Tensor],
    *,
    point_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """可変長頂点listをMVTN入力用の (B,point_samples,3) へ揃える。"""

    if point_samples <= 0:
        msg = "point_samples は1以上である必要があります。"
        raise ValueError(msg)
    if not vertices_list:
        msg = "vertices_list は空にできません。"
        raise ValueError(msg)

    packed = [_sample_vertices(vertices, point_samples=point_samples, device=device) for vertices in vertices_list]
    return torch.stack(packed, dim=0)


def camera_statistics(azimuths: torch.Tensor, elevations: torch.Tensor) -> dict[str, float]:
    """カメラ角度の分散と視点間距離を集計する。"""

    azimuths, elevations = _as_angle_batch(azimuths, elevations)
    distances = _pairwise_angle_distances(azimuths, elevations)
    if distances.numel() == 0:
        distance_min = 0.0
        distance_mean = 0.0
        distance_max = 0.0
    else:
        detached = distances.detach()
        distance_min = float(detached.min().item())
        distance_mean = float(detached.mean().item())
        distance_max = float(detached.max().item())
    return {
        "azimuth_mean": float(azimuths.detach().mean().item()),
        "azimuth_std": float(azimuths.detach().std(unbiased=False).item()),
        "elevation_mean": float(elevations.detach().mean().item()),
        "elevation_std": float(elevations.detach().std(unbiased=False).item()),
        "pairwise_distance_min": distance_min,
        "pairwise_distance_mean": distance_mean,
        "pairwise_distance_max": distance_max,
    }


def detect_view_collapse(azimuths: torch.Tensor, elevations: torch.Tensor, *, threshold_deg: float) -> bool:
    """最小視点間距離が閾値未満ならview collapseとして扱う。"""

    if threshold_deg <= 0.0:
        msg = "threshold_deg は0より大きい値である必要があります。"
        raise ValueError(msg)
    azimuths, elevations = _as_angle_batch(azimuths, elevations)
    distances = _pairwise_angle_distances(azimuths, elevations)
    if distances.numel() == 0:
        return False
    return bool((distances.detach() < threshold_deg).any().item())


def _sample_vertices(vertices: torch.Tensor, *, point_samples: int, device: torch.device) -> torch.Tensor:
    if vertices.ndim != 2 or vertices.shape[-1] != 3:
        msg = "各verticesは (N,3) のTensorである必要があります。"
        raise ValueError(msg)
    if vertices.shape[0] == 0:
        msg = "空のverticesはMVTNへ入力できません。"
        raise ValueError(msg)

    vertices = vertices.to(device=device, dtype=torch.float32)
    if vertices.shape[0] == point_samples:
        return vertices
    indices = torch.linspace(0, vertices.shape[0] - 1, steps=point_samples, device=device).round().long()
    return vertices.index_select(0, indices)


def _expand_base_angles(angles: torch.Tensor, *, batch_size: int, num_views: int) -> torch.Tensor:
    if angles.ndim == 1:
        if angles.shape[0] != num_views:
            msg = "base angleの視点数がnum_viewsと一致していません。"
            raise ValueError(msg)
        return angles.unsqueeze(0).expand(batch_size, -1)
    if angles.ndim == 2:
        if angles.shape != (batch_size, num_views):
            msg = "batch付きbase angleのshapeが (B,V) と一致していません。"
            raise ValueError(msg)
        return angles

    msg = "base angleは (V) または (B,V) のTensorである必要があります。"
    raise ValueError(msg)


def _as_angle_batch(azimuths: torch.Tensor, elevations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if azimuths.ndim == 1:
        azimuths = azimuths.unsqueeze(0)
    if elevations.ndim == 1:
        elevations = elevations.unsqueeze(0)
    if azimuths.shape != elevations.shape or azimuths.ndim != 2:
        msg = "azimuths/elevations は同じshapeの (V) または (B,V) Tensorである必要があります。"
        raise ValueError(msg)
    return azimuths, elevations


def _pairwise_angle_distances(azimuths: torch.Tensor, elevations: torch.Tensor) -> torch.Tensor:
    num_views = azimuths.shape[1]
    if num_views < 2:
        return azimuths.new_empty((azimuths.shape[0], 0))
    row_indices, col_indices = torch.triu_indices(num_views, num_views, offset=1, device=azimuths.device)
    azimuth_delta = _wrap_degrees(azimuths[:, row_indices] - azimuths[:, col_indices])
    elevation_delta = elevations[:, row_indices] - elevations[:, col_indices]
    return torch.sqrt(azimuth_delta.square() + elevation_delta.square())


def _wrap_degrees(angles: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angles + 180.0, 360.0) - 180.0
