from __future__ import annotations

import torch

from pokemon_3d_cls.models import CircularViewPredictor, camera_statistics, detect_view_collapse


def test_circular_view_predictor_allows_gradient_flow() -> None:
    predictor = CircularViewPredictor(num_views=4, point_samples=8, hidden_dim=16)
    vertices = torch.randn(2, 8, 3)
    base_azimuths = torch.tensor([0.0, 90.0, 180.0, 270.0])
    base_elevations = torch.zeros(4)

    azimuths, elevations, offsets = predictor(vertices, base_azimuths, base_elevations)
    loss = azimuths.mean() + elevations.mean() + offsets.square().mean()
    loss.backward()

    assert azimuths.shape == (2, 4)
    assert elevations.shape == (2, 4)
    assert any(parameter.grad is not None for parameter in predictor.parameters())


def test_camera_statistics_and_collapse_detection() -> None:
    azimuths = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    elevations = torch.zeros_like(azimuths)
    stats = camera_statistics(azimuths, elevations)

    assert stats["pairwise_distance_min"] == 1.0
    assert detect_view_collapse(azimuths, elevations, threshold_deg=5.0)
