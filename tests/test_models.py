from __future__ import annotations

import torch

from pokemon_3d_cls.models import build_model


def test_mvcnn_forward_shapes_for_multi_and_single_view() -> None:
    model = build_model(
        num_classes=3,
        backbone="simple_cnn",
        input_channels=1,
        feature_dim=16,
        pretrained=False,
        dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        multi_view_logits = model(torch.randn(2, 4, 1, 32, 32))
        single_view_logits = model(torch.randn(1, 1, 1, 32, 32))

    assert multi_view_logits.shape == (2, 3)
    assert single_view_logits.shape == (1, 3)
    assert model.num_classes == 3
