from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pokemon_3d_cls.models import ViewTransformerClassifier, build_model


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


def test_view_transformer_forward_backward_and_checkpoint_round_trip(tmp_path: Path) -> None:
    model = ViewTransformerClassifier(
        num_classes=3,
        num_views=4,
        backbone="simple_cnn",
        input_channels=1,
        feature_dim=16,
        pretrained=False,
        classifier_dropout=0.0,
        num_layers=2,
        num_heads=4,
        mlp_dim=32,
        transformer_dropout=0.0,
    )
    views = torch.randn(2, 4, 1, 32, 32)
    logits = model(views)
    logits.sum().backward()

    assert logits.shape == (2, 3)
    assert model.cls_token.grad is not None

    restored = ViewTransformerClassifier(
        num_classes=3,
        num_views=4,
        backbone="simple_cnn",
        input_channels=1,
        feature_dim=16,
        pretrained=False,
        classifier_dropout=0.0,
        num_layers=2,
        num_heads=4,
        mlp_dim=32,
        transformer_dropout=0.0,
    )
    checkpoint_path = tmp_path / "transformer.pt"
    torch.save(model.state_dict(), checkpoint_path)
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(views), restored(views))


def test_view_transformer_rejects_unexpected_view_count() -> None:
    model = ViewTransformerClassifier(
        num_classes=3,
        num_views=4,
        backbone="simple_cnn",
        input_channels=1,
        feature_dim=16,
        pretrained=False,
        num_heads=4,
        mlp_dim=32,
    )

    with pytest.raises(ValueError, match="Expected 4 views"):
        model(torch.randn(2, 3, 1, 32, 32))
