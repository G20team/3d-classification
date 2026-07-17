"""Classification models and MVTN helper APIs."""

from __future__ import annotations

from pokemon_3d_cls.models.camera import fixed_camera_angles
from pokemon_3d_cls.models.mvcnn import MVCNN, MVCNNClassifier, build_model
from pokemon_3d_cls.models.mvtn import (
    CircularViewPredictor,
    camera_statistics,
    detect_view_collapse,
    pack_vertices_for_mvtn,
)
from pokemon_3d_cls.models.view_transformer import ViewTransformerClassifier


def build_experiment_classifier(
    *,
    experiment_kind: str,
    num_classes: int,
    num_views: int,
    backbone: str,
    input_channels: int,
    feature_dim: int,
    pretrained: bool,
    dropout: float,
    transformer_num_layers: int = 2,
    transformer_num_heads: int = 8,
    transformer_mlp_dim: int = 2048,
    transformer_dropout: float = 0.1,
) -> MVCNNClassifier | ViewTransformerClassifier:
    """Build the classifier selected by an experiment configuration."""

    if experiment_kind == "view_transformer4":
        return ViewTransformerClassifier(
            num_classes=num_classes,
            num_views=num_views,
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
            classifier_dropout=dropout,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            mlp_dim=transformer_mlp_dim,
            transformer_dropout=transformer_dropout,
        )
    return MVCNNClassifier(
        num_classes=num_classes,
        backbone=backbone,
        input_channels=input_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        dropout=dropout,
    )

__all__ = [
    "CircularViewPredictor",
    "MVCNN",
    "MVCNNClassifier",
    "ViewTransformerClassifier",
    "build_model",
    "build_experiment_classifier",
    "camera_statistics",
    "detect_view_collapse",
    "fixed_camera_angles",
    "pack_vertices_for_mvtn",
]
