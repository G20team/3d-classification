"""Transformer aggregation for multi-view image features."""

from __future__ import annotations

import torch
import torch.nn as nn

from pokemon_3d_cls.models.mvcnn import build_encoder


class ViewTransformerClassifier(nn.Module):
    """Aggregate shared CNN view features with self-attention and a CLS token."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_views: int = 4,
        backbone: str = "resnet18",
        input_channels: int = 3,
        feature_dim: int = 512,
        pretrained: bool = True,
        classifier_dropout: float = 0.3,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 2048,
        transformer_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            msg = "num_classes must be at least 1."
            raise ValueError(msg)
        if num_views <= 1:
            msg = "num_views must be at least 2 for view-transformer aggregation."
            raise ValueError(msg)
        if feature_dim % num_heads != 0:
            msg = "feature_dim must be divisible by num_heads."
            raise ValueError(msg)
        if not 0.0 <= transformer_dropout <= 1.0:
            msg = "transformer_dropout must be between 0.0 and 1.0."
            raise ValueError(msg)

        self.num_classes = num_classes
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.encoder = build_encoder(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.view_position_embedding = nn.Parameter(torch.zeros(1, num_views + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(feature_dim, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.view_position_embedding, std=0.02)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Classify a fixed number of images shaped (B,V,C,H,W)."""

        if views.ndim != 5:
            msg = "views must be a tensor shaped (B,V,C,H,W)."
            raise ValueError(msg)
        batch_size, num_views, channels, height, width = views.shape
        if num_views != self.num_views:
            msg = f"Expected {self.num_views} views, but received {num_views}."
            raise ValueError(msg)

        flat_views = views.reshape(batch_size * num_views, channels, height, width)
        flat_features = self.encoder(flat_views)
        view_tokens = flat_features.reshape(batch_size, num_views, self.feature_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, view_tokens), dim=1)
        tokens = tokens + self.view_position_embedding
        encoded = self.transformer(tokens)
        descriptor = self.output_norm(encoded[:, 0])
        return self.classifier(descriptor)
