"""Multi-view CNN分類器。"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import ResNet

BackboneName = Literal["resnet18", "simple_cnn"]


class SimpleCNNEncoder(nn.Module):
    """テストやdebug実験向けの軽量CNN encoder。"""

    def __init__(self, *, input_channels: int, feature_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


def _build_resnet18_encoder(*, input_channels: int, feature_dim: int, pretrained: bool) -> ResNet:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    if input_channels != model.conv1.in_channels:
        model.conv1 = _adapt_first_conv(model.conv1, input_channels=input_channels, pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, feature_dim)
    return model


def _adapt_first_conv(conv: nn.Conv2d, *, input_channels: int, pretrained: bool) -> nn.Conv2d:
    padding = conv.padding if isinstance(conv.padding, str) else _pair2(conv.padding, "padding")
    new_conv = nn.Conv2d(
        input_channels,
        conv.out_channels,
        kernel_size=_pair2(conv.kernel_size, "kernel_size"),
        stride=_pair2(conv.stride, "stride"),
        padding=padding,
        bias=conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            channel_mean = conv.weight.mean(dim=1, keepdim=True)
            new_conv.weight.copy_(channel_mean.repeat(1, input_channels, 1, 1))
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)
    return new_conv


def _pair2(value: int | tuple[int, ...], name: str) -> int | tuple[int, int]:
    if isinstance(value, int):
        return value
    if len(value) != 2:
        msg = f"{name} はintまたは2要素tupleである必要があります。"
        raise ValueError(msg)
    return (value[0], value[1])


def build_encoder(
    *,
    backbone: BackboneName | str,
    input_channels: int,
    feature_dim: int,
    pretrained: bool,
) -> nn.Module:
    """backbone名から画像encoderを構築する。"""

    if backbone == "simple_cnn":
        return SimpleCNNEncoder(input_channels=input_channels, feature_dim=feature_dim)
    if backbone == "resnet18":
        return _build_resnet18_encoder(
            input_channels=input_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
        )

    msg = f"未知のbackboneです: {backbone}"
    raise ValueError(msg)


class MVCNN(nn.Module):
    """複数視点画像をview-wise max poolingで集約する分類器。"""

    def __init__(
        self,
        *,
        num_classes: int,
        backbone: BackboneName | str = "resnet18",
        input_channels: int = 3,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            msg = "num_classes は1以上である必要があります。"
            raise ValueError(msg)
        if input_channels <= 0:
            msg = "input_channels は1以上である必要があります。"
            raise ValueError(msg)
        if feature_dim <= 0:
            msg = "feature_dim は1以上である必要があります。"
            raise ValueError(msg)
        if not 0.0 <= dropout <= 1.0:
            msg = "dropout は0.0以上1.0以下である必要があります。"
            raise ValueError(msg)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.encoder = build_encoder(
            backbone=backbone,
            input_channels=input_channels,
            feature_dim=feature_dim,
            pretrained=pretrained,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """(B,V,C,H,W) または (B,C,H,W) の画像を分類する。"""

        if views.ndim == 4:
            views = views.unsqueeze(1)
        if views.ndim != 5:
            msg = "views は (B,V,C,H,W) または (B,C,H,W) のTensorである必要があります。"
            raise ValueError(msg)

        batch_size, num_views, channels, height, width = views.shape
        flat_views = views.reshape(batch_size * num_views, channels, height, width)
        flat_features = self.encoder(flat_views)
        features = flat_features.reshape(batch_size, num_views, self.feature_dim)
        pooled = features.max(dim=1).values
        return self.classifier(pooled)


class MVCNNClassifier(MVCNN):
    """mesh実験側で使う分類器名。実装はMVCNNと同一。"""


def build_model(
    *,
    num_classes: int,
    backbone: BackboneName | str = "resnet18",
    input_channels: int = 3,
    feature_dim: int = 512,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> MVCNN:
    """設定値からMVCNNを構築する。"""

    return MVCNN(
        num_classes=num_classes,
        backbone=backbone,
        input_channels=input_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        dropout=dropout,
    )
