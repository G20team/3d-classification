"""MVCNNモデル定義。"""

from __future__ import annotations

from typing import Literal, overload

import torch
import torch.nn as nn
import torchvision.models as models

BackboneName = Literal["resnet18", "simple_cnn"]


class SimpleCNN(nn.Module):
    """スクラッチ用の軽量CNNバックボーン。"""

    def __init__(self, input_channels: int = 1, feature_dim: int = 512) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNet18Backbone(nn.Module):
    """ImageNet事前学習済みResNet18を多視点用の特徴抽出器として使う。"""

    def __init__(self, input_channels: int = 1, feature_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        if input_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels,
                old_conv.out_channels,
                kernel_size=_size2(old_conv.kernel_size),
                stride=_size2(old_conv.stride),
                padding=_padding2(old_conv.padding),
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_weight.repeat(1, input_channels, 1, 1) / input_channels)
            resnet.conv1 = new_conv

        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, feature_dim)
        self.backbone = resnet

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class MVCNN(nn.Module):
    """Multi-View CNN本体。"""

    def __init__(
        self,
        *,
        num_classes: int,
        backbone: BackboneName = "resnet18",
        input_channels: int = 1,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            msg = "num_classes は1以上である必要があります。"
            raise ValueError(msg)

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone_name = backbone

        if backbone == "resnet18":
            self.encoder: nn.Module = ResNet18Backbone(
                input_channels=input_channels,
                feature_dim=feature_dim,
                pretrained=pretrained,
            )
        elif backbone == "simple_cnn":
            self.encoder = SimpleCNN(input_channels=input_channels, feature_dim=feature_dim)
        else:
            msg = f"未知のbackboneです: {backbone}"
            raise ValueError(msg)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_view_features(self, x: torch.Tensor) -> torch.Tensor:
        """各ビューを個別にエンコードする。"""

        if x.ndim != 5:
            msg = f"MVCNN入力は (B, V, C, H, W) である必要があります: shape={tuple(x.shape)}"
            raise ValueError(msg)
        batch_size, num_views, channels, height, width = x.shape
        flat_views = x.reshape(batch_size * num_views, channels, height, width)
        features = self.encoder(flat_views)
        return features.reshape(batch_size, num_views, -1)

    @staticmethod
    def view_pool(view_features: torch.Tensor) -> torch.Tensor:
        """ビュー方向のelement-wise max poolingを行う。"""

        pooled, _ = torch.max(view_features, dim=1)
        return pooled

    @overload
    def forward(self, x: torch.Tensor, return_embedding: Literal[False] = False) -> torch.Tensor:
        ...

    @overload
    def forward(self, x: torch.Tensor, return_embedding: Literal[True]) -> tuple[torch.Tensor, torch.Tensor]:
        ...

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        view_features = self.extract_view_features(x)
        embedding = self.view_pool(view_features)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def build_model(
    *,
    num_classes: int,
    backbone: BackboneName = "resnet18",
    input_channels: int = 1,
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


def _size2(value: int | tuple[int, ...]) -> int | tuple[int, int]:
    if isinstance(value, int):
        return value
    if len(value) != 2:
        msg = f"2次元サイズを想定しています: {value}"
        raise ValueError(msg)
    return (value[0], value[1])


def _padding2(value: str | int | tuple[int, ...]) -> str | int | tuple[int, int]:
    if isinstance(value, str | int):
        return value
    return _size2(value)
