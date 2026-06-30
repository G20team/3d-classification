"""
MVCNN (Multi-View CNN) for closed-set individual identification on silhouette images.

設計方針:
- backbone は 'resnet18' (ImageNet事前学習 + fine-tune) と 'simple_cnn' (スクラッチ軽量CNN) を
  引数一つで切り替え可能にしている。
- view-pooling は element-wise max。ビュー数が可変でも動作する
  (forward時に (B, V, C, H, W) の V 次元方向にmaxを取るだけなので、
  学習時はV=24、推論時はV=1(イラスト1枚)でも同じ forward が使える)。
- シルエット(1ch, 二値画像)を想定しているが、3chにも対応できるようにinput_channelsを公開している。
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """
    スクラッチ用の軽量CNNバックボーン。
    シルエットは色・テクスチャ情報を持たないので、ImageNet事前学習モデルほど深い
    特徴階層は不要という想定で、浅めの構成にしている。
    出力: (B, feature_dim) のベクトル
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 256, 1, 1)
        x = torch.flatten(x, 1)        # (B, 256)
        x = self.fc(x)                 # (B, feature_dim)
        return x


class ResNet18Backbone(nn.Module):
    """
    ImageNet事前学習済みResNet18をバックボーンとして使う。
    シルエットが1chの場合、最初のconv層を1ch入力用に置き換える
    (事前学習済み3ch重みをチャンネル方向に平均して初期化し、知識を一部活かす)。
    """

    def __init__(self, input_channels: int = 1, feature_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet18(weights=weights)

        if input_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(
                input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=False
            )
            if pretrained:
                with torch.no_grad():
                    # 3ch分の重みをチャンネル方向に平均して1ch(等)に詰め替える
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (out, 1, k, k)
                    new_conv.weight.copy_(mean_weight.repeat(1, input_channels, 1, 1) / input_channels)
            resnet.conv1 = new_conv

        # 最終fc層を feature_dim 出力の埋め込み層として置き換える
        in_features = resnet.fc.in_features
        resnet.fc = nn.Linear(in_features, feature_dim)
        self.backbone = resnet

    def forward(self, x):
        return self.backbone(x)  # (B, feature_dim)


class MVCNN(nn.Module):
    """
    Multi-View CNN本体。

    forward の入力形状: (B, V, C, H, W)
        B: バッチサイズ(個体数)
        V: ビュー数 (学習時24、推論時1でも可。可変でよい)
        C: チャンネル数 (シルエットなら1)
        H, W: 画像サイズ

    処理の流れ:
        1. (B, V, C, H, W) -> (B*V, C, H, W) にreshapeしてバックボーンに一括投入
        2. 各ビューの特徴ベクトル (B*V, feature_dim) を得る
        3. (B, V, feature_dim) に戻し、V次元でelement-wise maxを取って (B, feature_dim) に集約
        4. 分類ヘッド(全結合+softmax相当のlogits)に通してクラス数分のロジットを出力

    単一ビュー入力(イラスト実験など)の場合は V=1 として
    (B, 1, C, H, W) を渡せば同じforwardでそのまま動く。
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",      # 'resnet18' or 'simple_cnn'
        input_channels: int = 1,
        feature_dim: int = 512,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        backbone = backbone.lower()
        if backbone == "resnet18":
            self.encoder = ResNet18Backbone(
                input_channels=input_channels,
                feature_dim=feature_dim,
                pretrained=pretrained,
            )
        elif backbone == "simple_cnn":
            self.encoder = SimpleCNN(
                input_channels=input_channels,
                feature_dim=feature_dim,
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone!r} (use 'resnet18' or 'simple_cnn')")

        self.backbone_name = backbone

        # 分類ヘッド(第2段階CNN相当。MVCNN論文ではCNN2と呼ばれる部分に対応する全結合層)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    def extract_view_features(self, x):
        """
        各ビューを個別にエンコードする。
        x: (B, V, C, H, W) -> (B, V, feature_dim)
        """
        B, V, C, H, W = x.shape
        x = x.view(B * V, C, H, W)
        feats = self.encoder(x)              # (B*V, feature_dim)
        feats = feats.view(B, V, -1)          # (B, V, feature_dim)
        return feats

    def view_pool(self, view_feats):
        """
        view-pooling: ビュー方向(dim=1)のelement-wise max。
        view_feats: (B, V, feature_dim) -> (B, feature_dim)
        """
        pooled, _ = torch.max(view_feats, dim=1)
        return pooled

    def forward(self, x, return_embedding: bool = False):
        """
        x: (B, V, C, H, W)
        return_embedding=True の場合、分類logitsに加えて集約後の埋め込みベクトルも返す
        (後でmetric learning的な分析や類似度計算をしたくなった場合に備えて用意)
        """
        view_feats = self.extract_view_features(x)   # (B, V, feature_dim)
        embedding = self.view_pool(view_feats)         # (B, feature_dim)
        logits = self.classifier(embedding)             # (B, num_classes)

        if return_embedding:
            return logits, embedding
        return logits


def build_model(
    num_classes: int,
    backbone: str = "resnet18",
    input_channels: int = 1,
    feature_dim: int = 512,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> MVCNN:
    """モデル構築用のファクトリ関数。学習・評価スクリプトから呼び出す想定。"""
    return MVCNN(
        num_classes=num_classes,
        backbone=backbone,
        input_channels=input_channels,
        feature_dim=feature_dim,
        pretrained=pretrained,
        dropout=dropout,
    )


if __name__ == "__main__":
    # 動作確認用のスモークテスト
    # 学習時想定: バッチ2個体、各24ビュー、1ch、224x224
    dummy_multi = torch.randn(2, 24, 1, 224, 224)
    # 推論(イラスト)想定: バッチ1個体、1ビューのみ
    dummy_single = torch.randn(1, 1, 1, 224, 224)

    for bb in ["simple_cnn", "resnet18"]:
        model = build_model(num_classes=10, backbone=bb, input_channels=1, pretrained=False)
        model.eval()
        with torch.no_grad():
            out_multi = model(dummy_multi)
            out_single = model(dummy_single)
        print(f"[{bb}] multi-view output: {out_multi.shape}, single-view output: {out_single.shape}")
