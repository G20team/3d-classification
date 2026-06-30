"""
Dataset utilities for MVCNN individual identification.

想定するディレクトリ構造:
    dataset/
        1/
            1_001.png
            1_002.png
            ...
            1_024.png
        2/
            2_001.png
            ...

- フォルダ名(例: "1") = 個体ID = クラスラベル
- フォルダ内の各画像 = その個体を異なる角度から見たシルエットビュー

このスクリプトでは:
    1. MultiViewDataset: 1個体につき複数ビュー(画像群)をまとめて1サンプルとして返す
       (MVCNNの forward が要求する (V, C, H, W) 形式に対応)
    2. 角度ホールドアウト分割 (各個体のビューの一部をtestに回す。個体自体は分割しない)
    3. SingleImageDataset: イラスト実験用。1枚の画像を1サンプル(V=1)として返す
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def natural_sort_key(s: str):
    """ファイル名中の数字部分で自然順ソートするためのキー (1_001 < 1_002 < ... < 1_010)"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def list_individuals(root_dir: str) -> List[str]:
    """root_dir直下のフォルダ名(個体ID)を昇順で返す"""
    root = Path(root_dir)
    ids = [p.name for p in root.iterdir() if p.is_dir()]
    ids.sort(key=natural_sort_key)
    return ids


def list_views_for_individual(root_dir: str, individual_id: str) -> List[str]:
    """指定個体フォルダ内の画像パス一覧を、ファイル名の数字順で返す"""
    folder = Path(root_dir) / individual_id
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [str(p) for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort(key=natural_sort_key)
    return files


def build_default_transform(image_size: int = 224, train: bool = True) -> T.Compose:
    """
    シルエット画像用のtransform。
    - グレースケール1chに統一(L)
    - 学習時は軽いデータ拡張(回転ジッタ、アフィン、反転は個体形状次第なので控えめに)
    - 正規化は[0,1]の単純なToTensorのみ(シルエットなのでImageNetの平均分散は使わない)
    """
    base = [
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size)),
    ]
    if train:
        base += [
            T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)),
        ]
    base += [T.ToTensor()]  # -> (1, H, W), [0,1]
    return T.Compose(base)


class MultiViewDataset(Dataset):
    """
    1サンプル = 1個体 = (V, C, H, W) のテンソル + クラスラベル

    holdout_indices で各個体の「テストに回すビューのインデックス」を指定できる。
    例: 24ビュー中、4枚に1枚をテストに回す場合 -> holdout_indices = [0, 4, 8, 12, 16, 20]
    split='train' なら holdout_indices 以外を使い、split='test' なら holdout_indices のみ使う。
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",                 # 'train' or 'test'
        holdout_indices: Optional[List[int]] = None,
        image_size: int = 224,
        transform: Optional[T.Compose] = None,
    ):
        assert split in ("train", "test")
        self.root_dir = root_dir
        self.split = split
        self.holdout_indices = set(holdout_indices) if holdout_indices else set()

        self.individual_ids = list_individuals(root_dir)
        self.label_map = {iid: idx for idx, iid in enumerate(self.individual_ids)}

        self.transform = transform or build_default_transform(
            image_size=image_size, train=(split == "train")
        )

        # 各個体ごとに使うビューのパスを確定しておく
        self.samples: List[Tuple[str, List[str]]] = []  # (individual_id, [view_paths])
        for iid in self.individual_ids:
            all_views = list_views_for_individual(root_dir, iid)
            if split == "train":
                views = [v for i, v in enumerate(all_views) if i not in self.holdout_indices]
            else:
                views = [v for i, v in enumerate(all_views) if i in self.holdout_indices]
            if len(views) == 0:
                continue
            self.samples.append((iid, views))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        iid, view_paths = self.samples[idx]
        imgs = []
        for p in view_paths:
            img = Image.open(p).convert("RGB")  # 一旦RGBで開いてtransform内でGrayscaleに落とす
            img = self.transform(img)            # (1, H, W)
            imgs.append(img)
        views_tensor = torch.stack(imgs, dim=0)  # (V, 1, H, W)
        label = self.label_map[iid]
        return views_tensor, label

    def get_label_map(self):
        """label_map (個体ID文字列 -> クラスindex) を返す。推論結果の表示等に使う。"""
        return dict(self.label_map)


class SingleImageDataset(Dataset):
    """
    イラスト実験用。1枚の画像 = 1サンプル(V=1)として読み込む。
    フォルダ構成は問わず、画像パスのリストとラベル(分かる場合)を直接渡す形にしている。

    例:
        paths = ["illust/1_pikachu.png"]
        labels = [0]  # MultiViewDatasetのlabel_mapに対応するindexを事前に調べておく
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        image_size: int = 224,
        transform: Optional[T.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels if labels is not None else [-1] * len(image_paths)
        assert len(self.image_paths) == len(self.labels)
        # 単一画像はテスト/推論用途なのでデータ拡張なし(train=False)
        self.transform = transform or build_default_transform(image_size=image_size, train=False)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)              # (1, H, W)
        views_tensor = img.unsqueeze(0)          # (1, 1, H, W)  -> V=1
        label = self.labels[idx]
        return views_tensor, label


def make_holdout_indices(num_views: int = 24, stride: int = 4) -> List[int]:
    """
    角度ホールドアウト用のインデックスを規則的に作るヘルパー。
    例: num_views=24, stride=4 -> [0, 4, 8, 12, 16, 20] (6枚をテストに回す = 25%)
    """
    return list(range(0, num_views, stride))


if __name__ == "__main__":
    # 簡易動作確認(実データがある場合のみ動く。パスは適宜書き換えて使う想定)
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "dataset"
    if not os.path.exists(root):
        print(f"'{root}' が見つかりません。動作確認には実データのパスを渡してください。")
    else:
        holdout = make_holdout_indices(num_views=24, stride=4)
        train_ds = MultiViewDataset(root, split="train", holdout_indices=holdout)
        test_ds = MultiViewDataset(root, split="test", holdout_indices=holdout)
        print(f"individuals: {len(train_ds.individual_ids)}")
        print(f"train samples: {len(train_ds)}, test samples: {len(test_ds)}")
        x, y = train_ds[0]
        print(f"train sample[0]: views={x.shape}, label={y}")
        x, y = test_ds[0]
        print(f"test sample[0]: views={x.shape}, label={y}")
