"""多視点シルエット画像データセット。"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, cast

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
Transform = Callable[[Image.Image], torch.Tensor]


class DatasetError(ValueError):
    """データセット構成が不正なときの例外。"""


def natural_sort_key(value: str | Path) -> list[int | str]:
    """ファイル名中の数字を自然順で比較するためのキーを返す。"""

    return [int(token) if token.isdigit() else token for token in re.split(r"(\d+)", str(value))]


def list_individuals(root_dir: str | Path) -> list[str]:
    """root直下の個体IDディレクトリを自然順で返す。"""

    root = Path(root_dir)
    if not root.is_dir():
        msg = f"dataset rootが見つからないか、ディレクトリではありません: {root}"
        raise DatasetError(msg)

    individual_ids = [path.name for path in root.iterdir() if path.is_dir()]
    individual_ids.sort(key=natural_sort_key)
    if not individual_ids:
        msg = f"dataset root直下に個体ディレクトリがありません: {root}"
        raise DatasetError(msg)
    return individual_ids


def list_views_for_individual(root_dir: str | Path, individual_id: str) -> list[Path]:
    """指定個体ディレクトリ内の画像path一覧を自然順で返す。"""

    folder = Path(root_dir) / individual_id
    if not folder.is_dir():
        msg = f"個体ディレクトリが見つかりません: {folder}"
        raise DatasetError(msg)

    files = [path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
    files.sort(key=natural_sort_key)
    if not files:
        msg = f"画像ファイルが見つかりません: {folder}"
        raise DatasetError(msg)
    return files


def build_default_transform(image_size: int = 224, train: bool = True) -> Transform:
    """シルエット画像用の標準transformを作る。"""

    transforms: list[object] = [
        T.Grayscale(num_output_channels=1),
        T.Resize((image_size, image_size)),
    ]
    if train:
        transforms.append(T.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.95, 1.05)))
    transforms.append(T.ToTensor())
    return cast("Transform", T.Compose(transforms))


class MultiViewDataset(Dataset):
    """1サンプルを1個体の複数ビューとして返すDataset。"""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        split: Literal["train", "test"] = "train",
        holdout_indices: Sequence[int] | None = None,
        image_size: int = 224,
        transform: Transform | None = None,
        expected_num_views: int | None = None,
    ) -> None:
        if split not in ("train", "test"):
            msg = f"split は train / test のいずれかである必要があります: {split}"
            raise DatasetError(msg)
        if expected_num_views is not None and expected_num_views <= 0:
            msg = "expected_num_views は1以上である必要があります。"
            raise DatasetError(msg)

        self.root_dir = Path(root_dir)
        self.split = split
        self.holdout_indices = set(holdout_indices or [])
        self.individual_ids = list_individuals(self.root_dir)
        self.label_map = {individual_id: index for index, individual_id in enumerate(self.individual_ids)}
        self.transform: Transform = transform or build_default_transform(
            image_size=image_size,
            train=(split == "train"),
        )

        self.samples: list[tuple[str, list[Path]]] = []
        for individual_id in self.individual_ids:
            all_views = list_views_for_individual(self.root_dir, individual_id)
            if expected_num_views is not None and len(all_views) != expected_num_views:
                msg = (
                    f"{individual_id} のview数が設定と一致しません: "
                    f"expected={expected_num_views}, actual={len(all_views)}"
                )
                raise DatasetError(msg)

            if split == "train":
                selected_views = [path for index, path in enumerate(all_views) if index not in self.holdout_indices]
            else:
                selected_views = [path for index, path in enumerate(all_views) if index in self.holdout_indices]
            if not selected_views:
                msg = f"{individual_id} の {split} split に画像がありません。holdout設定を確認してください。"
                raise DatasetError(msg)
            self.samples.append((individual_id, selected_views))

        if not self.samples:
            msg = f"{split} split のサンプルが空です: {self.root_dir}"
            raise DatasetError(msg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        individual_id, view_paths = self.samples[index]
        images: list[torch.Tensor] = []
        for path in view_paths:
            with Image.open(path) as image:
                rgb_image = image.convert("RGB")
            images.append(self.transform(rgb_image))
        views_tensor = torch.stack(images, dim=0)
        return views_tensor, self.label_map[individual_id]

    def get_label_map(self) -> dict[str, int]:
        """個体ID文字列からクラスindexへの対応を返す。"""

        return dict(self.label_map)


class SingleImageDataset(Dataset):
    """1枚の画像をV=1の多視点サンプルとして返すDataset。"""

    def __init__(
        self,
        image_paths: Sequence[str | Path],
        *,
        labels: Sequence[int] | None = None,
        image_size: int = 224,
        transform: Transform | None = None,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        self.labels = list(labels) if labels is not None else [-1] * len(self.image_paths)
        if len(self.image_paths) != len(self.labels):
            msg = "image_paths と labels の長さが一致していません。"
            raise DatasetError(msg)
        if not self.image_paths:
            msg = "image_paths が空です。"
            raise DatasetError(msg)
        missing = [str(path) for path in self.image_paths if not path.is_file()]
        if missing:
            msg = f"画像ファイルが見つかりません: {missing[:3]}"
            raise DatasetError(msg)
        self.transform: Transform = transform or build_default_transform(image_size=image_size, train=False)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.image_paths[index]) as image:
            rgb_image = image.convert("RGB")
        image_tensor = self.transform(rgb_image)
        return image_tensor.unsqueeze(0), self.labels[index]


def make_holdout_indices(num_views: int = 24, stride: int = 4) -> list[int]:
    """角度ホールドアウト用のインデックスを規則的に作る。"""

    if num_views <= 0:
        msg = "num_views は1以上である必要があります。"
        raise ValueError(msg)
    if stride <= 0:
        msg = "stride は1以上である必要があります。"
        raise ValueError(msg)
    return list(range(0, num_views, stride))
