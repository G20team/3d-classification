from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pokemon_3d_cls.data import DatasetError, MultiViewDataset, make_holdout_indices, natural_sort_key


def test_multiview_dataset_split_and_label_map(tmp_path: Path) -> None:
    _write_dataset(tmp_path, individual_ids=["1", "2"], views=4)
    holdout = make_holdout_indices(num_views=4, stride=2)

    train_dataset = MultiViewDataset(
        tmp_path,
        split="train",
        holdout_indices=holdout,
        image_size=8,
        expected_num_views=4,
    )
    test_dataset = MultiViewDataset(
        tmp_path,
        split="test",
        holdout_indices=holdout,
        image_size=8,
        expected_num_views=4,
    )

    train_views, train_label = train_dataset[0]
    test_views, _test_label = test_dataset[0]

    assert train_dataset.get_label_map() == {"1": 0, "2": 1}
    assert train_label == 0
    assert train_views.shape == (2, 1, 8, 8)
    assert test_views.shape == (2, 1, 8, 8)


def test_multiview_dataset_rejects_empty_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="individual directories"):
        MultiViewDataset(tmp_path, split="train")


def test_multiview_dataset_rejects_view_count_mismatch(tmp_path: Path) -> None:
    _write_dataset(tmp_path, individual_ids=["1"], views=3)

    with pytest.raises(DatasetError, match="view count"):
        MultiViewDataset(tmp_path, split="train", expected_num_views=4)


def test_natural_sort_key_orders_numeric_tokens() -> None:
    names = ["1_10.png", "1_2.png", "1_1.png"]

    assert sorted(names, key=natural_sort_key) == ["1_1.png", "1_2.png", "1_10.png"]


def _write_dataset(root: Path, *, individual_ids: list[str], views: int) -> None:
    for individual_id in individual_ids:
        folder = root / individual_id
        folder.mkdir(parents=True)
        for index in range(views):
            image = Image.new("L", (8, 8), color=255)
            image.save(folder / f"{individual_id}_{index:03d}.png")
