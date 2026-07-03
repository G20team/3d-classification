# Single-view

Single-viewは、1つの固定視点だけで識別する基準条件です。
meshの正面方向は保証されないため、この条件は「正面画像」ではなく `single fixed view` として扱います。

## 目的

この条件では、マルチビュー統合や学習視点なしでどこまで識別できるかを測ります。
Fixed Ring-4やMVTNが改善した場合、その改善が単純に視点数を増やした効果なのか、視点学習による効果なのかを切り分けるための下限基準になります。

## Config

主なconfig:

```text
configs/debug_single_view.yaml
configs/single_view.yaml
```

重要な設定:

- `model.experiment_kind: single_view`
- `model.num_views: 1`
- `model.backbone: resnet18`
- `rendering.image_size: 224`
- `training.batch_size: 8`
- `training.epochs: 30`

debug configでは `class_limit` と `image_size` を小さくし、短時間で実装確認できるようにしています。

## 実行手順

先に [データ準備](../data_pipeline.md) が完了している必要があります。

debug subset:

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
```

本実験:

```bash
uv run python scripts/train.py --config configs/single_view.yaml
```

学習が完了すると、標準出力にrun directoryが表示されます。

```text
training finished: outputs/single_view/<timestamp>_seed0
```

## 評価

保存されたbest checkpointをtest splitで評価します。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/single_view/<timestamp>_seed0/checkpoints/best.ckpt \
  --split test
```

評価結果は既定で次へ保存されます。

```text
outputs/single_view/<timestamp>_seed0/eval_test/
```

## 確認ポイント

- `metrics.json` のvalidation macro-F1とtest macro-F1が極端に乖離していない。
- `confusion_matrix.png` で、似たシルエットのポケモンが混同されているか確認する。
- Single-viewだけ低い場合、未知姿勢に対して1視点では情報が足りない可能性がある。
- Single-viewがFixed Ring-4に近い場合、対象クラスが単視点でも十分識別可能か、4視点の配置やレンダリングに問題がないか確認する。

## 比較での使い方

Single-viewは、Fixed Ring-4とMVTNの改善幅を見るための基準です。
最終レポートでは、Single-viewからFixed Ring-4への改善を「視点数を増やした効果」として見て、
Fixed Ring-4からMVTNへの差分を「視点配置を学習した効果」として解釈します。
