# Learned Circular-4 MVTN

Learned Circular-4 MVTNは、固定Ring-4を初期配置として、mesh形状から4視点のazimuth/elevation補正を学習する条件です。

## 目的

この条件では、ポケモンごとの3D形状に応じて視点を調整することで、角度分布を揃えたsplit上で固定Ring-4より識別性能が改善するかを調べます。
分類器側はFixed Ring-4と同じMVCNN構造を使い、差分を視点配置に集中させます。

## MVTNの入力と出力

入力:

- 正規化済みmesh vertices
- 固定Ring-4のbase azimuth/elevation
- 姿勢split由来のyaw/elevation offset

出力:

- 補正後のazimuth
- 補正後のelevation
- 学習されたoffset

offsetは設定値で上限を持ちます。

```text
max_azimuth_offset_deg: 45.0
max_elevation_offset_deg: 25.0
```

elevationはレンダリングが破綻しにくいように範囲制限されます。

## Config

主なconfig:

```text
configs/debug_mvtn_circular4.yaml
configs/mvtn_circular4.yaml
```

重要な設定:

- `model.experiment_kind: mvtn_circular4`
- `model.num_views: 4`
- `model.mvtn.num_views: 4`
- `model.mvtn.point_samples: 512`
- `model.mvtn.hidden_dim: 128`
- `model.mvtn.collapse_threshold_deg: 5.0`
- `training.batch_size: 4`
- `training.epochs: 30`

Fixed Ring-4と比較するため、分類器のbackbone、feature_dim、dropout、optimizer設定は揃えます。

## 実行手順

debug subset:

```bash
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

本実験:

```bash
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

学習が完了すると、標準出力にrun directoryが表示されます。

```text
training finished: outputs/mvtn_circular4/<timestamp>_seed0
```

## 評価

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/mvtn_circular4/<timestamp>_seed0/checkpoints/best.ckpt \
  --split test
```

出力先:

```text
outputs/mvtn_circular4/<timestamp>_seed0/eval_test/
```

## Camera Log確認

MVTN条件では、分類性能だけでなく学習された視点の挙動も確認します。

主なファイル:

```text
camera_positions.json
learned_camera_visualization.png
```

確認する項目:

- `offset_abs_mean` が常に0に近すぎない。
- `pairwise_distance_min` が小さくなりすぎていない。
- `view_collapse` が頻発していない。
- epochが進むにつれて視点間距離やoffsetが極端に不安定になっていない。

`view_collapse` は、複数視点がほぼ同じ方向へ寄ってしまう状態の検出です。
collapseが起きている場合、MVTNが性能改善していても、解釈には注意が必要です。

## Fixed Ring-4との比較

MVTNの評価では、Single-viewではなくFixed Ring-4との比較を重視します。

見るべき差分:

- Top-1 Accuracy
- Top-5 Accuracy
- Macro-F1
- per-class F1
- confusion matrix上の混同パターン
- camera offsetとview collapseの有無

MVTNがFixed Ring-4を上回った場合でも、camera logを確認し、視点が意味のある範囲で動いているかを合わせて報告します。
