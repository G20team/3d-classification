# Fixed Ring-4 MVCNN

Fixed Ring-4 MVCNNは、円環状に固定した4視点をレンダリングし、共有encoderで抽出した特徴をview-wise max poolingで統合する条件です。

## 目的

この条件はMVTNと同じ4視点数を使う固定視点の基準です。
MVTNの性能を評価するときは、Single-viewではなくこの条件との比較が中心になります。

## 視点配置

既定のazimuthは次の4方向です。

```text
0, 90, 180, 270 degrees
```

elevationは0度です。
実際の学習データでは、姿勢split由来のyaw/elevation offsetが加わります。

## Config

主なconfig:

```text
configs/debug_fixed_ring4.yaml
configs/fixed_ring4.yaml
```

重要な設定:

- `model.experiment_kind: fixed_ring4`
- `model.num_views: 4`
- `model.backbone: resnet18`
- `rendering.image_size: 224`
- `training.batch_size: 4`
- `training.epochs: 30`

MVTN条件と比較しやすいように、backbone、feature_dim、dropout、optimizer設定は揃えます。

## 実行手順

debug subset:

```bash
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
```

本実験:

```bash
uv run python scripts/train.py --config configs/fixed_ring4.yaml
```

学習が完了すると、標準出力にrun directoryが表示されます。

```text
training finished: outputs/fixed_ring4/<timestamp>_seed0
```

## 評価

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/fixed_ring4/<timestamp>_seed0/checkpoints/best.ckpt \
  --split test
```

出力先:

```text
outputs/fixed_ring4/<timestamp>_seed0/eval_test/
```

## 確認ポイント

- Single-viewよりTop-1 AccuracyやMacro-F1が改善しているか。
- per-class metricsで、単視点では苦手だったクラスが改善しているか。
- confusion matrixで、形状が似たクラスの混同が減っているか。
- MVTNと比較するとき、画像解像度、視点数、backbone、optimizer、splitが揃っているか。

## MVTNとの比較

Fixed Ring-4はMVTNの初期配置と同じ思想の固定カメラ条件です。
MVTNがFixed Ring-4を上回る場合、単に4視点を使った効果ではなく、meshごとに視点を調整した効果の候補として解釈できます。

ただし、性能差が小さい場合は、MVTNの学習視点が固定配置から十分に動いているか、view collapseが起きていないか、
`camera_positions.json` と `learned_camera_visualization.png` を確認します。
