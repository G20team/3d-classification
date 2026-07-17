# 実験設計の概要

このドキュメントでは、Single-view、Fixed Ring-4 MVCNN、Learned Circular-4 MVTN、View Transformer-4を比較するための共通設計を説明します。
各条件の詳細な実行手順は個別ページを参照してください。

## 比較の考え方

比較したい差分は視点配置です。
そのため、Fixed Ring-4とMVTNではencoder、classifier、視点数、画像解像度、optimizer設定を原則として揃えます。
Single-viewは1視点しか使わない基準条件として扱います。

| 条件 | 視点数 | 視点配置 | 分類器 | 目的 |
| --- | ---: | --- | --- | --- |
| Single-view | 1 | 固定 | MVCNNClassifier | 1視点だけでどこまで識別できるかを測る |
| Fixed Ring-4 MVCNN | 4 | 固定円環 | MVCNNClassifier | 4固定視点の基準性能を測る |
| Learned Circular-4 MVTN | 4 | 学習補正付き円環 | MVCNNClassifier + CircularViewPredictor | 形状依存の視点調整が有効かを測る |
| View Transformer-4 | 4 | 固定円環 | CNN encoder + Transformer | view-wise max poolingをattention統合へ置換する効果を測る |

## 共通入力

全条件は次を前提にします。

- `data/manifests/selected_regular.jsonl`
- `data/mesh_cache/`
- `data/manifests/pose_splits_stratified_seed0.json`

未作成の場合は先に [データ準備](../data_pipeline.md) を実行してください。

## 共通出力

各runは `outputs/<condition_id>/<timestamp>_seed<seed>/` に保存されます。

```text
config.yaml
environment_report.json
metadata.json
metrics.json
per_class_metrics.csv
pose_metrics.csv
confusion_matrix.png
checkpoints/best.ckpt
logs/
rendered_examples/
camera_positions.json
learned_camera_visualization.png
```

`config.yaml` は実行時configのsnapshotです。
`metadata.json` にはseed、git commit、環境情報などが入ります。
`logs/` はTensorBoardで学習曲線を見るために使います。

## Debug Subset

本実験へ進む前に、10クラスだけのdebug subsetで4条件を確認します。
目的は高い汎化性能を出すことではなく、データ読み込み、レンダリング、学習、評価、保存処理、MVTNの勾配を通すことです。

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
uv run python scripts/train.py --config configs/debug_view_transformer4.yaml
```

成功条件:

- 4条件すべてで学習が開始できる。
- 10クラスsubsetで訓練lossが下がる。
- checkpoint、metrics、confusion matrix、config snapshotが保存される。
- MVTN条件でcamera offsetへ分類lossから勾配が流れる。
- MVTN条件で `camera_positions.json` が保存される。

debugが失敗した場合は、本実験へ進まずに `outputs/<debug_experiment>/.../` と
`outputs/environment_report.json` を確認します。

## 本実験

debug subsetが通ったら、全採用アセットで4条件を実行します。

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
uv run python scripts/train.py --config configs/view_transformer4.yaml
```

seedを変えた複数runを行う場合は、各configの `experiment.seed` を変更します。
runを識別しやすくしたい場合は `experiment.run_id` を設定します。

## モデル選択

学習中はvalidation macro-F1をモデル選択指標にし、最良checkpointを保存します。
最終比較では保存済みcheckpointをtest splitで評価します。

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

評価指標の読み方は [評価と結果解釈](../evaluation.md) を参照してください。

## 個別手順

- [Single-view](single_view.md)
- [Fixed Ring-4 MVCNN](fixed_ring4.md)
- [Learned Circular-4 MVTN](mvtn_circular4.md)
- [View Transformer-4](view_transformer4.md)

## 詳細解説

実験の目的、データ前提、手法、評価方法、結果解釈を詳しく確認したい場合は、次の詳細版を参照してください。

- [Single-view 詳細解説](single_view_details.md)
- [Fixed Ring-4 MVCNN 詳細解説](fixed_ring4_details.md)
- [Learned Circular-4 MVTN 詳細解説](mvtn_circular4_details.md)
