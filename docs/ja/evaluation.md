# 評価と結果解釈

このドキュメントでは、保存済みcheckpointの評価、metricsの読み方、4条件の比較方法をまとめます。

## Checkpoint評価

学習中はvalidation macro-F1が最良だったcheckpointを保存します。
保存済みcheckpointをtest splitで評価するには次を実行します。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/<condition_id>/<run_id>/checkpoints/best.ckpt \
  --split test
```

評価CLIの既定batch sizeは、WSLのメモリ不足による強制終了を避けるため `1` です。
余裕がある環境で高速化したい場合だけ、次のように明示的に増やしてください。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/<condition_id>/<run_id>/checkpoints/best.ckpt \
  --split test \
  --batch-size 2
```

出力先を明示したい場合:

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/<condition_id>/<run_id>/checkpoints/best.ckpt \
  --split test \
  --output-dir outputs/<condition_id>/<run_id>/eval_test
```

既定ではcheckpointの親run directoryに `eval_<split>/` が作成されます。

## 主な評価出力

```text
metrics.json
confusion_matrix.png
pose_metrics.csv
```

学習run directoryには次も保存されています。

```text
per_class_metrics.csv
camera_positions.json
learned_camera_visualization.png
```

`camera_positions.json` と `learned_camera_visualization.png` は主にMVTN条件で確認します。

## Metrics

見るべき指標:

- `Top-1 Accuracy`: 最上位予測が正解した割合。
- `Top-5 Accuracy`: 上位5候補に正解が含まれた割合。
- `Macro-F1`: クラスごとのF1を平均した値。クラス間の偏りに比較的強い。
- `per_class_metrics.csv`: どのポケモンで失敗しやすいか。
- `confusion_matrix.png`: 混同しやすいポケモンの組み合わせ。
- `pose_metrics.csv`: yaw/elevationごとの件数、Top-1、Top-5。角度別の難易度差を確認する。

Macro-F1はクラスごとに同じ重みを置くため、一部の頻出クラスだけがよく当たる状態を見抜きやすい指標です。
この実験ではvalidation macro-F1をモデル選択指標にしています。

## 条件間比較

比較は次の順序で行います。

1. Single-viewとFixed Ring-4を比較し、視点数を増やした効果を見る。
2. Fixed Ring-4とMVTNを比較し、視点配置を学習した効果を見る。
3. Fixed Ring-4とView Transformer-4を比較し、attention統合の効果を見る。
4. MVTNのcamera logを確認し、性能差がcollapseや不自然な視点集中で説明されないか確認する。

比較表の例:

| 条件                      | Seed | Top-1 | Top-5 | Macro-F1 | 備考                     |
| ------------------------- | ---: | ----: | ----: | -------: | ------------------------ |
| Single-view               |    0 |       |       |          |                          |
| Fixed Ring-4              |    0 |       |       |          |                          |
| Learned Circular-4 MVTN   |    0 |       |       |          | view collapse有無を記録  |
| View Transformer-4        |    0 |       |       |          | Fixed Ring-4との差を記録 |

複数seedを実行した場合は、平均と標準偏差を併記します。

## Confusion Matrixの読み方

`confusion_matrix.png` では、行が正解クラス、列が予測クラスです。
対角成分が強いほど正しく分類できています。

確認する観点:

- 似た体型やシルエットのクラス同士で混同しているか。
- Single-viewで混同していた組み合わせがFixed Ring-4で改善しているか。
- Fixed Ring-4で残った混同がMVTNで改善しているか。
- 特定クラスだけ極端に外していないか。

## MVTNの視点解釈

MVTNでは分類metricsだけでなく、視点の動きを確認します。

確認するファイル:

```text
camera_positions.json
learned_camera_visualization.png
```

確認する観点:

- `offset_abs_mean`: 学習視点が固定Ring-4からどの程度動いたか。
- `pairwise_distance_min`: 視点同士が近づきすぎていないか。
- `pairwise_distance_mean`: 全体として視点が分散しているか。
- `view_collapse`: 視点collapseが検出されていないか。

MVTNが高い性能を出しても、view collapseが頻発している場合は、単純に「良い視点を学習した」とは言い切れません。
分類性能とcamera logをセットで解釈します。

## レポート観点

最終レポートでは、少なくとも以下を含めると比較しやすくなります。

- 実行環境: `environment_report.json` の概要。
- データ条件: 採用クラス数、split設定、画像解像度。
- 実験条件: config名、seed、epoch数、batch size。
- 結果表: Top-1、Top-5、Macro-F1。
- 混同分析: confusion matrixとper-class metricsから見える失敗傾向。
- MVTN分析: camera offset、view collapse、learned camera visualization。
- 解釈: Single-viewからFixed Ring-4への改善、Fixed Ring-4からMVTNへの改善を分けて述べる。
