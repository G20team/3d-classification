# Learned Circular-4 MVTN 詳細解説

このドキュメントでは、`configs/mvtn_circular4.yaml` で実行するLearned Circular-4 MVTN条件について、目的、モデル構成、視点学習、評価方法、解釈上の注意を詳しく説明します。
短い実行手順だけを確認したい場合は [Learned Circular-4 MVTN](mvtn_circular4.md) を参照してください。

## 実験の位置づけ

Learned Circular-4 MVTNは、固定Ring-4を初期配置として、mesh形状に応じた4視点のazimuth/elevation補正を学習する条件です。
入力アセットは [Pokémon 3D assets](https://github.com/Pokemon-3D-api/assets) から取得したGLB形式の3Dモデルで、通常形アセットを監査・正規化したmesh cacheを使います。

この実験の中心的な問いは、限られた4視点という条件で、形状依存の視点調整が固定円環状カメラ配置より未知姿勢識別を改善できるかです。
分類器はFixed Ring-4と同じMVCNN構造を使い、差分を視点配置の学習に集中させます。

## MVTNで学習するもの

MVTN条件では、CNN分類器だけでなく、カメラ位置を決める小さなネットワークも学習します。
このネットワークは正規化済みmesh verticesから形状特徴を取り出し、4つのbase viewに対するazimuth/elevation offsetを出力します。

base viewはFixed Ring-4と同じ `0, 90, 180, 270 degrees` の円環配置です。
MVTNはそこから完全に自由な視点を選ぶのではなく、設定された上限内で補正を行います。
既定値では `max_azimuth_offset_deg: 45.0`、`max_elevation_offset_deg: 25.0` です。
この制約により、視点が極端に飛びすぎることを避け、Fixed Ring-4との比較可能性を保ちます。

## 入力と出力

MVTNの入力は主に次の情報です。

- 正規化済みmesh vertices
- Fixed Ring-4由来のbase azimuth/elevation
- 姿勢split由来のyaw/elevation offset

出力は次の情報です。

- 補正後のazimuth
- 補正後のelevation
- 学習されたazimuth/elevation offset
- 視点collapse確認用のcamera log

レンダリングされた4枚の画像は、Fixed Ring-4と同じMVCNN分類器へ渡されます。
したがって、分類損失はCNNと分類ヘッドだけでなく、視点offsetを出力するMVTNにも影響します。
分類しやすい画像を得られる視点へ移動するように、視点予測器が間接的に学習される設計です。

## モデル構成

モデル種別は `model.experiment_kind: mvtn_circular4` です。
分類側はFixed Ring-4と同じく、4ビューを共有CNN encoderに通し、view-wise max poolingで統合します。
視点側は `model.mvtn.point_samples: 512` でmeshから点をサンプリングし、`hidden_dim: 128` のネットワークでoffsetを予測します。

分類器のbackbone、feature dimension、dropout、optimizer、epoch数、画像解像度はFixed Ring-4と揃えます。
これにより、MVTNがFixed Ring-4を上回った場合に、視点数や分類器容量ではなく、視点配置を学習した効果として解釈しやすくなります。

## 学習時の注意点

MVTNは分類器よりも解釈が難しい条件です。
性能が上がっていても、それが直感的に意味のある視点学習によるものとは限りません。
たとえば、複数の視点が近い方向へ集まるview collapseが起きると、4視点を使っているつもりでも実質的な観測方向が減ってしまいます。

この実装では `collapse_threshold_deg: 5.0` を使い、視点同士が近づきすぎていないかをcamera logで確認します。
また、offsetが常に0に近い場合は、MVTNが固定配置からほとんど動いていない可能性があります。
逆にoffsetが大きく不安定に揺れる場合は、視点学習が分類器にとって安定した補助になっていない可能性があります。

## 評価方法

学習中はvalidation macro-F1でbest checkpointを保存し、最終評価ではtest splitで評価します。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/mvtn_circular4/<run_id>/checkpoints/best.ckpt \
  --split test
```

分類性能としてTop-1 Accuracy、Top-5 Accuracy、Macro-F1、per-class metrics、confusion matrixを確認します。
加えて、MVTNでは `camera_positions.json` と `learned_camera_visualization.png` を必ず確認します。
分類metricsとcamera logをセットで読むことで、「性能が良い」だけでなく「どのような視点挙動で性能が出たか」を説明できます。

## Camera logの読み方

`camera_positions.json` では、視点offsetや視点間距離に関する情報を確認します。
特に重要なのは次の項目です。

- `offset_abs_mean`: 固定Ring-4から平均的にどれくらい動いたか。
- `pairwise_distance_min`: もっとも近い視点同士の距離。
- `pairwise_distance_mean`: 視点全体がどの程度分散しているか。
- `view_collapse`: 視点collapseが検出されたか。

`offset_abs_mean` が小さすぎる場合、MVTNはFixed Ring-4とほぼ同じ挙動になっている可能性があります。
`pairwise_distance_min` が小さく、`view_collapse` が頻発する場合、視点が多様な方向を見ていない可能性があります。
この場合、Macro-F1が高くても、4視点を効果的に活用したとは言い切れません。

`learned_camera_visualization.png` は、学習された視点を視覚的に確認するための補助資料です。
数値ログだけでは見落としやすい、視点の偏りや不自然な集中を確認します。

## Fixed Ring-4との比較

MVTNの主な比較対象はFixed Ring-4です。
両者は4視点、同じbackbone、同じ分類ヘッド、同じoptimizer設定を使うため、性能差は視点配置を固定するか学習するかの差として解釈しやすくなります。

MVTNがFixed Ring-4を上回る場合、ポケモンの形状に応じて少し視点をずらすことが、識別に有利な部位を見つける助けになった可能性があります。
たとえば、固定配置では特徴が重なって見えにくいクラスで、学習視点が角や翼、尻尾、体型差などをより分離しやすい方向へ動いた可能性があります。

一方、MVTNがFixed Ring-4と同等または下回る場合も、失敗として単純に片付けるべきではありません。
固定4視点が十分強い、4視点という制約では学習視点の自由度が足りない、meshからの視点予測が安定しない、分類損失だけでは良い視点配置を学習しにくい、など複数の解釈が考えられます。

## 結果解釈の観点

最終レポートでは、まずSingle-viewからFixed Ring-4への改善を確認し、次にFixed Ring-4からMVTNへの差分を確認します。
MVTNだけをSingle-viewと比較すると、4視点化の効果と視点学習の効果が混ざってしまいます。
必ずFixed Ring-4を挟んで解釈します。

per-class metricsでは、Fixed Ring-4で苦手だったクラスがMVTNで改善しているかを見ます。
改善したクラスについては、camera logや可視化を確認し、視点offsetが意味のある範囲で動いているかを合わせて報告します。
悪化したクラスについては、視点が特徴部位を外していないか、collapseが起きていないか、学習が不安定でないかを確認します。

## この条件で分かること

- 固定4視点に対して、形状依存の視点補正が有効かを評価できます。
- 分類性能とcamera logを組み合わせ、視点学習の挙動を分析できます。
- Fixed Ring-4で残った混同が、学習視点で改善するかを確認できます。

## この条件では分からないこと

- 8視点以上や別の初期視点配置での最適性能は分かりません。
- 点群直接入力やView-GCNなど、別系統の3D認識手法との優劣は分かりません。
- 学習された視点が人間にとって常に意味的に分かりやすいとは限りません。
- 未知ポケモンへのopen-set汎化性能は評価していません。

