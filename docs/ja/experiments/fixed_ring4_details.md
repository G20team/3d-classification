# Fixed Ring-4 MVCNN 詳細解説

このドキュメントでは、`configs/fixed_ring4.yaml` で実行するFixed Ring-4 MVCNN条件について、設計意図、入力生成、モデル構成、評価方法、MVTNとの比較観点を詳しく説明します。
短い実行手順だけを確認したい場合は [Fixed Ring-4 MVCNN](fixed_ring4.md) を参照してください。

## 実験の位置づけ

Fixed Ring-4 MVCNNは、円環状に固定した4つの視点からmeshをレンダリングし、各画像の特徴を統合してポケモン名を識別する条件です。
このリポジトリでは、[Pokémon 3D assets](https://github.com/Pokemon-3D-api/assets) から取得したGLB形式の3Dアセットを前処理し、同一ポケモンを複数姿勢から観測するclosed-set cross-orientation asset identificationとして扱います。

この条件の主な役割は、MVTNと同じ視点数を使う固定視点の基準を作ることです。
MVTNがSingle-viewを上回っても、それだけでは視点学習が効いたとは言えません。
単に1視点から4視点に増えた効果かもしれないため、MVTNの比較相手としてFixed Ring-4を必ず置きます。

## 視点配置

既定の4視点は、水平円環上の `0, 90, 180, 270 degrees` です。
elevationは0度を基準にします。
この配置は、対象を周囲から均等に観測する単純で解釈しやすい設計です。

ただし、各splitではyaw/elevation offsetが加わります。
train、validation、testで姿勢条件を分けているため、モデルは同じポケモンIDを見ていても、testでは学習時と異なる角度ずれを持つレンダリング画像を分類することになります。
Fixed Ring-4は視点配置自体を学習しないので、未知姿勢に対する頑健性は、固定4視点のカバー範囲とCNN特徴の汎化に依存します。

## データと前処理

入力には `data/manifests/selected_regular.jsonl` と `data/mesh_cache` を使います。
`selected_regular.jsonl` は監査済みの通常形アセット一覧で、mesh cacheは正規化済みの幾何情報を保存したものです。
全条件で同じmanifest、同じmesh cache、同じ `pose_splits.json` を使うことで、条件間の差分を視点数と視点配置に集中させます。

Fixed Ring-4では1サンプルあたり4枚のレンダリング画像を生成します。
画像解像度は `224 x 224`、背景色は中間グレー、mesh色は白です。
色やテクスチャよりも、複数方向から見た形状、輪郭、陰影が分類情報の中心になります。

## モデル構成

モデル種別は `model.experiment_kind: fixed_ring4` です。
4枚のビューは同じCNN encoderに通されます。
つまり、各視点専用のencoderを持つのではなく、視点間で重みを共有した特徴抽出を行います。

各ビューから得られた特徴は、view-wise max poolingで統合されます。
max poolingは、ある視点で強く検出された形状特徴を統合特徴に残しやすい処理です。
たとえば、翼、尻尾、角、耳、背中の突起のような特徴が特定視点で目立つ場合、そのビューの活性が最終特徴に反映されやすくなります。

この条件では視点選択は固定で、学習されるのはCNN encoderと分類ヘッドです。
そのため、MVTNと比較する際は、backbone、feature dimension、dropout、optimizer、epoch数、画像解像度を揃えることが重要です。
既定configでは `resnet18`、`feature_dim: 512`、`dropout: 0.3`、`epochs: 30`、`learning_rate: 0.0001`、`weight_decay: 0.01` を使います。

## 評価方法

学習中はvalidation macro-F1でbest checkpointを選びます。
最終比較では、保存されたcheckpointをtest splitで評価します。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/fixed_ring4/<run_id>/checkpoints/best.ckpt \
  --split test
```

評価ではTop-1 Accuracy、Top-5 Accuracy、Macro-F1、per-class metrics、confusion matrixを確認します。
Single-viewとの比較では、複数視点化によってどの程度改善したかを見ます。
MVTNとの比較では、視点配置を固定した場合と学習した場合の差分を見ます。

## Single-viewとの比較

Single-viewからFixed Ring-4への改善は、主に視点数を増やした効果として解釈します。
ポケモンの3D形状は、角度によって見える部位が大きく変わる場合があります。
1視点では隠れていた特徴が、4視点のどれかに現れることで分類しやすくなる可能性があります。

改善が大きい場合は、対象タスクにおいて単視点の情報不足が支配的だったと考えられます。
改善が小さい場合は、単視点でも十分識別できるクラスが多い、4視点の固定配置が対象形状に合っていない、またはmax poolingが必要な情報を十分に統合できていない可能性があります。

## MVTNとの比較

MVTNはFixed Ring-4と同じ4視点を使いますが、mesh形状に応じてazimuth/elevationを補正します。
したがって、MVTNの主な比較相手はSingle-viewではなくFixed Ring-4です。
Fixed Ring-4を上回る場合、固定配置から動かした視点が分類に有利な情報を捉えた可能性があります。

ただし、MVTNとの差が小さい場合も重要です。
固定4視点が十分に強い基準であり、学習視点の追加自由度が大きな利益を生まない可能性があります。
また、MVTN側の視点offsetがほとんど動いていない、あるいはview collapseが起きている場合は、性能差だけでなくcamera logを含めて解釈する必要があります。

## 確認すべき失敗例

per-class metricsとconfusion matrixでは、Single-viewで混同していたクラスがFixed Ring-4で改善したかに注目します。
改善していれば、別方向から見た形状特徴が有効だったと説明しやすくなります。
改善していない場合は、4方向すべてで似たシルエットになるクラス、mesh正規化やレンダリングで特徴が見えにくいクラス、分類器の容量や学習データ量が不足しているクラスを疑います。

## この条件で分かること

- 1視点から4視点へ増やす効果を評価できます。
- MVTNと同じ視点数の固定基準を作れます。
- 複数視点統合によって改善するクラスと改善しないクラスを分析できます。

## この条件では分からないこと

- 形状ごとに最適な視点へ動かす効果は分かりません。
- 固定円環以外の視点配置がより良いかどうかは分かりません。
- 8視点以上の高密度マルチビュー条件との比較は行っていません。
- 未知ポケモンへのopen-set汎化性能は評価していません。
