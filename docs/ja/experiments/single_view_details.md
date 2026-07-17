# Single-view 詳細解説

このドキュメントでは、`configs/single_view.yaml` で実行するSingle-view条件について、目的、データ前提、手法、評価方法、結果解釈を詳しく説明します。
短い実行手順だけを確認したい場合は [Single-view](single_view.md) を参照してください。

## 実験の位置づけ

Single-viewは、1つの固定視点からレンダリングした画像だけでポケモン名を識別する基準条件です。
このリポジトリ全体の主題は、[Pokémon 3D assets](https://github.com/Pokemon-3D-api/assets) から取得したGLB形式の3Dアセットを、複数姿勢から画像化して分類することです。
ただしSingle-viewでは、マルチビュー統合や視点学習を使わず、もっとも単純な画像分類に近い設定にします。

この条件を置く理由は、4視点条件やMVTN条件の性能を解釈するための下限基準を作ることです。
Single-viewの性能が低く、Fixed Ring-4で改善する場合、視点数を増やして対象形状を多面的に観測することが有効だったと考えられます。
一方でSingle-viewが十分高い性能を出す場合、対象クラスの多くは1視点だけでも識別しやすく、マルチビュー化や視点学習による追加効果は小さく見える可能性があります。

重要なのは、この条件を「正面画像分類」と呼ばないことです。
取得元アセットの向きは必ずしも意味的な正面に揃っているとは限らないため、ここでの入力はあくまで `single fixed view` です。
角度分布を揃えた実験では、正面らしさではなく、固定した1つの観測方向に依存した場合の限界を見る条件として扱います。

## データ前提

入力アセットは、`scripts/fetch_assets.py` により `data/raw_assets` に取得したGLBファイルです。
その後、`scripts/audit_assets.py` で通常形として扱える候補を選別し、`data/manifests/selected_regular.jsonl` を作成します。
初期実験では、色違い、メガシンカ、リージョンフォーム、特殊形態、性別差分、同一National Dex IDの重複などを除外し、分類単位をできるだけ通常形ポケモンに揃えます。

学習時にはGLBを毎回直接処理するのではなく、`scripts/prepare_mesh_cache.py` で作成した `data/mesh_cache` を使います。
mesh cacheでは、複数geometryの統合、不要頂点や縮退面の除去、bounding box中心の原点移動、unit sphere相当のスケール正規化を行います。
これにより、レンダリング時の極端なスケール差や位置ずれを減らし、各実験条件が同じ前処理済みmeshを参照できるようにします。

train、validation、testの分割はポケモンIDと姿勢条件の組を単位に行います。
全splitに同じポケモンIDと17角度が含まれますが、同一ID・角度の組はsplit間で重複しません。
この設計により、未知ポケモン分類ではなく、既知カタログ内のポケモンを揃えた角度分布から識別できるかを評価します。

## 入力生成

Single-viewでは、各サンプルについて1つのカメラ位置からmeshをレンダリングします。
レンダリング解像度は `rendering.image_size: 224` で、分類器側の入力サイズも `data.image_size: 224` に揃えます。
背景色は中間グレー、mesh色は白を既定値とし、テクスチャの有無や色差ではなく形状由来のシルエットと陰影を主に使って識別する設定です。

姿勢splitで定義されたyaw/elevation offsetは、固定視点に対して加えられます。
そのため、validationやtestでは学習時とは異なる姿勢から同じポケモンを観測することになります。
Single-viewは視点の冗長性がないため、横向き、斜め向き、上下方向の変化に弱くなりやすい条件です。

## モデル構成

モデル種別は `model.experiment_kind: single_view` です。
backboneには `resnet18` を使い、事前学習重みを有効にします。
1枚のレンダリング画像をCNN encoderに入力し、得られた特徴を分類ヘッドへ渡してポケモンIDを予測します。

Fixed Ring-4やMVTNと比較しやすくするため、backbone、feature dimension、dropout、optimizer設定はできるだけ共通化します。
ただしSingle-viewは入力画像が1枚だけなので、batch sizeは4視点条件より大きく設定できます。
既定configでは `training.batch_size: 8`、`training.epochs: 30`、`training.learning_rate: 0.0001`、`training.weight_decay: 0.01` です。

## 評価方法

学習中はvalidation macro-F1をモデル選択指標として、最良checkpointを `checkpoints/best.ckpt` に保存します。
最終評価では、そのcheckpointをtest splitで評価します。

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/single_view/<run_id>/checkpoints/best.ckpt \
  --split test
```

主に確認する指標はTop-1 Accuracy、Top-5 Accuracy、Macro-F1、per-class metrics、confusion matrixです。
Top-1 Accuracyは最上位予測の正解率、Top-5 Accuracyは上位5候補内に正解が含まれる割合です。
Macro-F1はクラスごとのF1を平均するため、クラス数が多いこの実験で特定クラスだけに性能が偏っていないかを見るのに向いています。

## 結果解釈

Single-viewの結果は、単独で良し悪しを判断するより、Fixed Ring-4およびMVTNとの比較に使います。
Single-viewからFixed Ring-4へ大きく改善する場合、1枚では見えない部位やシルエットの曖昧さを、複数視点が補っている可能性があります。
逆に改善幅が小さい場合は、対象アセットが1視点でも十分識別できる、固定4視点の配置が有効でない、あるいはレンダリング画像が分類に必要な差分を十分表現できていない可能性があります。

confusion matrixでは、似た体型、似たシルエット、突起や翼などの特徴が特定角度で隠れるクラスに注目します。
Single-viewで混同しやすいペアは、後続のFixed Ring-4やMVTNで改善しているかを見ると、複数視点や視点学習の効果を説明しやすくなります。

## この条件で分かること

- 1固定視点だけで識別可能なクラスと、姿勢変化に弱いクラスを把握できます。
- マルチビュー条件の改善幅を測るための基準値を得られます。
- データ準備、レンダリング、学習、評価の一連の処理が最小構成で動くかを確認できます。

## この条件では分からないこと

- 4視点を使うこと自体の効果は分かりません。
- 視点配置を学習することの効果は分かりません。
- 未知ポケモンへのopen-set汎化性能は評価していません。
- アセットの色やテクスチャを積極的に使った識別性能は評価していません。
