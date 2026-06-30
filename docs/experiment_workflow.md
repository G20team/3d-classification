# Pokémon 3D MVTN実験の進め方

このドキュメントは、Pokemon-3D-api/assets のGLB形式3Dモデルを使って、
ポケモン名識別のマルチビュー認識実験を再現するための実行手順をまとめたものです。
単なるコマンド列ではなく、各段階で何を作り、何を確認すべきかも併記します。

## 実験の位置づけ

この実験は「未知ポケモンを当てる」open-set分類ではありません。学習時に登場した既知カタログ内の
ポケモンについて、見たことのない姿勢・角度からレンダリングされた画像を正しく識別できるかを調べます。
このタスクを **closed-set cross-orientation asset identification** と呼びます。

比較する条件は以下の3つです。

- **Single-view**: 1つの固定視点だけから識別する基準条件。
- **Fixed Ring-4 MVCNN**: 円環状に固定した4視点をレンダリングし、共有ResNet-18とview-wise max poolingで識別する条件。
- **Learned Circular-4 MVTN**: 固定Ring-4を初期値として、mesh形状から4視点のカメラ角度補正を学習する条件。

中心的な問いは、同じ4視点という制約下で、MVTNがポケモン形状に応じて視点を調整することで、
固定Ring-4より未知姿勢への識別性能を改善できるか、です。

## 前提

実行環境は `uv` とPython 3.10を前提にします。PyTorch3DはLinuxでは通常のPyPI wheelだけで入らない場合があるため、
利用環境のCUDA/PyTorchに合う公式wheel indexまたはsource buildで別途導入します。

アセット、mesh cache、学習出力は大きくなるためGit管理しません。すべて `data/` または `outputs/` 配下に作成されます。

## 1. 環境診断

最初に、Python、PyTorch、torchvision、CUDA、GPU、PyTorch3Dの状態を記録します。
この段階の目的は、後から実験結果を見返したときに「どの実行環境で得た結果か」を再現できるようにすることです。

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

出力:

```text
outputs/environment_report.json
```

確認ポイント:

- `python_version` が3.10系である。
- `torch_version` と `torchvision_version` が想定通りである。
- GPU実験を行う場合、`cuda_available` が `true` でGPU名とメモリが記録されている。
- PyTorch3D導入後は `pytorch3d_forward_render_ok` と `pytorch3d_camera_gradient_ok` が `true` になる。

PyTorch3Dが未導入の場合でも、この診断は失敗せず、`pytorch3d_error` に理由を記録します。

## 2. アセット取得と監査

次に、Pokemon-3D-api/assets のGLBアセットを取得し、実験に使える通常形ポケモンだけを選別します。
ここでは外部リポジトリの内部ファイル構成を決め打ちせず、`*.glb` を再帰的に探索します。

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl
```

主な出力:

```text
data/manifests/asset_audit.jsonl
data/manifests/selected_regular.jsonl
data/manifests/asset_audit_summary.json
data/manifests/asset_audit_summary.md
data/manifests/pokeapi_cache.json
```

`asset_audit.jsonl` には、読み込めたか、頂点数・面数、テクスチャ有無、推定したNational Dex ID、
PokeAPI由来の英語名、除外理由を1アセット1行で保存します。

`selected_regular.jsonl` は以降の実験で使う採用クラス一覧です。初期実験では、色違い、メガシンカ、
リージョンフォーム、特殊形態、性別差分、同一ID重複を除外します。

確認ポイント:

- 採用数が想定より極端に少なくない。
- `asset_audit_summary.md` の除外理由に、読み込み失敗やID不明が大量発生していない。
- 除外されたアセットにも理由が記録されており、無言で捨てられていない。

PokeAPIへアクセスできない環境では、既存の `pokeapi_cache.json` が必要です。

## 3. 可視確認

監査で採用されたアセットからランダムに抽出し、簡易turntable画像を1枚のcontact sheetとして出力します。
これは数値では見つけにくい問題、例えば極端に小さいmesh、上下が不自然なmesh、壊れた形状などを目視で確認するための段階です。

```bash
uv run python scripts/render_contact_sheet.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output outputs/asset_audit_contact_sheet.png \
  --num-samples 50
```

出力:

```text
outputs/asset_audit_contact_sheet.png
```

確認ポイント:

- ランダムサンプルが画像として表示される。
- 明らかに破綻したmeshや極端なスケールのアセットがない。
- この画像は最終評価ではなく、アセット監査の補助資料として扱う。

## 4. Mesh Cacheと姿勢Split作成

学習時に毎回GLBを直接読むと遅く、失敗の原因も増えます。そのため採用GLBを一度だけ正規化し、
PyTorchで読みやすいcacheとして保存します。

```bash
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache
```

mesh cacheでは、geometryを結合し、不要頂点や縮退面を除去し、bounding box中心を原点へ移動し、
unit sphere相当のスケールへ正規化します。

次に、train/validation/testの姿勢条件を作ります。

```bash
uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

重要なのは、split単位がポケモンIDではなく姿勢条件であることです。すべてのポケモンIDは
train/validation/testの全splitに含まれます。評価したいのは未知ポケモンへの一般化ではなく、
同じポケモンを未知姿勢から識別できるかです。

出力:

```text
data/mesh_cache/
data/mesh_cache/mesh_cache_manifest.jsonl
data/manifests/pose_splits.json
```

確認ポイント:

- `validate_splits.py` が成功する。
- train/validation/testでyaw/elevation条件が重複していない。
- `mesh_cache_manifest.jsonl` の件数が `selected_regular.jsonl` と大きくずれていない。

## 5. Debug Subset

本実験へ進む前に、10クラスだけを使った小規模debug subsetで実装全体の動作を確認します。
ここでの目的は高い汎化性能を出すことではなく、データ読み込み、レンダリング、学習、評価、保存処理、
MVTNの勾配が一通り機能することを確かめることです。

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

成功条件:

- 3条件すべてで学習が開始できる。
- 10クラスsubsetで訓練lossが下がる。
- Fixed Ring-4とSingle-viewが同じ分割・同じ評価指標で比較できる。
- MVTNのcamera offsetへ分類lossから勾配が流れる。
- MVTNの視点座標が初期Ring-4から変化する。
- `config.yaml`, `metrics.json`, checkpoint, camera logが保存される。

debugが失敗した場合は、本実験へ進まず、まず `outputs/<debug_experiment>/.../` のログと
`outputs/environment_report.json` を確認します。

## 6. 本実験

debug subsetが通ったら、全採用アセットで3条件を実行します。seedを変えた複数runを行い、
結果のばらつきも確認します。seedは各configの `experiment.seed` で管理します。

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

学習ではvalidation macro-F1をモデル選択指標にし、best checkpointを保存します。
評価は保存済みcheckpointに対してtest splitを指定して実行します。

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

主なrun出力:

```text
outputs/<experiment>/<timestamp>_seed<seed>/
├── config.yaml
├── environment_report.json
├── metadata.json
├── metrics.json
├── per_class_metrics.csv
├── confusion_matrix.png
├── checkpoints/best.ckpt
├── logs/
├── rendered_examples/
├── camera_positions.json
└── learned_camera_visualization.png
```

## 7. 結果確認

最終的には、3条件を同じ評価指標で比較します。

見るべき指標:

- `Top-1 Accuracy`: 最上位予測が正解した割合。
- `Top-5 Accuracy`: 上位5候補に正解が含まれた割合。
- `Macro-F1`: クラスごとのF1を平均した値。クラス間の偏りに比較的強い。
- `per_class_metrics.csv`: どのポケモンで失敗しやすいか。
- `confusion_matrix.png`: 混同しやすいポケモンの組み合わせ。
- `camera_positions.json`: MVTNがどのような視点へ動いたか。
- `learned_camera_visualization.png`: 固定Ring-4と学習視点の変化の概要。

解釈時の注意:

- 実験差分は原則として視点配置だけに限定する。
- Fixed Ring-4とMVTNでencoder、classifier、視点数、画像解像度、optimizer設定が揃っていることを確認する。
- MVTNが性能改善していても、view collapseが起きている場合はcamera logを確認する。
- Single-viewは「正面」とは呼ばず、meshの正面方向が保証されないため `single fixed view` として扱う。

## トラブルシュート

- PyTorch3D importに失敗する場合:
  `scripts/bootstrap_env.py` の `pytorch3d_error` を確認し、Python 3.10、PyTorch 2.4.1、CUDA wheelの組み合わせを見直す。
- PokeAPI取得に失敗する場合:
  `data/manifests/pokeapi_cache.json` があるか確認し、offline前提ならcacheを共有する。
- split検証に失敗する場合:
  `configs/splits.yaml` のyaw/elevation条件がtrain/validation/testで重複していないか確認する。
- debug subsetが過学習しない場合:
  render画像、label map、mesh cache、学習率、MVTN camera logを順に確認する。
