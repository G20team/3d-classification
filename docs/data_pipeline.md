# データ準備

このドキュメントでは、GLBアセットの取得から学習用mesh cacheと姿勢splitを作るまでの流れを説明します。

## 目的

実験では毎回GLBを直接読み込むのではなく、事前に以下を作成します。

- 監査済みの採用アセットmanifest
- 目視確認用のcontact sheet
- 正規化済みmesh cache
- train/validation/testの姿勢split

これにより、学習時の読み込み失敗を減らし、どのアセットと姿勢条件で実験したかを再現しやすくします。

## 1. アセット取得

Pokemon-3D-api/assets を浅いcloneで取得します。

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
```

既に `data/raw_assets` が存在する場合、スクリプトは再cloneせず終了します。
別の場所にアセットを置きたい場合は `--output` を変更します。

## 2. アセット監査

取得したGLBを再帰探索し、実験に使える通常形ポケモンだけを選別します。

```bash
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

`asset_audit.jsonl` には1アセット1行で、読み込み可否、頂点数、面数、テクスチャ有無、推定National Dex ID、
PokeAPI由来の英語名、除外理由が保存されます。

`selected_regular.jsonl` は以降の実験で使う採用クラス一覧です。
初期実験では、色違い、メガシンカ、リージョンフォーム、特殊形態、性別差分、同一ID重複を除外します。

PokeAPIへアクセスできない環境では、既存cacheだけを使います。

```bash
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl \
  --offline
```

確認する項目:

- `selected_regular.jsonl` の件数が極端に少なくない。
- `asset_audit_summary.md` の除外理由に読み込み失敗やID不明が大量発生していない。
- 除外されたアセットにも理由が記録されている。

## 3. 可視確認

採用アセットをランダム抽出し、簡易turntable画像をcontact sheetとして保存します。
数値だけでは見つけにくい破綻、極端なスケール、上下方向の違和感を確認するための工程です。

```bash
uv run python scripts/render_contact_sheet.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output outputs/asset_audit_contact_sheet.png \
  --num-samples 50 \
  --views 4 \
  --image-size 128
```

確認する項目:

- ランダムサンプルが画像として表示される。
- 明らかに壊れたmeshが混ざっていない。
- 極端に小さい、極端に大きい、上下が大きくずれているアセットが大量にない。

このcontact sheetは最終評価ではなく、アセット監査の補助資料です。

## 4. Mesh Cache作成

採用GLBを正規化し、PyTorchで読みやすいcacheとして保存します。

```bash
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache
```

mesh cacheでは、geometryを結合し、不要頂点や縮退面を除去し、bounding box中心を原点へ移動し、
unit sphere相当のスケールへ正規化します。

主な出力:

```text
data/mesh_cache/
data/mesh_cache/mesh_cache_manifest.jsonl
```

確認する項目:

- `mesh_cache_manifest.jsonl` の件数が `selected_regular.jsonl` と大きくずれていない。
- 読み込み失敗が出た場合、元GLBの監査結果と除外理由を確認する。

## 5. 姿勢Split作成

この実験ではポケモンIDではなく、姿勢条件でtrain/validation/testを分けます。
すべてのポケモンIDは全splitに含まれ、未知ポケモンではなく未知姿勢への一般化を評価します。

```bash
uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

`configs/splits.yaml` の既定値:

```text
train:      yaw = [-20, 0, 20], elevation = [-10, 0, 10]
validation: yaw = [-30, 30],    elevation = [-15, 15]
test:       yaw = [-45, 45],    elevation = [-25, 25]
```

主な出力:

```text
data/manifests/pose_splits.json
```

確認する項目:

- `validate_splits.py` が成功する。
- train/validation/testでyaw/elevation条件が重複していない。
- 本実験前にsplitを変更した場合、全条件で同じ `pose_splits.json` を使う。

## 次の段階

データ準備が終わったら、[実験設計の概要](experiments/index.md) に進み、まずdebug subsetを実行します。
