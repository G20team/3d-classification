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

mesh cacheを直接可視化することもできます。

```bash
uv run python scripts/inspect_mesh_cache.py \
  data/mesh_cache/0001_bulbasaur.pt \
  --output-root data/mesh_preview
```

このコマンドはcache内のtensor構造、頂点数、面数、bounding boxを表示し、確認用のPLYとHTMLを
`data/mesh_preview` に保存します。`torch.load(..., weights_only=False)` を使うため、
自分で作成した信頼できるcacheに対してのみ実行してください。

## 5. 姿勢Split作成

標準実験では「ポケモンID×姿勢条件」を単位にsplitを作ります。すべてのポケモンIDと17種類の姿勢条件を
各splitへ含め、角度分布の差によってvalidation/testだけが難しくならないようにします。同じID・同じ姿勢の
サンプルが複数splitへ入ることはありません。

```bash
uv run python scripts/build_splits.py --config configs/splits_stratified.yaml
uv run python scripts/validate_splits.py --splits data/manifests/pose_splits_stratified_seed0.json
```

各ポケモンについて17姿勢を次の件数で割り当てます。

```text
train: 9、validation: 4、test: 4
```

主な出力:

```text
data/manifests/pose_splits_stratified_seed0.json
```

確認する項目:

- `validate_splits.py` が成功する。
- 各ポケモンがtrain/validation/testに9/4/4件ずつ含まれる。
- 各splitが17姿勢をすべて含み、姿勢ごとの件数差が2件以内である。
- source manifestのSHA-256が一致し、同じサンプル割当がsplit間で重複しない。
- 全比較条件で同じ `pose_splits_stratified_seed0.json` を使う。

旧来の姿勢条件を完全に分離した `configs/splits.yaml` と `pose_splits.json` は、過去checkpointの再現用として残します。

## 6. 固定視点RGB Render Cache

Single-view、Fixed Ring-4、View Transformerでは、PyTorch3Dの固定視点RGB画像を一度だけ生成し、
30 epochで再利用します。PNGはsampleごとに全viewを横連結し、元manifest、split、描画設定、
カメラ角度のSHA-256から決まるディレクトリへ保存されます。

```bash
uv run python scripts/prepare_rgb_render_cache.py --config configs/single_view.yaml
uv run python scripts/prepare_rgb_render_cache.py --config configs/fixed_ring4.yaml
```

Fixed Ring-4とView Transformerは同じ固定4視点なので、2つ目のキャッシュを共有します。
MVTNは学習中にカメラ角度が変わり、rendererを通じた勾配が必要なため、このキャッシュを使用しません。
標準configでキャッシュが未生成の場合、学習開始時に生成コマンドを含むエラーが表示されます。

## 次の段階

データ準備が終わったら、[実験設計の概要](experiments/index.md) に進み、まずdebug subsetを実行します。
