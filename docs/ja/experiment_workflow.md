# 実験ワークフロー

このドキュメントは、環境構築から最終評価までの順序を確認するための入口です。
各段階の詳しい説明は分割されたドキュメントへ移しています。

## 全体像

1. [環境セットアップ](setup.md)
2. [データ準備](data_pipeline.md)
3. [debug subsetで3条件を確認](experiments/index.md#debug-subset)
4. [本実験を実行](experiments/index.md#本実験)
5. [checkpointをtest splitで評価](evaluation.md)
6. [3条件を比較して結果を解釈](evaluation.md#条件間比較)

## 実験の位置づけ

この実験は「未知ポケモンを当てる」open-set分類ではありません。
学習時に登場した既知カタログ内のポケモンについて、見たことのない姿勢・角度からレンダリングされた画像を
正しく識別できるかを調べます。このタスクを **closed-set cross-orientation asset identification** と呼びます。

比較する条件は以下の3つです。

- [Single-view](experiments/single_view.md): 1つの固定視点だけから識別する基準条件。
- [Fixed Ring-4 MVCNN](experiments/fixed_ring4.md): 円環状に固定した4視点を使うMVCNN条件。
- [Learned Circular-4 MVTN](experiments/mvtn_circular4.md): mesh形状から4視点の角度補正を学習する条件。

## 推奨コマンド列

環境診断:

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

データ準備:

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl
uv run python scripts/render_contact_sheet.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output outputs/asset_audit_contact_sheet.png \
  --num-samples 50
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache
uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

debug subset:

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

本実験:

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

評価:

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

## 次に読むもの

- 環境で詰まった場合: [環境セットアップ](setup.md#トラブルシュート)
- アセットやsplitを確認したい場合: [データ準備](data_pipeline.md)
- どの条件から実行すべきか迷う場合: [実験設計の概要](experiments/index.md)
- metricsの読み方を確認したい場合: [評価と結果解釈](evaluation.md)
