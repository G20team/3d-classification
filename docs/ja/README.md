# ポケモン3D MVTNマルチビュー個体識別実験環境

Pokemon-3D-api/assets のGLB形式3Dアセットを用い、複数視点からレンダリングした画像で
ポケモン名を識別する実験環境です。

このリポジトリで扱うタスクは未知ポケモン分類ではなく、学習時に登場した既知カタログ内の
ポケモンを未知姿勢から識別する **closed-set cross-orientation asset identification** です。
研究上の主な問いは、限られた4視点という条件で、MVTNがポケモン形状に応じてカメラ位置を学習することで、
固定円環状カメラ配置より未知姿勢識別を改善できるか、です。

## 比較条件

| 条件 | 概要 | 主なconfig |
| --- | --- | --- |
| Single-view | 1つの固定視点だけで識別する基準条件 | `configs/single_view.yaml` |
| Fixed Ring-4 MVCNN | 4つの固定円環視点をview-wise max poolingで統合する条件 | `configs/fixed_ring4.yaml` |
| Learned Circular-4 MVTN | 固定Ring-4を初期値に、mesh形状から4視点の角度補正を学習する条件 | `configs/mvtn_circular4.yaml` |

## ドキュメント

最初に読むもの:

- [環境セットアップ](setup.md): `uv`、Python 3.10、PyTorch3D、環境診断、開発用チェック。
- [実験ワークフロー](experiment_workflow.md): セットアップから評価までの全体順序。
- [データ準備](data_pipeline.md): アセット取得、監査、可視確認、mesh cache、姿勢split。

実験条件ごとの詳細:

- [実験設計の概要](experiments/index.md): 3条件の比較設計、共通設定、debug subset、本実験の進め方。
- [Single-view](experiments/single_view.md): 基準条件の目的、実行、評価。
- [Fixed Ring-4 MVCNN](experiments/fixed_ring4.md): 固定4視点条件の目的、実行、評価。
- [Learned Circular-4 MVTN](experiments/mvtn_circular4.md): 学習視点条件の目的、実行、camera log確認。
- [Single-view 詳細解説](experiments/single_view_details.md): 1固定視点条件の目的、前提、手法、評価観点。
- [Fixed Ring-4 MVCNN 詳細解説](experiments/fixed_ring4_details.md): 固定4視点MVCNN条件の設計意図と比較方法。
- [Learned Circular-4 MVTN 詳細解説](experiments/mvtn_circular4_details.md): 学習視点条件のモデル構成、camera log、解釈上の注意。
- [評価と結果解釈](evaluation.md): checkpoint評価、metrics、条件間比較、レポート観点。

## 最短実行例

詳細は各ドキュメントを参照してください。ここでは全体の流れだけを示します。

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl
uv run python scripts/render_contact_sheet.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output outputs/asset_audit_contact_sheet.png
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache
uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

## 主な出力

各runは `outputs/<condition_id>/<timestamp>_seed<seed>/` 配下に保存されます。

```text
config.yaml
environment_report.json
metadata.json
metrics.json
per_class_metrics.csv
confusion_matrix.png
checkpoints/best.ckpt
logs/
rendered_examples/
camera_positions.json
learned_camera_visualization.png
```

`logs/` はTensorBoard用、`config.yaml` と `environment_report.json` は再現性確認用です。
`camera_positions.json` と `learned_camera_visualization.png` は主にMVTN条件の視点学習を確認するために使います。

## リポジトリ構成

```text
configs/                  実験YAML
docs/                     手順と設計メモ
scripts/                  実行CLI
src/pokemon_3d_cls/
├── assets/               アセット監査とPokeAPI cache
├── experiments/          mesh render実験のDataset/学習/metrics
├── mesh/                 mesh正規化とcache
├── models/               encoder/MVCNN/MVTN/camera
├── rendering/            GLB補助とPyTorch3D renderer
├── config.py             設定読み込み
├── environment.py        環境診断
├── io.py                 JSON/YAML/CSV/JSONL
├── paths.py              project root基準のpath管理
└── splits.py             姿勢split
```

## 制約

- アセット、mesh cache、render cache、学習出力はGit管理しません。
- ViewFormer、View-GCN、点群直接入力、retrieval、open-set、8視点以上比較は今回の実装対象外です。
- Fixed Ring-4とMVTNではencoder、classifier、視点数、画像解像度、optimizer設定を揃え、差分を原則として視点配置だけにします。
- Single-viewは「正面」とは呼ばず、meshの正面方向が保証されないため `single fixed view` として扱います。

## 参考

- [Pokémon 3D assets](https://github.com/Pokemon-3D-api/assets)
- [MVCNN](https://arxiv.org/abs/1505.00880)
- [MVCNN PyTorch reference](https://github.com/RBirkeland/MVCNN-PyTorch)
- [MVTN](https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html)
- [MVTN official code](https://github.com/ajhamdi/MVTN)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [PokeAPI](https://pokeapi.co/docs/v2)
