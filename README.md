# ポケモン3D MVTNマルチビュー個体識別実験環境

Pokemon-3D-api/assets のGLB形式3Dアセットを用い、複数視点からレンダリングした画像から
ポケモン名を識別する実験環境です。タスクは未知ポケモン分類ではなく、既知カタログ内の
ポケモンを未知姿勢から識別する **closed-set cross-orientation asset identification** として扱います。

比較する条件は以下です。

- Single-view: single fixed view
- Fixed Ring-4 MVCNN: 固定4視点 + view-wise max pooling
- Learned Circular-4 MVTN: 学習可能な4視点 + view-wise max pooling

研究上の問いは、限られた視点数で3D形状に応じたカメラ位置を学習するMVTNが、固定円環状カメラ配置より未知姿勢識別を改善できるか、です。

## セットアップ

既存方針に合わせて `uv` を使います。

```bash
uv sync
```

PyTorch3Dのwheel互換性に合わせ、このリポジトリはPython 3.10を前提にします。`uv` は `.python-version` を見て
Python 3.10環境を作成します。

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

Linux版PyTorch3DはPyPI通常wheelではなく、公式の専用wheel indexまたはsource buildで導入します。
`uv sync` 完了後、利用環境のCUDA/PyTorchに合わせて以下のように追加してください。

```bash
uv run python -m pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/<py_cuda_torch>/download.html
```

`<py_cuda_torch>` は例として `py310_cu121_pyt241` のような形式です。導入後に
`uv run python scripts/bootstrap_env.py` でforward renderingとcamera gradientを確認します。

診断結果は `outputs/environment_report.json` に保存されます。

## 実験の流れ

共有用の短い手順は [docs/experiment_workflow.md](docs/experiment_workflow.md) を参照してください。

### 1. アセット取得

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
```

### 2. アセット監査

```bash
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl
```

監査では `*.glb` を再帰探索し、読み込み失敗、空mesh、NaN/Inf、面なしmesh、通常形以外、ID不明、重複IDを除外理由付きで記録します。採用クラスは `data/manifests/selected_regular.jsonl` に保存されます。

### 3. 可視確認

```bash
uv run python scripts/render_contact_sheet.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output outputs/asset_audit_contact_sheet.png \
  --num-samples 50
```

### 4. mesh cacheと姿勢split

```bash
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache

uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

split単位はポケモンIDではなく姿勢条件です。ポケモンIDはtrain/validation/testのすべてに含まれます。

### 5. debug subset

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

### 6. 本実験

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

評価:

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

## 出力

各runは以下のように保存されます。

```text
outputs/<experiment>/<timestamp>_seed<seed>/
├── config.yaml
├── environment_report.json
├── metadata.json
├── metrics.json
├── per_class_metrics.csv
├── confusion_matrix.png
├── checkpoints/
│   └── best.ckpt
├── logs/
├── rendered_examples/
├── camera_positions.json
└── learned_camera_visualization.png
```

## 構成

```text
configs/                  実験YAML
docs/                     共有用手順
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
- MVTNとFixed Ring-4ではencoder、classifier、視点数、画像解像度、最適化条件を揃え、差分を原則として視点配置だけにします。
- 実装上の判断と制約は [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) に記録しています。

## 参考

- Pokémon 3D assets: https://github.com/Pokemon-3D-api/assets
- MVCNN: https://arxiv.org/abs/1505.00880
- MVCNN PyTorch reference: https://github.com/RBirkeland/MVCNN-PyTorch
- MVTN: https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html
- MVTN official code: https://github.com/ajhamdi/MVTN
- PyTorch3D: https://github.com/facebookresearch/pytorch3d
- PokeAPI: https://pokeapi.co/docs/v2
