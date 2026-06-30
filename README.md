# ポケモン3Dモデル識別実験環境

`.glb` のポケモン3Dモデルからシルエット画像データセットを生成し、MVCNN
(Multi-View CNN) で個体識別を行うための実験環境です。

コードは `src/pokemon_3d_cls/` に分割し、実行入口は `scripts/` 配下の薄いCLIに
統一しています。設定は YAML を主入口にします。

## セットアップ

```bash
uv sync
```

開発時の基本コマンド:

```bash
uv run pytest
uv run ruff check .
uv run pyright
```

## データセット生成

`configs/generate_silhouettes.yaml` を編集してから実行します。

```bash
uv run python scripts/generate_silhouettes.py --config configs/generate_silhouettes.yaml
```

主な設定:

- `input.path`: `.glb` ファイル、または `.glb` を含むディレクトリ
- `output.dataset_root`: 生成画像の出力先
- `rendering.views`: 1モデルあたりの生成枚数
- `rendering.mode`: `quiz` / `turntable` / `sphere`
- `labels.mode`: `stem` / `species`

出力例:

```text
data/dataset/
├── 128-1/
│   ├── 128-1_000.png
│   └── ...
├── manifest.csv
└── generation_config.yaml
```

`DracoPy` が入っていれば、`KHR_draco_mesh_compression` で圧縮されたGLBにも対応します。

## 学習

`configs/train_mvcnn.yaml` を編集してから実行します。

```bash
uv run python scripts/train.py --config configs/train_mvcnn.yaml
```

既定では以下の設定です。

- backbone: `resnet18`
- pretrained: `true`
- image size: `224`
- views: `24`
- holdout stride: `4`
- output root: `outputs/runs`

学習結果は以下に保存されます。

```text
outputs/runs/<condition_id>/<run_id>/
├── config.yaml
├── label_map.json
├── metrics.json
├── confusion_matrix.pt
├── checkpoints/
│   └── best_model.pt
└── tensorboard/
```

TensorBoard:

```bash
uv run tensorboard --logdir outputs/runs
```

## 構成

```text
configs/                  YAML設定例
scripts/                  実行CLI
src/pokemon_3d_cls/
├── config.py             設定読み込みと型検証
├── data.py               Datasetとtransform
├── evaluation.py         精度・混同行列
├── generation.py         シルエット生成パイプライン
├── io.py                 JSON/YAML/CSV入出力
├── models.py             MVCNNとbackbone
├── paths.py              project root基準のpath管理
├── training.py           学習パイプライン
└── rendering/glb.py      GLB解析とレンダリング
tests/                    単体・スモークテスト
```

## 注意

- 実データ、生成済みdataset、学習出力は `.gitignore` 対象です。
- 旧トップレベルの `train.py` / `glb_silhouette_dataset.py` は正式入口から外し、
  `scripts/` 配下のCLIへ移行しました。
- ResNet18の事前学習重みを使う場合、初回実行時に `torchvision` が重みを取得することがあります。
