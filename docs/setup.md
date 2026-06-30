# 環境セットアップ

このドキュメントでは、実験を再現するためのPython環境、PyTorch3D導入、環境診断、開発用チェックをまとめます。

## 前提

- Pythonは3.10系を使います。
- パッケージ管理は `uv` を使います。
- 学習と評価は `uv run <command>` 経由で実行します。
- PyTorch3Dは通常のPyPI wheelだけで導入できない環境があるため、利用環境に合わせて別途導入します。

このリポジトリの `pyproject.toml` は `torch==2.4.1`、`torchvision==0.19.1` を前提にしています。
PyTorch3Dを使うmesh render実験では、Python、PyTorch、CUDA、PyTorch3D wheelの組み合わせを揃える必要があります。

## 基本セットアップ

```bash
uv python install 3.10
uv sync
```

`uv sync` が成功したら、まず環境診断を実行します。

```bash
uv run python scripts/bootstrap_env.py
```

診断結果は次に保存されます。

```text
outputs/environment_report.json
```

確認する項目:

- `python_version` が3.10系である。
- `torch_version` と `torchvision_version` が想定通りである。
- GPUを使う場合は `cuda_available` が `true` で、GPU名とメモリが記録されている。
- PyTorch3D導入後は `pytorch3d_forward_render_ok` と `pytorch3d_camera_gradient_ok` が `true` になる。

PyTorch3Dが未導入でも `bootstrap_env.py` 自体は失敗せず、`pytorch3d_error` に理由を記録します。
この挙動は、環境差分を成果物として残すためのものです。

## PyTorch3D

Linux版PyTorch3Dは通常のPyPIだけではwheelが見つからないことがあります。
`uv sync` 後、利用環境のCUDA/PyTorchに合う公式wheel indexまたはsource buildで導入してください。

例:

```bash
uv run python -m pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/<py_cuda_torch>/download.html
```

`<py_cuda_torch>` は `py310_cu121_pyt241` のような形式です。
GPUを使わない場合やwheelが合わない場合は、PyTorch3D公式手順に沿ってsource buildを検討します。

導入後は必ず再診断します。

```bash
uv run python scripts/bootstrap_env.py
```

PyTorch3Dが正しく動くと、forward renderingとcamera gradientの確認結果が `true` になります。
MVTN条件ではcamera角度に分類lossから勾配を流すため、camera gradientの確認が特に重要です。

## 開発用チェック

コード変更後は、最低限以下を実行します。

```bash
uv run ruff check .
uv run pyright
uv run pytest
```

このプロジェクトでは `ruff` と `pyright` の設定を `pyproject.toml` に集約しています。
VS Code側の設定は `.vscode/settings.json` を参照してください。

## 出力先

環境診断、学習結果、評価結果は `outputs/` 配下に保存されます。
アセットやmesh cacheは `data/` 配下に保存されます。
どちらも大きくなるためGit管理対象外です。

## トラブルシュート

`uv sync` が失敗する場合:

- Python 3.10が入っているか確認します。
- `pyproject.toml` の `requires-python = ">=3.10,<3.11"` に合っているか確認します。
- PyTorch3Dは通常依存に含めていないため、まずPyTorch3Dなしで `uv sync` が通る状態を作ります。

PyTorch3D importに失敗する場合:

- `outputs/environment_report.json` の `pytorch3d_error` を確認します。
- Python 3.10、PyTorch 2.4.1、CUDA、wheel indexの組み合わせを見直します。
- GPU driverやCUDA runtimeが実行環境に合っているか確認します。

GPUが使えない場合:

- `training.device` を `auto` にしている場合、CUDAが使えなければCPUへフォールバックします。
- CPUでも小さなdebug subsetは確認できますが、本実験は非常に遅くなる可能性があります。

IDEでimport警告が出る場合:

- `uv sync` が完了しているか確認します。
- VS Codeが `.venv` を見ているか確認します。
- `pyproject.toml` の `tool.pyright.extraPaths = ["src"]` が有効になっているか確認します。
