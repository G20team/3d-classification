# Pokemon 3D MVTN Multi-View Identification

This repository contains an experimental environment for identifying Pokemon from rendered views of GLB
models from [Pokemon-3D-api/assets](https://github.com/Pokemon-3D-api/assets).

The task is not open-set recognition of unseen Pokemon. It is **closed-set cross-orientation asset
identification**: the same Pokemon catalog appears during training and evaluation, while the camera
orientations are split so the model must identify known assets from unseen poses.

The main research question is whether a four-view MVTN setup can learn Pokemon-dependent camera offsets
that improve identification under unseen orientations compared with a fixed circular four-view camera.

## Experiment Conditions

| Condition | Summary | Main config |
| --- | --- | --- |
| Single-view | Baseline that classifies from one fixed view. | `configs/single_view.yaml` |
| Fixed Ring-4 MVCNN | MVCNN baseline that aggregates four fixed circular views with view-wise max pooling. | `configs/fixed_ring4.yaml` |
| Learned Circular-4 MVTN | MVTN condition initialized from Fixed Ring-4 and trained to predict four camera angle offsets from mesh geometry. | `configs/mvtn_circular4.yaml` |

## Documentation

Start here:

- [Setup](docs/setup.md): `uv`, Python 3.10, PyTorch3D, environment checks, and developer checks.
- [Experiment workflow](docs/experiment_workflow.md): end-to-end order from setup to evaluation.
- [Data pipeline](docs/data_pipeline.md): asset download, audit, visual inspection, mesh cache, and pose splits.

Experiment guides:

- [Experiment design overview](docs/experiments/index.md): shared design, debug subset, and full runs.
- [Single-view](docs/experiments/single_view.md): baseline purpose, commands, and evaluation.
- [Fixed Ring-4 MVCNN](docs/experiments/fixed_ring4.md): fixed four-view purpose, commands, and evaluation.
- [Learned Circular-4 MVTN](docs/experiments/mvtn_circular4.md): learned-view purpose, commands, and camera logs.
- [Single-view details](docs/experiments/single_view_details.md): assumptions, method, metrics, and interpretation.
- [Fixed Ring-4 MVCNN details](docs/experiments/fixed_ring4_details.md): design rationale and comparison points.
- [Learned Circular-4 MVTN details](docs/experiments/mvtn_circular4_details.md): model structure, camera logs, and interpretation caveats.
- [Evaluation](docs/evaluation.md): checkpoint evaluation, metrics, condition comparison, and reporting.

Japanese documentation is preserved under [docs/ja](docs/ja/README.md).

## Quick Start

See the documentation above for details. The minimal command sequence is:

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

## Main Outputs

Each run is written under `outputs/<condition_id>/<timestamp>_seed<seed>/`.

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

`logs/` is used by TensorBoard. `config.yaml` and `environment_report.json` support reproducibility.
`camera_positions.json` and `learned_camera_visualization.png` are mainly used to inspect learned camera
behavior in the MVTN condition.

## Repository Layout

```text
configs/                  Experiment YAML files
docs/                     Public documentation
scripts/                  Command-line entry points
src/pokemon_3d_cls/
├── assets/               Asset audit and PokeAPI cache helpers
├── experiments/          Dataset, training, and metrics for mesh-rendered experiments
├── mesh/                 Mesh normalization and caching
├── models/               Encoders, MVCNN, MVTN, and camera utilities
├── rendering/            GLB helpers and PyTorch3D renderer
├── config.py             Config loading
├── environment.py        Environment diagnostics
├── io.py                 JSON/YAML/CSV/JSONL I/O
├── paths.py              Project-root-based path handling
└── splits.py             Pose splits
```

## Public Release Notes

- Pokemon assets, mesh caches, render caches, and training outputs are not tracked in Git.
- This repository documents how to obtain assets from upstream sources; it does not redistribute Pokemon assets.
- ViewFormer, View-GCN, direct point-cloud input, retrieval, open-set recognition, and comparisons with eight or more views are out of scope for the current implementation.
- Fixed Ring-4 and MVTN should use matched encoder, classifier, view count, image resolution, and optimizer settings so the main difference is the camera policy.
- Single-view is described as a `single fixed view`, not a front view, because the semantic front direction of the source meshes is not guaranteed.

## References

- [Pokemon 3D assets](https://github.com/Pokemon-3D-api/assets)
- [MVCNN](https://arxiv.org/abs/1505.00880)
- [MVCNN PyTorch reference](https://github.com/RBirkeland/MVCNN-PyTorch)
- [MVTN](https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html)
- [MVTN official code](https://github.com/ajhamdi/MVTN)
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)
- [PokeAPI](https://pokeapi.co/docs/v2)
