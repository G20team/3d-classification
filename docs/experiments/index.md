# Experiment Design Overview

This project compares four conditions for closed-set Pokemon asset identification with matched pose distributions.
All conditions should use the same audited asset catalog, mesh cache, pose split, image resolution, and
training/evaluation protocol wherever possible.

## Shared Assumptions

- Input assets come from GLB files selected by the asset audit.
- Meshes are normalized and cached before training.
- Train, validation, and test contain the same 17-pose distribution without duplicating a Pokemon-pose sample.
- The task measures identification of known catalog items under pose-balanced sampling.
- Generated data and outputs are local artifacts and are not tracked in Git.

If manifests or pose splits have not been created yet, run the [data pipeline](../data_pipeline.md) first.

## Conditions

| Condition | Purpose | Config |
| --- | --- | --- |
| Single-view | Lower baseline using one fixed observation direction. | `configs/single_view.yaml` |
| Fixed Ring-4 MVCNN | Multi-view baseline with four fixed circular cameras. | `configs/fixed_ring4.yaml` |
| Learned Circular-4 MVTN | Learned camera condition initialized from Fixed Ring-4. | `configs/mvtn_circular4.yaml` |
| View Transformer-4 | Attention-based aggregation of four shared-CNN view features. | `configs/view_transformer4.yaml` |

## Debug Subset

Before full runs, verify all four conditions on the 10-class debug subset:

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
uv run python scripts/train.py --config configs/debug_view_transformer4.yaml
```

Use the debug subset to catch environment, data, rendering, and model-shape issues. If debug runs fail,
inspect the run directory under `outputs/<debug_experiment>/.../` before starting full experiments.

## Full Runs

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
uv run python scripts/train.py --config configs/view_transformer4.yaml
```

Keep the configs aligned across conditions so the main difference is the camera/view policy. In particular,
match the backbone, feature dimension, dropout, optimizer settings, split file, and image size unless a
specific ablation intentionally changes one of them.

## Evaluation

Evaluate each run from its best checkpoint:

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

See [Evaluation](../evaluation.md) for metric definitions and comparison guidance.

## Guides

- [Single-view](single_view.md)
- [Fixed Ring-4 MVCNN](fixed_ring4.md)
- [Learned Circular-4 MVTN](mvtn_circular4.md)
- [View Transformer-4](view_transformer4.md)

Detailed explanations:

- [Single-view details](single_view_details.md)
- [Fixed Ring-4 MVCNN details](fixed_ring4_details.md)
- [Learned Circular-4 MVTN details](mvtn_circular4_details.md)
