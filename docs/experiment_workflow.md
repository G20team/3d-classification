# Experiment Workflow

Run the project in this order:

1. [Setup](setup.md)
2. [Data pipeline](data_pipeline.md)
3. [Debug subset for all three conditions](experiments/index.md#debug-subset)
4. [Full experiments](experiments/index.md#full-runs)
5. [Checkpoint evaluation on the test split](evaluation.md)
6. [Comparison across conditions](evaluation.md#condition-comparison)

## Conditions

- [Single-view](experiments/single_view.md): baseline using one fixed view.
- [Fixed Ring-4 MVCNN](experiments/fixed_ring4.md): MVCNN using four fixed circular views.
- [Learned Circular-4 MVTN](experiments/mvtn_circular4.md): learned camera offsets predicted from mesh geometry.

## Command Sequence

Environment:

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

Data:

```bash
uv run python scripts/fetch_assets.py --output data/raw_assets
uv run python scripts/audit_assets.py \
  --asset-root data/raw_assets \
  --output data/manifests/asset_audit.jsonl
uv run python scripts/prepare_mesh_cache.py \
  --manifest data/manifests/selected_regular.jsonl \
  --output-root data/mesh_cache
uv run python scripts/build_splits.py --config configs/splits.yaml
uv run python scripts/validate_splits.py --config configs/splits.yaml
```

Debug runs:

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

Full runs:

```bash
uv run python scripts/train.py --config configs/single_view.yaml
uv run python scripts/train.py --config configs/fixed_ring4.yaml
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

Evaluation:

```bash
uv run python scripts/evaluate.py --checkpoint outputs/.../checkpoints/best.ckpt --split test
```

## When Something Fails

- Environment issues: [Setup](setup.md#troubleshooting)
- Asset or split issues: [Data pipeline](data_pipeline.md)
- Condition selection: [Experiment design overview](experiments/index.md)
- Metric interpretation: [Evaluation](evaluation.md)
