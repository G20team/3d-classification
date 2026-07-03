# Fixed Ring-4 MVCNN

Fixed Ring-4 MVCNN renders four circularly spaced views and aggregates view features with view-wise max
pooling. It is the main fixed-camera baseline for the MVTN condition.

## Purpose

This condition estimates the value of observing the same mesh from multiple directions while keeping camera
placement fixed. It separates the benefit of four views from the additional benefit of learned view
selection.

## Camera Setup

The base azimuths are arranged in a ring:

```text
0, 90, 180, 270 degrees
```

The base elevation is `0` degrees. Pose-split yaw/elevation offsets are applied on top of this base setup
during training and evaluation.

## Configs

Main config:

```text
configs/fixed_ring4.yaml
```

Debug config:

```text
configs/debug_fixed_ring4.yaml
```

Keep the backbone, feature dimension, dropout, and optimizer settings aligned with the MVTN config.

## Run

Debug:

```bash
uv run python scripts/train.py --config configs/debug_fixed_ring4.yaml
```

Full run:

```bash
uv run python scripts/train.py --config configs/fixed_ring4.yaml
```

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/fixed_ring4/.../checkpoints/best.ckpt \
  --split test
```

## What To Check

- Whether Top-1 accuracy and Macro-F1 improve over Single-view.
- Whether classes that were weak in Single-view improve with four views.
- Whether confusion between similarly shaped classes decreases.
- Whether all comparison settings match the MVTN condition.

## Comparison With MVTN

Fixed Ring-4 uses the same camera layout that initializes Learned Circular-4 MVTN. If MVTN outperforms this
condition, the difference is a candidate signal for the value of mesh-dependent camera adjustment. If the
difference is small, inspect whether MVTN camera offsets moved meaningfully and whether view collapse
occurred.
