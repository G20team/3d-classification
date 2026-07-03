# Single-view

Single-view is the baseline condition that classifies each asset from one fixed camera direction. It does
not use multi-view aggregation or learned camera placement.

## Purpose

This condition estimates how well the catalog can be identified from the smallest image input. It should be
used as the lower baseline for interpreting Fixed Ring-4 and MVTN results.

The condition should be described as a `single fixed view`, not as a front view, because the source meshes
do not guarantee a semantic front direction.

## Configs

Main config:

```text
configs/single_view.yaml
```

Debug config:

```text
configs/debug_single_view.yaml
```

## Run

Complete [data preparation](../data_pipeline.md) before training.

Debug:

```bash
uv run python scripts/train.py --config configs/debug_single_view.yaml
```

Full run:

```bash
uv run python scripts/train.py --config configs/single_view.yaml
```

The training command prints the run directory when it finishes.

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/single_view/.../checkpoints/best.ckpt \
  --split test
```

## What To Check

- Whether one fixed view is enough for many classes.
- Which classes are confused because important shape cues are hidden from the fixed direction.
- How much Fixed Ring-4 improves over this baseline.
