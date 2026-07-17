# Evaluation

Evaluation uses the best validation checkpoint saved during training and runs it on the selected split,
usually `test`.

## Run Evaluation

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/<condition>/<run_id>/checkpoints/best.ckpt \
  --split test
```

The command writes metrics and plots into the corresponding run directory unless another output path is
specified.

## Main Metrics

- Top-1 accuracy: the fraction of examples where the highest-scoring class is correct.
- Top-5 accuracy: the fraction of examples where the correct class appears in the five highest-scoring
  predictions.
- Macro-F1: the unweighted mean of per-class F1 scores.
- Per-class metrics: class-level precision, recall, and F1.
- Confusion matrix: pairwise class confusions.
- Per-pose metrics: sample count, Top-1, and Top-5 for each yaw/elevation condition in `pose_metrics.csv`.

Macro-F1 is important because the catalog has many classes and a high aggregate accuracy can hide weak
performance on a subset of classes.

## Condition Comparison

Compare conditions in this order:

1. Single-view vs. Fixed Ring-4: estimates the value of using multiple views.
2. Fixed Ring-4 vs. Learned Circular-4 MVTN: estimates the additional value of learned camera placement.
3. Fixed Ring-4 vs. View Transformer-4: estimates the value of attention-based view aggregation.
4. Per-class and confusion-matrix changes: identifies which shapes benefit or regress.

For the MVTN condition, do not rely only on classification metrics. Inspect:

```text
camera_positions.json
learned_camera_visualization.png
```

Check whether learned offsets move away from zero in a meaningful range and whether views collapse into
nearly identical directions.

## Reporting

A concise report should include:

- dataset and split versions,
- config files and run IDs,
- Top-1, Top-5, and Macro-F1 for all conditions,
- a comparison table,
- notable per-class improvements or regressions,
- MVTN camera-log observations,
- environment details from `environment_report.json`.
