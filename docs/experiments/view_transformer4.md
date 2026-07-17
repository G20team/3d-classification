# View Transformer-4

View Transformer-4 uses the same four rendered views and shared ResNet18 encoder as Fixed Ring-4, but
replaces view-wise max pooling with Transformer aggregation. This isolates the effect of attention-based
view fusion while keeping camera placement, feature dimension, and optimization settings aligned.

```bash
uv run python scripts/train.py --config configs/debug_view_transformer4.yaml
uv run python scripts/train.py --config configs/view_transformer4.yaml
uv run python scripts/evaluate.py \
  --checkpoint outputs/view_transformer4/<run_id>/checkpoints/best.ckpt \
  --split test
```

The four view tokens are combined with a CLS token and learned view-position embeddings, then processed by
a two-layer, eight-head pre-norm Transformer. Inspect `pose_metrics.csv` alongside the aggregate metrics to
check whether improvements are consistent across camera offsets.
