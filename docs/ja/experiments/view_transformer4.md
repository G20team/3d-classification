# View Transformer-4

View Transformer-4は、Fixed Ring-4と同じ4枚のレンダリング画像を共有ResNet18 encoderへ通し、得られた
view特徴をTransformerで統合する比較条件です。カメラ位置、backbone、feature dimension、optimizerは
Fixed Ring-4と揃え、view-wise max poolingをattentionへ変更した効果を調べます。

```bash
uv run python scripts/train.py --config configs/debug_view_transformer4.yaml
uv run python scripts/train.py --config configs/view_transformer4.yaml
uv run python scripts/evaluate.py \
  --checkpoint outputs/view_transformer4/<run_id>/checkpoints/best.ckpt \
  --split test
```

モデルは4個のview tokenにCLS tokenと学習可能なview位置埋め込みを加え、2層・8 headのpre-norm
Transformerへ入力します。最終CLS表現を分類ヘッドへ渡します。評価時は全体指標に加えて
`pose_metrics.csv` を確認し、改善が特定角度だけに偏っていないかを確認してください。
