# Single-view Details

This document explains the Single-view condition run with `configs/single_view.yaml`. For the shorter
command guide, see [Single-view](single_view.md).

## Role In The Study

Single-view is the simplest baseline: each example is rendered from one fixed camera direction and passed
to an image classifier. It does not use multi-view aggregation or learned view placement.

The condition is useful because it shows how much of the catalog is identifiable from a single observation.
If Fixed Ring-4 improves strongly over Single-view, multiple viewpoints are likely helping recover shape
cues hidden from the fixed direction. If the improvement is small, many classes may already be separable
from one view or the rendering setup may not expose useful additional cues.

The input should not be called a front-view image. The upstream asset orientation is not guaranteed to map
to a semantic front direction, so this condition is a `single fixed view`.

## Data Assumptions

Inputs are GLB assets fetched into `data/raw_assets`, audited into `data/manifests/selected_regular.jsonl`,
and normalized into `data/mesh_cache`.

The initial catalog is intended to focus on regular-form Pokemon by excluding shiny variants, mega forms,
regional forms, special forms, gender variants, and duplicate National Dex IDs where possible.

Train, validation, and test are split by pose conditions rather than by Pokemon IDs. The same Pokemon IDs
can appear in all splits, but yaw/elevation combinations must not overlap.

## Input Generation

Each sample is rendered from one camera location. The standard configs use `224` pixel images for both the
renderer and classifier input.

Pose-split yaw/elevation offsets are applied to the fixed camera. Validation and test therefore observe
the same catalog from orientations that differ from training.

## Model

The config uses:

- `model.experiment_kind: single_view`,
- a ResNet-18 backbone,
- pretrained image weights by default,
- a classifier head over the encoded image feature.

Backbone, feature dimension, dropout, and optimizer settings should stay close to the multi-view conditions
so comparisons remain interpretable.

## Evaluation

The best checkpoint is selected by validation Macro-F1 and saved as `checkpoints/best.ckpt`. Final
evaluation runs that checkpoint on the test split.

Important metrics are Top-1 accuracy, Top-5 accuracy, Macro-F1, per-class metrics, and the confusion
matrix.

## Interpretation

Single-view results are most useful when compared with Fixed Ring-4 and MVTN. Inspect classes with similar
silhouettes or shape cues that are hidden from the fixed camera. Improvements in later conditions can then
be tied to multi-view coverage or learned camera placement.

This condition does not evaluate open-set generalization to unseen Pokemon, the effect of four views, or
the value of learned camera placement.
