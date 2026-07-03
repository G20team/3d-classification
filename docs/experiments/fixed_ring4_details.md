# Fixed Ring-4 MVCNN Details

This document explains the Fixed Ring-4 MVCNN condition run with `configs/fixed_ring4.yaml`. For the
shorter command guide, see [Fixed Ring-4 MVCNN](fixed_ring4.md).

## Role In The Study

Fixed Ring-4 is the main fixed-camera multi-view baseline. It renders four views around each mesh and
aggregates per-view features with view-wise max pooling.

This condition answers whether multiple fixed views improve identification compared with Single-view. It
also provides the baseline needed to interpret MVTN, because Learned Circular-4 MVTN starts from the same
four-view circular layout.

## Data Assumptions

The condition uses the same audited asset catalog, normalized mesh cache, and pose splits as the other
conditions. Splits are based on yaw/elevation pose conditions, so the task remains closed-set
cross-orientation identification.

## Camera Setup

The base camera azimuths are:

```text
0, 90, 180, 270 degrees
```

Base elevation is `0` degrees. Pose-split yaw/elevation offsets are applied to this base layout, which
creates train/validation/test orientation differences without changing the catalog.

## Model

Each rendered view is encoded by the shared CNN backbone. View-wise max pooling aggregates the view
features into one feature vector, which is passed to the classifier head.

To keep the comparison with MVTN clean, match the backbone, feature dimension, dropout, optimizer,
resolution, split file, and view count.

## Evaluation

Evaluate the best validation checkpoint on the test split and compare against Single-view first. The most
important question is whether additional fixed views reduce errors caused by occluded or ambiguous shape
cues.

Then compare against MVTN. If MVTN improves, inspect camera logs to decide whether learned view placement
is a plausible explanation.

## Interpretation

Look for:

- Top-1 and Macro-F1 gains over Single-view,
- reduced confusion among similarly shaped classes,
- per-class improvements for classes that need side or rear cues,
- matched experimental settings when comparing with MVTN.

If Fixed Ring-4 is already strong, MVTN may have limited room to improve. If Fixed Ring-4 is weak, inspect
the rendered examples and mesh normalization before attributing the result only to the camera policy.
