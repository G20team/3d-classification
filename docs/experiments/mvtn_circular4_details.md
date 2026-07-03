# Learned Circular-4 MVTN Details

This document explains the Learned Circular-4 MVTN condition run with `configs/mvtn_circular4.yaml`. For
the shorter command guide, see [Learned Circular-4 MVTN](mvtn_circular4.md).

## Role In The Study

MVTN tests whether camera placement can be adapted to each mesh. The model starts from the Fixed Ring-4
camera layout and predicts bounded azimuth/elevation offsets from mesh geometry.

The condition should be compared primarily with Fixed Ring-4 MVCNN. Comparing only with Single-view would
mix the value of four views with the value of learned camera offsets.

## Data Assumptions

Inputs are normalized meshes from the same mesh cache used by the other conditions. The classifier observes
rendered images, while the MVTN camera module uses mesh geometry to predict view offsets.

Pose splits remain based on yaw/elevation conditions. The catalog is closed-set, but evaluation poses are
held out from training.

## Camera Module

Inputs:

- mesh vertices,
- base Fixed Ring-4 camera angles,
- pose-split yaw/elevation offsets.

Outputs:

- final azimuths,
- final elevations,
- learned offsets.

Offset ranges and elevation bounds should be controlled by config values so the renderer stays stable and
the learned cameras remain interpretable.

## Classifier

After camera prediction, the model renders four views and uses the same MVCNN-style aggregation as Fixed
Ring-4. Keep classifier settings aligned with the fixed baseline unless running an explicit ablation.

## Camera Diagnostics

MVTN runs should log camera behavior so classification changes can be interpreted. Inspect:

```text
camera_positions.json
learned_camera_visualization.png
```

Useful checks:

- mean absolute offsets are not always near zero,
- pairwise camera distances do not collapse,
- learned offsets stay within the expected range,
- camera behavior is not wildly unstable across epochs.

View collapse means multiple cameras converge to nearly the same direction. If collapse happens often,
classification metrics may still improve, but the learned views are less interpretable as a multi-view
strategy.

## Interpretation

Report MVTN results together with Fixed Ring-4 results and camera diagnostics. A useful improvement should
show both better metrics and plausible learned camera movement. If metrics improve while cameras barely
move, the gain may come from training noise or another implementation difference. If cameras move but
metrics do not improve, the learned views may not expose useful additional cues for this catalog.
