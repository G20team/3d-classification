# Learned Circular-4 MVTN

Learned Circular-4 MVTN initializes from the Fixed Ring-4 camera layout and learns azimuth/elevation
offsets from mesh geometry.

## Purpose

This condition tests whether Pokemon-dependent view placement improves identification under the shared,
pose-balanced split compared with the fixed four-view baseline.

The classifier side should remain aligned with Fixed Ring-4 MVCNN so the main difference is the camera
policy.

## Inputs and Outputs

Inputs:

- normalized mesh vertices,
- Fixed Ring-4 base azimuth/elevation,
- pose-split yaw/elevation offsets.

Outputs:

- adjusted azimuth,
- adjusted elevation,
- learned offsets.

Offsets are bounded by config values, and elevation is constrained to avoid unstable rendering directions.

## Configs

Main config:

```text
configs/mvtn_circular4.yaml
```

Debug config:

```text
configs/debug_mvtn_circular4.yaml
```

## Run

Debug:

```bash
uv run python scripts/train.py --config configs/debug_mvtn_circular4.yaml
```

Full run:

```bash
uv run python scripts/train.py --config configs/mvtn_circular4.yaml
```

## Evaluate

```bash
uv run python scripts/evaluate.py \
  --checkpoint outputs/mvtn_circular4/.../checkpoints/best.ckpt \
  --split test
```

## Camera Logs

For MVTN, inspect learned camera behavior in addition to classification metrics.

Important files:

```text
camera_positions.json
learned_camera_visualization.png
```

Check that:

- `offset_abs_mean` is not always near zero,
- `pairwise_distance_min` does not become too small,
- `view_collapse` is not frequent,
- offsets and view distances do not become extremely unstable over epochs.

`view_collapse` indicates that multiple views have moved toward nearly the same direction. If collapse
occurs, classification gains should be interpreted carefully.

## Comparison With Fixed Ring-4

The main comparison is MVTN vs. Fixed Ring-4, not MVTN vs. Single-view. Report metric differences together
with camera-log evidence so performance changes can be tied to learned view behavior.
