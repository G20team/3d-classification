"""実験条件ごとの固定カメラ角度。"""

from __future__ import annotations

from typing import Literal

import torch

ExperimentKind = Literal["single_view", "fixed_ring4", "mvtn_circular4"]


def fixed_camera_angles(
    experiment_kind: ExperimentKind | str,
    *,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """実験条件に対応するazimuth/elevation角度をdegreeで返す。"""

    if experiment_kind == "single_view":
        azimuths = torch.tensor([0.0], dtype=torch.float32, device=device)
        elevations = torch.tensor([0.0], dtype=torch.float32, device=device)
        return azimuths, elevations
    if experiment_kind in {"fixed_ring4", "mvtn_circular4"}:
        azimuths = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float32, device=device)
        elevations = torch.zeros(4, dtype=torch.float32, device=device)
        return azimuths, elevations

    msg = f"未知の実験条件です: {experiment_kind}"
    raise ValueError(msg)
