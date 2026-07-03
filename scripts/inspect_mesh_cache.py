"""CLI for inspecting mesh cache structure and writing simple previews."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import _bootstrap  # noqa: F401
import torch

from pokemon_3d_cls.paths import ensure_directory, find_project_root, resolve_project_path

MAX_PREVIEW_ITEMS = 5


@runtime_checkable
class PackedMesh(Protocol):
    """Minimal PyTorch3D Meshes API used by this CLI."""

    def verts_packed(self) -> torch.Tensor: ...

    def faces_packed(self) -> torch.Tensor: ...


def summarize(obj: object, name: str = "root", indent: int = 0) -> None:
    """Recursively print the high-level structure of a torch-loaded cache."""

    prefix = " " * indent

    if isinstance(obj, torch.Tensor):
        shape = tuple(obj.shape)
        print(f"{prefix}{name}: Tensor shape={shape}, dtype={obj.dtype}, device={obj.device}")
    elif isinstance(obj, Mapping):
        print(f"{prefix}{name}: dict(keys={list(obj.keys())})")
        for k, v in obj.items():
            summarize(v, name=str(k), indent=indent + 2)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}: {type(obj).__name__}(len={len(obj)})")
        for i, v in enumerate(obj[:MAX_PREVIEW_ITEMS]):
            summarize(v, name=f"[{i}]", indent=indent + 2)
        if len(obj) > MAX_PREVIEW_ITEMS:
            print(f"{prefix}  ...")
    else:
        print(f"{prefix}{name}: {type(obj).__name__} = {repr(obj)[:200]}")


def get_mesh_tensors(data: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract vertex and face tensors from a cache structure."""

    # PyTorch3D Meshes
    if isinstance(data, PackedMesh):
        return data.verts_packed(), data.faces_packed()

    # Dict format
    if isinstance(data, Mapping):
        vert_keys = ("verts", "vertices", "v")
        face_keys = ("faces", "triangles", "f")

        verts = next((data[k] for k in vert_keys if k in data), None)
        faces = next((data[k] for k in face_keys if k in data), None)

        if isinstance(verts, torch.Tensor) and isinstance(faces, torch.Tensor):
            return verts, faces

        # Nested format
        for value in data.values():
            try:
                return get_mesh_tensors(value)
            except ValueError:
                pass

    # (verts, faces) tuple/list format
    if isinstance(data, (list, tuple)) and len(data) >= 2:
        if isinstance(data[0], torch.Tensor) and isinstance(data[1], torch.Tensor):
            return data[0], data[1]

    raise ValueError(
        "Could not detect verts / faces."
        "Adjust get_mesh_tensors() to match the displayed cache structure."
    )


def normalize_shape(verts: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove a batch dimension only when it is 1, then validate mesh shapes."""

    # [1, V, 3] -> [V, 3]
    if verts.ndim == 3 and verts.shape[0] == 1:
        verts = verts[0]

    # [1, F, 3] -> [F, 3]
    if faces.ndim == 3 and faces.shape[0] == 1:
        faces = faces[0]

    if verts.ndim != 2 or verts.shape[-1] != 3:
        raise ValueError(f"Unexpected vertices shape: {tuple(verts.shape)}")

    if faces.ndim != 2 or faces.shape[-1] != 3:
        raise ValueError(f"Unexpected faces shape: {tuple(faces.shape)}")

    return verts, faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a mesh cache .pt file and write preview outputs.")
    parser.add_argument("pt_path", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("mesh_preview"))
    args = parser.parse_args()

    project_root = find_project_root(Path.cwd())
    pt_path = resolve_project_path(args.pt_path, project_root)
    output_root = ensure_directory(resolve_project_path(args.output_root, project_root))

    # Use weights_only=False only for trusted caches you created.
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    print("=== Cache structure ===")
    summarize(data)

    verts, faces = get_mesh_tensors(data)
    verts, faces = normalize_shape(verts, faces)

    verts_np = verts.detach().cpu().float().numpy()
    faces_np = faces.detach().cpu().long().numpy()

    print("\n=== Mesh summary ===")
    print(f"vertices: {verts_np.shape}")
    print(f"faces   : {faces_np.shape}")
    print(f"bbox min: {verts_np.min(axis=0)}")
    print(f"bbox max: {verts_np.max(axis=0)}")

    # PLY output
    try:
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=verts_np,
            faces=faces_np,
            process=False,
        )
        ply_path = output_root / f"{pt_path.stem}.ply"
        mesh.export(ply_path)
        print(f"\nPLY saved: {ply_path}")
    except ImportError:
        print("\nSkipped PLY output because trimesh is not installed.")
        print("If needed: uv add trimesh && uv sync")

    # Browser-readable HTML output
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Mesh3d(
                    x=verts_np[:, 0],
                    y=verts_np[:, 1],
                    z=verts_np[:, 2],
                    i=faces_np[:, 0],
                    j=faces_np[:, 1],
                    k=faces_np[:, 2],
                    flatshading=True,
                )
            ]
        )
        fig.update_layout(
            scene=dict(aspectmode="data"),
            margin=dict(l=0, r=0, t=30, b=0),
            title=pt_path.name,
        )

        html_path = output_root / f"{pt_path.stem}.html"
        fig.write_html(html_path)
        print(f"HTML saved: {html_path}")
    except ImportError:
        print("Skipped HTML output because plotly is not installed.")
        print("If needed: uv add plotly && uv sync")


if __name__ == "__main__":
    main()
