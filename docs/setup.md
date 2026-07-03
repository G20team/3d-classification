# Setup

This project uses `uv` for dependency management. Prefer `uv run <command>` for all commands so the
project environment is applied per command without keeping an activated shell open.

## Requirements

- Python 3.10
- CUDA-capable GPU for practical full experiments
- PyTorch 2.4.1 and torchvision 0.19.1
- PyTorch3D installed with a wheel or source build compatible with your Python, CUDA, and PyTorch versions

CPU execution can be useful for small checks, but full experiments are expected to be slow without a GPU.

## Environment

```bash
uv python install 3.10
uv sync
uv run python scripts/bootstrap_env.py
```

The diagnostic command writes an environment report that records Python, PyTorch, CUDA, PyTorch3D import
status, and related package versions.

## PyTorch3D

PyTorch3D wheel availability depends on the Python, CUDA, and PyTorch combination. If `uv sync` succeeds
but PyTorch3D is unavailable, install it separately using the official wheel index or a source build that
matches your environment.

Example pattern:

```bash
uv run pip install pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/<py_cuda_torch>/download.html
```

Replace `<py_cuda_torch>` with the wheel directory for your local environment. See the PyTorch3D project
for the current wheel matrix.

## Developer Checks

```bash
uv run ruff check .
uv run pytest
```

Use `uv run ruff check . --fix` only when you intentionally want the formatter/linter to update files.

## Troubleshooting

- If PyTorch3D cannot be imported, confirm Python 3.10, PyTorch 2.4.1, CUDA availability, and the wheel
  source used for installation.
- If asset download fails because PokeAPI is unavailable, use the cached manifest under
  `data/manifests/pokeapi_cache.json` when available.
- If training runs out of memory, start with the debug configs and lower `training.batch_size`.
- If rendered images look empty or badly framed, inspect the mesh cache and normalization outputs before
  training.
