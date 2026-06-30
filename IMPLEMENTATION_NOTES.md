# Implementation Notes

## References

- Pokémon 3D assets: https://github.com/Pokemon-3D-api/assets
- MVCNN paper: https://arxiv.org/abs/1505.00880
- MVCNN PyTorch reference: https://github.com/RBirkeland/MVCNN-PyTorch
- MVTN paper: https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html
- MVTN official code: https://github.com/ajhamdi/MVTN
- PyTorch3D: https://github.com/facebookresearch/pytorch3d
- PokeAPI: https://pokeapi.co/docs/v2

## Design Choices

- 既存リポジトリの方針に合わせ、Pipenvではなく `uv` を使う。
- PyTorch3Dのwheel互換性に合わせ、Python 3.10 + PyTorch 2.4.1 + torchvision 0.19.1を前提にする。
- Pokemon assetsの内部構成は決め打ちせず、`*.glb` 再帰走査を主入口にする。
- 初期実験ではテクスチャを使わず、単色meshをPyTorch3Dでレンダリングする。
- `front` という表現は避け、単視点条件は `single fixed view` と呼ぶ。
- MVTNは `learned_circular` 相当だけを実装し、`learned_direct` / `learned_spherical` は実装しない。

## Known Constraints

- PyTorch3Dのインストール可否はCUDA、PyTorch、Pythonの組み合わせに依存する。
- 2026-06-30時点ではPython 3.12向けPyTorch3D wheelが利用できなかったため、Python 3.10へ下げる。
- LinuxではPyPI上の `pytorch3d==0.7.4` 通常wheelがなく、`pyproject.toml` に通常依存として含めると `uv sync` が失敗する。そのためPyTorch3D本体は公式の専用wheel indexまたはsource buildで別途導入し、コード側は `scripts/bootstrap_env.py` で導入可否を診断する。
- PokeAPI取得に失敗した場合は `data/manifests/pokeapi_cache.json` が必要になる。
- debug subsetの過学習確認と全データ実験は、実GLBアセットとPyTorch3Dが揃った環境で実行する必要がある。
