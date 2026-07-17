from __future__ import annotations

import pytest

from pokemon_3d_cls.config import parse_generation_config, parse_mesh_experiment_config, parse_training_config


def test_parse_training_config_uses_defaults() -> None:
    config = parse_training_config({"experiment": {"condition_id": "baseline", "condition_name": "Baseline"}})

    assert config.experiment.condition_id == "baseline"
    assert config.data.dataset_root == "data/dataset"
    assert config.data.num_views == 24
    assert config.model.backbone == "resnet18"
    assert config.model.pretrained is True
    assert config.output.runs_root == "outputs/runs"


def test_parse_training_config_rejects_invalid_backbone() -> None:
    with pytest.raises(ValueError, match="backbone"):
        parse_training_config({"model": {"backbone": "vgg16"}})


def test_parse_generation_config_uses_defaults() -> None:
    config = parse_generation_config({})

    assert config.input.path == "data/models"
    assert config.output.dataset_root == "data/dataset"
    assert config.rendering.views == 36
    assert config.rendering.mode == "quiz"
    assert config.labels.mode == "stem"


def test_parse_generation_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        parse_generation_config({"rendering": {"mode": "random"}})


def test_parse_mesh_experiment_config_defaults_to_mesh_source() -> None:
    config = parse_mesh_experiment_config({"model": {"experiment_kind": "fixed_ring4"}})

    assert config.data.input_source == "mesh"
    assert config.data.render_cache_root == "data/render_cache"
    assert config.model.num_views == 4
    assert config.rendering.supersample == 2
    assert config.rendering.line_width == 0.4


def test_parse_mesh_experiment_config_supports_silhouette_cache_source() -> None:
    config = parse_mesh_experiment_config(
        {
            "data": {
                "input_source": "silhouette_cache",
                "render_cache_root": "data/render_cache_debug",
            },
            "model": {"experiment_kind": "single_view"},
            "rendering": {"supersample": 3, "line_width": 0.2},
        }
    )

    assert config.data.input_source == "silhouette_cache"
    assert config.data.render_cache_root == "data/render_cache_debug"
    assert config.model.num_views == 1
    assert config.rendering.supersample == 3
    assert config.rendering.line_width == 0.2


def test_parse_mesh_experiment_config_supports_rgb_cache_source() -> None:
    config = parse_mesh_experiment_config(
        {
            "data": {"input_source": "rgb_cache", "render_cache_root": "data/rgb_cache_debug"},
            "model": {"experiment_kind": "fixed_ring4"},
        }
    )

    assert config.data.input_source == "rgb_cache"
    assert config.data.render_cache_root == "data/rgb_cache_debug"


def test_parse_mesh_experiment_config_rejects_rgb_cache_for_mvtn() -> None:
    with pytest.raises(ValueError, match="online rendering"):
        parse_mesh_experiment_config(
            {
                "data": {"input_source": "rgb_cache"},
                "model": {"experiment_kind": "mvtn_circular4"},
            }
        )


def test_parse_mesh_experiment_config_supports_view_transformer() -> None:
    config = parse_mesh_experiment_config(
        {
            "model": {
                "experiment_kind": "view_transformer4",
                "transformer": {"num_layers": 3, "num_heads": 4, "mlp_dim": 1024, "dropout": 0.2},
            }
        }
    )

    assert config.model.experiment_kind == "view_transformer4"
    assert config.model.num_views == 4
    assert config.model.transformer.num_layers == 3
    assert config.model.transformer.num_heads == 4
    assert config.model.transformer.mlp_dim == 1024
    assert config.model.transformer.dropout == 0.2
