"""YAML設定の読み込みと型付き設定。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping, cast

from pokemon_3d_cls.io import read_yaml_mapping

BackboneName = Literal["resnet18", "simple_cnn"]
ViewpointMode = Literal["quiz", "turntable", "sphere"]
UpAxis = Literal["y", "z"]
LabelMode = Literal["stem", "species"]


@dataclass(frozen=True)
class ExperimentConfig:
    """実験条件の識別情報。"""

    condition_id: str = "mvcnn_baseline"
    condition_name: str = "MVCNN baseline"
    seed: int = 0
    run_id: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """学習データ設定。"""

    dataset_root: str = "data/dataset"
    illustrations_dir: str | None = None
    num_views: int = 24
    holdout_stride: int = 4
    image_size: int = 224
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """MVCNNモデル設定。"""

    backbone: BackboneName = "resnet18"
    pretrained: bool = True
    input_channels: int = 1
    feature_dim: int = 512
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """学習ループ設定。"""

    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "auto"


@dataclass(frozen=True)
class OutputConfig:
    """実験成果物の出力先設定。"""

    runs_root: str = "outputs/runs"


@dataclass(frozen=True)
class TrainRunConfig:
    """学習実行設定。"""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, object]:
        """メタデータ保存用に辞書化する。"""

        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True)
class GenerationInputConfig:
    """GLB入力設定。"""

    path: str = "data/models"


@dataclass(frozen=True)
class GenerationOutputConfig:
    """シルエットデータセット出力設定。"""

    dataset_root: str = "data/dataset"
    manifest_name: str = "manifest.csv"


@dataclass(frozen=True)
class RenderingConfig:
    """シルエットレンダリング設定。"""

    views: int = 36
    mode: ViewpointMode = "quiz"
    resolution: int = 256
    supersample: int = 2
    invert: bool = False
    pad: float = 1.25
    up: UpAxis = "y"
    base_azimuth: float = 0.0


@dataclass(frozen=True)
class LabelConfig:
    """生成データセットのラベル付け設定。"""

    mode: LabelMode = "stem"


@dataclass(frozen=True)
class GenerationConfig:
    """シルエット生成実行設定。"""

    seed: int = 0
    input: GenerationInputConfig = field(default_factory=GenerationInputConfig)
    output: GenerationOutputConfig = field(default_factory=GenerationOutputConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)

    def to_dict(self) -> dict[str, object]:
        """メタデータ保存用に辞書化する。"""

        return cast("dict[str, object]", asdict(self))


def load_training_config(path: str | Path) -> TrainRunConfig:
    """学習用YAML設定を読み込む。"""

    return parse_training_config(read_yaml_mapping(path))


def load_generation_config(path: str | Path) -> GenerationConfig:
    """シルエット生成用YAML設定を読み込む。"""

    return parse_generation_config(read_yaml_mapping(path))


def parse_training_config(raw: Mapping[str, object]) -> TrainRunConfig:
    """mappingから学習用の型付き設定を構築する。"""

    experiment = _mapping(raw.get("experiment", {}), "experiment")
    data = _mapping(raw.get("data", {}), "data")
    model = _mapping(raw.get("model", {}), "model")
    training = _mapping(raw.get("training", {}), "training")
    output = _mapping(raw.get("output", {}), "output")

    return TrainRunConfig(
        experiment=ExperimentConfig(
            condition_id=_str(experiment, "condition_id", "mvcnn_baseline"),
            condition_name=_str(experiment, "condition_name", "MVCNN baseline"),
            seed=_int(experiment, "seed", 0),
            run_id=_optional_str(experiment, "run_id"),
        ),
        data=DataConfig(
            dataset_root=_str(data, "dataset_root", "data/dataset"),
            illustrations_dir=_optional_str(data, "illustrations_dir"),
            num_views=_positive_int(data, "num_views", 24),
            holdout_stride=_positive_int(data, "holdout_stride", 4),
            image_size=_positive_int(data, "image_size", 224),
            num_workers=_non_negative_int(data, "num_workers", 0),
        ),
        model=ModelConfig(
            backbone=_backbone(model, "backbone", "resnet18"),
            pretrained=_bool(model, "pretrained", True),
            input_channels=_positive_int(model, "input_channels", 1),
            feature_dim=_positive_int(model, "feature_dim", 512),
            dropout=_bounded_float(model, "dropout", 0.3, min_value=0.0, max_value=1.0),
        ),
        training=TrainingConfig(
            batch_size=_positive_int(training, "batch_size", 4),
            epochs=_positive_int(training, "epochs", 30),
            learning_rate=_positive_float(training, "learning_rate", 1e-4),
            weight_decay=_non_negative_float(training, "weight_decay", 0.0),
            device=_str(training, "device", "auto"),
        ),
        output=OutputConfig(
            runs_root=_str(output, "runs_root", "outputs/runs"),
        ),
    )


def parse_generation_config(raw: Mapping[str, object]) -> GenerationConfig:
    """mappingからシルエット生成用の型付き設定を構築する。"""

    input_config = _mapping(raw.get("input", {}), "input")
    output = _mapping(raw.get("output", {}), "output")
    rendering = _mapping(raw.get("rendering", {}), "rendering")
    labels = _mapping(raw.get("labels", {}), "labels")

    return GenerationConfig(
        seed=_int(raw, "seed", 0),
        input=GenerationInputConfig(path=_str(input_config, "path", "data/models")),
        output=GenerationOutputConfig(
            dataset_root=_str(output, "dataset_root", "data/dataset"),
            manifest_name=_str(output, "manifest_name", "manifest.csv"),
        ),
        rendering=RenderingConfig(
            views=_positive_int(rendering, "views", 36),
            mode=_viewpoint_mode(rendering, "mode", "quiz"),
            resolution=_positive_int(rendering, "resolution", 256),
            supersample=_positive_int(rendering, "supersample", 2),
            invert=_bool(rendering, "invert", False),
            pad=_positive_float(rendering, "pad", 1.25),
            up=_up_axis(rendering, "up", "y"),
            base_azimuth=_float(rendering, "base_azimuth", 0.0),
        ),
        labels=LabelConfig(mode=_label_mode(labels, "mode", "stem")),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} はmappingである必要があります。"
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _str(mapping: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value:
        msg = f"{key} は空でない文字列である必要があります。"
        raise ValueError(msg)
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key} は文字列またはnullである必要があります。"
        raise ValueError(msg)
    return value


def _int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} は整数である必要があります。"
        raise ValueError(msg)
    return value


def _positive_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = _int(mapping, key, default)
    if value <= 0:
        msg = f"{key} は1以上の整数である必要があります。"
        raise ValueError(msg)
    return value


def _non_negative_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = _int(mapping, key, default)
    if value < 0:
        msg = f"{key} は0以上の整数である必要があります。"
        raise ValueError(msg)
    return value


def _float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} は数値である必要があります。"
        raise ValueError(msg)
    return float(value)


def _positive_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = _float(mapping, key, default)
    if value <= 0:
        msg = f"{key} は0より大きい数値である必要があります。"
        raise ValueError(msg)
    return value


def _non_negative_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = _float(mapping, key, default)
    if value < 0:
        msg = f"{key} は0以上の数値である必要があります。"
        raise ValueError(msg)
    return value


def _bounded_float(
    mapping: Mapping[str, object],
    key: str,
    default: float,
    *,
    min_value: float,
    max_value: float,
) -> float:
    value = _float(mapping, key, default)
    if not min_value <= value <= max_value:
        msg = f"{key} は {min_value} 以上 {max_value} 以下である必要があります。"
        raise ValueError(msg)
    return value


def _bool(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key} は真偽値である必要があります。"
        raise ValueError(msg)
    return value


def _backbone(mapping: Mapping[str, object], key: str, default: BackboneName) -> BackboneName:
    value = _str(mapping, key, default)
    if value in ("resnet18", "simple_cnn"):
        return cast("BackboneName", value)
    msg = f"{key} は resnet18 / simple_cnn のいずれかである必要があります。"
    raise ValueError(msg)


def _viewpoint_mode(mapping: Mapping[str, object], key: str, default: ViewpointMode) -> ViewpointMode:
    value = _str(mapping, key, default)
    if value in ("quiz", "turntable", "sphere"):
        return cast("ViewpointMode", value)
    msg = f"{key} は quiz / turntable / sphere のいずれかである必要があります。"
    raise ValueError(msg)


def _up_axis(mapping: Mapping[str, object], key: str, default: UpAxis) -> UpAxis:
    value = _str(mapping, key, default)
    if value in ("y", "z"):
        return cast("UpAxis", value)
    msg = f"{key} は y / z のいずれかである必要があります。"
    raise ValueError(msg)


def _label_mode(mapping: Mapping[str, object], key: str, default: LabelMode) -> LabelMode:
    value = _str(mapping, key, default)
    if value in ("stem", "species"):
        return cast("LabelMode", value)
    msg = f"{key} は stem / species のいずれかである必要があります。"
    raise ValueError(msg)
