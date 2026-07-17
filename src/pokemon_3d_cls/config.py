"""YAML config loading and typed settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping, cast

from pokemon_3d_cls.io import read_yaml_mapping

BackboneName = Literal["resnet18", "simple_cnn"]
SplitStrategy = Literal["exclusive_conditions", "stratified_samples"]
ViewpointMode = Literal["quiz", "turntable", "sphere"]
UpAxis = Literal["y", "z"]
LabelMode = Literal["stem", "species"]
ExperimentKind = Literal["single_view", "fixed_ring4", "mvtn_circular4", "view_transformer4"]
InputSource = Literal["mesh", "silhouette_cache", "rgb_cache"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment identity information."""

    condition_id: str = "mvcnn_baseline"
    condition_name: str = "MVCNN baseline"
    seed: int = 0
    run_id: str | None = None


@dataclass(frozen=True)
class DataConfig:
    """Training data settings."""

    dataset_root: str = "data/dataset"
    illustrations_dir: str | None = None
    num_views: int = 24
    holdout_stride: int = 4
    image_size: int = 224
    num_workers: int = 0


@dataclass(frozen=True)
class ModelConfig:
    """MVCNN model settings."""

    backbone: BackboneName = "resnet18"
    pretrained: bool = True
    input_channels: int = 1
    feature_dim: int = 512
    dropout: float = 0.3


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop settings."""

    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    device: str = "auto"


@dataclass(frozen=True)
class OutputConfig:
    """Output settings for experiment artifacts."""

    runs_root: str = "outputs/runs"


@dataclass(frozen=True)
class TrainRunConfig:
    """Training run settings."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary for metadata serialization."""

        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True)
class GenerationInputConfig:
    """GLB input settings."""

    path: str = "data/models"


@dataclass(frozen=True)
class GenerationOutputConfig:
    """Silhouette dataset output settings."""

    dataset_root: str = "data/dataset"
    manifest_name: str = "manifest.csv"


@dataclass(frozen=True)
class RenderingConfig:
    """Silhouette rendering settings."""

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
    """Labeling settings for generated datasets."""

    mode: LabelMode = "stem"


@dataclass(frozen=True)
class GenerationConfig:
    """Silhouette generation run settings."""

    seed: int = 0
    input: GenerationInputConfig = field(default_factory=GenerationInputConfig)
    output: GenerationOutputConfig = field(default_factory=GenerationOutputConfig)
    rendering: RenderingConfig = field(default_factory=RenderingConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary for metadata serialization."""

        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True)
class PoseSplitValues:
    """Yaw/elevation conditions for pose splits."""

    yaw_offsets: tuple[float, ...]
    elevation_offsets: tuple[float, ...]


@dataclass(frozen=True)
class SplitCounts:
    """Number of pose samples assigned to each split per class."""

    train: int = 9
    validation: int = 4
    test: int = 4


@dataclass(frozen=True)
class SplitConfig:
    """Pose split settings for closed-set cross-orientation experiments."""

    output_path: str = "data/manifests/pose_splits.json"
    strategy: SplitStrategy = "exclusive_conditions"
    manifest_path: str | None = None
    seed: int = 0
    split_counts: SplitCounts = field(default_factory=SplitCounts)
    pose_groups: tuple[PoseSplitValues, ...] = ()
    train: PoseSplitValues = field(
        default_factory=lambda: PoseSplitValues(
            yaw_offsets=(-20.0, 0.0, 20.0),
            elevation_offsets=(-10.0, 0.0, 10.0),
        )
    )
    validation: PoseSplitValues = field(
        default_factory=lambda: PoseSplitValues(
            yaw_offsets=(-30.0, 30.0),
            elevation_offsets=(-15.0, 15.0),
        )
    )
    test: PoseSplitValues = field(
        default_factory=lambda: PoseSplitValues(
            yaw_offsets=(-45.0, 45.0),
            elevation_offsets=(-25.0, 25.0),
        )
    )

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary for JSON/YAML serialization."""

        return cast("dict[str, object]", asdict(self))


@dataclass(frozen=True)
class MeshDataConfig:
    """Experiment data settings using mesh caches."""

    input_source: InputSource = "mesh"
    manifest_path: str = "data/manifests/selected_regular.jsonl"
    mesh_cache_root: str = "data/mesh_cache"
    splits_path: str = "data/manifests/pose_splits.json"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    image_size: int = 224
    num_workers: int = 0
    class_limit: int | None = None
    render_cache_root: str = "data/render_cache"


@dataclass(frozen=True)
class MVTNConfig:
    """MVTN settings equivalent to learned_circular."""

    enabled: bool = False
    num_views: int = 4
    max_azimuth_offset_deg: float = 45.0
    max_elevation_offset_deg: float = 25.0
    point_samples: int = 512
    hidden_dim: int = 128
    collapse_threshold_deg: float = 5.0


@dataclass(frozen=True)
class ViewTransformerConfig:
    """Transformer settings for aggregating multiple view features."""

    num_layers: int = 2
    num_heads: int = 8
    mlp_dim: int = 2048
    dropout: float = 0.1


@dataclass(frozen=True)
class RenderConfig:
    """Experiment rendering settings."""

    image_size: int = 224
    camera_distance: float = 2.7
    background_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    mesh_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    supersample: int = 2
    line_width: float = 0.4


@dataclass(frozen=True)
class MeshExperimentModelConfig:
    """Model settings for mesh-render classification experiments."""

    experiment_kind: ExperimentKind = "single_view"
    backbone: BackboneName = "resnet18"
    pretrained: bool = True
    feature_dim: int = 512
    dropout: float = 0.3
    num_views: int = 1
    mvtn: MVTNConfig = field(default_factory=MVTNConfig)
    transformer: ViewTransformerConfig = field(default_factory=ViewTransformerConfig)


@dataclass(frozen=True)
class MeshExperimentConfig:
    """Shared experiment settings for Single/Fixed Ring-4/MVTN."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: MeshDataConfig = field(default_factory=MeshDataConfig)
    model: MeshExperimentModelConfig = field(default_factory=MeshExperimentModelConfig)
    rendering: RenderConfig = field(default_factory=RenderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=lambda: OutputConfig(runs_root="outputs"))

    def to_dict(self) -> dict[str, object]:
        """Convert to a dictionary for metadata serialization."""

        return cast("dict[str, object]", asdict(self))


def load_training_config(path: str | Path) -> TrainRunConfig:
    """Load a training YAML config."""

    return parse_training_config(read_yaml_mapping(path))


def load_generation_config(path: str | Path) -> GenerationConfig:
    """Load a silhouette generation YAML config."""

    return parse_generation_config(read_yaml_mapping(path))


def load_split_config(path: str | Path) -> SplitConfig:
    """Load a pose split YAML config."""

    return parse_split_config(read_yaml_mapping(path))


def load_mesh_experiment_config(path: str | Path) -> MeshExperimentConfig:
    """Load a YAML config for mesh-render classification experiments."""

    return parse_mesh_experiment_config(read_yaml_mapping(path))


def parse_training_config(raw: Mapping[str, object]) -> TrainRunConfig:
    """Build typed training settings from a mapping."""

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
    """Build typed silhouette generation settings from a mapping."""

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


def parse_split_config(raw: Mapping[str, object]) -> SplitConfig:
    """Build pose split settings from a mapping."""

    train = _mapping(raw.get("train", {}), "train")
    validation = _mapping(raw.get("validation", {}), "validation")
    test = _mapping(raw.get("test", {}), "test")
    strategy = _split_strategy(raw, "strategy", "exclusive_conditions")
    split_counts = _mapping(raw.get("split_counts", {}), "split_counts")
    pose_groups = _pose_groups(raw.get("pose_groups", ()))
    config = SplitConfig(
        output_path=_str(raw, "output_path", "data/manifests/pose_splits.json"),
        strategy=strategy,
        manifest_path=_optional_str(raw, "manifest_path"),
        seed=_int(raw, "seed", 0),
        split_counts=SplitCounts(
            train=_positive_int(split_counts, "train", 9),
            validation=_positive_int(split_counts, "validation", 4),
            test=_positive_int(split_counts, "test", 4),
        ),
        pose_groups=pose_groups,
        train=_pose_split(train, default_yaw=(-20.0, 0.0, 20.0), default_elevation=(-10.0, 0.0, 10.0)),
        validation=_pose_split(validation, default_yaw=(-30.0, 30.0), default_elevation=(-15.0, 15.0)),
        test=_pose_split(test, default_yaw=(-45.0, 45.0), default_elevation=(-25.0, 25.0)),
    )
    if config.strategy == "stratified_samples":
        if config.manifest_path is None:
            msg = "manifest_path is required for stratified_samples."
            raise ValueError(msg)
        if not config.pose_groups:
            msg = "pose_groups must not be empty for stratified_samples."
            raise ValueError(msg)
        condition_count = sum(len(group.yaw_offsets) * len(group.elevation_offsets) for group in config.pose_groups)
        assigned_count = config.split_counts.train + config.split_counts.validation + config.split_counts.test
        if condition_count != assigned_count:
            msg = f"split_counts total ({assigned_count}) must match pose condition count ({condition_count})."
            raise ValueError(msg)
    return config


def parse_mesh_experiment_config(raw: Mapping[str, object]) -> MeshExperimentConfig:
    """Build Single/Fixed Ring-4/MVTN experiment settings from a mapping."""

    experiment = _mapping(raw.get("experiment", {}), "experiment")
    data = _mapping(raw.get("data", {}), "data")
    model = _mapping(raw.get("model", {}), "model")
    mvtn = _mapping(model.get("mvtn", {}), "model.mvtn")
    transformer = _mapping(model.get("transformer", {}), "model.transformer")
    rendering = _mapping(raw.get("rendering", {}), "rendering")
    training = _mapping(raw.get("training", {}), "training")
    output = _mapping(raw.get("output", {}), "output")
    kind = _experiment_kind(model, "experiment_kind", "single_view")
    default_views = 1 if kind == "single_view" else 4

    config = MeshExperimentConfig(
        experiment=ExperimentConfig(
            condition_id=_str(experiment, "condition_id", kind),
            condition_name=_str(experiment, "condition_name", kind),
            seed=_int(experiment, "seed", 0),
            run_id=_optional_str(experiment, "run_id"),
        ),
        data=MeshDataConfig(
            input_source=_input_source(data, "input_source", "mesh"),
            manifest_path=_str(data, "manifest_path", "data/manifests/selected_regular.jsonl"),
            mesh_cache_root=_str(data, "mesh_cache_root", "data/mesh_cache"),
            splits_path=_str(data, "splits_path", "data/manifests/pose_splits.json"),
            train_split=_str(data, "train_split", "train"),
            validation_split=_str(data, "validation_split", "validation"),
            test_split=_str(data, "test_split", "test"),
            image_size=_positive_int(data, "image_size", 224),
            num_workers=_non_negative_int(data, "num_workers", 0),
            class_limit=_optional_positive_int(data, "class_limit"),
            render_cache_root=_str(data, "render_cache_root", "data/render_cache"),
        ),
        model=MeshExperimentModelConfig(
            experiment_kind=kind,
            backbone=_backbone(model, "backbone", "resnet18"),
            pretrained=_bool(model, "pretrained", True),
            feature_dim=_positive_int(model, "feature_dim", 512),
            dropout=_bounded_float(model, "dropout", 0.3, min_value=0.0, max_value=1.0),
            num_views=_positive_int(model, "num_views", default_views),
            mvtn=MVTNConfig(
                enabled=(kind == "mvtn_circular4"),
                num_views=_positive_int(mvtn, "num_views", 4),
                max_azimuth_offset_deg=_positive_float(mvtn, "max_azimuth_offset_deg", 45.0),
                max_elevation_offset_deg=_positive_float(mvtn, "max_elevation_offset_deg", 25.0),
                point_samples=_positive_int(mvtn, "point_samples", 512),
                hidden_dim=_positive_int(mvtn, "hidden_dim", 128),
                collapse_threshold_deg=_positive_float(mvtn, "collapse_threshold_deg", 5.0),
            ),
            transformer=ViewTransformerConfig(
                num_layers=_positive_int(transformer, "num_layers", 2),
                num_heads=_positive_int(transformer, "num_heads", 8),
                mlp_dim=_positive_int(transformer, "mlp_dim", 2048),
                dropout=_bounded_float(transformer, "dropout", 0.1, min_value=0.0, max_value=1.0),
            ),
        ),
        rendering=RenderConfig(
            image_size=_positive_int(rendering, "image_size", 224),
            camera_distance=_positive_float(rendering, "camera_distance", 2.7),
            background_color=_float_tuple3(rendering, "background_color", (0.5, 0.5, 0.5)),
            mesh_color=_float_tuple3(rendering, "mesh_color", (1.0, 1.0, 1.0)),
            supersample=_positive_int(rendering, "supersample", 2),
            line_width=_positive_float(rendering, "line_width", 0.4),
        ),
        training=TrainingConfig(
            batch_size=_positive_int(training, "batch_size", 4),
            epochs=_positive_int(training, "epochs", 30),
            learning_rate=_positive_float(training, "learning_rate", 1e-4),
            weight_decay=_non_negative_float(training, "weight_decay", 0.0),
            device=_str(training, "device", "auto"),
        ),
        output=OutputConfig(runs_root=_str(output, "runs_root", "outputs")),
    )
    if config.data.input_source == "rgb_cache" and config.model.experiment_kind == "mvtn_circular4":
        msg = "rgb_cache cannot be used with mvtn_circular4 because learned camera angles require online rendering."
        raise ValueError(msg)
    return config


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        msg = f"{name} must be a mapping."
        raise ValueError(msg)
    return cast("Mapping[str, object]", value)


def _str(mapping: Mapping[str, object], key: str, default: str | None = None) -> str:
    value = mapping.get(key, default)
    if not isinstance(value, str) or not value:
        msg = f"{key} must be a non-empty string."
        raise ValueError(msg)
    return value


def _optional_str(mapping: Mapping[str, object], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key} must be a string or null."
        raise ValueError(msg)
    return value


def _int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{key} must be an integer."
        raise ValueError(msg)
    return value


def _positive_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = _int(mapping, key, default)
    if value <= 0:
        msg = f"{key} must be an integer greater than or equal to 1."
        raise ValueError(msg)
    return value


def _non_negative_int(mapping: Mapping[str, object], key: str, default: int) -> int:
    value = _int(mapping, key, default)
    if value < 0:
        msg = f"{key} must be an integer greater than or equal to 0."
        raise ValueError(msg)
    return value


def _optional_positive_int(mapping: Mapping[str, object], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{key} must be null or an integer greater than or equal to 1."
        raise ValueError(msg)
    return value


def _float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"{key} must be a number."
        raise ValueError(msg)
    return float(value)


def _positive_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = _float(mapping, key, default)
    if value <= 0:
        msg = f"{key} must be greater than 0."
        raise ValueError(msg)
    return value


def _non_negative_float(mapping: Mapping[str, object], key: str, default: float) -> float:
    value = _float(mapping, key, default)
    if value < 0:
        msg = f"{key} must be greater than or equal to 0."
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
        msg = f"{key} must be between {min_value} and {max_value}."
        raise ValueError(msg)
    return value


def _float_sequence(mapping: Mapping[str, object], key: str, default: tuple[float, ...]) -> tuple[float, ...]:
    value = mapping.get(key, default)
    if not isinstance(value, list | tuple) or isinstance(value, str):
        msg = f"{key} must be a list of numbers."
        raise ValueError(msg)
    floats: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            msg = f"{key} must be a list of numbers."
            raise ValueError(msg)
        floats.append(float(item))
    if not floats:
        msg = f"{key} must not be empty."
        raise ValueError(msg)
    return tuple(floats)


def _float_tuple(
    mapping: Mapping[str, object],
    key: str,
    default: tuple[float, ...],
    *,
    length: int,
) -> tuple[float, ...]:
    values = _float_sequence(mapping, key, default)
    if len(values) != length:
        msg = f"{key} must be a numeric list of length {length}."
        raise ValueError(msg)
    return values


def _float_tuple3(
    mapping: Mapping[str, object],
    key: str,
    default: tuple[float, float, float],
) -> tuple[float, float, float]:
    values = _float_tuple(mapping, key, default, length=3)
    return (values[0], values[1], values[2])


def _bool(mapping: Mapping[str, object], key: str, default: bool) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        msg = f"{key} must be a boolean."
        raise ValueError(msg)
    return value


def _pose_split(
    mapping: Mapping[str, object],
    *,
    default_yaw: tuple[float, ...],
    default_elevation: tuple[float, ...],
) -> PoseSplitValues:
    return PoseSplitValues(
        yaw_offsets=_float_sequence(mapping, "yaw_offsets", default_yaw),
        elevation_offsets=_float_sequence(mapping, "elevation_offsets", default_elevation),
    )


def _pose_groups(value: object) -> tuple[PoseSplitValues, ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, list | tuple) or isinstance(value, str):
        msg = "pose_groups must be a list of mappings."
        raise ValueError(msg)
    groups: list[PoseSplitValues] = []
    for index, item in enumerate(value):
        mapping = _mapping(item, f"pose_groups[{index}]")
        groups.append(
            PoseSplitValues(
                yaw_offsets=_float_sequence(mapping, "yaw_offsets", ()),
                elevation_offsets=_float_sequence(mapping, "elevation_offsets", ()),
            )
        )
    return tuple(groups)


def _backbone(mapping: Mapping[str, object], key: str, default: BackboneName) -> BackboneName:
    value = _str(mapping, key, default)
    if value in ("resnet18", "simple_cnn"):
        return cast("BackboneName", value)
    msg = f"{key} must be one of resnet18 / simple_cnn."
    raise ValueError(msg)


def _experiment_kind(mapping: Mapping[str, object], key: str, default: ExperimentKind) -> ExperimentKind:
    value = _str(mapping, key, default)
    if value in ("single_view", "fixed_ring4", "mvtn_circular4", "view_transformer4"):
        return cast("ExperimentKind", value)
    msg = f"{key} must be one of single_view / fixed_ring4 / mvtn_circular4 / view_transformer4."
    raise ValueError(msg)


def _split_strategy(mapping: Mapping[str, object], key: str, default: SplitStrategy) -> SplitStrategy:
    value = _str(mapping, key, default)
    if value in ("exclusive_conditions", "stratified_samples"):
        return cast("SplitStrategy", value)
    msg = f"{key} must be one of exclusive_conditions / stratified_samples."
    raise ValueError(msg)


def _input_source(mapping: Mapping[str, object], key: str, default: InputSource) -> InputSource:
    value = _str(mapping, key, default)
    if value in ("mesh", "silhouette_cache", "rgb_cache"):
        return cast("InputSource", value)
    msg = f"{key} must be one of mesh / silhouette_cache / rgb_cache."
    raise ValueError(msg)


def _viewpoint_mode(mapping: Mapping[str, object], key: str, default: ViewpointMode) -> ViewpointMode:
    value = _str(mapping, key, default)
    if value in ("quiz", "turntable", "sphere"):
        return cast("ViewpointMode", value)
    msg = f"{key} must be one of quiz / turntable / sphere."
    raise ValueError(msg)


def _up_axis(mapping: Mapping[str, object], key: str, default: UpAxis) -> UpAxis:
    value = _str(mapping, key, default)
    if value in ("y", "z"):
        return cast("UpAxis", value)
    msg = f"{key} must be one of y / z."
    raise ValueError(msg)


def _label_mode(mapping: Mapping[str, object], key: str, default: LabelMode) -> LabelMode:
    value = _str(mapping, key, default)
    if value in ("stem", "species"):
        return cast("LabelMode", value)
    msg = f"{key} must be one of stem / species."
    raise ValueError(msg)
