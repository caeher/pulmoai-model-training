"""Strict declarative CT configuration, reusing the existing YAML composer."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from respiratory_xray.config import ConfigError, compose


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "luna16"
    version: str = "user-verified-release"
    spacing_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    patch_size: int = 32
    patches_per_scan: int = 8
    positive_fraction: float = 0.5
    flip_probability: float = 0.5
    split_fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)


@dataclass(frozen=True)
class ModelConfig:
    name: str = "small_unet3d"
    base_channels: int = 8


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 20
    batch_size: int = 2
    lr: float = 0.001
    weight_decay: float = 0.00001
    grad_accum_steps: int = 1
    num_workers: int = 0
    amp: bool = True
    device: str = "auto"
    patience: int = 5
    positive_weight: float = 10.0
    cpu_threads: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    score_threshold: float = 0.1
    nms_distance_mm: float = 5.0
    max_candidates: int = 300
    overlap: float = 0.5
    operating_threshold: float = 0.5
    fp_rates: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    bootstrap_samples: int = 200


@dataclass(frozen=True)
class SeedsConfig:
    global_seed: int = 42
    split_seed: int = 1234
    deterministic: bool = True


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "json"
    experiment: str = "pulmoai-model-training"
    output_dir: str = "tracking"
    wandb_enabled: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    schema_version: int = 2

    def __post_init__(self) -> None:
        d, m, t, e = self.data, self.model, self.train, self.evaluation
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ConfigError("CT schema_version must be integer 2")
        if d.dataset not in {"luna16", "synthetic"}:
            raise ConfigError("Expected CT schema_version 2 and luna16 or synthetic dataset")
        if not d.version.strip():
            raise ConfigError("data.version must identify the dataset release")
        if len(d.spacing_xyz) != 3 or any(v <= 0 for v in d.spacing_xyz):
            raise ConfigError("spacing_xyz must contain three positive millimeter values")
        if len(d.hu_window) != 2 or d.hu_window[0] >= d.hu_window[1]:
            raise ConfigError("hu_window must have increasing bounds")
        if d.patch_size < 8 or d.patch_size % 4 or d.patches_per_scan < 1:
            raise ConfigError("patch_size must be >=8 and divisible by 4; patches_per_scan >=1")
        if not 0 <= d.positive_fraction <= 1 or not 0 <= d.flip_probability <= 1:
            raise ConfigError("Sampling and flip probabilities must be in [0,1]")
        if (
            len(d.split_fractions) != 3
            or any(v <= 0 for v in d.split_fractions)
            or not math.isclose(sum(d.split_fractions), 1.0)
        ):
            raise ConfigError("Three positive split_fractions must sum to one")
        if m.name != "small_unet3d" or m.base_channels < 2 or m.base_channels % 2:
            raise ConfigError("Only small_unet3d with an even base_channels >=2 is supported")
        if any(
            v < 1 for v in (t.epochs, t.batch_size, t.grad_accum_steps, t.patience, t.cpu_threads)
        ):
            raise ConfigError("Training counts and patience must be positive")
        if t.num_workers < 0 or not 0 < t.lr < 1 or t.weight_decay < 0:
            raise ConfigError("Invalid workers, learning rate, or weight decay")
        if t.positive_weight <= 0 or t.device not in {"auto", "cpu", "cuda"}:
            raise ConfigError("positive_weight must be positive; device must be auto/cpu/cuda")
        if not 0 <= e.score_threshold <= e.operating_threshold <= 1:
            raise ConfigError("Thresholds must satisfy 0 <= score <= operating <= 1")
        if e.nms_distance_mm <= 0 or e.max_candidates < 1 or not 0 <= e.overlap < 1:
            raise ConfigError("Invalid NMS distance, candidate limit, or tile overlap")
        if not e.fp_rates or any(v <= 0 for v in e.fp_rates) or e.bootstrap_samples < 0:
            raise ConfigError("fp_rates must be positive and bootstrap_samples nonnegative")
        if tuple(sorted(set(e.fp_rates))) != e.fp_rates:
            raise ConfigError("fp_rates must be unique and increasing")
        for seed in (self.seeds.global_seed, self.seeds.split_seed):
            if type(seed) is not int or not 0 <= seed < 2**32:
                raise ConfigError("Seeds must be integers in [0,2**32)")
        if self.tracking.backend not in {"json", "mlflow"} or self.tracking.wandb_enabled:
            raise ConfigError("CT tracking supports local JSON or MLflow; W&B is unavailable")
        for name, value in (
            ("experiment", self.tracking.experiment),
            ("output_dir", self.tracking.output_dir),
        ):
            if (
                not value
                or Path(value).is_absolute()
                or ":" in value
                or ".." in Path(value).parts
                or "\\" in value
            ):
                raise ConfigError(f"tracking.{name} must be a safe relative name/path")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def preprocessing_signature(self) -> dict[str, Any]:
        return {
            "modality": "CT",
            "spacing_xyz": list(self.data.spacing_xyz),
            "hu_window": list(self.data.hu_window),
            "target": "spherical_proxy_v1",
        }


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    classes = {
        "data": DataConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "evaluation": EvaluationConfig,
        "seeds": SeedsConfig,
        "tracking": TrackingConfig,
    }
    if not isinstance(raw, dict) or set(raw) - set(classes) - {"schema_version"}:
        raise ConfigError("Unknown CT configuration sections")
    values: dict[str, Any] = {}
    try:
        for section, cls in classes.items():
            supplied = raw.get(section, {})
            if not isinstance(supplied, dict):
                raise ConfigError(f"{section} must be a mapping")
            defaults = cls()
            fields = cls.__dataclass_fields__
            if set(supplied) - set(fields):
                raise ConfigError(f"Unknown {section} keys: {set(supplied) - set(fields)}")
            kwargs = {}
            for key, value in supplied.items():
                default = getattr(defaults, key)
                if isinstance(default, tuple):
                    if not isinstance(value, (list, tuple)):
                        raise ConfigError(f"{section}.{key} must be a sequence")
                    if any(type(v) not in (int, float) or not math.isfinite(v) for v in value):
                        raise ConfigError(f"{section}.{key} must contain finite numbers")
                    value = tuple(value)
                elif type(default) is bool:
                    if type(value) is not bool:
                        raise ConfigError(f"{section}.{key} must be a boolean")
                elif type(default) is int:
                    if type(value) is not int:
                        raise ConfigError(f"{section}.{key} must be an integer")
                elif type(default) is float:
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ConfigError(f"{section}.{key} must be finite")
                elif not isinstance(value, str):
                    raise ConfigError(f"{section}.{key} must be a string")
                kwargs[key] = value
            values[section] = cls(**kwargs)
        return ExperimentConfig(**values, schema_version=raw.get("schema_version", 2))
    except (TypeError, AttributeError) as exc:
        raise ConfigError(f"Invalid CT configuration: {exc}") from exc


def load_config(
    config_dir: str | Path = "configs", name: str = "experiment", overrides: dict | None = None
) -> ExperimentConfig:
    return config_from_dict(compose(config_dir, name, overrides))
