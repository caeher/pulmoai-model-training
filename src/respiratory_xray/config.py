"""Versioned experiment configuration (issue 003).

Plain PyYAML loader with strict dataclass schemas. The layout under ``configs/``
is Hydra-compatible (``experiment.yaml`` with ``defaults``), but Hydra is optional.
Local paths and credentials live in ``configs/local.yaml`` (gitignored) and are
never merged into the publishable config.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SECTIONS = ("data", "model", "train", "evaluation", "seeds", "tracking")
_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
_SECRET_KEYS = re.compile(r"(token|password|secret|api[_-]?key|credential)", re.I)


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


def _rel_path(value: Any, name: str, required: bool = True) -> str | None:
    if value is None:
        if required:
            raise ConfigError(f"{name} is required")
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{name} must be a non-empty path string")
    if _URL.match(value):
        raise ConfigError(f"{name} must be a local path, not a URL")
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:", value):
        raise ConfigError(f"{name} must be relative in publishable config; use local.yaml")
    return value


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 224
    channels: int = 1
    interpolation: str = "bilinear"
    normalization: str = "xrv"  # xrv: [-1024, 1024]; imagenet: mean/std
    mean: tuple[float, ...] = ()
    std: tuple[float, ...] = ()
    center_crop: bool = True
    clahe: bool = False

    def __post_init__(self) -> None:
        if not 64 <= self.image_size <= 1024:
            raise ConfigError("preprocessing.image_size must be in [64, 1024]")
        if self.channels not in (1, 3):
            raise ConfigError("preprocessing.channels must be 1 or 3")
        if self.interpolation not in {"nearest", "bilinear", "bicubic", "area"}:
            raise ConfigError("preprocessing.interpolation invalid")
        if self.normalization not in {"xrv", "imagenet", "minmax"}:
            raise ConfigError("preprocessing.normalization invalid")
        if len(self.mean) != len(self.std):
            raise ConfigError("preprocessing.mean and std must have equal length")
        if any(s <= 0 for s in self.std):
            raise ConfigError("preprocessing.std must be positive")


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    classes: tuple[str, ...]
    split_manifest: str
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    allow_download: bool = False
    view_positions: tuple[str, ...] = ("PA", "AP")

    def __post_init__(self) -> None:
        if not self.dataset:
            raise ConfigError("data.dataset is required")
        if not self.classes:
            raise ConfigError("data.classes must be a non-empty ordered list")
        if len(set(self.classes)) != len(self.classes):
            raise ConfigError("data.classes must be unique")
        _rel_path(self.split_manifest, "data.split_manifest")
        if self.allow_download:
            raise ConfigError("data.allow_download must stay false (no automatic downloads)")


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_classes: int
    pretrained: str | None = None
    weights_path: str | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.name not in {"densenet121", "resnet50", "efficientnet_b0", "dummy"}:
            raise ConfigError(f"model.name unsupported: {self.name}")
        if self.num_classes < 1:
            raise ConfigError("model.num_classes must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError("model.dropout must be in [0, 1)")
        _rel_path(self.weights_path, "model.weights_path", required=False)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    amp: bool = True
    grad_accum_steps: int = 1
    num_workers: int = 2
    checkpoint_dir: str = "outputs/checkpoints"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ConfigError("train.epochs/batch_size/grad_accum_steps must be >= 1")
        if not 0 < self.lr < 1:
            raise ConfigError("train.lr must be in (0, 1)")
        if self.weight_decay < 0 or self.num_workers < 0:
            raise ConfigError("train.weight_decay/num_workers must be >= 0")
        if self.optimizer not in {"adamw", "adam", "sgd"}:
            raise ConfigError("train.optimizer invalid")
        _rel_path(self.checkpoint_dir, "train.checkpoint_dir")


@dataclass(frozen=True)
class EvaluationConfig:
    split: str = "validation"
    metrics: tuple[str, ...] = ("auroc", "auprc", "ece")
    selection_metric: str = "auroc"
    output_dir: str = "reports"

    def __post_init__(self) -> None:
        if self.split not in {"validation", "test", "external"}:
            raise ConfigError("evaluation.split invalid")
        if self.selection_metric not in self.metrics:
            raise ConfigError("evaluation.selection_metric must be listed in metrics")
        _rel_path(self.output_dir, "evaluation.output_dir")


@dataclass(frozen=True)
class SeedsConfig:
    global_seed: int = 42
    deterministic: bool = True
    split_seed: int = 1234

    def __post_init__(self) -> None:
        for name in ("global_seed", "split_seed"):
            v = getattr(self, name)
            if type(v) is not int or not 0 <= v < 2**32:
                raise ConfigError(f"seeds.{name} must be an int in [0, 2**32)")


@dataclass(frozen=True)
class TrackingConfig:
    backend: str = "auto"  # auto | mlflow | json
    experiment: str = "respiratory-xray-dl"
    output_dir: str = "mlruns"
    wandb_enabled: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "mlflow", "json"}:
            raise ConfigError("tracking.backend must be auto, mlflow or json")
        _rel_path(self.output_dir, "tracking.output_dir")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    seeds: SeedsConfig
    tracking: TrackingConfig
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.model.num_classes != len(self.data.classes):
            raise ConfigError("model.num_classes must equal len(data.classes)")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def preprocessing_signature(self) -> dict[str, Any]:
        """Everything required to rebuild preprocessing and output order exactly."""
        return {
            "classes": list(self.data.classes),
            "preprocessing": _jsonable(asdict(self.data.preprocessing)),
            "view_positions": list(self.data.view_positions),
        }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _build(cls: type, values: Any, scope: str) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"{scope} must be a mapping")
    known = set(cls.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"{scope}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for k, v in values.items():
        if isinstance(v, list):
            v = tuple(v)
        kwargs[k] = v
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{scope}: {exc}") from exc


def _check_no_secrets(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _SECRET_KEYS.search(str(k)):
                raise ConfigError(f"secret-like key '{path}{k}' not allowed in publishable config")
            _check_no_secrets(v, f"{path}{k}.")
    elif isinstance(obj, list):
        for v in obj:
            _check_no_secrets(v, path)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    _check_no_secrets(raw)
    raw = copy.deepcopy(raw)
    missing = [s for s in SECTIONS if s not in raw]
    if missing:
        raise ConfigError(f"missing sections: {missing}")
    extra = set(raw) - set(SECTIONS) - {"schema_version", "defaults"}
    if extra:
        raise ConfigError(f"unknown top-level keys {sorted(extra)}")
    data = dict(raw["data"])
    data["preprocessing"] = _build(
        PreprocessingConfig, data.get("preprocessing", {}), "data.preprocessing"
    )
    return ExperimentConfig(
        data=_build(DataConfig, data, "data"),
        model=_build(ModelConfig, raw["model"], "model"),
        train=_build(TrainConfig, raw["train"], "train"),
        evaluation=_build(EvaluationConfig, raw["evaluation"], "evaluation"),
        seeds=_build(SeedsConfig, raw["seeds"], "seeds"),
        tracking=_build(TrackingConfig, raw["tracking"], "tracking"),
        schema_version=raw.get("schema_version", 1),
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a mapping")
    return loaded


def compose(
    config_dir: str | Path, name: str = "experiment", overrides: dict | None = None
) -> dict[str, Any]:
    """Minimal Hydra-style composition: ``defaults: [{data: nih_baseline}, ...]``."""
    config_dir = Path(config_dir)
    root = load_yaml(config_dir / f"{name}.yaml")
    merged: dict[str, Any] = {}
    for entry in root.pop("defaults", []) or []:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ConfigError("defaults entries must be single-key mappings")
        ((group, option),) = entry.items()
        part = load_yaml(config_dir / group / f"{option}.yaml")
        merged = _deep_merge(merged, part)
    merged = _deep_merge(merged, root)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def load_config(
    config_dir: str | Path = "configs", name: str = "legacy_cxr", overrides: dict | None = None
) -> ExperimentConfig:
    return config_from_dict(compose(config_dir, name, overrides))


def load_local_settings(path: str | Path = "configs/local.yaml") -> dict[str, Any]:
    """Machine-specific paths/credentials. Never logged nor merged into the experiment."""
    p = Path(path)
    return load_yaml(p) if p.exists() else {}
