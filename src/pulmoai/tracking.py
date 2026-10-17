"""Local, privacy-preserving experiment tracking (issue 003).

Backends: ``json`` (always available, files under ``mlruns/json``) and ``mlflow``
(local file store, used only if mlflow is installed). W&B is opt-in and disabled
by default; nothing here sends data over the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pulmoai.config import ExperimentConfig

_FORBIDDEN_KEY = re.compile(
    r"(token|password|secret|api[_-]?key|credential|patient|study_?id|subject_?id|"
    r"accession|mrn|image_?path|volume_?path|seriesuid|dicom|pixel)",
    re.I,
)
_IMAGE_EXT = (
    ".png",
    ".jpg",
    ".jpeg",
    ".dcm",
    ".dicom",
    ".tif",
    ".tiff",
    ".bmp",
    ".npy",
    ".mhd",
    ".raw",
    ".nii",
    ".nii.gz",
    ".npz",
)
REDACTED = "[REDACTED]"


class TrackingError(RuntimeError):
    pass


def redact(obj: Any) -> Any:
    """Recursively replace sensitive keys/values."""
    if isinstance(obj, dict):
        return {
            k: (REDACTED if _FORBIDDEN_KEY.search(str(k)) else redact(v)) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    if isinstance(obj, str) and obj.lower().endswith(_IMAGE_EXT):
        return REDACTED
    return obj


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        sha = out.stdout.strip() if out.returncode == 0 else "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
        if sha != "unknown" and dirty.stdout.strip():
            sha += "-dirty"
        return sha
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
        "gpu": None,
        "cpu_count": os.cpu_count(),
    }
    from importlib.metadata import PackageNotFoundError, version

    info["dependencies"] = {}
    for package in ("numpy", "scipy", "SimpleITK", "PyYAML", "pillow", "mlflow"):
        try:
            info["dependencies"][package] = version(package)
        except PackageNotFoundError:
            info["dependencies"][package] = None
    try:
        import torch  # noqa: PLC0415

        info["torch"] = str(torch.__version__)
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - torch optional
        pass
    return info


def peak_vram_mb() -> float | None:
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return None


class _JsonBackend:
    def __init__(self, root: Path, experiment: str):
        self.root = root / "json" / experiment

    def run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    def start(self, run_id: str) -> None:
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)

    def write(self, run_id: str, record: dict[str, Any]) -> None:
        d = self.run_dir(run_id)
        tmp = d / "run.json.tmp"
        tmp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, d / "run.json")

    def read(self, run_id: str) -> dict[str, Any]:
        p = self.run_dir(run_id) / "run.json"
        if not p.exists():
            raise TrackingError(f"run {run_id} not found")
        return json.loads(p.read_text(encoding="utf-8"))


class _MlflowMirror:
    """Mirrors params/metrics/tags to a local MLflow file store."""

    def __init__(self, root: Path, experiment: str):
        import mlflow  # noqa: PLC0415

        self.mlflow = mlflow
        store = (root / "mlflow").resolve()
        store.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        # local file store (works with mlflow-skinny; no server, no network)
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(store.as_uri())
        mlflow.set_experiment(experiment)
        self.active = None

    def start(self, run_name: str, mlflow_run_id: str | None) -> str:
        self.active = self.mlflow.start_run(run_id=mlflow_run_id, run_name=run_name)
        return self.active.info.run_id

    def params(self, flat: dict[str, Any]) -> None:
        self.mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})

    def metric(self, key: str, value: float, step: int | None) -> None:
        self.mlflow.log_metric(key, value, step=step)

    def tags(self, tags: dict[str, Any]) -> None:
        self.mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def end(self, status: str) -> None:
        self.mlflow.end_run(status="FINISHED" if status == "FINISHED" else "FAILED")


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class Tracker:
    """Records one run. Use :meth:`run` as a context manager."""

    def __init__(
        self,
        config: ExperimentConfig,
        root: str | Path | None = None,
        backend: str | None = None,
        repo_dir: str | Path | None = None,
    ):
        self.config = config
        tc = config.tracking
        if tc.wandb_enabled and os.environ.get("RXDL_ALLOW_WANDB") != "1":
            raise TrackingError(
                "W&B is opt-in: set RXDL_ALLOW_WANDB=1 explicitly; never log restricted data"
            )
        self.root = Path(root if root is not None else tc.output_dir)
        choice = backend or tc.backend
        if choice == "auto":
            choice = "mlflow" if mlflow_available() else "json"
        if choice == "mlflow" and not mlflow_available():
            raise TrackingError("mlflow backend requested but mlflow is not installed")
        self.backend_name = choice
        self.store = _JsonBackend(self.root, tc.experiment)
        self.mirror = _MlflowMirror(self.root, tc.experiment) if choice == "mlflow" else None
        self.repo_dir = repo_dir
        self.run_id: str | None = None
        self.record: dict[str, Any] = {}
        self._t0 = 0.0

    # ---- lifecycle -------------------------------------------------------
    def start(self, run_name: str = "run", resume_id: str | None = None) -> str:
        self._t0 = time.monotonic()
        if resume_id:
            rec = self.store.read(resume_id)
            if rec["config_fingerprint"] != self.config.fingerprint():
                raise TrackingError("cannot resume: config fingerprint differs")
            rec["status"] = "RUNNING"
            rec.pop("error", None)
            rec["resume_count"] = rec.get("resume_count", 0) + 1
            self.run_id, self.record = resume_id, rec
        else:
            self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
            self.store.start(self.run_id)
            self.record = {
                "run_id": self.run_id,
                "run_name": run_name,
                "status": "RUNNING",
                "backend": self.backend_name,
                "package_version": _package_version(),
                "git_commit": git_commit(self.repo_dir),
                "environment": environment_info(),
                "seed": self.config.seeds.global_seed,
                "config": redact(self.config.to_dict()),
                "config_fingerprint": self.config.fingerprint(),
                "preprocessing_signature": self.config.preprocessing_signature(),
                "hashes": {},
                "metrics": {},
                "artifacts": {},
                "selection_decisions": [],
                "wandb_enabled": self.config.tracking.wandb_enabled,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "wall_time_s": 0.0,
                "resume_count": 0,
            }
        if self.mirror:
            mid = self.mirror.start(run_name, self.record.get("mlflow_run_id"))
            if not resume_id:
                self.record["mlflow_run_id"] = mid
                self.mirror.params(_flatten(self.record["config"]))
                self.mirror.tags(
                    {
                        "git_commit": self.record["git_commit"],
                        "seed": self.record["seed"],
                        "config_fingerprint": self.record["config_fingerprint"],
                    }
                )
        self._flush()
        return self.run_id

    def end(self, status: str = "FINISHED", error: BaseException | None = None) -> None:
        self._require()
        self.record["status"] = status
        self.record["wall_time_s"] = round(
            self.record.get("wall_time_s", 0.0) + time.monotonic() - self._t0, 3
        )
        self.record["peak_vram_mb"] = peak_vram_mb()
        self.record["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if error is not None:
            # Exception messages may contain identifiers embedded within prose or paths.
            self.record["error"] = {"type": type(error).__name__}
        self._flush()
        if self.mirror:
            self.mirror.end(status)

    @contextmanager
    def run(self, run_name: str = "run", resume_id: str | None = None):
        self.start(run_name, resume_id)
        try:
            yield self
        except BaseException as exc:
            self.end("FAILED", exc)
            raise
        else:
            self.end("FINISHED")

    # ---- logging ---------------------------------------------------------
    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self._require()
        self._check_key(key)
        value = float(value)
        if not math.isfinite(value):
            raise TrackingError(f"metric {key} must be finite")
        self.record["metrics"].setdefault(key, []).append({"step": step, "value": value})
        if self.mirror:
            self.mirror.metric(key, value, step)
        self._flush()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            self.log_metric(k, v, step)

    def log_hash(self, name: str, path: str | Path) -> str:
        """Record sha256 of a manifest/weights file (the path itself is not stored)."""
        self._require()
        digest = sha256_file(path)
        self.record["hashes"][name] = digest
        self._flush()
        return digest

    def log_artifact(self, name: str, payload: dict[str, Any]) -> None:
        """Aggregated artifacts only (e.g. metric tables). Per-sample data is rejected."""
        self._require()
        self._check_key(name)
        if _contains_rows(payload):
            raise TrackingError("artifacts must be aggregated; per-sample records are not allowed")
        self.record["artifacts"][name] = redact(payload)
        self._flush()

    def log_selection(self, decision: str, reason: str, metric: str, value: float) -> None:
        self._require()
        self.record["selection_decisions"].append(
            {"decision": decision, "reason": reason, "metric": metric, "value": float(value)}
        )
        self._flush()

    # ---- helpers ---------------------------------------------------------
    def _check_key(self, key: str) -> None:
        if _FORBIDDEN_KEY.search(key):
            raise TrackingError(f"refusing to log sensitive key '{key}'")

    def _require(self) -> None:
        if self.run_id is None:
            raise TrackingError("tracker not started")

    def _flush(self) -> None:
        self.store.write(self.run_id, self.record)

    def load(self, run_id: str | None = None) -> dict[str, Any]:
        return self.store.read(run_id or self.run_id)


def _contains_rows(obj: Any) -> bool:
    if isinstance(obj, list) and len(obj) > 0 and all(isinstance(x, dict) for x in obj):
        return any(_FORBIDDEN_KEY.search(k) for x in obj for k in x) or len(obj) > 1000
    if isinstance(obj, dict):
        return any(_contains_rows(v) for v in obj.values())
    return False


def _package_version() -> str:
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("pulmoai-model-training")
    except Exception:  # noqa: BLE001
        from pulmoai import __version__  # noqa: PLC0415

        return __version__
