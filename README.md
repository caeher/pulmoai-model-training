# PulmoAI Model Training

**PulmoAI — Pulmonary Nodule Detection and Analysis with Deep Learning.**

Research tooling for the CT model lifecycle: local data preparation, reproducible
training, independent evaluation, inference, and artifact export. Nodule detection
supports research on early lung cancer detection; it does not diagnose cancer.
Detector scores are uncalibrated candidate scores, not malignancy probabilities.
This project has no clinically validated model or demonstrated clinical benefit.

The existing Python/PyYAML configuration and local tracking foundation is retained.
The original `respiratory_xray` namespace and CXR configurations remain available
for compatibility; no CXR models or weights had been implemented. CT and CXR are
separate tasks and never share weights automatically. All project content is English.

## Implemented scope

- LUNA16 MHD/RAW CT reading through SimpleITK; internal HU NPZ caches and fixtures.
- Explicit patient mapping, reproducible patient splits, checksums, metadata checks,
  and exact duplicate checks across splits. No automatic dataset acquisition.
- HU clipping/normalization, spacing-aware resampling, native coordinate restoration,
  spherical proxy supervision, ignored regions, deterministic patch sampling and flips.
- One small PyTorch 3D U-Net trained from scratch; weighted voxel BCE, AdamW,
  gradient accumulation, optional CUDA AMP, fixed validation patches, early stopping,
  best/last checkpoints and epoch-boundary recovery.
- Full-volume overlapping tiles, local maxima, physical-distance NMS, JSON predictions,
  optional PNG slice previews, and inference-only PyTorch state-dict export.
- Internal sensitivity, FP/scan, FROC, CPM, precision/recall/F1 and patient bootstrap.
  The internal evaluator is **not the official LUNA16 benchmark**.
- Local JSON tracking; optional existing MLflow mirror. No remote tracking.

Only synthetic volumes have been used to verify this implementation. Real CT
training, CUDA training, official LUNA16 cross-validation, DICOM/LIDC XML ingestion,
NIfTI, malignancy classification and a separate false-positive classifier remain
unverified or unimplemented. See [the initial audit](docs/AUDIT.md),
[research plan](docs/PLAN.md), and [delivery report](docs/IMPLEMENTATION_REPORT.md).

## Installation

Use Python 3.11 or newer in a local environment. Choose CPU or an appropriate CUDA
PyTorch wheel using the [official installer](https://pytorch.org/get-started/locally/).
The commands below use CPU and work from the repository root in PowerShell or bash
(after activating the environment for your shell).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,ct]"
python -m pulmoai --help
python -m pulmoai validate-config
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

On bash, activate with `source .venv/bin/activate`. The base installation needs only
PyYAML; the `ct` extra adds NumPy, SciPy, PyTorch, SimpleITK and Pillow. Importing
`pulmoai` or loading configuration does not import PyTorch or initialize workloads.
`pip install -e ".[tracking]"` adds optional MLflow. No model or dataset downloads
occur on import, during training, or in the tests.

## Verified synthetic workflow

These are real training/evaluation operations on six tiny generated CT volumes,
used solely to check software. Their scores are not evidence of medical performance.
Use new output directories on each independent run.

```powershell
python -m pulmoai synthetic --config smoke --output data/processed/synthetic
python -m pulmoai validate-data --manifest data/processed/synthetic/manifest.json
python -m pulmoai train --config smoke --manifest data/processed/synthetic/manifest.json --output artifacts/smoke
python -m pulmoai evaluate --checkpoint artifacts/smoke/best.pt --manifest data/processed/synthetic/manifest.json --output artifacts/smoke/evaluation
python -m pulmoai infer --checkpoint artifacts/smoke/best.pt --image data/processed/synthetic/synthetic_4.npz --output artifacts/smoke/prediction.json --preview artifacts/smoke/preview.png
python -m pulmoai export --checkpoint artifacts/smoke/best.pt --output artifacts/smoke/export/detector.pt
```

## Real CT workflow

Acquire permitted LUNA16 files manually and build a verified patient mapping before
preparation. Set `data.version` in `configs/experiment.yaml` to your recorded release
identifier. Keep that configuration fixed once preparation begins.

```powershell
python -m pulmoai prepare --scans data/raw/luna16 --annotations data/raw/annotations.csv --patients data/raw/patients.csv --output data/processed/luna16
python -m pulmoai validate-data --manifest data/processed/luna16/manifest.json
python -m pulmoai train --manifest data/processed/luna16/manifest.json --output artifacts/luna16-baseline
python -m pulmoai evaluate --checkpoint artifacts/luna16-baseline/best.pt --manifest data/processed/luna16/manifest.json --split validation --output artifacts/luna16-baseline/validation
python -m pulmoai evaluate --checkpoint artifacts/luna16-baseline/best.pt --manifest data/processed/luna16/manifest.json --split test --output artifacts/luna16-baseline/test
```

Include `--excluded data/raw/annotations_excluded.csv` during preparation when
available. See [dataset contracts and access](docs/DATASETS.md),
[training and reproducibility](docs/TRAINING.md), and
[inference JSON and evaluation semantics](docs/INFERENCE.md). Test is for a frozen
protocol only. Tune candidate extraction and model settings using validation.

## Repository layout

```text
configs/                       CT experiment, synthetic smoke, preserved CXR groups
src/pulmoai/                   CT config, datasets, preprocessing, model, training,
                               evaluation, inference, tracking and utilities
src/respiratory_xray/           preserved legacy configuration/safety/tracking API
scripts/                       environment diagnostics and preparation compatibility entry point
tests/                         offline synthetic unit and integration tests
docs/                          audit, current pipeline guides, plan and historical CXR backlog
data/{raw,interim,processed}/   local private data, ignored by Git
artifacts/                     local checkpoints, logs, predictions and exports, ignored by Git
```

Runtime directories are created when commands need them. Do not commit scans,
patient mappings, identifiers, credentials, or weights. Predictions and previews
may contain restricted information even without patient names. Aggregate tracker
records exclude patient IDs and image content; per-scan files stay local.

The original code is MIT licensed. Preserve its copyright notice; datasets,
annotations and third-party weights retain their own terms and attribution.
LUNA16 is derived from LIDC-IDRI and cannot serve as an independent external cohort
against its parent dataset. No benign/malignant task is implemented.
