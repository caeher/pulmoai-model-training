# PulmoAI implementation and validation report

Delivered 2026-10-17. All maintained project documentation, code comments, CLI
messages and local backlog content are in English. The existing project was
inspected before implementation; [AUDIT.md](AUDIT.md) records its original state.

## Delivered changes and decisions

The project is now packaged as `pulmoai-model-training` version 0.2.0. Its new
`pulmoai` namespace implements the CT lifecycle while preserving the original
`respiratory_xray` namespace, CXR configuration groups, network guard, diagnostic
scripts, tests and research proposals. The old tracker import forwards to the
hardened shared implementation. The original YAML composer is reused, with a
separate CT schema; CT and CXR schemas are deliberately incompatible.

There were no existing model implementations, weights or datasets to remove or
reuse. The single CT baseline is a native PyTorch 3D U-Net with 19,425 parameters
at its default width of 8. It uses spherical proxy targets derived from nodule
centers/diameters, weighted BCE and AdamW. This is an architectural starting point,
not a validated segmentation, malignancy classifier or clinically useful detector.
PyTorch avoids an unnecessary stack migration; MONAI and false-positive
classifiers are deferred until a real baseline justifies a comparison.

LUNA16 MHD/RAW is the implemented external input format. Data preparation retains
original and resampled geometry, validates annotations and explicit patient
mappings, records release/hashes and creates deterministic patient holdouts. Cache
validation also rejects duplicate decoded voxel values across splits. LIDC-IDRI
was evaluated as the parent source, but DICOM/XML consensus processing is not
implemented. No datasets or pretrained weights were downloaded.

Training supports CPU/CUDA selection, deterministic sampling/shuffling, matched
flips, accumulation, CUDA-only AMP, fixed validation patches, validation-loss
selection, early stopping and atomic checkpoint files. Recovery restores a
completed epoch including optimizer/scaler/RNG, best weights and committed JSON
metric history. A missing best.pt after a write interruption can be reconstructed
from last.pt. Completed or stopped runs cannot be extended under a changed config.
The optional MLflow mirror remains append-only; the JSON history is authoritative.

Inference uses full-volume overlapping tiles, local maxima and physical-distance
NMS, returning physical XYZ and native continuous voxel coordinates. Export writes
a versioned PyTorch inference bundle and JSON metadata. An optional PNG contact
sheet helps inspect native slice coordinates. These outputs can be consumed by a
future API without coupling the trainer to a server framework.

Internal evaluation includes sensitivity, FP/scan, FROC, CPM, detection
precision/recall/F1 and patient-cluster bootstrap. Matching, duplicate handling,
score ties, exclusions and interpolation are explicit in [INFERENCE.md](INFERENCE.md).
This is not the official LUNA16 evaluator or its ten-fold benchmark.

## File inventory

| Files added or changed | Purpose |
| --- | --- |
| `pyproject.toml`, `README.md`, `.gitignore`, `LICENSE` | PulmoAI metadata/CLI/extras, English documentation, local artifact exclusions; original copyright preserved. |
| `.github/workflows/ci.yml` | Install CPU PyTorch and CT dependencies for offline synthetic checks on Windows/Linux. Remote CI execution was not performed here. |
| `configs/experiment.yaml`, `configs/smoke.yaml`, `configs/legacy_cxr.yaml`, `configs/README.md` | Operational CT settings, tiny CPU verification, preserved legacy composition and configuration guide. |
| `configs/local.example.yaml`, `configs/tracking/local.yaml`, `configs/train/baseline.yaml` | CT local-path example, PulmoAI legacy tracking name, removal of an unverified GPU-fit comment. Other CXR groups retained. |
| `src/pulmoai/__init__.py`, `__main__.py`, `cli.py`, `config.py`, `safety.py`, `tracking.py` | Package entry points, strict CT configuration, network guard and shared local tracking. |
| `src/pulmoai/datasets/{__init__,manifest,patches,synthetic}.py` | MHD ingestion, patient/duplicate checks, sampling, ignored/padded masks and synthetic fixtures. |
| `src/pulmoai/preprocessing/{__init__,volume}.py` | CT geometry, HU normalization, resampling, safe NPZ caches and decoded-voxel hashes. |
| `src/pulmoai/models/{__init__,unet}.py` | Compact 3D candidate model. |
| `src/pulmoai/training/{__init__,engine}.py` | Training, validation, recovery and export. |
| `src/pulmoai/evaluation/{__init__,metrics,runner}.py` | Internal detection evaluation, bootstrap and JSON/CSV outputs. |
| `src/pulmoai/inference/{__init__,detector,preview}.py` | Reusable detector and local PNG slice inspection. |
| `src/pulmoai/utils/{__init__,io,runtime}.py` | Atomic JSON and deterministic device/runtime setup. |
| `src/respiratory_xray/config.py`, `tracking.py` | Explicit legacy CXR default config and compatibility tracker imports; formatting normalized. |
| `scripts/prepare_data.py`, `acquire_data.py`, `check_optional_imports.py`, `scripts/README.md` | Operational preparation wrapper, guarded acquisition placeholder, expanded environment diagnostic and English usage. CUDA smoke script retained. |
| `tests/test_ct_config.py`, `test_ct_data.py`, `test_detection.py`, `test_pipeline.py` | New CT configuration, geometry/data, metrics and end-to-end/recovery tests. |
| `tests/test_tracking.py`, `tests/conftest.py` | Shared tracker privacy/nonfinite checks; restore original socket functions after offline tests. Existing config/scaffold tests retained. |
| `docs/AUDIT.md`, `DATASETS.md`, `TRAINING.md`, `INFERENCE.md`, `IMPLEMENTATION_REPORT.md` | Initial audit, implemented contracts, commands, technical decisions and this report. |
| `docs/PLAN.md`, `docs/environment/windows-cuda.md` | English current plan, preserved CXR research rationale/references and clearly historical CUDA evidence. |
| `docs/issues/README.md` and all twenty issue documents | English translation/adaptation with accurate status; Spanish filenames renamed to their English equivalents to satisfy the language requirement. No research proposal was represented as a completed model. |

The repository started without any Git commit and with all supplied files
untracked. No commit, branch, PR, dataset deletion or experiment deletion was
performed. Git-based diffs against an initial revision are consequently unavailable;
this inventory and the audit describe the evolution. Tracking records source
revision as `unknown` until a source revision is committed.

## Verified execution

Environment used: Windows, Python 3.13.14, PyTorch 2.14.0+cpu, NumPy 2.4.4,
SciPy 1.17.1, SimpleITK 2.5.6, PyYAML 6.0.3, Pillow 12.2.0, pytest 8.3.5 and
Ruff 0.15.22. CUDA was unavailable through the installed CPU build. The local
`.venv` used system site packages to reuse the existing PyTorch installation;
new dependencies and the editable project were installed in that environment.
The README documents a conventional fresh environment for another machine.

Validation commands:

```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m pip check
python -m pulmoai validate-config
python -m pulmoai validate-config --config smoke
```

The synthetic suite has **62 passing tests and one optional MLflow test skipped**
because MLflow is not installed. Lint and formatting pass; pip reports no broken
requirements. The host's unrelated pytest-asyncio plugin emits a deprecation
warning; no async plugin is required by this project.

Tests cover rotated/reflected anisotropic physical transforms and actual resample
values, HU range, invalid modality/metadata, synthetic MHD/RAW ingestion, patient
mapping and grouping, checksum tampering, duplicate decoded pixels, aligned
patch/label transforms, fixed validation, score ties, negative scans, exclusions,
duplicate detections, null metrics, bootstrap, native-coordinate restoration,
plateaus/NMS, tile edge coverage, incompatible artifacts and nonfinite predictions.
Repeated CPU training produces identical model tensors. A forced interruption
immediately after last.pt is committed is recovered with identical final model
weights to uninterrupted training. The test also checks rollback of partially
logged later metrics and clearing the old failure status. All checkpoint loads
keep `weights_only=True`.

Actual CLI smoke artifacts are local and ignored by Git:

- `data/processed/synthetic-final/manifest.json`: six tiny generated CT volumes,
  with separate train/validation/test patients and positive/negative scans.
- `artifacts/smoke-final/best.pt`, `last.pt`, `config.json`, `tracking/`: two actual
  CPU training epochs and run provenance.
- `artifacts/smoke-final/evaluation/`: aggregate metrics and private JSON/CSV predictions.
- `artifacts/smoke-final/prediction.json`, `preview.png`: inference and inspected preview.
- `artifacts/smoke-final/export/detector.pt`, `detector.json`: exported inference bundle.

The two-epoch synthetic detector has internal CPM 0.0 on its two synthetic test
scans. This is an executed software smoke result, not an estimate of clinical or
real CT performance, and does not demonstrate a useful detector. Geometry/metric
correctness is separately checked against known synthetic expectations. No real
patient images, pretrained weights or fabricated medical metrics were used.

## Remaining work for real experiments

1. Review/retain the exact LUNA16 and parent LIDC-IDRI terms and attribution,
   obtain data manually, record release, and build a verified series-to-patient
   mapping. Include the complete compatible inclusion/exclusion references.
2. Set the dataset version in YAML, prepare a patient-disjoint manifest, review
   scan/nodule counts, duplicates, HU, acquisition geometry and transformed labels
   on permitted real examples. Audit exclusions and annotation completeness.
3. Profile a small real CT pilot on the intended CPU/GPU. Measure RAM, VRAM,
   throughput and convergence; tune patches, sampling and candidate extraction
   using validation only. Default batch/patch sizes are not a measured 4 GB guarantee.
4. Freeze model/settings and evaluate the independent holdout. For benchmark
   comparisons, implement the official fold protocol and run the organizers'
   evaluator rather than presenting the internal score as an official result.
5. Produce a model card and cohort sheet tied to actual runs and uncertainty.
   Add external CT validation only after source/annotation/license and independence
   audits. LUNA16 and LIDC-IDRI are overlapping sources, not external cohorts.

Direct DICOM/LIDC XML/NIfTI support, acquisition adapters, official fold orchestration,
lung segmentation, diameter regression, separate false-positive classification,
malignancy labels/classification, calibration/OOD, Gradio, ONNX/TorchScript and a
clinical validation study remain outside the delivered MVP. Optional MLflow and
CUDA training are not verified here. Full-volume arrays and decoded patch caches
still impose RAM/I/O costs. No patient reidentification, clinical diagnosis,
malignancy probability or clinical deployment claim is made.
