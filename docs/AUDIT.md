# Initial repository audit

Audit date: 2026-10-17. This records the state inspected before implementation.
The Git repository had no commits and all supplied files were untracked. No user
files, datasets, experiments, or model weights were deleted.

| Area | Verified initial state | Decision |
| --- | --- | --- |
| Package | `src/respiratory_xray`, version 0.1.0, Python >=3.11 | Preserve the import namespace for compatibility; add `pulmoai` as the CT interface. |
| Configuration | Working PyYAML composition, frozen dataclasses, fingerprints, private settings separation | Reuse composition and fingerprint conventions; add a separate strict CT schema. |
| Tracking | Working JSON run records, optional local MLflow, environment and file hashes, resume/failure handling | Reuse and harden; use JSON explicitly for the default MVP. W&B has a gate but no implemented backend. |
| Safety | Network opt-in guard and unconditional CI denial; no downloader | Preserve; keep acquisition manual. |
| Data | NIH CXR configuration only; other CXR resources appear in planning documents | Keep CXR configuration separately. Add LUNA16 MHD/RAW ingestion with explicit patient mapping. |
| Image formats | None actually implemented; JPG/PNG/DICOM were planned | Add CT MHD/RAW and an internal NPZ fixture/cache format. Do not claim DICOM/XML/NIfTI support. |
| Models | Empty model namespace; DenseNet/XRV identifiers only | No existing architecture or weights to reuse. Add one compact PyTorch 3D U-Net. |
| Training | Empty namespace; YAML hyperparameters only | Implement patch training, validation, AMP, checkpoints, exact epoch-boundary resume, early stopping. |
| Preprocessing | CXR schema only, no operational transforms | Add HU windowing, physical resampling, spatial metadata, spherical proxy targets, matched flips. |
| Evaluation / inference | Empty namespaces | Add full-volume tiled detection, physical NMS, internal FROC/CPM, patient bootstrap, structured JSON and CSV. |
| Hardware | Historical CUDA report; scripts for CUDA and optional imports | Preserve diagnostics, distinguish historical report from this session. Current Python reports torch 2.14.0+cpu and no CUDA. |
| Dependencies | Only PyYAML >=6 core; dev pytest >=8, Ruff >=0.5, pre-commit >=3.7; optional MLflow >=2.12 | Keep lightweight import behavior; declare NumPy/SciPy/PyTorch/SimpleITK as CT extras. Existing tests already require undeclared NumPy. |
| Quality | Three test modules plus network-blocking fixture; Windows/Linux CI; no real-data tests | Retain and extend with synthetic geometry, leakage, metrics and end-to-end checks. |
| Documentation | Spanish README, research plan, twenty local CXR backlog issues, historical environment report | Translate preserved research material to English and label historical proposals; document current CT behavior separately. |

No dataset, image reader, optimizer, loss, callback, trained checkpoint, training
recovery implementation, GPU training result, or performance claim existed in the
source. Existing training/evaluation/model YAML files describe intended CXR work,
not implemented functionality. Import diagnostics do not establish model viability.

The CT baseline uses centers and diameters as spherical proxy supervision, not
expert segmentation. A small 3D model preserves volumetric context and avoids an
unverified dependency or pretrained-weight migration. A separate false-positive
classifier and MONAI architectures remain future experiments. Real CT quality,
memory use, convergence and clinical applicability require independent validation.
