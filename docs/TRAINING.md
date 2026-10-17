# Training, artifacts and reproducibility

## Baseline decision

There were no existing model implementations or usable checkpoints. CXR DenseNet
and TorchXRayVision identifiers are retained only in `configs/legacy_cxr.yaml` and
its groups. CXR projections and CT volumes have incompatible input and label
contracts; no CXR weights are loaded by the CT model.

The MVP is a small native PyTorch 3D U-Net: one encoder block, a pooled bottleneck,
trilinear upsampling, a skip connection, decoder block and one-logit voxel head.
GroupNorm supports small batches. Default width is 8 channels and the training
patch is 32 cubed voxels. It trains from scratch using weighted voxel BCE and
AdamW. Target spheres use the included centers and diameters; excluded spheres
and padded voxels contribute no loss. These are approximate occupancy targets,
not validated anatomical segmentations or malignancy labels.

A 2D model would discard through-plane context unless a separate aggregation
scheme were introduced. A 3D candidate classifier would require a separate
candidate generator. A MONAI 3D U-Net is a reasonable later comparator, but adding
it now would duplicate the chosen architecture and introduce a dependency without
an established baseline. No transfer-learning source has been verified compatible.
This choice minimizes implementation and model size; it does not assert good
sensitivity, convergence, or suitability for a particular GPU.

## Configuration and execution

`configs/experiment.yaml` is the CT configuration; `configs/smoke.yaml` supplies
small CPU fixtures. `pulmoai.config.load_config` reuses the original Hydra-style
YAML composer with a separate CT dataclass schema. Unknown keys, nonfinite values,
incorrect types, invalid geometry/sampling settings and secrets are rejected.
Machine-specific paths are explicit CLI arguments; no credentials or home-directory
configuration are loaded. Optional `configs/local.yaml` remains a legacy helper
and is not automatically merged or logged.

```powershell
python -m pulmoai validate-config
python -m pulmoai train --manifest data/processed/luna16/manifest.json --output artifacts/baseline
```

The config and cached manifest must agree on dataset/release and preprocessing.
Only training and validation patches are loaded by the optimization loop. Training
patches randomly sample the scan or a positive center, with configured positive
fraction and matched image/target/valid-mask flips. Validation uses a fixed seed
and no augmentation; it remains identical across epochs. Flips are a baseline
assumption that requires review on real acquisitions. No intensity jitter,
elastic transform, scheduler or learned lung segmentation is implemented.

`device: auto` selects CUDA when available and otherwise CPU. Explicit `cuda` fails
if unavailable. AMP and its gradient scaler apply only on CUDA. Gradient
accumulation handles the final incomplete group. Validation proxy loss selects
`best.pt`; `last.pt` is saved after each completed epoch. Early stopping counts
consecutive epochs without improvement. Detection FROC is computed by the separate
evaluator, not substituted with patch loss. Dense validation evaluation should
confirm whether loss-based selection is useful before a real study.

## Resume and tracking

```powershell
python -m pulmoai train --manifest data/processed/luna16/manifest.json --output artifacts/baseline --resume artifacts/baseline/last.pt
```

Resume is for an interrupted run in its original output directory, with identical
configuration, manifest and device type. The configured epoch budget must not
already be exhausted and early stopping must not already have fired. Recovery
restores model, optimizer, scaler, CPU/CUDA RNG, best loss, patience count, epoch
and tracker run ID. The last checkpoint also contains the best model snapshot and
a committed JSON metric history: recovery repairs best.pt and discards partially
logged later epochs. The optional MLflow mirror is append-only and may retain
repeated steps after an interrupted logging operation; JSON is authoritative. Sampling/shuffling are deterministic functions of seed, epoch
and patch index. Partial epochs are restarted from the preceding complete epoch.
Do not roll back to an older checkpoint after later epochs have been logged.
A completed run cannot be extended by changing its config under the same run ID.

Run records live under `<output>/<tracking.output_dir>/json/<experiment>/<run-id>`;
the default tracking subdirectory is `tracking`. Records include resolved config,
fingerprint, source revision, dependency/hardware details, seed, manifest and
checkpoint hashes, loss history, selection decisions, time and peak CUDA memory
when available. The source revision is `unknown` when Git has no commit. Dataset
IDs and per-scan predictions are excluded from aggregate tracking. Exception
records keep the error type only. Set `tracking.backend: mlflow` after installing
the tracking extra to mirror metrics into the existing local MLflow file store.
W&B has no CT implementation and is rejected by the CT schema.

## Export and compatibility

```powershell
python -m pulmoai export --checkpoint artifacts/baseline/best.pt --output artifacts/baseline/export/detector.pt
```

Export writes a versioned PyTorch state-dict bundle plus JSON metadata with hash,
config and training-manifest hash. It omits optimizer/scaler/RNG/run records.
Inference uses `torch.load(..., weights_only=True)`, validates task and configuration
fingerprint, constructs the known architecture and loads the state dictionary.
This is a Python/PyTorch integration artifact, not ONNX, TorchScript or a clinical
product. Loading still requires compatible local dependencies. Use only trusted
local artifacts and compare the metadata hash when transferring a bundle.

## Reproducibility limits

Seeds cover Python, NumPy, CPU/CUDA PyTorch, split assignment, patch selection and
DataLoader shuffling. Deterministic PyTorch algorithms and cuDNN settings are
explicit. See [PyTorch reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html):
bitwise reproducibility across releases/devices is not guaranteed. On CUDA,
set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python if a deterministic CUDA operation
requires it. Preserve installed versions with `python -m pip list --format=freeze` in private
run artifacts. A Git revision is also required for a source-complete reproduction. GPU training/AMP and the 4 GB memory budget require measurement.
Start with a few real scans, audit transforms visually and profile time, RAM and
VRAM before a full run. No dataset or pretrained weights are downloaded by these
commands. The original CUDA diagnostic remains `python scripts/smoke_cuda.py`.
