# Preserved CXR backlog

These twenty local planning items were not published as remote issues. They are
preserved in English with updated status and filenames. CXR proposals remain
independent of the operational CT MVP. No data acquisition or clinical use is
authorized merely by a checklist. See [the plan](../PLAN.md).

| ID | Title | Status |
| --- | --- | --- |
| [001](001-repo-scaffolding.md) | Repository scaffolding and quality | Historical infrastructure implemented; extended by the CT MVP. |
| [002](002-windows-cuda-environment.md) | Windows and PyTorch CUDA environment | Historical diagnostic recorded; CT CUDA training remains unverified. |
| [003](003-tracking-configs.md) | Configuration and experiment tracking | Original infrastructure implemented; separate CT schema and pipeline integrated. |
| [004](004-data-access-licenses.md) | Data access, licenses and provenance | CXR work pending; CT source guidance documented, local user access still required. |
| [005](005-dataset-acquisition.md) | Explicit dataset acquisition adapters | Pending; acquisition placeholder always exits without downloading. |
| [006](006-label-harmonization.md) | CXR taxonomy, availability masks and uncertainty | CXR work pending; CT uses a separate nodule center/diameter contract. |
| [007](007-patient-splits.md) | Patient splits and leakage audit | CXR work pending; CT patient split and exact cache duplicate checks implemented. |
| [008](008-pipeline-cache-loaders.md) | CXR preprocessing, augmentation and caches | CXR work pending; LUNA16 CT preprocessing implemented separately. |
| [009](009-baseline-densenet.md) | DenseNet-121 ImageNet CXR baseline | Pending; no existing DenseNet implementation or weights were removed. |
| [010](010-torchxrayvision-finetune.md) | TorchXRayVision fine-tuning | Pending. |
| [011](011-model-comparators.md) | ConvNeXt, EfficientNet and optional embeddings | Pending. |
| [012](012-cxr-evaluation-metrics.md) | CXR evaluation and statistical uncertainty | CXR work pending; CT internal evaluator implemented and synthetically tested. |
| [013](013-external-validation.md) | Independent external validation and separate RSNA benchmark | Pending for both CXR and external CT. LUNA16 is derived from LIDC-IDRI. |
| [014](014-calibration-ensemble.md) | Calibration, thresholds and optional ensembles | Pending. |
| [015](015-gradcam-explainability.md) | Grad-CAM and explanation failure controls | Pending; CT slice previews are coordinate inspection aids, not Grad-CAM. |
| [016](016-subgroup-fairness.md) | Subgroup performance and underdiagnosis | Pending. |
| [017](017-uncertainty-ood.md) | Uncertainty, OOD detection and abstention | Pending. |
| [018](018-cli-demo-gradio.md) | Research inference CLI and optional local Gradio demo | CT CLI, JSON and PNG inspection implemented; CXR CLI and Gradio remain pending. |
| [019](019-model-card-datasheets.md) | Model cards, dataset sheets and reporting matrix | Guides and limitations documented; real trained-model cards await real experiments. |
| [020](020-clinical-validation-plan.md) | Future clinical validation protocol | Not performed; requires an independent approved study. |
