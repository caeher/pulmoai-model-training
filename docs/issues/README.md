# CXR planning issues

These twenty planning items are published as GitHub issues and preserved here
in English. CXR proposals remain independent of the operational CT MVP. No data
acquisition or clinical use is authorized merely by a checklist. See
[the plan](../PLAN.md).

| ID | Title | Status |
| --- | --- | --- |
| [#1](https://github.com/caeher/pulmoai-model-training/issues/1) | [Repository scaffolding and quality](001-repo-scaffolding.md) | Historical infrastructure implemented; extended by the CT MVP. |
| [#2](https://github.com/caeher/pulmoai-model-training/issues/2) | [Windows and PyTorch CUDA environment](002-windows-cuda-environment.md) | Historical diagnostic recorded; CT CUDA training remains unverified. |
| [#3](https://github.com/caeher/pulmoai-model-training/issues/3) | [Configuration and experiment tracking](003-tracking-configs.md) | Original infrastructure implemented; separate CT schema and pipeline integrated. |
| [#4](https://github.com/caeher/pulmoai-model-training/issues/4) | [Data access, licenses and provenance](004-data-access-licenses.md) | CXR work pending; CT source guidance documented, local user access still required. |
| [#5](https://github.com/caeher/pulmoai-model-training/issues/5) | [Explicit dataset acquisition adapters](005-dataset-acquisition.md) | Pending; acquisition placeholder always exits without downloading. |
| [#6](https://github.com/caeher/pulmoai-model-training/issues/6) | [CXR taxonomy, availability masks and uncertainty](006-label-harmonization.md) | CXR work pending; CT uses a separate nodule center/diameter contract. |
| [#7](https://github.com/caeher/pulmoai-model-training/issues/7) | [Patient splits and leakage audit](007-patient-splits.md) | CXR work pending; CT patient split and exact cache duplicate checks implemented. |
| [#8](https://github.com/caeher/pulmoai-model-training/issues/8) | [CXR preprocessing, augmentation and caches](008-pipeline-cache-loaders.md) | CXR work pending; LUNA16 CT preprocessing implemented separately. |
| [#9](https://github.com/caeher/pulmoai-model-training/issues/9) | [DenseNet-121 ImageNet CXR baseline](009-baseline-densenet.md) | Pending; no existing DenseNet implementation or weights were removed. |
| [#10](https://github.com/caeher/pulmoai-model-training/issues/10) | [TorchXRayVision fine-tuning](010-torchxrayvision-finetune.md) | Pending. |
| [#11](https://github.com/caeher/pulmoai-model-training/issues/11) | [ConvNeXt, EfficientNet and optional embeddings](011-model-comparators.md) | Pending. |
| [#12](https://github.com/caeher/pulmoai-model-training/issues/12) | [CXR evaluation and statistical uncertainty](012-cxr-evaluation-metrics.md) | CXR work pending; CT internal evaluator implemented and synthetically tested. |
| [#13](https://github.com/caeher/pulmoai-model-training/issues/13) | [Independent external validation and separate RSNA benchmark](013-external-validation.md) | Pending for both CXR and external CT. LUNA16 is derived from LIDC-IDRI. |
| [#14](https://github.com/caeher/pulmoai-model-training/issues/14) | [Calibration, thresholds and optional ensembles](014-calibration-ensemble.md) | Pending. |
| [#15](https://github.com/caeher/pulmoai-model-training/issues/15) | [Grad-CAM and explanation failure controls](015-gradcam-explainability.md) | Pending; CT slice previews are coordinate inspection aids, not Grad-CAM. |
| [#16](https://github.com/caeher/pulmoai-model-training/issues/16) | [Subgroup performance and underdiagnosis](016-subgroup-fairness.md) | Pending. |
| [#17](https://github.com/caeher/pulmoai-model-training/issues/17) | [Uncertainty, OOD detection and abstention](017-uncertainty-ood.md) | Pending. |
| [#18](https://github.com/caeher/pulmoai-model-training/issues/18) | [Research inference CLI and optional local Gradio demo](018-cli-demo-gradio.md) | CT CLI, JSON and PNG inspection implemented; CXR CLI and Gradio remain pending. |
| [#19](https://github.com/caeher/pulmoai-model-training/issues/19) | [Model cards, dataset sheets and reporting matrix](019-model-card-datasheets.md) | Guides and limitations documented; real trained-model cards await real experiments. |
| [#20](https://github.com/caeher/pulmoai-model-training/issues/20) | [Future clinical validation protocol](020-clinical-validation-plan.md) | Not performed; requires an independent approved study. |
