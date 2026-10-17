# PulmoAI research and implementation plan

Updated 2026-10-17. This plan separates the operational CT MVP from the earlier
CXR research proposals. It reports no real-data performance or clinical benefit.
See [AUDIT](AUDIT.md) for the initial state and [the delivery report](IMPLEMENTATION_REPORT.md)
for verification evidence. Existing CXR models were proposals, not implementations.

## Current CT research question

Can a compact 3D candidate detector identify pulmonary nodule centers with useful
sensitivity at a controlled number of false positives per CT study under a limited
compute budget? Establish this through patient-disjoint retrospective experiments
before considering clinical validation. Detecting a nodule does not diagnose
cancer. No malignancy classifier, cancer probability, treatment recommendation,
or automatic discharge decision belongs to this MVP.

The current baseline is documented in [TRAINING](TRAINING.md). Local LUNA16 MHD
preparation, patient splitting, proxy-sphere training, checkpoint recovery,
detection evaluation, inference and export are implemented and tested synthetically.
The next required work is authorized real-data acquisition, patient mapping,
annotation/geometry QA, memory profiling and actual training/evaluation. The
[LUNA16 procedure](https://luna16.grand-challenge.org/Procedure/) and
[evaluator](https://luna16.grand-challenge.org/Evaluation/) define the official
benchmark; the internal holdout evaluator is a development tool.

| Stage | Evidence required before proceeding |
| --- | --- |
| Data readiness | Source/version/license register, correct HU, patient mapping, included/excluded annotations, verified spatial transforms and no known cross-split duplicates. |
| Pilot training | Real patch inspection, positive/background balance, convergence, actual CPU/GPU memory/time and failure log; synthetic tests alone do not qualify. |
| Model selection | Architecture, patch size, sampling, score threshold, candidate cap and NMS settings selected only with validation. |
| Locked test | Frozen weights, protocol, cohort, matching rules and thresholds; sensitivity/FP-per-study/FROC with uncertainty and failure analysis. |
| Official benchmark | Organizers' reference/exclusion tables and evaluator, official fold procedure, full provenance and no patient/pretraining leakage. |
| Extension | Independent CT cohort after annotation/license audit; optional false-positive classifier or MONAI comparator only if the baseline motivates it. |
| Clinical study | Separate intended-use protocol, professional review, ethics/legal approvals and independent prospective validation. Not executed here. |

## Preserved CXR research direction (historical, independent)

The original proposal concerned adult frontal AP/PA chest radiographs, multilabel
findings, calibration, exploratory saliency and uncertainty. Pediatric/lateral
images were outside its proposed initial scope. The research question was how
transfer learning, resolution and calibration affect discrimination across
institutions within a nominal 4 GB GPU budget. It did not establish early
detection, reduced mortality, diagnostic accuracy or a functioning triage device.
No CXR model, harmonizer, dataset loader or demo has been implemented. The English
[local backlog](issues/README.md) preserves this work for later independent use.

### Architecture literature and original decisions

- Rajpurkar et al. (2017), [CheXNet](https://arxiv.org/abs/1711.05225), is a historical
  DenseNet/ChestX-ray14 pneumonia baseline, not evidence of general clinical superiority.
- Huang et al. (2017), [DenseNet](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html),
  motivated the proposed DenseNet-121 baseline at 224 pixels.
- Tan and Le, [EfficientNet](https://proceedings.mlr.press/v97/tan19a.html) and
  [EfficientNetV2](https://proceedings.mlr.press/v139/tan21a.html), and Liu et al.,
  [ConvNeXt](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html),
  motivated budget-matched CNN comparisons. Published efficiency does not guarantee
  a fit in a 4 GB GPU. Start at 224 before considering 384/512 pixels.
- [ViT](https://openreview.net/forum?id=YicbFdNTTy) and
  [Swin](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html)
  remain literature comparisons, not implemented training routes.
- Cohen et al., [TorchXRayVision](https://proceedings.mlr.press/v172/cohen22a.html)
  and its [model documentation](https://mlmed.org/torchxrayvision/models.html)
  require compatible channels, normalization, native resolution and output order.
  Aggregated pretraining can contaminate downstream test cohorts.
- Optional representation-learning references retained from the original plan:
  [MoCo-CXR](https://proceedings.mlr.press/v143/sowrirajan21a.html),
  [BioViL](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960001.pdf),
  [CheXzero](https://doi.org/10.1038/s41551-022-00936-9),
  [RAD-DINO](https://doi.org/10.1038/s42256-024-00965-w),
  [RAD-DINO model card](https://huggingface.co/microsoft/rad-dino),
  [ELIXR preprint](https://arxiv.org/abs/2308.01317), and
  [CXR Foundation](https://developers.google.com/health-ai-developer-foundations/cxr-foundation).
  These are not installed models or verified compatible CT weights. Frozen embeddings
  would still require a provenance, licensing and resource audit.

### Original CXR dataset candidates

The following access summaries are retained as historical planning information,
not a current acquisition instruction or legal conclusion. Verify the exact release
and terms at the linked primary source before use. All remain unintegrated.

| Dataset | Original proposed role and annotation limits | Access review needed |
| --- | --- | --- |
| [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data) | Approximately 112,120 CXR; noisy report-derived labels; proposed baseline train | Linked distribution declares CC0; verify selected source and version. |
| [CheXpert](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) | Approximately 224,316 CXR; positive, negative and uncertain report labels | Stanford registration and research use agreement. |
| [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) | Approximately 377,110 images; optional report-label expansion, distinct from DICOM parent | PhysioNet credentialing, training and DUA; retain release license. |
| [RSNA Pneumonia](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) | Approximately 30,000 challenge images; opacity boxes; NIH subset, not external validation | Kaggle challenge terms and RSNA attribution; do not inherit NIH licensing for annotations automatically. |
| [COVID-QU-Ex](https://www.kaggle.com/dsv/3122958) | Approximately 33,920 images; mixed sources, segmentation masks, shortcut/duplicate risks | Linked release declares CC BY-SA 4.0; original source rights also require review. |
| [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/) | Approximately 18,000 images; radiologist local/global labels; potential external cohort | PhysioNet credentialing, training and DUA. Not anonymously downloadable by assumption. |
| [PadChest](https://bimcv.cipf.es/bimcv-projects/padchest/) | Approximately 160,000 images; mixed views and manual/automatic labels; optional external cohort | BIMCV application and research agreement; complete archive was estimated near 1 TB and exceeds the original disk budget. |

Original dataset references: [NIH](https://arxiv.org/abs/1705.02315),
[CheXpert](https://ojs.aaai.org/index.php/AAAI/article/view/3834),
[MIMIC-CXR](https://doi.org/10.1038/s41597-019-0322-0),
[RSNA](https://doi.org/10.1148/ryai.2019180041),
[COVID-QU-Ex](https://doi.org/10.1016/j.compbiomed.2021.105002),
[VinDr-CXR](https://doi.org/10.1038/s41597-022-01498-w),
[PadChest](https://doi.org/10.1016/j.media.2020.101797).
CXR resources cannot be substituted for volumetric CT nodule supervision.

### Original proposed CXR protocol

The proposed taxonomy was Pneumonia, Effusion, Atelectasis, Cardiomegaly,
Consolidation, Edema, Pneumothorax, Nodule/Mass and No Finding. The preserved
configuration actually has eight outputs and omits Nodule/Mass; this inconsistency
is historical and is not silently interpreted as a working nine-output model.
CheXpert Lung Lesion is broader than NIH Nodule/Mass and must initially be masked
for that task. No Finding is not the complement of selected findings or proof of
health. Effusion/Pleural Effusion equivalence requires documented source review.

Keep original and harmonized states, source and availability masks. Missing,
uncertain and negative labels are different. Nodule OR Mass is positive if either
is known positive and negative only if both are known negative. Compare U-Ones,
U-Zeros and U-Ignore only for explicit uncertain labels; masked BCE excludes
unknown labels. Do not change test reference labels to improve apparent results.

Original split proposal: 70/10/10/10 percent by patient for training, validation,
calibration and internal test, with stratification when feasible. It is not the
CT split implementation. Preserve official splits when compatible; audit patient,
study, exact/near-duplicate and pretraining overlap. Namespace source identifiers,
but do not assume this establishes cross-source independence. Use the official
NIH/RSNA mapping; a same-source RSNA benchmark is not independent external validation.
Reserve a fresh cohort if test was used to decide later changes.

Keep ImageNet and TorchXRayVision preprocessing distinct; preserve aspect ratio
and anatomy, review AP/PA orientation, and test 224/384/512 pixel costs. DICOM HU
or photometric handling cannot be assumed for CXR. Cache keys should include source,
resolution and transformations; assess workers, pinning, RAM and throughput on
Windows. Original proposals included gradual unfreezing, masked BCE, positive
weighting estimated on train, AMP, small batches, accumulation and safe normalization.
Nothing here asserts that an arbitrary architecture fits the target GPU.

Evaluate per-class AUROC/AUPRC with known-label masks, support and prevalence;
report null for unestimable classes. AUPRC depends on prevalence. Freeze thresholds
on calibration and report actually attained sensitivity/specificity, PPV/NPV and
FNR on test. Use patient-cluster confidence intervals, paired comparisons where
appropriate, and distinguish sampling uncertainty from across-seed variation.
Subgroup reports require support, missing-value reporting and uncertainty; do not
infer protected attributes from images or claim fairness from similar AUROC alone.

Proposed extensions remain independent future work: an ensemble only when justified
by validation and latency, global positive temperature scaling before any per-class
calibration, Grad-CAM with randomization/occlusion and localization controls, and
OOD/abstention policies developed without test. Excluded/rejected cases must remain
in the global denominators. Extreme probabilities or a plausible saliency map do
not establish trustworthy behavior. An OOD benchmark must have a defined reference
standard, not just a hospital label.

Literature retained for these risks: [Zech et al.](https://doi.org/10.1371/journal.pmed.1002683),
[DeGrave et al.](https://doi.org/10.1038/s42256-021-00338-7),
[Seyyed-Kalantari et al.](https://doi.org/10.1038/s41591-021-01595-0),
[temperature scaling](https://proceedings.mlr.press/v70/guo17a.html),
[Grad-CAM](https://openaccess.thecvf.com/content_ICCV_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html),
and [saliency limitations](https://doi.org/10.1038/s42256-022-00536-x).

## Reporting, risks and future clinical validation

Maintain model cards and dataset sheets from the start: intended use, exclusions,
cohort, reference standard, rights, acquisition/preprocessing, weights hash,
training provenance, metrics with uncertainty, subgroup support and observed errors.
Use [Model Cards](https://doi.org/10.1145/3287560.3287596),
[Datasheets](https://doi.org/10.1145/3458723),
[CLAIM 2024](https://doi.org/10.1148/ryai.240300),
[TRIPOD+AI](https://doi.org/10.1136/bmj-2023-078378) and
[STARD](https://doi.org/10.1136/bmj.h5527) as reporting references, without treating
checklists as evidence of regulatory approval or clinical usefulness.

Primary risks remain OOM, slow I/O, incompatible dependencies, late data access,
annotation noise, spatial mistakes, source shortcuts, patient/pretraining leakage,
small subgroups, test overfitting and data exposure in tracking/previews. Mitigate
with pilots, explicit compatibility contracts, access logs, immutable manifests,
validation-only selection, uncertainty reporting, local storage and clear failure
analysis. Neither a disclaimer nor MIT licensing authorizes clinical deployment.

Any future clinical study needs a specific population, users and setting,
professional consultation, adjudicated reference, blinded multicenter/temporal
validation, endpoint-based sample size, ethics/consent/privacy review, applicable
regulatory review, stopping criteria and drift/version monitoring. A silent
prospective phase must precede a separately approved interventional study. Relevant
endpoints can include sensitivity, false negatives, alarm burden, reading time,
coverage and harm. No such study, private-data collection, recruitment or deployment
is performed or authorized by this repository. General reference portals retained
from the original proposal: [FDA SaMD](https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd)
and [EU MDR](https://eur-lex.europa.eu/eli/reg/2017/745/oj); jurisdiction and device
classification have not been determined.
