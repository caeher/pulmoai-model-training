# Inference and evaluation contracts

## Reusable inference

```python
from pulmoai.inference.detector import Detector

detector = Detector.from_checkpoint("artifacts/baseline/export/detector.pt", device="cpu")
result = detector.predict("data/raw/luna16/example.mhd", "artifacts/prediction.json")
```

```powershell
python -m pulmoai infer --checkpoint artifacts/baseline/export/detector.pt --image data/raw/luna16/example.mhd --output artifacts/prediction.json --preview artifacts/prediction.png
```

This explicitly reads a compatible HU CT, resamples using saved settings, and runs
one patch at a time on the selected device. Overlapping logits are averaged on
CPU, then sigmoid is applied. Local maxima above `score_threshold` become candidate
centers; plateaus collapse to one point. Physical-distance NMS suppresses nearby
centers, with at most `max_candidates` outputs. Lower thresholds and a different
cap alter sensitivity and false positives and must be fixed before test evaluation.
The candidate cap can truncate valid detections. No lung ROI detector, diameter
regressor, calibration, uncertainty estimator or false-positive classifier exists.

The optional local PNG contact sheet displays up to six scored native voxel
planes, with an in-plane spacing correction and a center marker. It is an
inspection aid, not a radiological viewer. Its axes follow the stored voxel
orientation, including oblique acquisitions. It does not implement standardized
radiological display orientation or measure nodule extent.

## JSON version 1

A synthetic illustrative excerpt (not a measured model result):

```json
{
  "schema_version": 1,
  "task": "ct_nodule_candidates",
  "modality": "CT",
  "coordinate_system": "LPS",
  "coordinate_units": "mm",
  "voxel_order": "xyz",
  "voxel_index_base": 0,
  "score_meaning": "uncalibrated nodule candidate score, not malignancy probability",
  "research_only": true,
  "candidates": [
    {"center_xyz_mm": [-30.0, 20.0, -100.0], "center_voxel_xyz": [12.5, 22.0, 9.0], "score": 0.7}
  ]
}
```

The actual document also includes package version, model SHA-256, configuration
fingerprint, original and processed geometry (ZYX shape, XYZ origin and spacing,
row-major direction), preprocessing and candidate policy. `center_voxel_xyz` is a
continuous, zero-based index in the original input grid. `center_xyz_mm` uses the
original physical frame. The model does not predict boxes, lesion diameters or
malignancy. An empty candidates array means no retained candidates under the
chosen policy; it does not rule out disease. No source filename or patient ID is
included in individual inference JSON. Physical metadata and image previews may
still be sensitive; keep outputs private.

## Independent evaluation

```powershell
python -m pulmoai evaluate --checkpoint artifacts/baseline/best.pt --manifest data/processed/luna16/manifest.json --split test --output artifacts/baseline/test
```

The evaluator accepts validation or test only, requires the frozen training
manifest hash and uses the selected split only for predictions/metrics. Negative
scans remain in the FP-per-scan denominator. Results contain scan/nodule counts,
TP/FP/FN, sensitivity, FP/scan, detection precision/recall/F1, the complete internal
FROC, sensitivity at configured FP budgets, mean CPM, cohort metadata and hashes.
Undefined metrics return JSON null with a reason when no positive nodules exist.
A patient-cluster bootstrap produces a percentile 95% CPM interval when at least
two patients and two valid replicates are available. Replicates without positive
nodules are omitted and their count is reported. This interval can be unstable in
small cohorts; the default 200 replicates is a development setting.

Predictions match nearest unmatched included nodule centers within diameter/2,
ordered by descending score. Included matches take precedence over excluded
regions. Additional detections of an already matched nodule count as FP under this
internal protocol. Detections hitting excluded regions are ignored. All equal
scores enter the FROC together. CPM uses the greatest observed sensitivity at or
below each FP budget, without linear interpolation. The point above all scores
has zero detections. Scores below the extraction threshold are unavailable.

`metrics.json` is aggregate. Private `predictions.json` preserves per-series
candidates and native coordinates; `predictions.csv` writes the LUNA-style columns
`seriesuid,coordX,coordY,coordZ,probability`. The historical CSV column name
`probability` means candidate score, never probability of cancer. Scans without
candidates do not produce CSV rows but remain in the manifest denominator.

This evaluator is deliberately labeled `pulmoai_internal_froc_v1` and sets
`official_luna16_score: false`. Official LUNA16 evaluation differs in duplicate
handling/interpolation, scan bootstrap and ten-fold procedure; use the
[organizers' evaluator](https://luna16.grand-challenge.org/Evaluation/) and complete
reference/exclusion files for benchmark claims. This internal holdout is not
independent external validation against LIDC-IDRI, because LUNA16 derives from it.
No thresholds, NMS settings, cap, architecture or exclusions should be selected
using test outcomes. External-cohort evaluation needs a separate prospective
compatibility and independence audit and is not implemented by this command.
