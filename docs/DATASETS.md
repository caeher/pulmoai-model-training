# CT datasets and preparation

## Sources, rights and compatibility

| Resource | Relevant structure | Integration status |
| --- | --- | --- |
| [LUNA16](https://luna16.grand-challenge.org/Data/) | Chest CT volumes in MHD/RAW, included/excluded center/diameter CSV annotations, ten published subsets | Local MHD preparation implemented; no downloader and no automatic challenge-fold evaluation. |
| [LIDC-IDRI at TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) | CT DICOM, multi-reader XML annotations and acquisition metadata; the collection also contains radiographs | Source evaluated; direct DICOM/XML conversion, reader consensus and series-to-patient mapping extraction are not implemented. |

TCIA's collection page lists CC BY 3.0 and requires dataset citation and its data
usage policy. Retain the exact terms for the version and distribution acquired.
The LUNA16 portal provides derived data: review its access instructions and terms
as well as the parent collection attribution before obtaining files. No license is
accepted automatically. Do not assume the code's MIT license covers images.
Record the collection citation (Armato et al.), TCIA citation, release, source URL,
access date, attribution, license, checksums and local preprocessing version.
No additional public cohort is enabled without a compatible license and annotation audit.

LUNA16 contains 888 CT scans derived from LIDC-IDRI and uses nodules defined by its
reference standard. Read the [official procedure](https://luna16.grand-challenge.org/Procedure/)
and [evaluation rules](https://luna16.grand-challenge.org/Evaluation/).
Its ten-fold challenge protocol differs from this MVP's patient-based holdout.
LIDC reader suspicion ratings are not confirmed cancer labels. Never infer
benign/malignant labels from the presence of a nodule or a detector score.

## Local input contract

Acquire only the data you are authorized to use. Do not run a massive download.
Place selected CT series under `data/raw/luna16/subset*/<seriesuid>.mhd` with each
paired RAW file at the location referenced by its header. Each MHD filename stem
must be a unique series UID. Data must already be scalar CT in Hounsfield units
and geometry must use millimeters in the LPS coordinate frame. The reader does
not convert arbitrary MHD intensity units to HU and cannot establish modality
from a filename. Verify provenance. PNG/JPEG CXR images are unsupported.

Included and optional excluded annotation CSV columns:

```csv
seriesuid,coordX,coordY,coordZ,diameter_mm
example-series,-30.5,20.0,-100.0,6.0
```

Coordinates are physical XYZ, never array ZYX. The positive labels are nodule
centers and diameters. Spheres are proxy targets, not expert segmentation masks.
Excluded regions are ignored during loss/evaluation unless an included nodule is
matched first. An omitted exclusion file changes the reference standard and is
recorded in the manifest and metrics.

Provide an explicit private patient mapping covering exactly the selected series:

```csv
seriesuid,patient_id
example-series,pseudonymous-patient-1
another-series,pseudonymous-patient-1
```

Build it from verified LIDC metadata. The tool never substitutes series UID for
patient identity. Multiple CT acquisitions from one patient must remain together.
When using a subset, filter both annotation tables and the mapping to that exact
inventory; unrecognized annotation series are rejected rather than silently lost.
At least three patients are required. For a meaningful study many more are needed.

```powershell
python -m pulmoai prepare --scans data/raw/luna16 --annotations data/raw/annotations.csv --excluded data/raw/annotations_excluded.csv --patients data/raw/patients.csv --output data/processed/luna16
python -m pulmoai validate-data --manifest data/processed/luna16/manifest.json
```

Set the dataset release in YAML first. Output must be empty or new; a failed
preparation leaves partial local files and no usable final manifest. Inspect the
failure and retry in a new output directory. Inputs are never overwritten.

## Spatial and split guarantees

Physical coordinates follow `origin + direction @ (index_xyz * spacing_xyz)`.
SimpleITK returns NumPy arrays in ZYX order. Origin and direction remain fixed
during linear resampling; output size covers the original voxel-center extent.
Outside samples use -1000 HU. HU clipping defaults to [-1000,400] then maps to
[0,1]. Original and processed geometry are both retained; predictions invert the
original transform to recover continuous native zero-based voxel coordinates.
See [SimpleITK spatial concepts](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html).

Patients are sorted, shuffled with the recorded split seed, and assigned to
train/validation/test using the configured fractions with at least one patient in
each. This is not stratified and does not recreate the official LUNA16 subsets.
Retain the immutable generated manifest. Counts and prevalence should be audited
before fixing the research protocol, without choosing splits by model performance.

Preparation checks dimensions, finite values, spacing, direction, annotation
bounds, mapping completeness and split isolation. Validation checks cache SHA-256,
geometry consistency and duplicate decoded voxel values across splits, even when
container metadata differs. This does not
detect all re-encoded/near duplicates or incorrect source patient mappings.
The manifest records release, annotation/mapping hashes, cached-volume hashes,
seed and preprocessing signature. Private original acquisition metadata remain
with the source data; the cache retains geometry, not complete DICOM headers.

The internal NPZ format contains `hu` (float32 ZYX), `units="HU"`, `origin_xyz`,
`spacing_xyz` and flattened row-major `direction`. It contains no Python pickles.
It is used for local caches and synthetic tests, not as a generic interchange
format for unverified clinical data. No NIfTI/DICOM adapter is claimed.

Budget storage for both the downloaded volumes and float32 resampled caches.
Preparation processes one volume at a time, but resampling and dense inference
still require RAM proportional to a complete scan. Compressed caches are decoded
per patch; large-cohort I/O optimization remains future work.
