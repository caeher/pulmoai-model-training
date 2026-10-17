# Patient splits and leakage audit

**Historical milestone:** M1

**Status:** CXR work pending; CT patient split and exact cache duplicate checks implemented.

## Description

CXR proposals require patient/study and pretraining isolation. CT holdout splitting now has its own implementation.

## Tasks

- Preserve compatible official CXR splits or audit a proposed 70/10/10/10 patient split.
- Namespace source IDs and audit exact/near duplicates.
- Verify NIH/RSNA mapping before benchmark use.
- Audit checkpoint pretraining overlap.
- Keep patient mappings private.

## Acceptance criteria

No known patient overlap across development/calibration/test. RSNA remains a same-source NIH benchmark.

## Dependencies

005, 006. See the [index](README.md) for linked items.
