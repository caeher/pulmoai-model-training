# Independent external validation and separate RSNA benchmark

**Historical milestone:** M4

**Status:** Pending for both CXR and external CT. LUNA16 is derived from LIDC-IDRI.

## Description

CXR external cohorts require access, semantic compatibility and independence from all pretraining.

## Tasks

- Freeze weights, transforms, calibration, thresholds, ensemble and rejection policy.
- Audit test/pretraining overlap.
- Evaluate compatible VinDr/PadChest labels only with authorized access.
- Use NIH mapping for RSNA and label it same-source.
- Report site-specific support, prevalence, uncertainty and limitations.

## Acceptance criteria

No model or threshold selection on test. Obtain a fresh cohort after test-informed changes.

## Dependencies

004, 007, 012; freeze 014-017 before test. See the [index](README.md) for linked items.
