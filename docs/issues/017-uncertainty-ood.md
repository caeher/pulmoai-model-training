# Uncertainty, OOD detection and abstention

**Historical milestone:** M5

**Status:** Pending.

## Description

A high score does not prove in-domain input or clinical safety. No OOD estimator is implemented.

## Tasks

- Compare ensemble disagreement and training-embedding distances.
- Build a defined development OOD benchmark separate from test.
- Choose abstention thresholds in validation/calibration.
- Report risk/coverage, subgroup coverage and rejected-case errors.
- Preserve all rejections in global denominators.

## Acceptance criteria

Unknown inputs must not be converted to No Finding. Do not claim uncertainty coverage from entropy alone.

## Dependencies

012, 014. See the [index](README.md) for linked items.
