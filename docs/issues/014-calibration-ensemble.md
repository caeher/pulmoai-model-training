# Calibration, thresholds and optional ensembles

**Historical milestone:** M5

**Status:** Pending.

## Description

Calibration does not make a detector score a malignancy probability. No calibrator is implemented.

## Tasks

- Compare a single CXR model and an optional validation-selected ensemble.
- Run members sequentially and measure latency.
- Fit positive global temperature before considering per-class scaling.
- Select and freeze threshold goals only in calibration.
- Export weights hash, output order and policy.

## Acceptance criteria

Report actually reached external specificity and calibration, including failures or no improvement.

## Dependencies

009, 010, 011, 012. See the [index](README.md) for linked items.
