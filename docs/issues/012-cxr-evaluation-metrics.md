# CXR evaluation and statistical uncertainty

**Historical milestone:** M4

**Status:** CXR work pending; CT internal evaluator implemented and synthetically tested.

## Description

This is the preserved multilabel evaluation proposal. The CT detector has separate FROC metrics.

## Tasks

- Implement masked per-class AUROC/AUPRC and prevalence baselines.
- Freeze calibration thresholds then report attained sensitivity/specificity, PPV/NPV and FNR.
- Add ECE/Brier/reliability plots with support.
- Bootstrap by patient and separate seed variation.
- Test perfect, constant, one-class and missing-label cases.

## Acceptance criteria

Unestimable classes return a reason instead of fabricated scores. Keep unit and cohort denominators explicit.

## Dependencies

006, 007 for core; 009-011 for model results. See the [index](README.md) for linked items.
