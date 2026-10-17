# Subgroup performance and underdiagnosis

**Historical milestone:** M5

**Status:** Pending.

## Description

Descriptive subgroup analysis requires adequate support and uncertainty.

## Tasks

- Document available sex/age/view metadata and missing values.
- Prespecify age groups and minimum support.
- Use frozen global thresholds for sensitivity/FNR, AUROC/AUPRC and calibration.
- Bootstrap by patient.
- Do not infer protected attributes from images.

## Acceptance criteria

Report small/unstable groups without rankings or claims of fairness based on similar AUROC alone.

## Dependencies

007, 012, 014. See the [index](README.md) for linked items.
