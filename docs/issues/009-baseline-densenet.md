# DenseNet-121 ImageNet CXR baseline

**Historical milestone:** M2

**Status:** Pending; no existing DenseNet implementation or weights were removed.

## Description

This remains a future CXR baseline, not the current CT architecture.

## Tasks

- Adapt to verified output taxonomy with masked BCE.
- Start at 224 pixels with head training and gradual unfreezing.
- Handle AMP, accumulation and small-batch normalization.
- Estimate class weights from training only and compare prevalence baseline.
- Track validation selection, seed, time, memory and checkpoints.

## Acceptance criteria

Use identical verified splits/classes across comparisons and report OOM or convergence failures.

## Dependencies

003, 008, 012 core. See the [index](README.md) for linked items.
