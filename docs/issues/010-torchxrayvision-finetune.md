# TorchXRayVision fine-tuning

**Historical milestone:** M3

**Status:** Pending.

## Description

Evaluate radiographic transfer only within an independent CXR task. CT never loads these weights.

## Tasks

- Audit checkpoint hash/license and pretraining cohorts.
- Match channels, native normalization, resolution and output order.
- Compare frozen head and gradual fine-tuning.
- Evaluate 384/512 only after measuring native-resolution costs.
- Select uncertainty policies in validation.

## Acceptance criteria

Label contaminated/unknown-provenance results exploratory. Do not promise a full fine-tune fits 4 GB.

## Dependencies

009. See the [index](README.md) for linked items.
