# Research inference CLI and optional local Gradio demo

**Historical milestone:** M6

**Status:** CT CLI, JSON and PNG inspection implemented; CXR CLI and Gradio remain pending.

## Description

The new CT CLI works independently of the historical CXR demo proposal.

## Tasks

- For future CXR inference validate weights hash, class order, preprocessing/calibration and OOD policy.
- Support explicit CPU/CUDA selection and readable failures.
- Keep a Gradio demo local with no public links or persisted uploads.
- Use synthetic examples and measure latency/artifact size.

## Acceptance criteria

The same artifact/input should produce matching CLI/demo results, with no final diagnosis or automatic clearance.

## Dependencies

013-017. See the [index](README.md) for linked items.
