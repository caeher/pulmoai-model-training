# CXR preprocessing, augmentation and caches

**Historical milestone:** M1

**Status:** CXR work pending; LUNA16 CT preprocessing implemented separately.

## Description

The CXR image pipeline was proposed but never implemented. CT preprocessing is a distinct module.

## Tasks

- Add authorized JPG/PNG/DICOM readers with photometric/metadata checks.
- Filter population and AP/PA views.
- Preserve aspect ratio and anatomy with model-specific normalization.
- Measure 224/384/512 pixel costs, workers, memory and I/O.
- Version cache keys and train-only conservative augmentation.

## Acceptance criteria

Synthetic shape/range/orientation checks and permitted real-image visual QA must precede claims of compatibility.

## Dependencies

002, 005, 006, 007. See the [index](README.md) for linked items.
