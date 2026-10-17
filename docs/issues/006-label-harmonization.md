# CXR taxonomy, availability masks and uncertainty

**Historical milestone:** M1

**Status:** CXR work pending; CT uses a separate nodule center/diameter contract.

## Description

Historical proposed taxonomy has nine findings; the retained example config has eight. No harmonizer exists.

## Tasks

- Version label definitions and source mappings.
- Preserve positive, negative, uncertain and missing states.
- Implement U-Ones/U-Zeros/U-Ignore for explicit uncertainty only.
- Define Nodule OR Mass logic and initially mask broader CheXpert Lung Lesion.
- Document No Finding contradictions and masked BCE support.

## Acceptance criteria

Unknown labels must not act as negatives. Test truth tables and samples without eligible labels.

## Dependencies

004. See the [index](README.md) for linked items.
