# Grad-CAM and explanation failure controls

**Historical milestone:** M5

**Status:** Pending; CT slice previews are coordinate inspection aids, not Grad-CAM.

## Description

Saliency is exploratory, not a clinical segmentation or causal explanation.

## Tasks

- Specify layer/output and spatial transform restoration.
- Check that visualization does not alter predictions.
- Add randomized-weight and occlusion controls.
- Evaluate with compatible boxes and localization limits.
- Include failures and permitted examples only.

## Acceptance criteria

Plausible-looking examples alone cannot establish reliability; use validation for development.

## Dependencies

010, 012. See the [index](README.md) for linked items.
