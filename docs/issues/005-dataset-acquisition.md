# Explicit dataset acquisition adapters

**Historical milestone:** M1

**Status:** Pending; acquisition placeholder always exits without downloading.

## Description

No dataset downloader is implemented. Manual authorized acquisition remains the current route.

## Tasks

- Design resource-specific adapters only after access review.
- Add dry runs, selected downloads, size estimates, checksums and recovery.
- Keep credentials explicit and private.
- Do not bypass agreements/CAPTCHA or download all datasets by default.
- Budget disk and defer oversized archives.

## Acceptance criteria

Acquisition must be explicit, gated, resumable and testable without network access.

## Dependencies

002, 004. See the [index](README.md) for linked items.
