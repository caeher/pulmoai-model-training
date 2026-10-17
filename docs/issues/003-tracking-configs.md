# Configuration and experiment tracking

**Historical milestone:** M0

**Status:** Original infrastructure implemented; separate CT schema and pipeline integrated.

## Description

The original YAML composer, fingerprints and local tracker are reused by PulmoAI.

## Tasks

- Keep public config separate from explicit private settings.
- Record source/environment, seeds, manifest/checkpoint hashes, timing and memory.
- Test failure and resume behavior.
- Keep remote tracking disabled.

## Acceptance criteria

Resolved settings reconstruct preprocessing and outputs without logging identifiers or credentials.

## Dependencies

001, 002. See the [index](README.md) for linked items.
