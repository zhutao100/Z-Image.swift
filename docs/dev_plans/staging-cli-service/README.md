# Staging CLI Service Plan

Status: completed on March 13, 2026

## Decision

Chosen option: warm worker daemon

Why:

- the current pipelines still unload heavy modules after each run, so a thin daemon alone would not actually keep the service warm
- the repo already has the right library entry points, so the missing piece is shared CLI parsing plus serving-oriented residency controls
- a local Unix socket keeps the transport small and avoids pulling HTTP concerns into a CLI-first package

## Delivery phases

1. shared CLI parsing/building plus a local `ZImageServe` daemon/client for ad hoc generation requests
2. serving residency policy and warm worker reuse with idle eviction and memory-pressure fallback
3. JSON batch, markdown ingestion, and operational commands

## Completion note

The chosen warm worker daemon shipped as:

- shared CLI parsing and request building in `Sources/ZImageCLICommon/`
- `ZImageServe` plus the local socket protocol in `Sources/ZImageServe/` and `Sources/ZImageServeCore/`
- serving residency controls in `Sources/ZImage/Pipeline/RuntimeOptions.swift`
- JSON batch manifests, markdown fenced-command ingestion, and `status` / `cancel` / `shutdown`

Recorded verification:

- fast coverage for parser, batch JSON, and markdown ingestion
- staged E2E coverage for daemon lifecycle, batch, markdown, status, and shutdown
- repeated-request warm-serving probe against cached `mzbac/z-image-turbo-8bit`
- manual active-job cancel probe against the same cached profile

## Verification bar

- keep `ZImageCLI` behavior-compatible
- add fast coverage for parser, JSON, and markdown ingestion
- add staged end-to-end coverage for daemon lifecycle and client submission
- run a repeated-request warm-serving check against a locally cached model profile so reuse is measured rather than inferred

## Source docs

- [requirements.md](requirements.md)
- [design_options.md](design_options.md)
