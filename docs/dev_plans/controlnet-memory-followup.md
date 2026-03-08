# ControlNet Memory Follow-Up Plan

This plan turns the still-open findings in `docs/debug_notes/controlnet-memory-analysis.md` into a measured next sequence after the completed March 7, 2026 remediation work.

Execution status: phase 1 rejected on March 8, 2026; phase 2 remains open.

## Goal

Reduce the remaining control-path peak memory pressure without changing control-image semantics or fixed-seed output quality.

Current measured reference state from phase 3:

- `/usr/bin/time -l` maximum resident set size: `42,656,612,352` bytes
- `/usr/bin/time -l` peak memory footprint: `59,328,863,512` bytes
- fixed-seed output SHA-256: `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70`

## Scope

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- targeted lifecycle helpers if they reduce duplicate load and unload logic
- optional control and inpaint VAE encode work under `Sources/ZImage/Model/VAE/*`
- targeted unit coverage under `Tests/ZImageTests/`
- supporting docs under `docs/`

## Non-Goals

- changing ControlNet conditioning math
- changing scheduler behavior
- broad refactors across the base text-to-image pipeline unless needed to keep loader conventions aligned

## Validation Protocol

Every phase must be evaluated before it is wrapped.

Repo verification:

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath .build/xcode
```

High-resolution memory probe:

```bash
/usr/bin/time -l ./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --guidance 0 \
  --seed 1234 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-followup/<phase>_memory.png
```

Fixed-seed quality probe:

```bash
/usr/bin/time -l ./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "a stone archway covered in moss, cinematic lighting" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 512 \
  --height 512 \
  --steps 4 \
  --guidance 0 \
  --seed 1234 \
  --no-progress \
  --output /tmp/zimage-control-followup/<phase>_quality.png
```

Required metrics per phase:

1. `prompt-embeddings.after-clear-cache`
2. `control-context.after-baseline-reduction`
3. `control-context.after-clear-cache`
4. `denoising.before-start`
5. `decode.after-eval`
6. `/usr/bin/time -l` maximum resident set size
7. `/usr/bin/time -l` peak memory footprint
8. output SHA-256 and image drift versus the phase 3 baseline and the immediately previous phase

Preferred image metrics:

- mean absolute pixel error
- max absolute pixel delta
- PSNR

Acceptance bar:

- no broken image, NaN-like failure, or shape mismatch
- no unexpected regression in the high-resolution memory probe
- phase 1 should reduce or eliminate avoidable loader churn
- later phases should only proceed if the previous phase leaves a meaningful memory target unmet
- output semantics should stay bit-identical or differ only by a trivial and explainable amount

## Phase 1: Defer Transformer And ControlNet Loading

Status: rejected on March 8, 2026

Objective:

- stop loading the transformer and optional ControlNet before prompt encoding and control-context construction

Files:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- keep tokenizer and text encoder lifecycle unchanged unless the refactor proves a tighter policy is simpler
- resolve model paths and weight metadata early, but instantiate transformer and ControlNet modules only when denoising is about to start
- remove the current unload and reload churn if the modules can simply remain absent until needed
- keep externally visible CLI and pipeline behavior unchanged

Acceptance criteria:

- the control path no longer performs an unnecessary transformer and ControlNet load before prompt embedding work
- the measured pre-control baseline and/or denoising start boundary improves versus the phase 3 reference
- fixed-seed output remains effectively unchanged

Execution result:

- Attempt A deferred both transformer and ControlNet until denoising.
- Attempt B kept the original transformer lifecycle and deferred only ControlNet.
- Both attempts changed the fixed-seed output from phase 3 SHA-256 `2afd1fa9ba4398ad2b8b53510f44d602d5d7d5cc2631cee99d35c6d0752f8f70` to `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`.
- The resulting drift versus the saved phase 3 image was not trivial:
  - MAE: `20.8288`
  - max absolute pixel delta: `197`
  - PSNR: `18.2115 dB`
- Attempt B still improved prompt-stage residency versus phase 3:
  - `prompt-embeddings.after-clear-cache`: `33.81 GiB -> 29.48 GiB`
  - `maximum resident set size`: `42,656,612,352 -> 38,369,869,824`
  - `peak memory footprint`: `59,328,863,512 -> 59,325,996,240`
- The quality regression failed the acceptance bar, so the deferred-loading implementation was not landed.

## Phase 2: Consolidate Lifecycle Boundaries And Telemetry

Status: proposed

Objective:

- make remaining memory jumps attributable and keep loader policy easy to reason about

Files:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- targeted shared helpers if needed
- `docs/`

Implementation notes:

- add or tighten telemetry around module load, unload, and first-use boundaries only where phase 1 still leaves attribution gaps
- consolidate duplicated loader sequencing into a narrow helper if phase 1 introduces parallel load paths
- do not widen the refactor into generic loader abstractions unless duplication is real and current

Acceptance criteria:

- logs clearly isolate the residual lifecycle transitions that still matter
- loader sequencing is simpler than the pre-phase code, not more abstract
- memory and quality stay at least as good as phase 1

## Phase 3: Optional Tiled Control And Inpaint VAE Encode

Status: proposed, gated on phase 1 and 2 results

Objective:

- reduce the remaining monolithic control-context build spike if lifecycle cleanup is not sufficient

Files:

- `Sources/ZImage/Model/VAE/*`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Tests/ZImageTests/`
- `docs/`

Implementation notes:

- keep the latent-space math aligned with the existing control and inpaint encode path
- prefer the smallest practical tiling or striping strategy that can be validated with fixed-seed comparisons
- treat this as a structural phase only if the deferred-loading work still leaves the high-resolution probe above the practical target

Acceptance criteria:

- the control-context build path shows a material peak-memory reduction beyond phase 1 and 2
- quality drift is zero or small enough to quantify and justify
- the implementation does not duplicate a second independent encode stack unnecessarily

## Execution Log

- Phase 1: attempted on March 8, 2026 and rejected.
  - Attempt A, defer transformer and ControlNet:
    - `prompt-embeddings.after-clear-cache`: resident `6.73 GiB`, active `2.50 MiB`, cache `0 B`
    - `control-context.after-baseline-reduction`: resident `2.61 GiB`, active `67.87 MiB`, cache `0 B`
    - `control-context.after-clear-cache`: resident `318.80 MiB`, active `71.36 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.50 GiB`, active `29.19 GiB`, cache `65.30 MiB`
    - `decode.after-eval`: resident `417.31 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `32.67 GiB`
    - `/usr/bin/time -l` maximum resident set size: `38,384,238,592` bytes
    - `/usr/bin/time -l` peak memory footprint: `59,325,389,960` bytes
    - output SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`
  - Attempt B, defer ControlNet only:
    - `prompt-embeddings.after-clear-cache`: resident `29.48 GiB`, active `22.93 GiB`, cache `0 B`
    - `control-context.after-baseline-reduction`: resident `269.34 MiB`, active `67.87 MiB`, cache `0 B`
    - `control-context.after-clear-cache`: resident `321.64 MiB`, active `71.36 MiB`, cache `0 B`
    - `denoising.before-start`: resident `29.48 GiB`, active `29.19 GiB`, cache `65.30 MiB`
    - `decode.after-eval`: resident `418.94 MiB`, active `127.48 MiB`, cache `39.00 GiB`, MLX peak `32.67 GiB`
    - `/usr/bin/time -l` maximum resident set size: `38,369,869,824` bytes
    - `/usr/bin/time -l` peak memory footprint: `59,325,996,240` bytes
    - output SHA-256: `b5f1585314323c7e12f3a4871644346ac9d5f2470cfbf74c11935e9f2c558b98`
  - Assessment:
    - the prompt-stage baseline does improve when ControlNet is deferred
    - the output drift is identical across both variants, which localizes the regression to ControlNet deferral rather than transformer deferral
    - peak memory footprint stays effectively flat, so there is no reason to accept the quality regression
