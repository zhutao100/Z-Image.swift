# Control-Context Memory Remediation Plan

This plan translates the latest control-path memory analysis into a concrete implementation sequence for the Swift codebase.

Scope:

- `ZImageCLI control`
- `ZImageControlPipeline.buildControlContext(...)`
- `AutoencoderKL.encode(...)`
- validation against the local Diffusers reference

## Problem Statement

At `1536x2304`, control-mode runs show a steep resident-memory increase during:

- `Loading text encoder...` (`~20 GB -> ~30 GB`)
- `Building control context...` (`~30 GB -> ~100 GB`)

The final control-context tensor is small (`[1, 33, 1, 288, 192]`), so the surge occurs while building it, not while storing it.

## Validated Findings

### 1) The hot path is `vae.encode(...)` during control-context construction

Relevant code paths:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `buildControlContext(...)`
  - `encodeImageToLatents(...)`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `VAEEncoder`
  - `VAEMidBlock`
  - `VAESelfAttention`

### 2) The VAE mid-block attention is a prime memory hotspot

At `1536x2304`, the control-image latent grid is:

- `2304 / 8 = 288`
- `1536 / 8 = 192`
- latent tokens: `288 * 192 = 55,296`

The VAE encoder mid-block attention operates over that token count. This is large enough that attention workspaces can become a dominant transient-memory contributor.

### 3) The final control context is not the cause of the peak

Logged shape:

- `[1, 33, 1, 288, 192]`
- `1,824,768` elements total

Approximate storage:

- bf16: `~3.5 MiB`
- fp32: `~7 MiB`

### 4) The current code already fixed one earlier issue

The control-image encode path now uses `vae.dtype` rather than hard-coded fp32 image tensors. That means the remaining major suspects are:

- VAE attention workspace
- VAE convolution/downsample workspace
- MLX cache / lazy-evaluation overlap
- extra resident baseline from still-loaded modules

### 5) Diffusers parity should be treated carefully

The local Diffusers control pipeline also VAE-encodes the control image, so the Swift port is following the same high-level algorithm.

Do not assume the observed difference versus Diffusers-on-MPS is explained solely by “PyTorch flash attention”. The MPS comparison should instead be treated as evidence that backend/runtime behavior and implementation details differ materially enough to justify targeted memory work in the Swift path.

## Goals

### Primary goals

1. Reduce peak resident memory during `buildControlContext(...)`.
2. Keep the control path functional at `1536x2304` on high-end Apple Silicon without pathological memory spikes.
3. Preserve output parity as much as practical relative to the current Swift behavior and the Diffusers reference.

### Non-goals

- Full architectural rewrite of the VAE.
- Custom Metal kernels in the first remediation pass.
- Changing external CLI semantics unless a new opt-in safety flag is introduced.

## Patch Strategy

The work is split into diagnostic, mitigation, and structural phases. Earlier phases should land independently so they can be measured in isolation.

---

## Phase 0 — Add Reproducible Memory Instrumentation

### Objective

Make the control-path peak measurable by phase so later changes can be compared cleanly.

### Files

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImageCLI/main.swift`
- optionally a small helper under `Sources/ZImage/Util/`

### Changes

Add phase-level memory logging around:

- after text embeddings are evaluated
- immediately before `buildControlContext(...)`
- immediately before `vae.encode(...)`
- immediately after `MLX.eval(...)` of the encoded latents / control context
- immediately after `Memory.clearCache()`
- immediately before denoising begins
- immediately after final decode

Report at least:

- process resident memory (best-effort)
- `Memory.peakMemory`
- `Memory.activeMemory`
- `Memory.cacheMemory`

If the current MLX Swift surface does not expose all counters directly, log the ones that are available and leave the rest as TODOs.

### Acceptance criteria

- A single control run produces stable phase markers.
- The instrumentation is cheap enough to leave in debug builds or behind a verbose flag.

---

## Phase 1 — Add A Diagnostic Switch For VAE Mid-Block Attention

### Objective

Quickly determine how much of the peak is attributable to VAE self-attention versus the surrounding convolution path.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImageCLI/main.swift`
- `docs/CLI.md`

### Changes

Add a temporary diagnostic path that disables VAE mid-block attention for control-image encoding only.

Preferred shape:

- internal-only pipeline option first
- optional CLI flag only if needed for repeated manual validation

Possible forms:

- `--debug-disable-control-vae-attention`
- or a non-public constant / test hook while iterating

### Validation

Run control-context build twice at `1536x2304`:

- baseline
- with VAE mid-block attention disabled

### Acceptance criteria

- If the peak collapses substantially, treat VAE attention as the primary hotspot.
- If the peak drops only modestly, keep attention work in scope but elevate convolution/downsample workspace as a co-primary issue.

---

## Phase 2 — Lower The Baseline Before Control-Context Build

### Objective

Reduce avoidable resident memory before entering the most expensive VAE-encode phase.

### Files

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

### Changes

After text embeddings are fully evaluated:

1. release text-encoder state if any remaining ownership still exists
2. explicitly release transformer / controlnet modules before `buildControlContext(...)`
3. clear MLX cache before entering `vae.encode(...)`
4. reload transformer / controlnet only after the control context has been built

The control pipeline already has reloading mechanics for later phases. Reuse them rather than adding a second loading architecture.

### Notes

This phase does not fix the peak itself. It reduces the floor under the peak and lowers swap/compression risk.

### Acceptance criteria

- Lower pre-control-context resident memory than current baseline.
- No externally visible change in CLI behavior.
- Control generation still succeeds without requiring the caller to change flags.

---

## Phase 3 — Implement Query-Chunked VAE Attention

### Objective

Cap the largest transient attention workspace in `VAESelfAttention` without changing model weights.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- tests under `Tests/ZImageTests/` as appropriate

### Changes

Replace the single full-query SDPA call with a chunked query loop:

1. reshape Q/K/V exactly as today
2. split the query sequence dimension into chunks
3. run `MLXFast.scaledDotProductAttention(...)` per chunk against full K/V
4. concatenate outputs along the query dimension
5. preserve existing output projection and residual behavior

### Configuration

Add an internal tunable query chunk size, for example:

- default experimental values: `512`, `1024`, `2048`
- no public CLI surface initially

### Design constraints

- preserve exact tensor semantics and output layout
- avoid introducing extra copies beyond what concatenation requires
- keep code path easy to revert or disable if numerical issues appear

### Validation

Compare:

- peak memory
- runtime
- output drift versus unchunked attention

### Acceptance criteria

- Peak memory during `buildControlContext(...)` drops materially.
- Output drift stays within an acceptable tolerance for the control pipeline.
- Runtime regression is documented and acceptable relative to the memory win.

---

## Phase 4 — Add Tiled VAE Encode For Large Control Images

### Objective

Reduce VAE-encode activation/workspace growth for large control images, independent of whether attention or convolutions dominate the peak.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `docs/ARCHITECTURE.md`
- `docs/CLI.md` if a user-visible switch is added

### Changes

Add an encode path that splits large images into overlapping tiles before VAE encode, then stitches the latent tiles.

Recommended policy:

- automatic enablement above a pixel threshold
- optional internal override during development

Important details:

- overlap must be sufficient to avoid obvious seam artifacts
- stitching/blending should occur in latent space
- tile size should be chosen with Apple Silicon unified-memory constraints in mind

### Rationale

This is the most direct analogue to Diffusers’ VAE tiling strategy and is the strongest general-purpose fix if high-resolution VAE encode remains unstable.

### Acceptance criteria

- Large-image control encode completes with materially lower peak memory.
- Artifacting at tile seams is either absent or quantitatively negligible.
- Small-image runs keep the simpler non-tiled path.

---

## Phase 5 — Split VAE Lifecycle Into Encoder-Only / Decoder-Only Phases

### Objective

Avoid carrying the full `AutoencoderKL` residency when only the encoder or decoder half is needed.

### Files

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- related weight-loading code if needed
- `docs/ARCHITECTURE.md`

### Changes

Refactor the control pipeline lifecycle so that:

- the encoder is loaded for control-context construction
- the encoder can be released afterward
- the decoder is loaded later for final image decode

This may be implemented as:

- explicit encoder/decoder submodules inside the existing VAE wrapper, or
- staged loading helpers that construct only the needed half at each phase

### Acceptance criteria

- Lower steady-state resident memory outside the encode hotspot.
- No regression in final image decode behavior.
- Lifecycle remains understandable and testable.

---

## Phase 6 — Finalize Runtime Policy And Documentation

### Objective

Turn the successful experiments into a stable default behavior and document the tradeoffs.

### Files

- `docs/CLI.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/dev_plans/ROADMAP.md`
- tests/docs as needed

### Changes

Document:

- when chunked attention is enabled
- when tiled encode is enabled
- any debug flags retained after the work
- expected runtime versus memory tradeoffs for large control images

Also update roadmap priority if the remediation lands partially.

### Acceptance criteria

- The docs match the actual runtime policy.
- Follow-up contributors can reproduce the memory analysis workflow without reconstructing context from chat logs.

## Suggested Implementation Order

1. Phase 0 — instrumentation
2. Phase 1 — diagnostic no-attention run
3. Phase 2 — unload/reload baseline reduction
4. Phase 3 — query-chunked VAE attention
5. Phase 4 — tiled VAE encode
6. Phase 5 — encoder/decoder lifecycle split
7. Phase 6 — docs and cleanup

## File-Level Patch Checklist

### Likely code changes

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - phase markers
  - pre-encode module release / post-encode reload
  - optional runtime policy for tiled encode
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - optional mid-block attention disablement hook for diagnosis
  - query-chunked `VAESelfAttention`
  - tiled encode support
  - optional encoder/decoder lifecycle split
- `Sources/ZImageCLI/main.swift`
  - temporary diagnostic flags only if required

### Likely test additions

- shape/consistency tests for chunked attention output
- regression tests for tiled encode stitching dimensions
- integration-level smoke test for large control-context construction if practical

### Likely docs updates

- `docs/CLI.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/dev_plans/ROADMAP.md`

## Risks And Tradeoffs

### Query-chunked attention

- lower peak memory
- higher runtime
- possible numerical drift depending on chunk size and kernel behavior

### Tiled encode

- strongest memory reduction
- more implementation complexity
- risk of seam artifacts if overlap/blending is wrong

### Encoder/decoder lifecycle split

- lower steady-state residency
- touches model-loading lifecycle and therefore increases refactor risk

## Recommended First Success Criteria

Treat the remediation as successful when all of the following are true for the `1536x2304` control case:

1. Control-context construction peak memory is materially below the current `~100 GB` observation.
2. The run completes without pathological swap/compression behavior.
3. Output quality remains acceptable for existing control examples.
4. The final runtime policy is documented in `docs/`.

## Relationship To Existing Notes

- `docs/debug_notes/control-memory-phased-plan.md` records the earlier phased work and its original diagnosis.
- This plan supersedes that note for future control-memory remediation because the current repo state already incorporates some of the earlier fixes.
