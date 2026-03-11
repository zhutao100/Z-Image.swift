# Runtime Precision Parity Follow-Up (Archived)

Status: completed on March 11, 2026.

This plan is archived because the remaining RoPE numeric-staging audit has now been executed against the latest codebase.

## What The Audit Validated

The repo still has source-backed coverage for the earlier March 2026 parity fixes:

- `Tests/ZImageTests/Support/PipelinePrecisionTests.swift`
  - runtime-dtype normalization at transformer and ControlNet ingress
  - timestep-frequency casting before the first timestep MLP layer
- `Tests/ZImageTests/Support/QwenEncoderAttentionMaskTests.swift`
  - boolean causal-plus-padding prompt masking on the non-generation text path

The former open RoPE follow-up is now covered by `Tests/ZImageTests/Transformer/ZImageRoPEParityTests.swift`, which validates:

- Diffusers-style RoPE table construction for representative axes
- Diffusers-style rotary application on `float32` inputs
- Diffusers-style float32 rotary compute with cast-back to `bfloat16` inputs

## Final Outcome

The audit found one concrete runtime gap on the denoiser/control path:

- `ZImageAttentionUtils.applyRotary(...)` promoted `bfloat16` query/key tensors to `float32` and returned `float32` tensors

That behavior is now fixed. The shared rotary helper now:

- performs the rotation in `float32`
- casts the rotated tensors back to the original query/key dtype

The audit did not justify a separate `ZImageRopeEmbedder` table-generation change. The existing table construction already matched the Diffusers-style float64-backed, float32-materialized reference within tight tolerance in the new unit coverage.

## Verification

Targeted repo checks:

- `swift test --filter ZImageRoPEParityTests`
- `swift test --filter PipelinePrecisionTests`
- `swift test --filter QwenEncoderAttentionMaskTests`

Fixed-seed runtime probes:

- Base probe
  - model: `Tongyi-MAI/Z-Image-Turbo`
  - prompt: `a brass compass on a wooden desk, dramatic sunlight, product photo`
  - size: `512x512`
  - steps: `4`
  - guidance: `0`
  - seed: `1234`
  - output sha256: `492c9ae847506e06259a21f6779fdbea7600fdbf2121297a37dc42bfdcd76749`
- Control probe
  - model: `Tongyi-MAI/Z-Image-Turbo`
  - control weights: `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`
  - control file: `Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors`
  - control image: `images/canny.jpg`
  - prompt: `a stone archway covered in moss, cinematic lighting`
  - size: `512x512`
  - steps: `4`
  - guidance: `0`
  - control scale: `0.75`
  - seed: `1234`
  - output sha256: `273e080928424feb8563ba4b6cba227d9ae0aa88d538d538a2eaa5835260dc7d`

## Re-Entry Criteria

Reopen precision implementation work only when one of these is true:

- a code change touches denoiser numerics, rotary table construction, rotary application, or prompt masking
- a Swift-vs-Diffusers drift report needs renewed source-backed confirmation
- a new backend or weight-format change makes the current dtype assumptions questionable

For methodology, use [../../golden_checks.md](../../golden_checks.md). The current source-backed status note lives in [../../context/zimage_runtime_precision_parity_report.md](../../context/zimage_runtime_precision_parity_report.md).
