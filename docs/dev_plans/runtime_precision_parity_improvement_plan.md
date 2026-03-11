# Runtime Precision Parity Follow-Up

This is the current follow-up note for precision-parity work. The March 7, 2026 fixes for the first three confirmed mismatches have already landed.

## Already Landed

The repo now has the following parity-oriented fixes in place:

- explicit runtime-dtype normalization at transformer and ControlNet ingress
- explicit dtype normalization for the timestep-frequency tensor before the first timestep MLP layer
- boolean causal-plus-padding prompt-attention masking on the non-generation text-encoding path

Those behaviors are implemented in:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`

## Remaining Open Work

The remaining documented runtime parity work is now a narrow RoPE numeric-staging audit, not a confirmed algorithm mismatch.

The current denoiser and ControlNet transformer RoPE path appears structurally aligned with the Diffusers Z-Image reference, so the open questions are reduced to items such as:

- whether Diffusers float64-backed RoPE precompute leaves a measurable intermediate-tensor signature that Swift float32 table construction does not
- whether the lack of an explicit float32 cast inside Swift `applyRotary(...)` leaves a measurable Q/K delta versus the Diffusers path
- whether any such deltas are large enough to justify a runtime change after probe validation

Changing RoPE behavior without intermediate probes is still risky because end-to-end image deltas alone are too noisy to isolate the effect.

There is also an important nuance to keep in mind during parity work:

- `weightsVariant` selects which files are loaded, but it is not a global runtime precision switch by itself

## Re-Entry Criteria

Reopen parity implementation work when one of these is true:

- a code change touches denoiser numerics, rotary table construction, rotary application, or prompt masking
- a Swift-vs-Diffusers drift report needs source-backed confirmation
- a new backend or weight-format change makes the current dtype assumptions questionable

## Validation Recipe

Repo verification:

```bash
swift test
./scripts/build.sh
```

Reference fixed-seed probes:

- Base probe:
  - model: `Tongyi-MAI/Z-Image-Turbo`
  - prompt: `a brass compass on a wooden desk, dramatic sunlight, product photo`
  - size: `512x512`
  - steps: `4`
  - guidance: `0`
  - seed: `1234`
- Control probe:
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

When you need deeper diagnosis, use [../golden_checks.md](../golden_checks.md) and the current status note in [../context/zimage_runtime_precision_parity_report.md](../context/zimage_runtime_precision_parity_report.md).

## Next Recommended Step

If parity work resumes, the next step should still be an intermediate-tensor RoPE probe, but it should be framed as a numeric drift audit rather than a search for a still-open algorithm mismatch. Validate the RoPE tables plus the pre-RoPE and post-RoPE Q/K tensors against the local Diffusers checkout first, then decide whether the runtime path needs a behavioral change.
