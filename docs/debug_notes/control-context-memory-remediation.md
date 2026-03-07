# Control-Context Memory Remediation

Validated on March 7, 2026 against the current Swift repo state.

This note replaces older conclusions that became partially stale after the control-path unload work, the VAE attention chunking change, and the phase-memory telemetry landed.

## Validation Basis

- Source inspection of:
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift`
  - `Sources/ZImage/Model/Transformer/ZImageControlTransformerBlock.swift`
- Source inspection of the local Diffusers reference:
  - `~/workspace/custom-builds/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`
- One measured high-resolution control probe on March 7, 2026:

```bash
./.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --guidance 0 \
  --seed 1234 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-remediation/phase0_memory.png
```

Diffusers was not executed in this validation pass. Any statement below about Diffusers behavior is source-based unless explicitly marked as measured.

## Current Verdict

The large control-path memory footprint is still not caused by the stored `controlContext` tensor itself. The current dominant contributors are:

1. full-resolution VAE encode during `buildControlContext(...)`
2. retained MLX cache after control-context materialization and before transformer/controlnet reload
3. repeated ControlNet hint stacking during denoising

The final control tensor shape from the measured high-resolution run was `[1, 33, 1, 288, 192]`. That is only a few MiB in bf16 or fp32. The large footprint comes from how it is built and what remains cached afterward, not from the tensor that is ultimately kept.

## What Is Already True In The Current Repo

These older failure modes are no longer current:

- `ZImageControlPipeline` unloads the transformer, ControlNet, and active LoRA state before `buildControlContext(...)`.
- prompt-embedding cache cleanup already happens before the control build begins.
- the VAE mid-block self-attention path is query-chunked by default via `VAEAttention.defaultQueryChunkSize = 1024`.
- `--log-control-memory` already emits resident, active, cache, and peak markers around the main control-path phases.

Those mitigations are already visible in the measured baseline:

- `control-context.after-baseline-reduction`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`
- `control-context.before-build`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`

So the current problem is not "the transformer stayed resident during control-context build" and not "the VAE attention path is still fully unchunked."

## What Is Still True In The Current Repo

### 1. There is still no hard cache-release barrier after control-context materialization

In `generateCore(...)`, the control path does:

- `let result = try buildControlContext(...)`
- `MLX.eval(result)`
- `controlContext = result.asType(vae.dtype)`
- immediately proceeds toward transformer/controlnet reload

There is still no `Memory.clearCache()` between the typed control-context handoff and the reload of the heavy denoising modules.

The measured probe shows why this still matters:

- `control-context.after-eval`: resident `464.34 MiB`, active `172.72 MiB`, cache `28.07 GiB`
- `denoising.before-start`: resident `29.63 GiB`, active `29.34 GiB`, cache `28.08 GiB`

Interpretation: the control-context build itself finishes with low resident bytes but a very large retained MLX cache, and that cache is still present when transformer/controlnet residency returns.

### 2. The control pipeline still keeps a full `AutoencoderKL` resident

`ZImageControlPipeline` currently loads a full `AutoencoderKL` even though the control build only needs the encoder and the final image write only needs the decoder.

By contrast, `ZImagePipeline` already uses `AutoencoderDecoderOnly` for the normal text-to-image path.

That means the control path still has an obvious lifecycle-splitting opportunity:

- load an encoder-only VAE on demand for control/inpaint encode
- release it after control-context materialization
- defer decoder-only residency until the final decode

### 3. Denoising still pays for stacked ControlNet hint transport

`ZImageControlTransformerBlock` still returns `MLX.stacked(allC, axis: 0)` after appending the new skip hint and updated control state.

That means each control layer repeatedly rebuilds a stacked tensor containing:

- all prior hints
- the new hint
- the current control state

This is structurally heavier than carrying the current control state plus an incrementally grown hint list, and it directly targets denoising memory rather than the control-context build spike.

## What The Measured Baseline Says

Measured high-resolution control probe, March 7, 2026:

- `prompt-embeddings.after-clear-cache`: resident `36.08 GiB`, active `29.34 GiB`, cache `0 B`
- `control-context.after-baseline-reduction`: resident `371.59 MiB`, active `162.37 MiB`, cache `0 B`
- `control-context.after-eval`: resident `464.34 MiB`, active `172.72 MiB`, cache `28.07 GiB`
- `denoising.before-start`: resident `29.63 GiB`, active `29.34 GiB`, cache `28.08 GiB`
- `decode.after-eval`: resident `124.06 MiB`, active `192.86 MiB`, cache `38.95 GiB`, MLX peak `39.03 GiB`
- `/usr/bin/time -l` maximum resident set size: `42,832,363,520` bytes
- `/usr/bin/time -l` peak memory footprint: `112,574,979,616` bytes

Two points matter here:

1. The build-time baseline reduction is working. The repo now really does collapse resident memory before the control VAE encode.
2. The build output still leaves behind enough retained cache to make the reload boundary expensive.

The `/usr/bin/time -l` footprint is much higher than the MLX "peak" counter, which is expected. They are not the same metric.

## Diffusers Parity Check

The local Diffusers control pipeline still does the same high-level control-image preparation:

- preprocess control image
- `self.vae.encode(control_image)`
- shift/scale to latent space
- `unsqueeze(2)` for control-context shape

That means the Swift port is not paying the control-context cost because it invented a different algorithm. The remaining gap is about residency policy and data transport, not about basic control-image semantics.

## Ranked Remediation Order

1. Add a post-build materialization barrier:
   - cast to the stored dtype
   - `MLX.eval` the typed tensor
   - `Memory.clearCache()`
   - log a new phase such as `control-context.after-clear-cache`
2. Split the control VAE lifecycle:
   - encoder-only on demand for control/inpaint encode
   - decoder-only deferred until final decode
3. Remove stacked hint transport:
   - keep current control state separate from accumulated hints
   - stop rebuilding `MLX.stacked(...)` at every control block
4. Only consider tiled/sliced encode if the first three phases still leave an unacceptable high-resolution spike

The measured execution plan for these changes lives in `docs/dev_plans/control-context-memory-remediation.md`.
