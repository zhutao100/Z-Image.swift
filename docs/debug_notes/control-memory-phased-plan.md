# Control Pipeline Memory Analysis And Phased Plan

This note records the original diagnosis for the resident-memory surge during `ZImageCLI control` at `1536x2304` and the phased remediation plan that followed.

Status on `2026-03-07`:

- Phase 1 implemented: text encoder is released before control-context build.
- Phase 2 implemented: control context is built in the VAE dtype.
- Phase 3 implemented: ControlNet is a separate module that shares base transformer modules.

Unless stated otherwise, the root-cause descriptions below refer to the pre-fix control-pipeline architecture that motivated the phased changes.

## Observed Behavior

Reported run before the fixes:

- Model + VAE + control transformer loaded: a little above `~20 GB`
- During `Loading text encoder...`: gradually climbs to `~30 GB`
- During `Building control context...`: quickly surges to `~100 GB`

The relevant runtime path is in `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`.

## Root Causes

### 1) The final control-context tensor is small; the spike happens while building it

The logged control-context shape is `[1, 33, 1, 288, 192]`.

- Elements: `1 * 33 * 1 * 288 * 192 = 1,824,768`
- Approx size:
  - bf16: `~3.48 MiB`
  - fp32: `~6.96 MiB`

This rules out the final control-context tensor itself as the source of the `~100 GB` peak.

### 2) Control-context construction currently runs the VAE encoder on full-resolution fp32 inputs

`encodeImageToLatents(...)` and the inpaint path in `buildControlContext(...)` currently create image tensors as `.float32` before calling `vae.encode(...)`.

Relevant code:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
  - `encodeImageToLatents(...)`
  - `buildControlContext(...)`
- `Sources/ZImage/Util/ImageIO.swift`
  - `normalizeForEncoder(...)`

By contrast, the Diffusers reference prepares control images in `self.vae.dtype`, not hard-coded fp32.

Implication:

- Early VAE activations and temporary workspaces are larger than necessary.
- Untyped scalars can keep parts of the graph in fp32 even when the model weights are bf16.

### 3) The control VAE encoder includes a latent-space self-attention hotspot

The full control pipeline uses `AutoencoderKL`, not the decoder-only VAE.

Relevant code:

- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
  - `VAEEncoder`
  - `VAEMidBlock`
  - `VAESelfAttention`

For `1536x2304`, the control-image latent resolution is:

- `latentH = 2304 / 8 = 288`
- `latentW = 1536 / 8 = 192`

The VAE mid-block attention therefore operates on:

- `288 * 192 = 55,296` latent tokens

Back-of-the-envelope sizing:

- QKV storage at `C=512`
  - bf16: about `0.16 GiB`
  - fp32: about `0.32 GiB`
- Full attention matrix, if materialized
  - bf16: about `5.7 GiB`
  - fp32: about `11.4 GiB`

Even when the backend uses a fused attention kernel, this stage is large enough to be a major contributor. The early full-resolution conv activations are also multi-GB each at fp32.

### 4) The control pipeline does not explicitly release the text encoder after embedding generation

The base pipeline drops the text encoder and clears the MLX cache immediately after prompt embedding generation.

Relevant code:

- Base pipeline:
  - `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- Control pipeline:
  - `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

The control pipeline currently keeps the text-encoding phase more memory-resident than necessary before moving into control-context construction. That aligns with the reported `~20 GB -> ~30 GB` climb during text-encoder load.

### 5) The Swift control path keeps a higher baseline than Diffusers

Before Phase 3, the Swift implementation built a full control transformer and applied base transformer weights into it.

Relevant code:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Weights/ZImageControlWeightsMapping.swift`

Diffusers instead uses a separate ControlNet model that shares core transformer modules from the base transformer.

Relevant reference:

- `~/workspace/custom-builds/diffusers/src/diffusers/models/controlnets/controlnet_z_image.py`
- `~/workspace/custom-builds/diffusers/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`

Implication:

- The Swift control path starts from a higher resident-memory floor before denoising begins.

## Phased Plan

Implementation status:

- Phase 1 completed
- Phase 2 completed
- Phase 3 completed

### Phase 1: Release text-encoder memory earlier

Goal:

- Free the text encoder and clear MLX cache immediately after prompt embeddings are evaluated in the control pipeline.

Expected impact:

- Reduce the pre-control-context baseline.
- Address the reported `~20 GB -> ~30 GB` growth during the text-encoder phase.

Scope:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

Verification:

- Unit tests
- Build
- Confirm the control pipeline now mirrors the base pipeline’s lifecycle conventions

### Phase 2: Build control context in VAE dtype

Goal:

- Stop feeding fp32 image tensors into `vae.encode(...)` in the control/inpaint path.
- Make scalar math dtype-safe so the graph stays in the VAE dtype instead of silently promoting to fp32.

Expected impact:

- Lower VAE activation and workspace size during control-context construction.
- Reduce the largest single contributor to the observed peak.

Scope:

- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Util/ImageIO.swift`
- Any nearby helper that currently injects untyped fp32 scalars into the VAE encode path

Verification:

- Unit tests
- Build
- Confirm dtype usage is consistent with the existing bf16 conventions already used elsewhere in the project

### Phase 3: Split ControlNet into a separate module that shares base transformer modules

Goal:

- Refactor the control path so the base transformer and ControlNet are separate modules, matching the upstream Diffusers architecture more closely.
- Share the common transformer modules instead of duplicating them into a monolithic control transformer model.

Expected impact:

- Lower the steady-state memory floor before and during denoising.
- Reduce duplication between the base transformer path and the control path.

Scope:

- `Sources/ZImage/Model/Transformer/*`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Weights/ZImageControlWeightsMapping.swift`
- Tests and docs as needed

Verification:

- Unit tests
- Build
- Confirm the refactor does not change external CLI behavior

## Notes On Prioritization

The first two phases are low risk and directly target the reported memory growth. The third phase is the structural fix that addresses the higher baseline versus Diffusers, but it is intentionally staged last because it touches model boundaries, weight loading, and denoising integration.
