# Z-Image.swift Runtime Precision Parity Report

This note summarizes the current source-backed precision-parity state of the Swift + MLX implementation against the directly relevant Diffusers Z-Image pipelines.

Status: refreshed against the current repo state on March 11, 2026.

For probe design and fixture methodology, use [../golden_checks.md](../golden_checks.md). For the currently open follow-up work, use [../dev_plans/runtime_precision_parity_improvement_plan.md](../dev_plans/runtime_precision_parity_improvement_plan.md).

## Scope

Swift files that matter most for the current parity story:

- `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- `Sources/ZImage/Pipeline/PipelineUtilities.swift`
- `Sources/ZImage/Pipeline/FlowMatchScheduler.swift`
- `Sources/ZImage/Model/Transformer/ZImageTimestepEmbedder.swift`
- `Sources/ZImage/Model/Transformer/ZImageRopeEmbedder.swift`
- `Sources/ZImage/Model/Transformer/ZImageAttentionUtils.swift`
- `Sources/ZImage/Model/TextEncoder/TextEncoder.swift`
- `Sources/ZImage/Model/VAE/AutoencoderKL.swift`
- `Sources/ZImage/Model/VAE/AutoencoderDecoder.swift`
- `Sources/ZImage/Weights/ZImageWeightsMapper.swift`

Diffusers reference targets:

- `src/diffusers/pipelines/z_image/pipeline_z_image.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py`
- `src/diffusers/pipelines/z_image/pipeline_z_image_controlnet_inpaint.py`
- `src/diffusers/models/transformers/transformer_z_image.py`
- `src/diffusers/models/controlnets/controlnet_z_image.py`

## Confirmed Parity In The Current Repo

### 1. Standard non-quantized weight loading is still BF16-oriented

The standard weight loaders still default to `.bfloat16`, including ControlNet loading. That keeps the default Swift loading path aligned with the common Diffusers Z-Image usage pattern.

### 2. Scheduler state remains FP32-oriented, with explicit denoiser ingress normalization

Base and control latents still start from the default `MLXRandom.normal(...)` floating dtype, and the scheduler preserves the latent dtype across steps.

The important current behavior change is that both pipelines now cast the model-facing latent tensor to the runtime dtype of the loaded module at the forward boundary through `PipelineUtilities.castModelInputToRuntimeDTypeIfNeeded(...)`.

### 3. Timestep features are normalized before the first timestep MLP layer

`ZImageTimestepEmbedder` now casts the sinusoidal frequency embedding to the first MLP layer runtime dtype before `mlp.0`, which aligns the ingress behavior with the intended Diffusers comparison.

### 4. Prompt encoding now uses boolean causal-plus-padding masking on the non-generation path

The prompt-encoding attention mask is no longer an additive hidden-dtype mask on the main non-generation path. The current implementation builds a combined boolean keep mask for prompt encoding while leaving generation-time masking behavior separate.

### 5. VAE encode/decode boundaries remain dtype-aware

The encode and decode paths still normalize image and latent tensors to the active VAE dtype at the relevant boundaries, which keeps the Swift path aligned with the typical Diffusers VAE handling.

## Confirmed Remaining Differences

### 1. RoPE construction and application still differ from Diffusers

This is the main remaining documented runtime parity gap.

The Swift runtime still constructs and applies rotary embeddings differently from the Diffusers reference path. The implementation difference is real, but the practical output impact still needs intermediate-tensor validation before any runtime change is made.

### 2. `weightsVariant` is file selection, not a full runtime precision policy

`weightsVariant` chooses which component files are loaded from a snapshot. It is useful for selecting `fp16` or `bf16` shards, but it is not a global runtime compute-dtype switch by itself.

## Practical Reading Of The Current State

The repo has already closed the three most actionable parity gaps from the earlier March 2026 pass:

- denoiser ingress dtype normalization
- timestep-MLP ingress dtype normalization
- boolean prompt masking

That means parity work should no longer focus on those items unless a regression reopens them. The next meaningful parity step is a RoPE probe that compares intermediate tensors against the local Diffusers checkout before changing runtime behavior.

## References

External references that are still useful when validating assumptions:

- MLX data types: <https://ml-explore.github.io/mlx/build/html/python/data_types.html>
- MLX `random.normal`: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.normal.html>
- MLX fast SDPA: <https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html>
- PyTorch SDPA: <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
