I inspected the Swift repo, the local Diffusers reference, and current external backend docs. I did not execute the model here, so this is a source-based diagnosis.

## Conclusion

The **100 GB-looking spike is not explained by the final `control_context` tensor**. The dominant causes are:

1. **Full-resolution VAE encode during control-context build** in `ZImageControlPipeline.buildControlContext(...)`.
2. **Large post-build MLX/Metal cache retention before transformer/controlnet are reloaded**, which can make Activity Monitor show a much larger footprint than the live tensors alone.
3. **Additional control-mode denoising overhead** from the control transformer’s stacked hint tensors, which normal mode does not have.

## What the code confirms

### 1) The hot path is still full-resolution VAE encode

For pure control mode, Swift does:

* `buildControlContext(...)`
* `encodeImageToLatents(...)`
* `vae.encode(normalized)`

That is the same high-level algorithm Diffusers uses in `pipeline_z_image_controlnet.py`: it preprocesses the control image, runs `vae.encode(control_image)`, rescales, then unsqueezes to the control-context shape.

So the core issue is **not** that Swift is doing a fundamentally different control algorithm.

### 2) The final control tensor is tiny

Your logged shape is:

* `[1, 33, 1, 288, 192]`

That is only about:

* **~3.4 MiB in BF16**
* **~6.8 MiB in FP32**

So the spike is clearly happening **while constructing it**, not from retaining it.

### 3) The old “easy” causes are already mitigated in this repo

The current source already has both of these mitigations:

* unload transformer/controlnet before `buildControlContext(...)`
* chunk VAE mid-block attention queries (`defaultQueryChunkSize = 1024`)

So this is **not** just “transformer wasn’t unloaded” or “attention was fully quadratic with no chunking”.

### 4) The biggest remaining code-level amplifier is cache retention after build

This is the most important concrete finding in the current source.

In `generateCore(...)`, after:

* `let result = try buildControlContext(...)`
* `MLX.eval(result)`
* `controlContext = result.asType(vae.dtype)`

the pipeline **does not call `Memory.clearCache()` before reloading transformer/controlnet**.

That matters because the repo’s own remediation note says the retained policy still left roughly **~28 GiB of MLX cache after control-context build** on the reference run. If that cache stays resident while transformer + controlnet are reloaded, the apparent process footprint can balloon even though the live control tensor is tiny.

This matches your symptom very well.

### 5) Control denoising is intrinsically heavier than normal denoising

Even after the control context is built, control mode has an extra memory burden that normal mode does not:

* `ZImageControlTransformerBlock` repeatedly does `MLX.stacked(allC, axis: 0)`
* the control config uses **15 `controlLayersPlaces`** and **2 `controlRefinerLayersPlaces`**
* at `1536x2304`, the control/image sequence length is `13824`, and each `[seq, dim]` BF16 hidden state at `dim=3840` is about **~100 MiB**

So stacked control-hint state can easily reach **~1.5+ GiB per stacked tensor family** before counting attention/FFN intermediates. That explains why control mode remains materially heavier than normal mode during denoising.

## Why Diffusers/MPS can look much smaller

Two things are likely being conflated.

First, **metric mismatch**. Apple’s VM Tracker / Metal memory guidance distinguishes process footprint from simple heap allocations, and VM Tracker’s footprint includes **noncompressed plus compressed/swapped dirty memory**. Activity Monitor likewise exposes compressed memory and swap, not just “currently live tensor bytes.” ([Apple Developer][1])

Second, PyTorch on MPS explicitly separates:

* `current_allocated_memory()` = memory occupied by tensors, **excluding cached allocator pools**
* `driver_allocated_memory()` = total Metal-driver allocation, **including cached pools and MPS/MPSGraph allocations** ([PyTorch Docs][2])

So “Diffusers looked smaller on MPS” is not necessarily an apples-to-apples comparison unless you compare the same metric on both sides.

There is also an ecosystem difference: the current Z-Image guidance in DiffSynth recommends **CPU offloading / VRAM management** for Z-Image rather than keeping everything resident. The Swift port does not yet have an equivalent sequential offload policy. ([GitHub][3])

Also, your requested resolution is **within** the model family’s published range, so this is not simply “unsupported resolution abuse.” ([Hugging Face][4])

## Root-cause buckets

### Confirmed

* `buildControlContext(...)` is dominated by **full-resolution VAE encode**
* the final stored control-context tensor is **not** the memory problem
* the repo already unloads transformer/controlnet before build
* the repo already chunks VAE mid-block attention
* there is **no cache clear after control-context build and before transformer/controlnet reload**
* control denoising adds major extra memory via **stacked hint tensors**

### Likely, but needs measurement on your machine

* The **~100 GB Activity Monitor peak** is probably a **compound footprint**:

  * VAE encode workspace/cache retained after build
  * then transformer/controlnet reload
  * then possibly initial denoising setup
* MLX/Metal caching and Activity Monitor update cadence are likely making the build/reload boundary look like one giant spike

## Best improvement options

### 1) Add a hard cache-release barrier immediately after control-context build

Highest ROI, lowest risk.

Right after control-context materialization:

* assign `controlContext`
* `MLX.eval(controlContext)`
* `Memory.clearCache()`
* log a new phase like `control-context.after-clear-cache`

This directly targets the most plausible reason your footprint still looks enormous after the repo’s prior mitigations.

### 2) Split VAE lifecycle into encoder-only for control build

Load only the VAE encoder for control/inpaint encode, then release it before reloading transformer/controlnet. Keep decoder loading deferred until final decode.

That prevents encoder residency/workspace from overlapping with the next heavy phase.

### 3) Add tiled/sliced VAE encode for control images

This is the strongest structural fix for the build spike itself.

The current pure-control path still runs the encoder over the full 1536×2304 image in one pass. Tiling with overlap would trade some latency and implementation complexity for a much flatter peak.

### 4) Rework control hint transport to avoid `stacked(allC)`

This targets denoising memory, not build memory.

Instead of repeatedly stacking full-sequence tensors, return only the newly emitted hint and the latest control state, or carry a lightweight list/tuple-like structure until the final dictionary emission.

That should materially reduce control-mode denoising overhead.

### 5) Optional: add offload policies for text encoder / controlnet

This is secondary for your specific build spike, but useful overall:

* sequential text-encoder load/unload
* optional controlnet offload outside active denoise
* optional GGUF / quantized text encoder path for prompt encoding

## What I would measure next

Use the existing flag that is already in this repo:

```bash
.build/xcode/Build/Products/Release/ZImageCLI control \
  --cw alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --cf Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors \
  --control-image "[control_image_path]" \
  --width 1536 \
  --height 2304 \
  --prompt "[prompt]" \
  --negative-prompt "[negative_prompt]" \
  --steps 1 \
  --log-control-memory \
  --output /tmp/zimage-control-test.png
```

The key comparison is:

* `control-context.after-baseline-reduction`
* `control-context.after-eval`
* a new `control-context.after-clear-cache` phase if you add it
* `denoising.before-start`

If `after-eval` is huge and `after-clear-cache` collapses, you have isolated the main issue.

The single most actionable takeaway is this: **the current repo still appears to be missing a post-control-build cache flush before transformer/controlnet reload, and that is the first thing I would change.**

[1]: https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app?utm_source=chatgpt.com "Analyzing the memory usage of your Metal app"
[2]: https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory.html "torch.mps.current_allocated_memory — PyTorch 2.10 documentation"
[3]: https://github.com/modelscope/DiffSynth-Studio/blob/main/README.md "DiffSynth-Studio/README.md at main · modelscope/DiffSynth-Studio · GitHub"
[4]: https://huggingface.co/Tongyi-MAI/Z-Image "Tongyi-MAI/Z-Image · Hugging Face"
