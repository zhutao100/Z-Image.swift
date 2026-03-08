I inspected the current Swift code path and compared it against the upstream Diffusers control pipeline and current backend memory docs. I did **not** run the model here, so this is a **source-based diagnosis**, but the code evidence is strong.

## Diagnosis

The surge is **primarily triggered by control-context construction**, not by the final stored `control_context` tensor.

For your exact run, the hot path is:

* `ZImageControlPipeline.generateCore(...)`
* `buildControlContext(...)`
* `encodeImageToLatents(...)`
* `vae.encode(normalized)`

in:

* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:369-392`
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:440-531`
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:900-926`

### 1) The final control tensor is tiny; the spike happens while building it

Your log shows:

* `Control context built, shape: [1, 33, 1, 288, 192]`

That tensor is only:

* `1 * 33 * 1 * 288 * 192 = 1,824,768` elements
* about **3.5 MiB in BF16**
* about **7.0 MiB in FP32**

So the 100 GB-looking jump is **not** from retaining the final tensor.

### 2) The dominant trigger is the **full-resolution VAE encoder path**

For pure control mode, current Swift does exactly this:

```swift
let imageArray = try QwenImageIO.resizedPixelArray(... dtype: vaeDType)
let normalized = QwenImageIO.normalizeForEncoder(imageArray)
let encodedLatents = vae.encode(normalized)
```

That is in `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:377-392`.

For your requested `1536 x 2304`, the control VAE latent grid is:

* `latentH = 2304 / 8 = 288`
* `latentW = 1536 / 8 = 192`

So the VAE mid-block attention runs on:

* `288 * 192 = 55,296` tokens

The current VAE attention implementation is chunked, but it is still operating on very large tensors:

* `Sources/ZImage/Model/VAE/AutoencoderKL.swift:51-103`
* `Sources/ZImage/Model/VAE/AutoencoderKL.swift:123-145`

This means the code is no longer suffering from the earlier “accidental fp32 control-image path” issue; that was already fixed. The remaining problem is more structural: **large encoder activations + large attention workspaces + retained cache/workspace**.

### 3) Control mode has a much higher baseline than non-control mode even before the spike

This matters for your comparison against normal mode.

The **normal** pipeline loads a **decoder-only VAE**:

* `Sources/ZImage/Pipeline/ZImagePipeline.swift:172-185`

The **control** pipeline loads the **full AutoencoderKL** (encoder + decoder):

* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:306-320`

So control mode is inherently carrying more VAE memory than normal mode. Then it also loads transformer + controlnet on top:

* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:722-780`

That explains why your baseline is already ~20 GB before text encoding, whereas normal mode is materially lighter.

### 4) The current load order is inefficient and amplifies peak memory

Right now the control pipeline:

1. loads full VAE
2. loads transformer
3. loads controlnet
4. loads text encoder
5. encodes prompt
6. **unloads transformer/controlnet**
7. builds control context with VAE encode
8. **reloads transformer/controlnet**

That is visible in:

* preload: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:722-780`
* unload before control build: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:904-909`
* reload after control build: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:960-986`

So the code is doing avoidable memory churn: it pays the cost of loading the heaviest modules early, only to unload them for control-context construction, then load them again.

This is likely why your text-encoder phase climbs from ~20 GB to ~30 GB: the text encoder is being loaded **on top of** full VAE + transformer + controlnet residency.

### 5) There is still no hard cache-release barrier after control-context build

After the control context is built, current code does:

```swift
let result = try buildControlContext(...)
MLX.eval(result)
logControlMemory("control-context.after-eval", ...)
controlContext = result.asType(vae.dtype)
```

But it does **not** call `Memory.clearCache()` there before transformer/controlnet reload.

That is `Sources/ZImage/Pipeline/ZImageControlPipeline.swift:912-925`.

This is important because the visible “memory footprint” in Apple tools is not the same as just “live tensor bytes.” Apple’s memory-footprint accounting includes dirty, compressed, and swapped pages, and on Apple silicon it includes GPU objects as part of the process footprint. PyTorch’s MPS docs also distinguish “tensor memory” from total driver allocation, with the latter explicitly including cached allocator pools and MPS/MPSGraph allocations. ([Apple Developer][1])

So even if the live tensor graph after control build is small, **retained MLX/Metal cache/workspace can still keep Activity Monitor very high**, especially right before heavy modules are reloaded.

### 6) Even after the build phase, control denoising is intrinsically heavier than normal denoising

The control path repeatedly grows stacked hint tensors through `MLX.stacked(allC, axis: 0)`:

* `Sources/ZImage/Model/Transformer/ZImageControlTransformerBlock.swift:111-145`

and it does that across:

* 15 control layers
* 2 control refiner layers

from:

* `Sources/ZImage/Model/Transformer/ZImageControlTransformer2D.swift:19-20`

At your latent size, the image-token count is:

* `(288 / 2) * (192 / 2) = 13,824`

With hidden size `3840`, a single BF16 `[batch, seq, dim]` hidden state is about **100 MiB**. Once the control blocks grow stacked tensors like `[num_hints, batch, seq, dim]`, transient memory gets into the **multi-GiB** range very quickly. This is a real steady-state control-mode overhead that normal mode does not pay.

## Comparison with Diffusers

The upstream Diffusers Z-Image ControlNet pipeline uses the **same high-level algorithm** for the control image: preprocess image, run `vae.encode(control_image)`, rescale latents, then feed them to ControlNet. So the Swift port is not wrong in principle here. ([GitHub][2])

Two differences matter, though:

* Diffusers has documented **tiled VAE encode/decode** support; its `tiled_encode` is explicitly described as splitting the input into overlapping tiles to keep memory use stable with image size. ([Hugging Face][3])
* Diffusers also documents CPU/model offloading to reduce memory usage by moving components on and off device only when needed. ([Hugging Face][4])

So the Swift port is matching the reference math, but it does **not** yet have the same mature memory-management toolbox.

## What is **not** the root cause

These earlier hypotheses are no longer the main issue in the current code:

* **Hard-coded fp32 control image inputs**: fixed for the control encode path; current code uses `dtype: vae.dtype`.
* **Transformer/controlnet still resident during control build**: current code does unload them before `buildControlContext(...)`.
* **Final `control_context` tensor too large**: false; it is tiny.

## Highest-value improvement options

### A. Add a hard cache barrier immediately after control-context build

This is the first change I would make.

Right after control-context materialization:

```swift
let result = try buildControlContext(...)
controlContext = result.asType(vae.dtype)
MLX.eval(controlContext)
Memory.clearCache()
logControlMemory("control-context.after-clear-cache", enabled: logPhaseMemory)
```

Why this is high ROI:

* minimal code change
* directly targets the most plausible reason Activity Monitor remains inflated
* gives you an immediate validation point with `--log-control-memory`

### B. Reorder lifecycle so transformer/controlnet are loaded **after** prompt encoding and control-context build

Current order is wasteful. Better order:

1. tokenizer
2. full VAE encoder path or encoder-only VAE
3. text encoder
4. prompt embeddings
5. build control context
6. clear cache
7. load transformer
8. load controlnet
9. denoise
10. unload transformer/controlnet
11. decode

That should reduce:

* the ~20 GB baseline before text encoding
* the ~20 → 30 GB text-encoder climb
* load/unload/reload churn

### C. Split VAE into **encoder-only** for control build and **decoder-only** for final decode

This is the most structurally correct fix.

Normal mode already benefits from decoder-only VAE residency. Control mode should do something similar:

* transient **encoder-only** VAE for `buildControlContext(...)`
* persistent **decoder-only** VAE for final decode

That removes encoder residency from the denoising/decode phases and makes control mode much closer to normal mode’s memory profile.

### D. Add tiled VAE encode for control/inpaint images

This is the strongest fix for the actual build spike.

Diffusers documents tiled encode specifically as a way to keep memory use stable as image size grows. ([Hugging Face][3])

This is the right answer if you want to keep supporting very large control images without huge transient peaks.

Tradeoff:

* more implementation complexity
* some risk of tile-boundary artifacts unless overlap/blending is done carefully
* slight output mismatch versus monolithic encode

### E. Refactor control hint transport to avoid growing `stacked(allC)` tensors

This targets denoising memory, not the build spike.

Current design repeatedly carries a growing stacked tensor through control blocks. A leaner design would:

* keep only the latest control state
* emit a hint for the current target layer
* store required hints in a dictionary or preallocated buffer
* avoid repeatedly re-stacking prior hints as a leading dimension

That will not fix the control-context build surge, but it should reduce the gap between control mode and normal mode during denoising.

### F. Optional: add offload / quantization policies for control-specific modules

Secondary options:

* model-offload-like lifecycle for text encoder / controlnet
* quantized text encoder path
* quantized controlnet path if quality stays acceptable

These are valid, but I would do them **after** A/B/C.

## My priority order

1. **Post-build `Memory.clearCache()` barrier + telemetry marker**
2. **Lifecycle reorder to defer transformer/controlnet load**
3. **Encoder-only / decoder-only VAE split**
4. **Tiled VAE encode**
5. **Control hint-stack refactor**

## What I would measure next

Use the existing probe:

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

Then add two extra markers:

* `control-context.after-clear-cache`
* `transformer-controlnet.after-reload`

Interpretation:

* if `after-eval` is huge and `after-clear-cache` drops sharply, cache retention is a major amplifier
* if the peak still occurs inside `buildControlContext(...)` before the clear, then VAE encode itself is the dominant spike
* if denoising remains much heavier than normal mode even after that, the `stacked(allC)` design is the next target

## Bottom line

The most likely root-cause stack is:

1. **full-resolution VAE encoder work** during control-context build
2. **retained MLX/Metal cache/workspace** because there is no post-build cache barrier
3. **unnecessarily high baseline and churn** from loading transformer/controlnet before they are needed
4. **extra control-mode denoising overhead** from stacked hint tensors

So the fastest credible improvement is **not** another dtype tweak. It is:

* **flush cache after control-context build**
* **defer transformer/controlnet loading**
* then move to **encoder/decoder VAE split** and **tiled control encode**

That is where I would focus engineering effort.

[1]: https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app?utm_source=chatgpt.com "Analyzing the memory usage of your Metal app"
[2]: https://raw.githubusercontent.com/huggingface/diffusers/refs/heads/main/src/diffusers/pipelines/z_image/pipeline_z_image_controlnet.py "raw.githubusercontent.com"
[3]: https://huggingface.co/docs/diffusers/en/api/models/autoencoderkl "AutoencoderKL"
[4]: https://huggingface.co/docs/diffusers/optimization/memory?utm_source=chatgpt.com "Reduce memory usage"
