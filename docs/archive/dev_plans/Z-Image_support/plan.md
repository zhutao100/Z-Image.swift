Below is a historical implementation plan for adding first-class support for [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image) to `zimage.swift`.

The codebase now ships that support, so this document is archived for context only. Current behavior is documented in:

- [README.md](../../../README.md)
- [../../CLI.md](../../CLI.md)
- [../../MODELS_AND_WEIGHTS.md](../../MODELS_AND_WEIGHTS.md)
- [../../ARCHITECTURE.md](../../ARCHITECTURE.md)

The original plan content is kept below unchanged except for this notice.

---

Below is an implementation-oriented plan to add **first-class support for [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image)** (the new, non-Turbo variant) to the existing **`zimage.swift`** Swift/MLX port.

---

## 0) Terminology (avoid “variant” ambiguity)

Two different things are commonly called “variant”:

1. **Model family / checkpoint**: `Tongyi-MAI/Z-Image-Turbo` vs `Tongyi-MAI/Z-Image` (Base).
2. **HF weights “variant” (precision)**: `bf16`, `fp16`, etc. This changes filenames like `*.bf16.safetensors` and the corresponding `*.index.json`.

In this plan:

- **“model” / “family”** refers to Turbo vs Base.
- **“weightsVariant”** refers to HF precision variants (`bf16`, `fp16`, …).

---

## 1) Establish what “support Z-Image” means in this codebase

### What already looks compatible

From the two model snapshots you attached, `Z-Image` and `Z-Image-Turbo` share the same high-level component layout (`transformer/`, `text_encoder/`, `vae/`, `scheduler/`, `tokenizer/`). The project already:

* resolves model snapshots from a local directory or HF model id (`ModelResolution` / `PipelineSnapshot`)
* loads configs from JSON (`ZImageModelConfigs`)
* resolves shard lists dynamically via the `*.safetensors.index.json` weight maps (`ZImageFiles.resolveTransformerWeights/resolveTextEncoderWeights`)
* implements the Z-Image denoising loop and (optional) CFG path

So “adding support” is primarily:

1. **first-class model selection + presets** (so users get sane defaults when choosing `Z-Image`),
2. **robust weight resolution** (avoid mixed bf16/fp16 shard loads; deterministic index selection),
3. **parity with the official pipeline semantics** that matter for Z-Image in practice:

   - steps semantics (Turbo’s “9 steps → ~8 effective updates” behavior)
   - CFG truncation / CFG normalization
4. **tests/docs/examples** that prove it works and are runnable by contributors.

---

## 2) Add a model registry (Turbo vs Base vs future variants)

### Goal

Avoid “stringly-typed” model ids scattered across CLI/pipeline and ensure the defaults change when the user selects `Z-Image`.

### Work items

**A. Create a small model registry type**
Add a new file, e.g. `Sources/ZImage/Support/ZImageModelRegistry.swift`:

* `enum ZImageKnownModel`

  * `.zImageTurbo` → `"Tongyi-MAI/Z-Image-Turbo"`
  * `.zImage` → `"Tongyi-MAI/Z-Image"`
  * (optional) `.zImageTurbo8bit` → `"mzbac/z-image-turbo-8bit"` (existing tests use this)

* `struct ZImagePreset`

  * `recommendedSteps`
  * `recommendedGuidance`
  * `recommendedResolution` (keep current 1024 defaults for Turbo; for `Z-Image` you may prefer 1024 too, but document its recommended operating range)
  * `recommendedMaxSequenceLength` (default 512; document that 1024 is useful for long prompts)
  * `defaultNegativePrompt` (often `""`)
  * (optional) `recommendedCFGTruncation` / `recommendedCFGNormalization` defaults for “known” presets

**B. Update `ZImageRepository` / cache naming**
Right now `ZImageRepository.id` and `defaultCacheDirectory()` are Turbo-specific. Refactor to:

* keep Turbo as the default (to preserve current behavior),
* add a helper:

  * `static func defaultCacheDirectory(for modelId: String, base: URL = ...) -> URL`

    * e.g. `.../z-image-turbo` vs `.../z-image`

This prevents confusing cache layouts when users switch between `Z-Image-Turbo` and `Z-Image`.

**C. Make “variant reuse” aware of `Z-Image`**
Update `areZImageVariants(_:_:)` in:

* `ZImagePipeline.swift`
* `ZImageControlPipeline.swift`

so the base model id is included in the “same-family” check, allowing clean in-place weight swapping between `Z-Image-Turbo` and `Z-Image` (instead of forcing a full teardown/rebuild).

**D. Ensure defaults propagate through *all* entry points**

Default model id/revision and caching decisions currently flow through more than CLI + `ZImageRepository`.
Explicitly include these in the refactor checklist so Turbo-default leakage doesn’t persist:

* `Sources/ZImage/Pipeline/PipelineSnapshot.swift` (default model selection)
* `Sources/ZImage/Weights/ModelResolution.swift` (default model id/revision and cache lookup)

---

## 3) Make weight/index resolution robust to HF precision variants (bf16/fp16/…)

Today the “dynamic” weight resolvers still assume fixed filenames for index.json (e.g. `transformer/diffusion_pytorch_model.safetensors.index.json`).
This becomes fragile as soon as a repo contains multiple weight variants (bf16/fp16/etc.), because directory scans can accidentally select a mixed set.
`ModelResolution` also downloads `["*.safetensors", "*.json", "tokenizer/*"]`, which will pull **all** variants if present.

### Work items

**A. Introduce an explicit `weightsVariant`**

Add an optional `weightsVariant: String?` to the model selection/config surface area (CLI + library). This should map to HF’s “variant” concept (e.g. `"bf16"`).

**B. Make index selection variant-aware**

Update `ZImageFiles.resolveTransformerWeights` / `resolveTextEncoderWeights` (and VAE if needed) to accept `weightsVariant` and:

* Prefer `*.{weightsVariant}.safetensors.index.json` when `weightsVariant != nil`
* Fall back to the non-variant index filename if the variant index isn’t present
* When scanning directories, only accept candidates whose filenames match the chosen variant (to prevent mixing)

**C. Make downloads variant-aware**

Update `ModelResolution` to allow downloading only the selected variant:

* If `weightsVariant != nil`, use patterns like:

  * `"*.\(weightsVariant).safetensors"`
  * `"*.\(weightsVariant).safetensors.index.json"`
  * plus the required `*.json` configs + tokenizer files

* If multiple variants exist and the user didn’t specify one, keep a deterministic default (prefer non-variant) and log a warning listing discovered variants.

**D. Add guardrails**

If `weightsVariant` is set but required components don’t have matching weights (transformer/text_encoder/vae), fail with a clear error that points out the mismatch (avoid partial/mixed loads).

---

## 4) Verify and align step semantics with the reference Z-Image pipeline

Turbo’s model card guidance says `num_inference_steps=9` “results in ~8 DiT forwards”. The current Swift implementation runs exactly `request.steps` denoise iterations, and the current scheduler does not naturally terminate at `sigma=0`, so the last step is *not* a no-op.

The reference Diffusers pipeline forces the schedule to end at `sigma=0` (`scheduler.sigma_min = 0.0`) and appends a terminal sigma; this makes the last `dt` zero, so the final iteration doesn’t change latents (and can be skipped as a perf optimization without changing output).

### Work items

**A. Decide what the public “steps” semantic is**

* Keep CLI/library semantics aligned with Diffusers: `--steps` maps to `num_inference_steps`.
* Document that (for this scheduler) the last iteration can be a no-op when the schedule ends at `sigma=0`, which is why “9 steps → ~8 effective updates” can be true.

**B. Update the Swift scheduler/pipeline to match**

* Extend `FlowMatchEulerScheduler` to support an explicit terminal sigma (e.g. `sigmaMinOverride = 0.0`).
* In both `ZImagePipeline` and `ZImageControlPipeline`, construct the scheduler with terminal sigma = 0.0 (matching Diffusers pipeline behavior) unless there’s evidence Base needs different semantics.
* Optional optimization: if `dt == 0` on the last step, skip the transformer forward for that iteration. Call this out explicitly as “perf optimization; numerically equivalent because the update is a no-op.”

**C. Add a fast unit test for step semantics**

Add a unit test (no weights) that validates:

* scheduler produces `numInferenceSteps` timesteps and `numInferenceSteps + 1` sigmas
* terminal sigma is 0 and last `dt` is 0 when `sigmaMinOverride == 0`
* Turbo preset’s “9 steps → ~8 effective updates” statement is true under the chosen semantics

---

## 5) Add the missing `Z-Image` inference knobs: CFG truncation + CFG normalization

The official Z-Image pipeline supports:

* `cfg_truncation`: disables CFG after a normalized time threshold
* `cfg_normalization`: renormalizes/clamps the CFG output norm relative to the original “positive” prediction norm ([Hugging Face][1])

Even if the `Tongyi-MAI/Z-Image` model card’s minimal recommendation doesn’t require them, adding them makes your Swift port “real pipeline compatible”.

### Work items

**A. Extend request types**
Update `ZImageGenerationRequest` (and the corresponding control request if separate) to include:

* `cfgTruncation: Float?`

  * semantics: threshold in `[0, 1]` on `t_norm` (same normalization you already compute: `(1000 - t)/1000`)
  * default: `nil` (or `1.0`) meaning “never truncate”

* `cfgNormalization: Bool`

  * semantics: match Diffusers API (boolean). When enabled, apply the same renormalization logic as reference.
  * default: `false` (disabled)

* (optional, advanced) `cfgNormalizationFactor: Float?`

  * semantics: treat Diffusers’ bool as `k = 1.0`; allow advanced callers to override `k` if desired.
  * default: `nil` meaning “use `1.0` when `cfgNormalization == true`”

**B. Implement in both denoising loops**
You currently have duplicated denoising code in:

* `ZImagePipeline.generateCore(...)`
* `ZImageControlPipeline` (two denoise paths)

Implement the same logic in both:

1. compute `tNorm` (you already compute `normalizedTimestep`)
2. compute `currentGuidanceScale`:

   * start with `request.guidanceScale`
   * if `cfgTruncation != nil` and `tNorm > cfgTruncation` ⇒ set `currentGuidanceScale = 0`
3. `applyCFG = doCFG && currentGuidanceScale > 0`
4. if `applyCFG`:

   * run the 2× batch
   * compute `pred = pos + currentGuidanceScale * (pos - neg)` (this matches the official Z-Image pipeline behavior) ([Hugging Face][1])
   * if `cfgNormalization`:

     * `k = cfgNormalizationFactor ?? 1.0`
     * `ori = l2Norm(pos)` and `new = l2Norm(pred)` using a **global** vector norm over all elements (Diffusers uses `torch.linalg.vector_norm`).
     * `maxNew = ori * k`
     * if `new > maxNew`: `pred *= maxNew / (new + eps)` (`eps` to avoid division by zero) ([Hugging Face][1])

This gives you parity with the pipeline’s behavior and avoids edge-case “blow-ups” at high guidance.
