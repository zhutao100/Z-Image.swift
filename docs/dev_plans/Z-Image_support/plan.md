## Recommended direction

This should be treated as a **“finish first-class Base support”** project, not a brand-new model port.

The current upstream `Tongyi-MAI/Z-Image` checkpoint still uses the same high-level Diffusers component graph as Turbo—`ZImagePipeline`, `ZImageTransformer2DModel`, `Qwen3Model`, `Qwen2Tokenizer`, and `AutoencoderKL`—so the Swift repo’s existing architecture is the right one to extend rather than fork. The important upstream behavioral deltas are that Base uses full CFG/negative prompting, recommends roughly **28–50** steps with guidance around **3–5**, and ships a scheduler config with `shift: 6.0`. ([Hugging Face][1])

There is also a current upstream parity target worth honoring: Diffusers’ `ZImagePipeline` for Base exposes `cfg_truncation` and `cfg_normalization`, explicitly sets `scheduler.sigma_min = 0.0`, and applies CFG as `pos + scale * (pos - neg)` with optional norm renormalization. ([GitHub][2])

## What the repo already has

The repo already contains most of the structural groundwork:

* `Sources/ZImage/Support/ZImageModelRegistry.swift` already knows both `Tongyi-MAI/Z-Image-Turbo` and `Tongyi-MAI/Z-Image`.
* `Sources/ZImage/Weights/ModelConfigs.swift` loads model behavior from snapshot configs rather than hardcoding Turbo-only architecture.
* `Sources/ZImage/Weights/ModelPaths.swift` already has dynamic shard resolution and `weightsVariant` handling.
* `README.md`, `docs/CLI.md`, and `docs/MODELS_AND_WEIGHTS.md` already acknowledge Base support.
* `docs/dev_plans/ROADMAP.md` says the main remaining near-term issue is that the CLI still boots with Turbo-oriented defaults.

So the missing work is mostly **productization, parity, and cleanup**.

## The main gaps

### 1. CLI behavior is still Turbo-biased

This is the biggest user-facing problem.

`Sources/ZImageCLI/main.swift` seeds width, height, steps, and guidance from `ZImageModelMetadata`, which is Turbo-oriented. That matches the repo docs, which explicitly say Base users must still pass `--steps` and `--guidance` manually.

### 2. Diffusers parity knobs for Base are missing

The library request types and pipelines do not currently expose or implement Base-oriented CFG controls like:

* `cfgTruncation`
* `cfgNormalization`
* optional normalization factor override

That is the most meaningful runtime gap versus upstream Base semantics. ([GitHub][2])

### 3. The Base fixtures are stale

The most obvious example is `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/...`:

* fixture transformer config says `dim: 4096`
* fixture scheduler says `shift: 3.0`

But the upstream `Z-Image` model and the HF files, show `dim: 3840` and `shift: 6.0`. ([Hugging Face][3])

That means part of the repo’s “Base support” test surface is currently validating against an outdated approximation, not the real checkpoint.

### 4. The upstream reference docs in the repo need re-baselining

Your local `docs/z-image.md` appears partially stale. One concrete example: it describes an example using `cfg_normalization=True`, while the current upstream model card example shows `cfg_normalization=False`. ([Hugging Face][4])

That is not a runtime blocker, but it will confuse future work unless cleaned up.

---

## Proposed implementation plan

## Phase 0 — Rebaseline against the real Base checkpoint

**Goal:** make the repo’s Base assumptions accurate before adding more behavior.

### Work

* Replace the `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/...` config fixtures with values copied from the `Z-Image` model.
* Update `SnapshotModelConfigsTests.swift` to assert the real Base values:

  * transformer `dim == 3840`
  * scheduler `shift == 6.0`
  * `useDynamicShifting == false`
* Add a shard-layout assertion for Base:

  * transformer is 2 shards
  * text encoder is 3 shards
* Audit `docs/z-image.md` against the current HF model, and mark any stale statements.

### Files

* `Tests/ZImageTests/Fixtures/Snapshots/ZImageBase/...`
* `Tests/ZImageTests/Config/SnapshotModelConfigsTests.swift`
* `Tests/ZImageTests/Weights/ModelPathsResolutionTests.swift`
* `docs/z-image.md`

### Exit criteria

* The repo’s Base fixtures reflect the actual upstream checkpoint layout and scheduler config.
* No code behavior change yet.

---

## Phase 1 — Make Base a first-class model selection in the CLI

**Goal:** `--model Tongyi-MAI/Z-Image` should behave sensibly without requiring users to manually override Turbo defaults.

### Work

* Refactor `Sources/ZImageCLI/main.swift` so it can distinguish:

  * values explicitly provided by the user
  * values still unset and eligible for model-aware defaults
* After parsing `--model`, apply `ZImagePreset.defaults(for:)` only for fields the user did **not** set:

  * width
  * height
  * steps
  * guidance
  * max sequence length if you want that to become model-aware later
* Apply the same policy to the `control` subcommand.
* Update help text and examples so the CLI no longer implies Turbo defaults universally.

### Files

* `Sources/ZImageCLI/main.swift`
* optionally `Sources/ZImage/Support/ZImageModelRegistry.swift` if preset shape needs extension
* `README.md`
* `docs/CLI.md`
* `docs/MODELS_AND_WEIGHTS.md`

### Exit criteria

* `ZImageCLI -m Tongyi-MAI/Z-Image -p "..."` defaults to Base-friendly steps/guidance.
* Turbo default behavior remains unchanged when `--model` is omitted.

---

## Phase 2 — Add Base parity knobs to the library and CLI

**Goal:** match upstream Base inference behavior closely enough that the Swift port is not “Base in name only.”

### Work

Add to both request types:

* `cfgTruncation: Float?`
* `cfgNormalization: Bool`
* `cfgNormalizationFactor: Float?` as an advanced override

Implement the logic in both:

* `Sources/ZImage/Pipeline/ZImagePipeline.swift`
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`

Refactor the shared math into `PipelineUtilities.swift` so Base and Control paths do not drift.

### Behavior to implement

* compute normalized timestep `t_norm`
* if CFG is active and `cfgTruncation` is set and `t_norm > cfgTruncation`, disable CFG for that step
* compute CFG as:

  * `pred = pos + currentGuidanceScale * (pos - neg)`
* if normalization is enabled:

  * compute global vector norm of `pos`
  * compute global vector norm of `pred`
  * clamp `pred` so its norm does not exceed `oriNorm * factor`

This mirrors the current Diffusers reference closely. ([GitHub][2])

### CLI surface

Add flags such as:

* `--cfg-truncation`
* `--cfg-normalization`
* optionally `--cfg-normalization-factor`

### Files

* `Sources/ZImage/Pipeline/ZImagePipeline.swift`
* `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
* `Sources/ZImage/Pipeline/PipelineUtilities.swift`
* `Sources/ZImageCLI/main.swift`
* `docs/CLI.md`
* `README.md`

### Exit criteria

* Base runs can express the same CFG controls as upstream Diffusers.
* Turbo remains unaffected when guidance is `0`.

---

## Phase 3 — Tighten scheduler and parity validation

**Goal:** confirm the current scheduler semantics are actually good enough for Base and Turbo, rather than rewriting them preemptively.

### Work

* Keep the current `FlowMatchEulerScheduler` unless parity tests show real divergence.
* Add explicit tests for:

  * Base config loading `shift == 6.0`
  * Turbo config loading its own scheduler behavior
  * final sigma equals `0`
  * timestep/sigma monotonicity
* Optionally add a tiny parity harness that compares Swift scheduler output against a saved Python reference trace for:

  * Base 28-step config
  * Turbo 9-step config

### Important note

The old archived plan treated terminal-sigma behavior as unfinished work. The current scheduler already appends a final `0` sigma and has tests around that. So I would not prioritize scheduler refactoring until a parity harness shows a real mismatch.

### Files

* `Tests/ZImageTests/Scheduler/FlowMatchSchedulerTests.swift`
* maybe new parity fixture data under `Tests/ZImageTests/Fixtures/`

### Exit criteria

* Scheduler behavior is explicitly validated for both Base and Turbo.
* No speculative scheduler rewrite.

---

## Phase 4 — Add real Base smoke coverage

**Goal:** ensure “supported” means “actually usable.”

### Work

* Add an env-gated Base integration test:

  * local/HF snapshot
  * conservative resolution
  * deterministic seed
  * verifies generation completes and output dimensions are correct
* Keep it out of default CI unless weights are present.

A reasonable pattern is:

* default fast unit tests stay cheap
* `ZIMAGE_RUN_BASE_INTEGRATION=1` enables the heavy test

### Files

* `Tests/ZImageIntegrationTests/PipelineIntegrationTests.swift`
* or a new `BaseModelIntegrationTests.swift`

### Exit criteria

* There is at least one path that exercises real Base loading and denoising.

---

## Phase 5 — Documentation cleanup and historical-plan retirement

**Goal:** remove ambiguity for future maintainers and agents.

### Work

* Update:

  * `README.md`
  * `docs/CLI.md`
  * `docs/MODELS_AND_WEIGHTS.md`
  * `docs/z-image.md`
* Move any now-obsolete archived assumptions into a short “historical only” note.
* Update `docs/dev_plans/ROADMAP.md` so Base support is no longer framed as “mostly missing” once phases 1–4 land.

### Exit criteria

* Docs match real behavior.
* No stale “Base fixture” or “Turbo-only default” surprises remain.

---

## Suggested PR breakdown

### PR 1

Fixture and docs rebaseline:

* Base snapshot fixtures
* snapshot tests
* upstream-reference doc sync

### PR 2

Model-aware CLI presets:

* text-to-image
* control subcommand
* help/docs updates

### PR 3

Base parity knobs:

* `cfgTruncation`
* `cfgNormalization`
* shared helper
* unit tests

### PR 4

Parity validation:

* scheduler/parity tests
* optional Python trace comparison

### PR 5

Integration + final docs:

* env-gated Base smoke test
* README / CLI / models docs cleanup

---

## Non-goals for this plan

I would explicitly keep these out of scope for the initial Base-support effort:

* separate Base-specific transformer/text encoder/VAE implementations
* `Z-Image-Omni-Base` or `Z-Image-Edit`
* major CLI parser replacement
* ControlNet feature expansion beyond what is necessary for Base parity

Those are follow-on projects.

---

## Bottom line

The shortest correct plan is:

1. **rebaseline stale Base fixtures/docs**
2. **make CLI defaults model-aware**
3. **add Diffusers-parity CFG knobs**
4. **validate Base with a real smoke test**

That gets you from “partially supported if you know the magic flags” to “first-class Base model support” without unnecessary architecture churn.

[1]: https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/model_index.json "model_index.json · Tongyi-MAI/Z-Image at main"
[2]: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py "diffusers/src/diffusers/pipelines/z_image/pipeline_z_image.py at main · huggingface/diffusers · GitHub"
[3]: https://huggingface.co/Tongyi-MAI/Z-Image/blob/main/transformer/config.json "transformer/config.json · Tongyi-MAI/Z-Image at main"
[4]: https://huggingface.co/Tongyi-MAI/Z-Image "Tongyi-MAI/Z-Image · Hugging Face"
