# AGENTS.md (LLM / Agent Guide)

This guide is for agents working inside `zimage.swift`. Keep changes grounded in the current codebase, prefer the smallest doc set needed for the task, and update the docs whenever user-visible behavior changes.

## Read First By Task

- Any task: `README.md`, then `docs/README.md`
- Triage or bugfix: relevant failing test file, then `docs/ARCHITECTURE.md`; if the issue is about model loading, also read `docs/MODELS_AND_WEIGHTS.md`
- CLI work: `docs/CLI.md` and `Sources/ZImageCLI/main.swift`
- Pipeline or feature work: `docs/ARCHITECTURE.md`, then the relevant files under `Sources/ZImage/Pipeline/`, `Sources/ZImage/Model/`, and `Sources/ZImage/Weights/`
- Release or packaging work: `docs/DEVELOPMENT.md`, `.github/workflows/ci.yml`, `scripts/build.sh`
- Docs-only work: `README.md`, `docs/README.md`, then the specific source-of-truth doc for the area you are editing
- Upstream model layout questions: `docs/z-image-turbo.md` and `docs/z-image.md`

## Repo Map

- `Sources/ZImageCLI/`
  - `main.swift`: CLI parsing, help text, and subcommands
- `Sources/ZImage/`
  - `Pipeline/`: `ZImagePipeline`, `ZImageControlPipeline`, scheduler wiring, snapshot helpers
  - `Model/`: Qwen text encoder, diffusion transformer, VAE
  - `Weights/`: cache lookup, Hugging Face download, safetensors reading, AIO detection, tensor mapping
  - `Quantization/`: quantization manifests and conversion commands
  - `LoRA/`: LoRA and LoKr loading/application
  - `Support/`: model metadata and model registry/presets
  - `Util/`: image I/O and control-memory telemetry
- `Tests/`
  - `ZImageTests/`: default unit-test suite
  - `ZImageIntegrationTests/`: heavier tests that require weights
  - `ZImageE2ETests/`: builds and runs the CLI
- `docs/`
  - core reference docs plus targeted plans, background notes, and archive material
- `scripts/`
  - build and developer helper scripts

## Current Source Of Truth

- CLI flags and help output: `Sources/ZImageCLI/main.swift`
- Text-to-image API: `Sources/ZImage/Pipeline/ZImagePipeline.swift`
- ControlNet and inpainting API: `Sources/ZImage/Pipeline/ZImageControlPipeline.swift`
- Known model ids and presets: `Sources/ZImage/Support/ZImageModelRegistry.swift`
- Default model id and weight-file resolution: `Sources/ZImage/Weights/ModelPaths.swift`
- Snapshot resolution, cache lookup, and Hugging Face download behavior: `Sources/ZImage/Weights/ModelResolution.swift` and `Sources/ZImage/Weights/HuggingFaceHub.swift`
- AIO checkpoint detection and canonicalization: `Sources/ZImage/Weights/AIOCheckpoint.swift`
- Quantization manifests and application: `Sources/ZImage/Quantization/ZImageQuantization.swift` and `Sources/ZImage/Weights/WeightsMapping.swift`
- CI build and release packaging: `.github/workflows/ci.yml`
- Formatting and local hooks: `.pre-commit-config.yaml` and `scripts/precommit_swift_format_autostage.sh`

## Conventions And Expectations

- Keep changes small and targeted unless the code clearly needs a broader refactor.
- Check for parallel implementations before stopping:
  - `ZImagePipeline` and `ZImageControlPipeline`
  - `ZImageTransformer2DModel` and `ZImageControlTransformer2DModel`
  - CLI help text and docs
- When changing user-visible behavior, update the matching docs in the same change:
  - CLI flags or examples: `README.md`, `docs/CLI.md`, `Sources/ZImageCLI/main.swift`
  - Model loading semantics: `docs/MODELS_AND_WEIGHTS.md`, `docs/ARCHITECTURE.md`, relevant files in `Sources/ZImage/Weights/`
  - Build/test/release workflow: `docs/DEVELOPMENT.md`, `.github/workflows/ci.yml`, and helper scripts
- Prefer `docs/` as the detailed explanation layer. Keep `README.md` short and link outward.
- Do not treat `docs/archive/` as current truth. It is historical context only.
- Do not assume the CLI applies model-aware presets automatically. Today its defaults are Turbo-oriented even when `--model Tongyi-MAI/Z-Image` is used.

## Build, Test, And Lint

Build the release CLI:

```bash
./scripts/build.sh
```

CI-equivalent explicit build:

```bash
xcodebuild build -scheme ZImageCLI -configuration Release -destination 'platform=macOS' -derivedDataPath ./dist -skipPackagePluginValidation ENABLE_PLUGIN_PREPAREMLSHADERS=YES CLANG_COVERAGE_MAPPING=NO
```

Run the default verification suite:

```bash
xcodebuild test -scheme zimage.swift-Package -destination 'platform=macOS' -enableCodeCoverage NO -only-testing:ZImageTests
```

## CI And Release Expectations

- GitHub Actions builds a macOS release binary on pushes to `main`
- The workflow currently publishes or updates the `nightly` prerelease with `zimage.macos.arm64.zip`
- The build uses Xcode 16.0 and enables the MLX shader-preparation plugin in non-interactive mode

If you change packaging, artifact names, build flags, or release behavior, update `docs/DEVELOPMENT.md` and `.github/workflows/ci.yml` together.

## Focused Workflows

- Control-memory work: start with `docs/DEVELOPMENT.md`, then `docs/dev_plans/controlnet-memory-followup.md`, `docs/debug_notes/control-context-memory-remediation.md`, and `Sources/ZImage/Util/ControlMemoryTelemetry.swift`
- Precision or numerical-parity work: read `docs/golden_checks.md` and the notes under `docs/context/`
- Model-loading bugs: inspect `Tests/ZImageTests/Weights/*` before changing resolver logic
- Base vs Turbo behavior: inspect `Sources/ZImage/Support/ZImageModelRegistry.swift` and `Tests/ZImageTests/Support/ZImageModelRegistryTests.swift`

## External Reference

Some implementations were validated against Hugging Face Diffusers:

- Local checkout: `~/workspace/custom-builds/diffusers`
- Upstream: `https://github.com/huggingface/diffusers`

Use the Diffusers reference when checking weight naming, scheduler behavior, or pipeline parity. When running Python tooling for that comparison, use `PYENV_VERSION=venv313 pyenv exec ...`.

## Useful Local Resources

- Inspect `.safetensors` contents:
  - `~/bin/stls.py --format toon <file.safetensors>`
- Common Hugging Face snapshot roots:
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots`
  - `~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/snapshots`
