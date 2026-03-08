# Documentation

This folder is the reference set for the current `zimage.swift` codebase. Start with the root `README.md` if you want the shortest runnable path.

## Core Docs

1. [CLI.md](CLI.md)
   Build, run, and inspect `ZImageCLI`, including `control`, `quantize`, and `quantize-controlnet`.
2. [MODELS_AND_WEIGHTS.md](MODELS_AND_WEIGHTS.md)
   Model ids, local-path handling, Hugging Face cache behavior, AIO checkpoints, transformer overrides, and quantization manifests.
3. [ARCHITECTURE.md](ARCHITECTURE.md)
   Current code structure, data flow, and source-of-truth files.
4. [DEVELOPMENT.md](DEVELOPMENT.md)
   Build, test, lint, CI, and targeted validation workflows.
5. [dev_plans/ROADMAP.md](dev_plans/ROADMAP.md)
   Short prioritized list of next iterations.

## Read By Task

- Running the CLI or updating help text:
  - [CLI.md](CLI.md)
  - `Sources/ZImageCLI/main.swift`
- Model loading, caches, or safetensors behavior:
  - [MODELS_AND_WEIGHTS.md](MODELS_AND_WEIGHTS.md)
  - `Sources/ZImage/Weights/*`
- General code navigation:
  - [ARCHITECTURE.md](ARCHITECTURE.md)
- Build, CI, packaging, or release work:
  - [DEVELOPMENT.md](DEVELOPMENT.md)
  - `.github/workflows/ci.yml`
- Upstream checkpoint structure:
  - [z-image-turbo.md](z-image-turbo.md)
  - [z-image.md](z-image.md)

## Supporting Context

- [golden_checks.md](golden_checks.md): guidance for numerical-parity fixtures and diagnostics
- [context/zimage_runtime_precision_parity_report.md](context/zimage_runtime_precision_parity_report.md): current precision-parity findings
- [context/mlx_pytorch_bf16_inference_dtype_deep_dive.md](context/mlx_pytorch_bf16_inference_dtype_deep_dive.md): backend-level BF16 notes
- [context/precision_formats_on_apple_silicon.md](context/precision_formats_on_apple_silicon.md): broader Apple Silicon precision background

## Active Focused Plans

- [dev_plans/controlnet-memory-followup.md](dev_plans/controlnet-memory-followup.md): targeted follow-up for the control-path memory story
- [dev_plans/runtime_precision_parity_improvement_plan.md](dev_plans/runtime_precision_parity_improvement_plan.md): next precision-parity fixes under consideration

Completed investigations that still explain the current control-memory policy remain in:

- [debug_notes/control-context-memory-remediation.md](debug_notes/control-context-memory-remediation.md)
- [debug_notes/controlnet-memory-analysis.md](debug_notes/controlnet-memory-analysis.md)
- [dev_plans/control-context-memory-remediation.md](dev_plans/control-context-memory-remediation.md)

## Archive

Historical investigations and completed implementation plans live under [archive/](archive/README.md). They can still be useful, but they are not the source of truth for current behavior.
