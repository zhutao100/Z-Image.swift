# Roadmap

This is the short, current priority list for the repo. It is intentionally small and should only contain work that still makes sense from the current codebase state.

## Near Term

1. **Reduce duplicated loading and denoising logic**
   - The main overlap is still between `ZImagePipeline` and `ZImageControlPipeline`, plus the base/control transformer variants.
2. **Improve CLI ergonomics**
   - Better validation and clearer errors for missing required flags and bad values
   - Revisit whether manual parsing in `Sources/ZImageCLI/main.swift` is still worth keeping
3. **Tighten model-resolution UX**
   - Clearer feedback for local-path mistakes, auth failures, and `--weights-variant` mismatches
4. **Harden local-path Base ergonomics**
   - Known `Tongyi-MAI` ids now get Base-aware presets, but local Base directories and unknown aliases still require explicit sampling flags

## Follow-On Work

5. **Expose more of the library-only control features in the CLI**
   - Control-path LoRA and prompt enhancement are implemented in the library request type but not yet surfaced in `ZImageCLI control`
6. **Add a first-party app example**
   - The package declares an iOS library target, but the repo still has no maintained sample app
7. **Consider batch or multi-image generation**

## Ongoing Maintenance

- Keep `README.md`, `docs/CLI.md`, and CLI help text in sync.
- Use `ZImageCLI control --log-control-memory` with the `1536x2304` reference probe when changing control-memory-sensitive paths.
- Archive completed plans instead of leaving them in `docs/dev_plans/` as if they were still active.
