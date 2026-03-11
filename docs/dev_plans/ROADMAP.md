# Roadmap

This is the short, current priority list for the repo. It is intentionally small and should only contain work that still makes sense from the current codebase state.

## Near Term

1. **Reduce duplicated pipeline logic**
   - The largest overlap is still between `ZImagePipeline` and `ZImageControlPipeline`, plus the base/control transformer variants.
2. **Improve CLI ergonomics**
   - Better validation and clearer errors for missing required flags, bad values, and unsupported path forms
   - Revisit whether manual parsing in `Sources/ZImageCLI/main.swift` is still worth keeping
3. **Tighten model-resolution UX**
   - Clearer feedback for local-path mistakes, auth failures, and `--weights-variant` mismatches
4. **Harden local-path Base ergonomics**
   - Known ids get Base-aware presets, but local Base directories and unknown aliases still require explicit sampling flags

## Follow-On Work

5. **Finish the next precision-parity pass**
   - The remaining documented runtime gap is RoPE parity and the intermediate-tensor probes needed to validate it safely
6. **Expose more of the library-only control features in the CLI**
   - Control-path LoRA and prompt enhancement exist in the library request type but are not surfaced in `ZImageCLI control`
7. **Add a first-party app example**
   - The package declares an iOS library target, but the repo still has no maintained sample app
8. **Consider batch or multi-image generation**

## Ongoing Maintenance

- Keep `README.md`, `docs/CLI.md`, and CLI help text in sync.
- Use `ZImageCLI control --log-control-memory` with the `1536x2304` reference probe when changing control-memory-sensitive paths.
- Keep completed investigations clearly marked as historical so the active docs stay forward-looking.
