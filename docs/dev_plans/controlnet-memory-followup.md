# ControlNet Memory Follow-Up

This is the current control-memory status note for the repo. It replaces the older "multi-phase plan" framing now that the main March 2026 fixes have landed.

## Current Status

The control pipeline already keeps the main pre-denoising lifecycle boundaries intentionally narrow:

- prompt embeddings are built before transformer and ControlNet loading
- the control and inpaint path uses an encoder-only VAE that is released after the typed control context is materialized
- MLX cache is cleared before denoiser modules are loaded
- the final image decode uses a decoder-only VAE
- `--log-control-memory` exposes the supported runtime probe for those boundaries

Measured follow-up result retained from the March 8, 2026 high-resolution rerun:

- `prompt-embeddings.after-clear-cache`: `6.26 GiB` resident after deferred loading
- `control-context.after-clear-cache`: about `315 MiB`
- `transformer.denoising-load.after-apply`: about `23.24 GiB` resident
- `controlnet.denoising-load.after-apply`: about `29.49 GiB` resident
- `/usr/bin/time -l` peak memory footprint: about `59.3 GiB`

The practical takeaway is that the stored control context is no longer the dominant limiter. The remaining large jump is the live denoiser residency, not control-context storage.

## What Is Still Worth Doing

Only reopen this area if one of these is true:

- a regression changes the control-memory markers materially
- a target machine or workflow still cannot tolerate the retained `1536x2304` high-resolution footprint
- a larger loader or denoiser refactor changes residency policy and needs fresh measurements

The next meaningful reduction would need to target denoiser residency or offload policy. Tiled control or inpaint VAE encode is not the default next step for the current measured workload.

## Validation Recipe

Repo verification:

```bash
swift test
./scripts/build.sh
```

High-resolution probe:

```bash
.build/xcode/Build/Products/Release/ZImageCLI control \
  --prompt "memory validation" \
  --control-image images/canny.jpg \
  --controlnet-weights alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --control-file Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors \
  --width 1536 \
  --height 2304 \
  --steps 1 \
  --guidance 0 \
  --seed 1234 \
  --log-control-memory \
  --no-progress \
  --output /tmp/zimage-control-memory-check.png
```

Watch these markers:

- `prompt-embeddings.after-clear-cache`
- `control-context.after-baseline-reduction`
- `control-context.after-clear-cache`
- `transformer.denoising-load.after-apply`
- `controlnet.denoising-load.after-apply`
- `denoising.before-start`
- `decode.after-eval`

For the phase-by-phase March 2026 implementation history, use the historical notes under `docs/debug_notes/`.
