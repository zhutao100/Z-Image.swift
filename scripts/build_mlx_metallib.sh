#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build and install MLX Metal shader library (mlx.metallib) next to SwiftPM-built executables.

Why:
  mlx-swift expects a colocated 'mlx.metallib' (or a SwiftPM bundle 'default.metallib').
  When building via SwiftPM, the metallib is not produced automatically.

Usage:
  scripts/build_mlx_metallib.sh [-c debug|release]
EOF
}

config="debug"
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configuration)
      config="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ "$config" != "debug" && "$config" != "release" ]]; then
  echo "Invalid configuration: $config (expected 'debug' or 'release')" >&2
  exit 2
fi

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

bin_path="$(swift build -c "$config" --show-bin-path)"
mkdir -p $bin_path

mlx_checkout="$root_dir/.build/checkouts/mlx-swift"
mlx_src="$mlx_checkout/Source/Cmlx/mlx"
kernels_dir="$mlx_src/mlx/backend/metal/kernels"

if [[ ! -d "$kernels_dir" ]]; then
  echo "mlx-swift kernel sources not found at: $kernels_dir" >&2
  echo "Run 'swift build' first to populate .build/checkouts." >&2
  exit 1
fi

out_dir="$root_dir/.build/mlx-metallib/$config"
rm -rf "$out_dir"
mkdir -p "$out_dir"

metal_flags=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
  -mmacosx-version-min=14.0
)

# Match the MLX CMake "nojit" metallib set (excluding fence.metal, which requires Metal 3.2+).
kernels=(
  arg_reduce
  conv
  gemv
  layer_norm
  random
  rms_norm
  rope
  scaled_dot_product_attention

  arange
  binary
  binary_two
  copy
  fft
  reduce
  quantized
  fp_quantized
  scan
  softmax
  logsumexp
  sort
  ternary
  unary

  "steel/conv/kernels/steel_conv"
  "steel/conv/kernels/steel_conv_general"

  "steel/gemm/kernels/steel_gemm_fused"
  "steel/gemm/kernels/steel_gemm_gather"
  "steel/gemm/kernels/steel_gemm_masked"
  "steel/gemm/kernels/steel_gemm_splitk"
  "steel/gemm/kernels/steel_gemm_segmented"

  gemv_masked
  "steel/attn/kernels/steel_attention"
)

air_files=()
for kernel in "${kernels[@]}"; do
  src="$kernels_dir/$kernel.metal"
  if [[ ! -f "$src" ]]; then
    echo "Missing kernel source: $src" >&2
    exit 1
  fi

  air="$out_dir/$(basename "$kernel").air"
  echo "metal: $kernel"
  xcrun -sdk macosx metal "${metal_flags[@]}" -c "$src" -I"$mlx_src" -o "$air"
  air_files+=("$air")
done

out_lib="$bin_path/mlx.metallib"
echo "metallib: $out_lib"
xcrun -sdk macosx metallib "${air_files[@]}" -o "$out_lib"

echo "OK: wrote $out_lib"
