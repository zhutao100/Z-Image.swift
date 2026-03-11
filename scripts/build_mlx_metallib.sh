#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

usage() {
  cat <<'USAGE'
Build and install MLX Metal shader library (mlx.metallib) for SwiftPM-built executables.

Why:
  mlx-swift's SwiftPM build excludes the Metal kernel sources and nojit_kernels.cpp,
  so plain `swift build` / `swift test` does not produce the default MLX metallib.
  This script builds a no-JIT metallib from the vendored MLX kernel sources and places
  it next to the SwiftPM binary path as `mlx.metallib`, which current MLX will load first.

Usage:
  scripts/build_mlx_metallib.sh [options]

Options:
  -c, --configuration <debug|release>  Build configuration (default: debug)
      --project-root <path>            Swift package root (defaults to detected root)
      --mlx-swift-path <path>          Explicit mlx-swift checkout/path override
      --bin-path <path>                Explicit output directory for SwiftPM binaries
      --output <path>                  Explicit output metallib path
      --deployment-target <version>    macOS deployment target (minimum: 14.0)
      --force                          Rebuild even if inputs are unchanged
      --clean                          Remove cached intermediates before building
      --verbose                        Print tool invocations
      --debug-metal                    Add Metal source/line debug info
      --print-plan                     Print the selected build plan and exit
  -h, --help                           Show this help

Examples:
  scripts/build_mlx_metallib.sh
  scripts/build_mlx_metallib.sh -c release --force
  scripts/build_mlx_metallib.sh --mlx-swift-path ../mlx-swift
USAGE
}

log() {
  printf '%s\n' "$*"
}

warn() {
  printf 'warning: %s\n' "$*" >&2
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run() {
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    printf '+ ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
  fi
  "$@"
}

canonical_dir() {
  (
    cd "$1" >/dev/null 2>&1
    pwd -P
  )
}

version_ge() {
  # Returns success when $1 >= $2 for dotted numeric versions.
  # Compatible with macOS's bash 3.2.
  local lhs="$1"
  local rhs="$2"
  local lhs_parts rhs_parts idx lhs_val rhs_val IFS=.
  read -r -a lhs_parts <<< "$lhs"
  read -r -a rhs_parts <<< "$rhs"

  idx=0
  while :; do
    lhs_val=0
    rhs_val=0
    if [[ $idx -lt ${#lhs_parts[@]} ]]; then
      lhs_val="${lhs_parts[$idx]}"
    fi
    if [[ $idx -lt ${#rhs_parts[@]} ]]; then
      rhs_val="${rhs_parts[$idx]}"
    fi

    lhs_val="${lhs_val%%[^0-9]*}"
    rhs_val="${rhs_val%%[^0-9]*}"
    [[ -n "$lhs_val" ]] || lhs_val=0
    [[ -n "$rhs_val" ]] || rhs_val=0

    if (( lhs_val > rhs_val )); then
      return 0
    fi
    if (( lhs_val < rhs_val )); then
      return 1
    fi

    if (( idx >= ${#lhs_parts[@]} - 1 && idx >= ${#rhs_parts[@]} - 1 )); then
      return 0
    fi
    idx=$((idx + 1))
  done
}

hash_stdin() {
  if command_exists shasum; then
    shasum -a 256 | awk '{print $1}'
  elif command_exists sha256sum; then
    sha256sum | awk '{print $1}'
  elif command_exists openssl; then
    openssl dgst -sha256 -r | awk '{print $1}'
  else
    die "No SHA-256 tool available (need shasum, sha256sum, or openssl)."
  fi
}

hash_file() {
  local file="$1"
  if command_exists shasum; then
    shasum -a 256 "$file" | awk '{print $1}'
  elif command_exists sha256sum; then
    sha256sum "$file" | awk '{print $1}'
  elif command_exists openssl; then
    openssl dgst -sha256 -r "$file" | awk '{print $1}'
  else
    die "No SHA-256 tool available (need shasum, sha256sum, or openssl)."
  fi
}

cleanup() {
  local exit_code=$?
  if [[ -n "${TMP_DIR:-}" && -d "${TMP_DIR:-}" ]]; then
    rm -rf "$TMP_DIR"
  fi
  if [[ -n "${LOCK_DIR:-}" && -d "${LOCK_DIR:-}" ]]; then
    rmdir "$LOCK_DIR" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT

MIN_DEPLOYMENT_TARGET="14.0"
project_root=""
mlx_swift_path=""
bin_path=""
output_path=""
config="debug"
deployment_target=""
deployment_target_source=""
FORCE=0
CLEAN=0
VERBOSE=0
DEBUG_METAL=0
PRINT_PLAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configuration)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      config="$2"
      shift 2
      ;;
    --project-root)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      project_root="$2"
      shift 2
      ;;
    --mlx-swift-path)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      mlx_swift_path="$2"
      shift 2
      ;;
    --bin-path)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      bin_path="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      output_path="$2"
      shift 2
      ;;
    --deployment-target)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      deployment_target="$2"
      deployment_target_source="--deployment-target"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    --debug-metal)
      DEBUG_METAL=1
      shift
      ;;
    --print-plan)
      PRINT_PLAN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

case "$config" in
  debug|release) ;;
  *) die "Invalid configuration: $config (expected debug or release)" ;;
esac

validate_version_string() {
  local value="$1"
  local label="$2"
  [[ "$value" =~ ^[0-9]+([.][0-9]+)*$ ]] || die "Invalid $label: $value"
}

resolve_deployment_target() {
  local requested="$deployment_target"
  local source="$deployment_target_source"

  if [[ -z "$requested" ]]; then
    if [[ -n "${SWIFTPM_MLX_METAL_DEPLOYMENT_TARGET:-}" ]]; then
      requested="$SWIFTPM_MLX_METAL_DEPLOYMENT_TARGET"
      source="SWIFTPM_MLX_METAL_DEPLOYMENT_TARGET"
    elif [[ -n "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
      requested="$MACOSX_DEPLOYMENT_TARGET"
      source="MACOSX_DEPLOYMENT_TARGET"
    else
      requested="$MIN_DEPLOYMENT_TARGET"
      source="default"
    fi
  fi

  validate_version_string "$requested" "deployment target"
  if ! version_ge "$requested" "$MIN_DEPLOYMENT_TARGET"; then
    if [[ "$source" == "MACOSX_DEPLOYMENT_TARGET" ]]; then
      warn "Ignoring MACOSX_DEPLOYMENT_TARGET=$requested because MLX requires macOS $MIN_DEPLOYMENT_TARGET or newer; using $MIN_DEPLOYMENT_TARGET"
      requested="$MIN_DEPLOYMENT_TARGET"
      source="default"
    else
      die "Unsupported deployment target $requested; MLX requires macOS $MIN_DEPLOYMENT_TARGET or newer"
    fi
  fi

  deployment_target="$requested"
  DEPLOYMENT_TARGET_SOURCE="$source"
}

validate_sdk_support() {
  validate_version_string "$SDK_VERSION" "macOS SDK version"
  if ! version_ge "$SDK_VERSION" "$MIN_DEPLOYMENT_TARGET"; then
    die "Installed macOS SDK $SDK_VERSION is too old; MLX requires SDK $MIN_DEPLOYMENT_TARGET or newer"
  fi
  if ! version_ge "$SDK_VERSION" "$deployment_target"; then
    die "Deployment target $deployment_target exceeds installed macOS SDK $SDK_VERSION"
  fi
}

detect_project_root() {
  local script_path script_dir script_parent git_root
  script_path="${BASH_SOURCE[0]}"
  script_dir="$(canonical_dir "$(dirname "$script_path")")"
  script_parent="$(canonical_dir "$script_dir/..")"

  if [[ -f "$script_parent/Package.swift" ]]; then
    printf '%s\n' "$script_parent"
    return 0
  fi
  if [[ -f "$script_dir/Package.swift" ]]; then
    printf '%s\n' "$script_dir"
    return 0
  fi
  if command_exists git; then
    git_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
    if [[ -n "$git_root" && -f "$git_root/Package.swift" ]]; then
      printf '%s\n' "$git_root"
      return 0
    fi
  fi
  printf '%s\n' "$(pwd -P)"
}

is_mlx_swift_root() {
  local candidate="$1"
  [[ -d "$candidate" ]] || return 1
  [[ -f "$candidate/Package.swift" ]] || return 1
  [[ -d "$candidate/Source/Cmlx/mlx/mlx/backend/metal/kernels" ]] || return 1
  grep -q 'name:[[:space:]]*"mlx-swift"' "$candidate/Package.swift" 2>/dev/null
}

find_mlx_swift_root() {
  local root="$1"
  local candidate package_file
  local candidate_bases

  for candidate in \
    "$root/.build/checkouts/mlx-swift" \
    "$root/.swiftpm/checkouts/mlx-swift" \
    "$root/mlx-swift" \
    "$root/../mlx-swift"
  do
    if is_mlx_swift_root "$candidate"; then
      printf '%s\n' "$(canonical_dir "$candidate")"
      return 0
    fi
  done

  for candidate_bases in "$root/.build/checkouts" "$root/.swiftpm/checkouts"; do
    if [[ -d "$candidate_bases" ]]; then
      while IFS= read -r package_file; do
        candidate="$(dirname "$package_file")"
        if is_mlx_swift_root "$candidate"; then
          printf '%s\n' "$(canonical_dir "$candidate")"
          return 0
        fi
      done < <(find "$candidate_bases" -mindepth 2 -maxdepth 2 -type f -name Package.swift 2>/dev/null | LC_ALL=C sort)
    fi
  done

  return 1
}

probe_metal_version() {
  local result
  if ! command_exists xcrun; then
    die "xcrun not found; Xcode command line tools are required"
  fi
  result="$(printf '__METAL_VERSION__\n' | xcrun -sdk macosx metal -E -x metal -P "-mmacosx-version-min=$deployment_target" - 2>/dev/null | tail -n 1 | tr -d '[:space:]')"
  if [[ "$result" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$result"
  else
    warn "Unable to determine __METAL_VERSION__; assuming 0"
    printf '0\n'
  fi
}

append_kernel_if_present() {
  local kernel="$1"
  local src="$KERNELS_DIR/$kernel.metal"
  if [[ -f "$src" ]]; then
    KERNELS+=("$kernel")
  else
    warn "Skipping optional kernel that is not present in this MLX snapshot: $kernel"
  fi
}

select_kernels() {
  KERNELS=(
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
    steel/conv/kernels/steel_conv
    steel/conv/kernels/steel_conv_general
    steel/gemm/kernels/steel_gemm_fused
    steel/gemm/kernels/steel_gemm_gather
    steel/gemm/kernels/steel_gemm_masked
    steel/gemm/kernels/steel_gemm_splitk
    steel/gemm/kernels/steel_gemm_segmented
    gemv_masked
    steel/attn/kernels/steel_attention
  )

  append_kernel_if_present "steel_conv_3d"

  if version_ge "$METAL_VERSION" "320"; then
    append_kernel_if_present "fence"
  fi

  if version_ge "$METAL_VERSION" "400" && version_ge "$SDK_VERSION" "26.2"; then
    append_kernel_if_present "steel/gemm/kernels/steel_gemm_fused_nax"
    append_kernel_if_present "steel/gemm/kernels/steel_gemm_gather_nax"
    append_kernel_if_present "steel/gemm/kernels/steel_gemm_splitk_nax"
    append_kernel_if_present "quantized_nax"
    append_kernel_if_present "fp_quantized_nax"
    append_kernel_if_present "steel/attn/kernels/steel_attention_nax"
  fi
}

validate_required_kernels() {
  local kernel src missing=0
  for kernel in "${KERNELS[@]}"; do
    src="$KERNELS_DIR/$kernel.metal"
    if [[ ! -f "$src" ]]; then
      printf 'missing kernel source: %s\n' "$src" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || die "One or more kernel sources are missing"
}

print_plan() {
  local kernel
  log "project_root=$PROJECT_ROOT"
  log "mlx_swift_root=$MLX_SWIFT_ROOT"
  log "mlx_src=$MLX_SRC"
  log "kernels_dir=$KERNELS_DIR"
  log "config=$config"
  log "deployment_target=$deployment_target"
  log "deployment_target_source=$DEPLOYMENT_TARGET_SOURCE"
  log "sdk_version=$SDK_VERSION"
  log "metal_version=$METAL_VERSION"
  log "bin_path=$BIN_PATH"
  log "output_path=$OUTPUT_PATH"
  log "cache_dir=$CACHE_DIR"
  log "kernels:"
  for kernel in "${KERNELS[@]}"; do
    log "  - $kernel"
  done
}

fingerprint() {
  local extra_files file
  {
    printf 'config=%s\n' "$config"
    printf 'deployment_target=%s\n' "$deployment_target"
    printf 'sdk_version=%s\n' "$SDK_VERSION"
    printf 'metal_version=%s\n' "$METAL_VERSION"
    printf 'debug_metal=%s\n' "$DEBUG_METAL"
    printf 'metal_flags='
    printf '%s|' "${METAL_FLAGS[@]}"
    printf '\n'
    printf 'kernels='
    printf '%s|' "${KERNELS[@]}"
    printf '\n'
    printf 'tools.metal=%s\n' "$(xcrun -f metal)"
    printf 'tools.metallib=%s\n' "$(xcrun -f metallib)"
    printf 'script=%s\n' "$(hash_file "$0")"

    extra_files=(
      "$MLX_SWIFT_ROOT/Package.swift"
      "$MLX_SRC/mlx/backend/metal/kernels/CMakeLists.txt"
    )
    for file in "${extra_files[@]}"; do
      if [[ -f "$file" ]]; then
        printf '%s %s\n' "$(hash_file "$file")" "$file"
      fi
    done

    while IFS= read -r file; do
      [[ -n "$file" ]] || continue
      printf '%s %s\n' "$(hash_file "$file")" "$file"
    done < <(find "$KERNELS_DIR" -type f \( -name '*.metal' -o -name '*.h' \) -print | LC_ALL=C sort)
  } | hash_stdin
}

build_kernel_air() {
  local kernel="$1"
  local src="$KERNELS_DIR/$kernel.metal"
  local safe_name="${kernel//\//__}"
  local air="$TMP_DIR/$safe_name.air"

  run xcrun -sdk macosx metal "${METAL_FLAGS[@]}" -c "$src" -I"$MLX_SRC" -o "$air"
  AIR_FILES+=("$air")
}

PROJECT_ROOT="${project_root:-$(detect_project_root)}"
[[ -d "$PROJECT_ROOT" ]] || die "Project root does not exist: $PROJECT_ROOT"
PROJECT_ROOT="$(canonical_dir "$PROJECT_ROOT")"
[[ -f "$PROJECT_ROOT/Package.swift" ]] || die "No Package.swift found in project root: $PROJECT_ROOT"
resolve_deployment_target

if [[ -n "$mlx_swift_path" ]]; then
  is_mlx_swift_root "$mlx_swift_path" || die "Invalid --mlx-swift-path: $mlx_swift_path"
  MLX_SWIFT_ROOT="$(canonical_dir "$mlx_swift_path")"
else
  MLX_SWIFT_ROOT="$(find_mlx_swift_root "$PROJECT_ROOT" || true)"
  [[ -n "$MLX_SWIFT_ROOT" ]] || die "Unable to locate mlx-swift checkout. Pass --mlx-swift-path explicitly."
fi

MLX_SRC="$MLX_SWIFT_ROOT/Source/Cmlx/mlx"
KERNELS_DIR="$MLX_SRC/mlx/backend/metal/kernels"
[[ -d "$KERNELS_DIR" ]] || die "Kernel source directory not found: $KERNELS_DIR"

if [[ -n "$bin_path" ]]; then
  BIN_PATH="$bin_path"
else
  BIN_PATH="$(cd "$PROJECT_ROOT" && swift build -c "$config" --show-bin-path)"
fi
BIN_PATH="$(mkdir -p "$BIN_PATH" && canonical_dir "$BIN_PATH")"

if [[ -n "$output_path" ]]; then
  case "$output_path" in
    /*) OUTPUT_PATH="$output_path" ;;
    *) OUTPUT_PATH="$PROJECT_ROOT/$output_path" ;;
  esac
else
  OUTPUT_PATH="$BIN_PATH/mlx.metallib"
fi

OUTPUT_DIR="$(dirname "$OUTPUT_PATH")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(canonical_dir "$OUTPUT_DIR")"
OUTPUT_PATH="$OUTPUT_DIR/$(basename "$OUTPUT_PATH")"

CACHE_DIR="$PROJECT_ROOT/.build/mlx-metallib/$config"
STAMP_FILE="$CACHE_DIR/fingerprint.sha256"
MANIFEST_FILE="$CACHE_DIR/build-plan.txt"
LOCK_DIR="$CACHE_DIR/.lock"

if (( CLEAN == 1 )) && [[ -d "$CACHE_DIR" ]]; then
  rm -rf "$CACHE_DIR"
fi
mkdir -p "$CACHE_DIR"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  die "Another metallib build appears to be in progress for $config"
fi

SDK_VERSION="$(xcrun -sdk macosx --show-sdk-version)"
validate_sdk_support
METAL_VERSION="$(probe_metal_version)"

METAL_FLAGS=(
  -x
  metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
  "-mmacosx-version-min=$deployment_target"
)

if [[ "$config" == "debug" ]] && version_ge "$METAL_VERSION" "320"; then
  METAL_FLAGS+=( -fmetal-enable-logging )
fi
if (( DEBUG_METAL == 1 )); then
  METAL_FLAGS+=( -gline-tables-only -frecord-sources )
fi

select_kernels
validate_required_kernels

print_plan > "$MANIFEST_FILE"
if (( PRINT_PLAN == 1 )); then
  cat "$MANIFEST_FILE"
  exit 0
fi

FINGERPRINT="$(fingerprint)"
if (( FORCE == 0 )) \
  && [[ -f "$OUTPUT_PATH" ]] \
  && [[ -f "$STAMP_FILE" ]] \
  && [[ "$(cat "$STAMP_FILE")" == "$FINGERPRINT" ]]; then
  log "Up to date: $OUTPUT_PATH"
  exit 0
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-metallib.XXXXXX")"
AIR_FILES=()

log "Building mlx.metallib"
log "  config: $config"
log "  sdk: $SDK_VERSION"
log "  metal version: $METAL_VERSION"
log "  output: $OUTPUT_PATH"

for kernel in "${KERNELS[@]}"; do
  log "  metal: $kernel"
  build_kernel_air "$kernel"
done

TMP_OUTPUT="$OUTPUT_DIR/.mlx.metallib.tmp.$$"
run xcrun -sdk macosx metallib "${AIR_FILES[@]}" -o "$TMP_OUTPUT"
mv -f "$TMP_OUTPUT" "$OUTPUT_PATH"
printf '%s\n' "$FINGERPRINT" > "$STAMP_FILE"

log "OK: wrote $OUTPUT_PATH"
