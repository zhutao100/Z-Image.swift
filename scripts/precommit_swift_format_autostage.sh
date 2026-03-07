#!/usr/bin/env bash
set -euo pipefail

echo "swift-format-autostage PID=$$ args=$*" >&2

# --- tool discovery ---
# Prefer an installed swift-format in PATH, otherwise use Xcode toolchain.
if command -v swift-format >/dev/null 2>&1; then
  SWIFT_FORMAT=(swift-format)
elif command -v xcrun >/dev/null 2>&1; then
  SWIFT_FORMAT_PATH="$(xcrun --find swift-format 2>/dev/null || true)"
  if [[ -n "${SWIFT_FORMAT_PATH}" ]]; then
    SWIFT_FORMAT=("${SWIFT_FORMAT_PATH}")
  else
    echo "error: swift-format not found (neither in PATH nor via xcrun). Install swift-format." >&2
    exit 2
  fi
else
  echo "error: swift-format not found (no swift-format in PATH, and xcrun unavailable)." >&2
  exit 2
fi

# --- file collection helpers (NUL-delimited for safety) ---
collect_staged_swift_files() {
  # staged A/C/M/R only; NUL-delimited
  git diff --cached --name-only -z --diff-filter=ACMR -- '*.swift' 2>/dev/null || true
}

collect_all_tracked_swift_files() {
  git ls-files -z -- '*.swift' 2>/dev/null || true
}

read_nul_list_into_array() {
  local -n _out="$1"
  _out=()
  while IFS= read -r -d '' f; do
    _out+=("$f")
  done
}

# Prefer staged files (commit flow). If none, fall back to all tracked (manual/all-files flow).
staged_nul="$(collect_staged_swift_files)"
declare -a files=()
if [[ -n "$staged_nul" ]]; then
  read_nul_list_into_array files < <(printf '%s' "$staged_nul")
else
  read_nul_list_into_array files < <(collect_all_tracked_swift_files)
fi

if [[ "${#files[@]}" -eq 0 ]]; then
  exit 0
fi

# --- formatting ---
# Optional: set SWIFTFORMAT_IGNORE_UNPARSABLE=1 to avoid blocking on WIP syntax breaks.
format_args=(format --in-place --parallel)
if [[ "${SWIFTFORMAT_IGNORE_UNPARSABLE:-0}" == "1" ]]; then
  format_args+=(--ignore-unparsable-files)
fi

# Chunk to avoid argv limits on very large repos.
run_in_chunks() {
  local -a cmd=("$@")  # includes tool + args, files appended per chunk
  local chunk_size=80
  local i=0
  while [[ $i -lt ${#files[@]} ]]; do
    "${cmd[@]}" "${files[@]:i:chunk_size}"
    i=$((i + chunk_size))
  done
}

run_in_chunks "${SWIFT_FORMAT[@]}" "${format_args[@]}"

# --- autostage (safe) ---
# pre-commit discourages modifying the staging area :contentReference[oaicite:3]{index=3},
# but since you want autostage, we must prevent index lock contention.
git_dir="$(git rev-parse --git-dir)"
lock_dir="${git_dir}/.swift-format-precommit.lockdir"

acquire_lock() {
  local attempts=0
  local max_attempts=200
  while ! mkdir "$lock_dir" 2>/dev/null; do
    attempts=$((attempts + 1))
    if [[ $attempts -ge $max_attempts ]]; then
      echo "error: could not acquire hook lock: $lock_dir" >&2
      return 1
    fi
    sleep 0.05
  done
  trap 'rmdir "$lock_dir" 2>/dev/null || true' EXIT
}

git_add_with_retry() {
  local -a delays=(0.05 0.05 0.1 0.1 0.2 0.2 0.4 0.4 0.8 0.8 1 1)
  local out
  for d in "${delays[@]}"; do
    if out="$(git add -- "${files[@]}" 2>&1)"; then
      return 0
    fi
    if [[ "$out" == *"index.lock"* ]]; then
      sleep "$d"
      continue
    fi
    echo "$out" >&2
    return 1
  done

  # last attempt (print output if still failing)
  if ! out="$(git add -- "${files[@]}" 2>&1)"; then
    echo "$out" >&2
    return 1
  fi
}

# Optional: set SWIFTFORMAT_AUTOSTAGE=0 to disable staging (useful in CI/manual runs).
if [[ "${SWIFTFORMAT_AUTOSTAGE:-1}" == "1" ]]; then
  acquire_lock
  git_add_with_retry
fi

# --- lint (strict gate) ---
lint_args=(lint --strict --parallel)
if [[ "${SWIFTFORMAT_IGNORE_UNPARSABLE:-0}" == "1" ]]; then
  lint_args+=(--ignore-unparsable-files)
fi

run_in_chunks "${SWIFT_FORMAT[@]}" "${lint_args[@]}"
