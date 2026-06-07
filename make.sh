#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT/output"

usage() {
  cat <<EOF
Usage: $0 <test/cpp_test/SomeTest.cpp>

Build a CUDA GTest binary and write it to output/<name>.

Examples:
  $0 test/cpp_test/cuRmsNormCUDATest.cpp
  $0 test/cpp_test/cuReorderTest.cpp
  $0 test/cpp_test/cuDeepSeekUnPermutationCUDATest.cpp

Environment:
  CUDA_ARCH   Override GPU arch (default: auto-detect via nvidia-smi, fallback sm_89)
  PYTHON      Python interpreter for PyTorch include/lib paths (default: python3)
EOF
  exit 1
}

detect_cuda_arch() {
  if [[ -n "${CUDA_ARCH:-}" ]]; then
    echo "$CUDA_ARCH"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    if [[ -n "$cap" ]]; then
      echo "sm_${cap//./}"
      return
    fi
  fi

  echo "sm_89"
}

needs_torch() {
  grep -q '#include <torch/torch.h>' "$1"
}

resolve_source() {
  local src="$1"

  if [[ ! -f "$src" && -f "$ROOT/$src" ]]; then
    src="$ROOT/$src"
  fi

  if [[ ! -f "$src" ]]; then
    echo "error: source file not found: $1" >&2
    exit 1
  fi

  if [[ "$src" != /* ]]; then
    src="$(cd "$(dirname "$src")" && pwd)/$(basename "$src")"
  fi

  if [[ "$src" != *.cpp ]]; then
    echo "error: expected a .cpp source file: $src" >&2
    exit 1
  fi

  echo "$src"
}

[[ $# -eq 1 ]] || usage

SRC="$(resolve_source "$1")"
BASENAME="$(basename "$SRC" .cpp)"
OUT="$OUTPUT_DIR/$BASENAME"
ARCH="$(detect_cuda_arch)"

mkdir -p "$OUTPUT_DIR"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found in PATH" >&2
  exit 1
fi

NVCC_FLAGS=(-x cu -std=c++17 -O3 "-arch=$ARCH")
INCLUDES=(-I"$ROOT" -I"$ROOT/kernel" -I/usr/include)
LIBS=(-L/usr/lib/x86_64-linux-gnu -lgtest -lgtest_main -lpthread)

echo "Building $BASENAME"
echo "  source : $SRC"
echo "  output : $OUT"
echo "  arch   : $ARCH"

if needs_torch "$SRC"; then
  PYTHON="${PYTHON:-python3}"

  if [[ -f "$ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  fi

  if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "error: $SRC requires PyTorch; activate .venv or install torch first" >&2
    exit 1
  fi

  mapfile -t TORCH_INCLUDES < <("$PYTHON" - <<'PY'
from torch.utils.cpp_extension import include_paths
for path in include_paths():
    print(path)
PY
)

  TORCH_LIB="$("$PYTHON" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

  for path in "${TORCH_INCLUDES[@]}"; do
    INCLUDES+=("-I$path")
  done

  LIBS=(-L"$TORCH_LIB" "${LIBS[@]}" -ltorch_cpu -lc10)

  nvcc "${NVCC_FLAGS[@]}" "$SRC" "${INCLUDES[@]}" "${LIBS[@]}" -o "$OUT"

  echo
  echo "Built: $OUT"
  echo "Run:"
  echo "  export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
  echo "  $OUT"
else
  nvcc "${NVCC_FLAGS[@]}" "$SRC" "${INCLUDES[@]}" "${LIBS[@]}" -o "$OUT"

  echo
  echo "Built: $OUT"
  echo "Run:"
  echo "  $OUT"
fi
