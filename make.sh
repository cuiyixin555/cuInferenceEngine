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
  VERBOSE     Set to 1 to show nvcc verbose output (-v)
  TMPDIR      Temp directory for nvcc intermediates (default: /tmp/cuInferenceEngine-build)
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

find_kernel_cu() {
  local test_stem="${BASENAME%Test}"
  local kernel_cu="$ROOT/kernel/${test_stem}Kernel.cu"
  if [[ -f "$kernel_cu" ]]; then
    echo "$kernel_cu"
  fi
}

cuda_home() {
  if [[ -n "${CUDA_HOME:-}" ]]; then
    echo "$CUDA_HOME"
    return
  fi
  dirname "$(dirname "$(command -v nvcc)")"
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

run_nvcc() {
  local torch_build="$1"
  shift

  export TMPDIR="${TMPDIR:-/tmp/cuInferenceEngine-build}"
  mkdir -p "$TMPDIR"

  local nvcc_args=(-objtemp)
  if [[ "${VERBOSE:-0}" == "1" ]]; then
    nvcc_args+=(-v)
  fi

  echo "  tmpdir : $TMPDIR"
  if [[ "$torch_build" == "1" ]]; then
    echo "  note   : nvcc is compiling (PyTorch headers are large; first build often takes 10-20 min)"
    if [[ "$ROOT" == /mnt/* ]]; then
      echo "  tip    : /mnt/c/ on WSL is slow; clone the repo to ~/ for much faster builds"
    fi
    echo "  tip    : run with VERBOSE=1 to see nvcc progress"
  else
    echo "  note   : nvcc compiling..."
  fi

  nvcc "${nvcc_args[@]}" "$@"
}

build_cuda_split() {
  local kernel_cu="$1"
  local use_torch="$2"
  local obj_dir="${TMPDIR:-/tmp/cuInferenceEngine-build}/$BASENAME"
  mkdir -p "$obj_dir"

  local cuda_root
  cuda_root="$(cuda_home)"
  local host_includes=("${INCLUDES[@]}" "-I$cuda_root/include")
  local host_obj="$obj_dir/${BASENAME}.o"
  local kernel_obj="$obj_dir/$(basename "$kernel_cu" .cu).o"

  local cxx_flags=(-std=c++17 -O3 -fPIC)
  local nvcc_host_flags=(-Xcompiler -std=c++17 -Xcompiler -fPIC)
  local link_libs=(-L"$cuda_root/lib64" -L/usr/lib/x86_64-linux-gnu
    -lcudart -lgtest -lgtest_main -lpthread)

  if [[ "$use_torch" == "1" ]]; then
    local torch_cxx_abi
    torch_cxx_abi="$("$PYTHON" - <<'PY'
import torch
print(int(torch._C._GLIBCXX_USE_CXX11_ABI))
PY
)"
    cxx_flags+=("-D_GLIBCXX_USE_CXX11_ABI=${torch_cxx_abi}")
    nvcc_host_flags+=("-Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=${torch_cxx_abi}")
    link_libs=(-L"$TORCH_LIB" "${link_libs[@]}" -ltorch_cpu -ltorch -lc10)
    echo "  mode   : split build (g++ for PyTorch host code, nvcc for CUDA kernel)"
  else
    echo "  mode   : split build (g++ for host code, nvcc for CUDA kernel)"
  fi

  echo "  kernel : $kernel_cu"

  export TMPDIR="${TMPDIR:-/tmp/cuInferenceEngine-build}"
  mkdir -p "$TMPDIR"

  echo "  step 1 : nvcc compiling CUDA kernel..."
  nvcc -objtemp "${NVCC_FLAGS[@]}" "${nvcc_host_flags[@]}" -c "$kernel_cu" \
    "${INCLUDES[@]}" -o "$kernel_obj"

  echo "  step 2 : g++ compiling test..."
  g++ "${cxx_flags[@]}" -c "$SRC" "${host_includes[@]}" -o "$host_obj"

  echo "  step 3 : linking..."
  g++ "${cxx_flags[@]}" "$host_obj" "$kernel_obj" "${link_libs[@]}" -o "$OUT"
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

  LIBS=(-L"$TORCH_LIB" "${LIBS[@]}" -ltorch_cpu -ltorch -lc10)

  KERNEL_CU="$(find_kernel_cu || true)"
  if [[ -n "$KERNEL_CU" ]]; then
    build_cuda_split "$KERNEL_CU" 1
  else
    echo "  warning: no kernel/${BASENAME%Test}Kernel.cu found; falling back to single nvcc build"
    run_nvcc 1 "${NVCC_FLAGS[@]}" "$SRC" "${INCLUDES[@]}" "${LIBS[@]}" -o "$OUT"
  fi

  echo
  echo "Built: $OUT"
  echo "Run:"
  echo "  export LD_LIBRARY_PATH=\"$TORCH_LIB:\$LD_LIBRARY_PATH\""
  echo "  $OUT"
else
  KERNEL_CU="$(find_kernel_cu || true)"
  if [[ -n "$KERNEL_CU" ]]; then
    build_cuda_split "$KERNEL_CU" 0
  else
    run_nvcc 0 "${NVCC_FLAGS[@]}" "$SRC" "${INCLUDES[@]}" "${LIBS[@]}" -o "$OUT"
  fi

  echo
  echo "Built: $OUT"
  echo "Run:"
  echo "  $OUT"
fi
