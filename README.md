# Inference Engine Kernel with CUDA Core Compute Libraries (CCCL)

Welcome to the inference engine kernel with CUDA Core Compute Libraries where my mission is to make inference operator kernel delightful.

This repository contains implementations of deep learning algorithm operators based on the CUDA CCCL library, including RmsNorm, AddRmsNorm, and Moe, etc.

All repo code can be run on WSL Ubuntu24.04 with Windows11

## How to setup environment

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libgtest-dev python3 python3-pip

python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu131
```

## How to build

```bash
cd /mnt/c/Users/xincu/cuInferenceEngine

TORCH_INC=$(python - <<'PY'
from torch.utils.cpp_extension import include_paths
print(" ".join(f"-I{x}" for x in include_paths()))
PY
)

TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
```
```bash
you can find the NVIDIA ARCH Info with the cmd following as
nvidia-smi --query-gpu=compute_cap --format=csv
```

```bash
nvcc -x cu -std=c++17 -O3 -arch=sm_89 \
  cuRmsNormCUDATest.cpp \
  -I. -I/usr/include \
  $TORCH_INC \
  -L"$TORCH_LIB" \
  -L/usr/lib/x86_64-linux-gnu \
  -ltorch_cpu -lc10 -lgtest -lgtest_main -lpthread \
  -o cuRmsNormCUDATest
```

## How to run

```bash
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# run sanity test
./cuRmsNormCUDATest --gtest_filter=brRmsNormCUDAKernelTest.sanity

# run regression test
./cuRmsNormCUDATest --gtest_filter=brRmsNormCUDAKernelTest.regression

# run perf test
./cuRmsNormCUDATest --gtest_filter=brRmsNormCUDAKernelTest.perf
```

