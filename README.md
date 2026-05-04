# Inference Engine Kernel with CUDA Core Compute Libraries (CCCL)

Welcome to the inference engine kernel with CUDA Core Compute Libraries where my mission is to make inference operator kernel delightful.

This repository contains implementations of deep learning algorithm operators based on the CUDA CCCL library, including RmsNorm, AddRmsNorm, and Moe, etc.

All repo code can be run on WSL Ubuntu24.04 with Windows11

## How to setup environment

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build libgtest-dev python3 python3-pip

python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu131

TORCH_INC=$(python3 - <<'PY'
import torch
from torch.utils.cpp_extension import include_paths
print(" ".join(f"-I{x}" for x in include_paths()))
PY
)

TORCH_LIB=$(python3 - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
```

## How to run kernel operators

```bash
nvcc -x cu -std=c++17 -O3 -arch=sm_89 \
  cuRmsNormCUDATest.cpp \
  -I. -I/usr/include \
  -I/mnt/c/Users/admin/cuixin/DeepSeek-V4-Flash/.venv311/lib/python3.11/site-packages/torch/include \
  -I/mnt/c/Users/admin/cuixin/DeepSeek-V4-Flash/.venv311/lib/python3.11/site-packages/torch/include/torch/csrc/api/include \
  -L/mnt/c/Users/admin/cuixin/DeepSeek-V4-Flash/.venv311/lib/python3.11/site-packages/torch/lib \
  -L/usr/lib/x86_64-linux-gnu \
  -ltorch -ltorch_cpu -lc10 -lgtest -lgtest_main -lpthread \
  -o cuRmsNormCUDATest

export LD_LIBRARY_PATH=/mnt/c/Users/admin/cuixin/DeepSeek-V4-Flash/.venv311/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH
./cuRmsNormCUDATest --gtest_filter=brRmsNormCUDAKernelTest.sanity
```
