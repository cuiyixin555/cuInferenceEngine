#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

cudaError_t launchRmsNormCUDAKernel(__nv_bfloat16 *output,
                                    const __nv_bfloat16 *input,
                                    const float *weight, int32_t N,
                                    int32_t H, int32_t W, float eps,
                                    cudaStream_t stream = nullptr);
