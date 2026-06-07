#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

cudaError_t launchElementwiseAddF32(float *a, float *b, float *c, int N,
                                    cudaStream_t stream = nullptr);

cudaError_t launchElementwiseAddF32x4(float *a, float *b, float *c, int N,
                                      cudaStream_t stream = nullptr);

cudaError_t launchElementwiseAddF16(half *a, half *b, half *c, int N,
                                    cudaStream_t stream = nullptr);

cudaError_t launchElementwiseAddF16x2(half *a, half *b, half *c, int N,
                                      cudaStream_t stream = nullptr);

cudaError_t launchElementwiseAddF16x8(half *a, half *b, half *c, int N,
                                      cudaStream_t stream = nullptr);

cudaError_t launchElementwiseAddF16x8Pack(half *a, half *b, half *c, int N,
                                          cudaStream_t stream = nullptr);
