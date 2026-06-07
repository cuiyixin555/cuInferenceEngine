// MIT License

// Copyright (c) 2026 CUI Xin (崔 欣)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "cuElementwiseCUDAKernel_launch.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static bool compareFloatVectors(const std::vector<float> &result,
                                const std::vector<float> &golden) {
  int64_t mismatch = 0;
  float maxDiff = 0.0f;

  for (size_t i = 0; i < result.size(); ++i) {
    const float diff = std::fabs(result[i] - golden[i]);
    if (diff > 0.0f) {
      mismatch++;
    }
    maxDiff = std::max(maxDiff, diff);
  }

  if (mismatch > 0) {
    printf("Mismatches: %ld / %ld (%.4f%%), maxDiff=%.6f\n", mismatch,
           static_cast<int64_t>(result.size()),
           100.0 * static_cast<double>(mismatch) /
               static_cast<double>(result.size()),
           maxDiff);
    return false;
  }
  return true;
}

static bool compareHalfVectors(const std::vector<half> &result,
                               const std::vector<half> &golden) {
  int64_t mismatch = 0;
  float maxDiff = 0.0f;

  for (size_t i = 0; i < result.size(); ++i) {
    const float diff =
        std::fabs(__half2float(result[i]) - __half2float(golden[i]));
    if (diff > 0.0f) {
      mismatch++;
    }
    maxDiff = std::max(maxDiff, diff);
  }

  if (mismatch > 0) {
    printf("Mismatches: %ld / %ld (%.4f%%), maxDiff=%.6f\n", mismatch,
           static_cast<int64_t>(result.size()),
           100.0 * static_cast<double>(mismatch) /
               static_cast<double>(result.size()),
           maxDiff);
    return false;
  }
  return true;
}

enum class ElementwiseKernel {
  F32,
  F32x4,
  F16,
  F16x2,
  F16x8,
  F16x8Pack,
};

static bool runElementwiseKernel(ElementwiseKernel kernel, float *d_a, float *d_b,
                                 float *d_c, int N) {
  switch (kernel) {
  case ElementwiseKernel::F32:
    return launchElementwiseAddF32(d_a, d_b, d_c, N) == cudaSuccess;
  case ElementwiseKernel::F32x4:
    return launchElementwiseAddF32x4(d_a, d_b, d_c, N) == cudaSuccess;
  default:
    return false;
  }
}

static bool runElementwiseKernel(ElementwiseKernel kernel, half *d_a, half *d_b,
                                 half *d_c, int N) {
  switch (kernel) {
  case ElementwiseKernel::F16:
    return launchElementwiseAddF16(d_a, d_b, d_c, N) == cudaSuccess;
  case ElementwiseKernel::F16x2:
    return launchElementwiseAddF16x2(d_a, d_b, d_c, N) == cudaSuccess;
  case ElementwiseKernel::F16x8:
    return launchElementwiseAddF16x8(d_a, d_b, d_c, N) == cudaSuccess;
  case ElementwiseKernel::F16x8Pack:
    return launchElementwiseAddF16x8Pack(d_a, d_b, d_c, N) == cudaSuccess;
  default:
    return false;
  }
}

static bool testElementwiseAddF32(ElementwiseKernel kernel, int N,
                                  int warmup_time = 3, int repeat_time = 5) {
  printf("[elementwiseAddF32] kernel=%d, N=%d\n", static_cast<int>(kernel), N);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<float> hostA(N), hostB(N), golden(N);
  for (int i = 0; i < N; ++i) {
    hostA[i] = dist(gen);
    hostB[i] = dist(gen);
    golden[i] = hostA[i] + hostB[i];
  }

  const int64_t bytes = static_cast<int64_t>(N) * sizeof(float);
  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(
      cudaMemcpy(d_a, hostA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b, hostB.data(), bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup_time; ++i) {
    if (!runElementwiseKernel(kernel, d_a, d_b, d_c, N)) {
      fprintf(stderr, "kernel launch failed\n");
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
      return false;
    }
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float msecTotal = 0.0f;
  for (int i = 0; i < repeat_time; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    if (!runElementwiseKernel(kernel, d_a, d_b, d_c, N)) {
      fprintf(stderr, "kernel launch failed\n");
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
      return false;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecOnce = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecOnce, start, stop));
    msecTotal += msecOnce;
  }
  printf("kernel avg time: %.4f ms\n", msecTotal / repeat_time);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<float> result(N);
  CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool pass = compareFloatVectors(result, golden);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return pass;
}

static bool testElementwiseAddF16(ElementwiseKernel kernel, int N,
                                  int warmup_time = 3, int repeat_time = 5) {
  printf("[elementwiseAddF16] kernel=%d, N=%d\n", static_cast<int>(kernel), N);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<half> hostA(N), hostB(N), golden(N);
  for (int i = 0; i < N; ++i) {
    hostA[i] = __float2half(dist(gen));
    hostB[i] = __float2half(dist(gen));
    golden[i] = __hadd(hostA[i], hostB[i]);
  }

  const int64_t bytes = static_cast<int64_t>(N) * sizeof(half);
  half *d_a = nullptr;
  half *d_b = nullptr;
  half *d_c = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_a), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_b), bytes));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_c), bytes));

  CUDA_CHECK(
      cudaMemcpy(d_a, hostA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_b, hostB.data(), bytes, cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup_time; ++i) {
    if (!runElementwiseKernel(kernel, d_a, d_b, d_c, N)) {
      fprintf(stderr, "kernel launch failed\n");
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
      return false;
    }
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  float msecTotal = 0.0f;
  for (int i = 0; i < repeat_time; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    if (!runElementwiseKernel(kernel, d_a, d_b, d_c, N)) {
      fprintf(stderr, "kernel launch failed\n");
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      cudaFree(d_a);
      cudaFree(d_b);
      cudaFree(d_c);
      return false;
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecOnce = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecOnce, start, stop));
    msecTotal += msecOnce;
  }
  printf("kernel avg time: %.4f ms\n", msecTotal / repeat_time);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::vector<half> result(N);
  CUDA_CHECK(cudaMemcpy(result.data(), d_c, bytes, cudaMemcpyDeviceToHost));

  bool pass = compareHalfVectors(result, golden);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return pass;
}

TEST(brElementwiseCUDAKernelTest, sanityF32) {
  for (int N : {1, 3, 7, 255, 256, 257, 1023, 1024, 1025, 4096}) {
    EXPECT_TRUE(testElementwiseAddF32(ElementwiseKernel::F32, N));
    EXPECT_TRUE(testElementwiseAddF32(ElementwiseKernel::F32x4, N));
  }
}

TEST(brElementwiseCUDAKernelTest, sanityF16) {
  for (int N : {1, 3, 7, 255, 256, 257, 1023, 1024, 1025, 4096}) {
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x2, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x8, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x8Pack, N));
  }
}

TEST(brElementwiseCUDAKernelTest, perf) {
  const std::vector<int> sizes = {1024 * 1024, 2048 * 2048, 4096 * 4096};
  for (int N : sizes) {
    EXPECT_TRUE(testElementwiseAddF32(ElementwiseKernel::F32, N));
    EXPECT_TRUE(testElementwiseAddF32(ElementwiseKernel::F32x4, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x2, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x8, N));
    EXPECT_TRUE(testElementwiseAddF16(ElementwiseKernel::F16x8Pack, N));
  }
}
