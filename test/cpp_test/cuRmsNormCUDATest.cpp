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

#include "cuRmsNormCUDAKernel.h"
#include "gtest/gtest.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <vector>


// -------------------------------
// CUDA error checking macro
// -------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ------------------------------------------------------------------
// Compare BF16 results usings PyTorch: |a - b| <= atol + rtol * |b|
// Returns true if mismatch rate is below threshold (0.1%)
// ------------------------------------------------------------------
static bool compareBF16Tensors(const torch::Tensor &result,
                               const torch::Tensor &golden, float rtol = 0.016f,
                               float atol = 1e-3f) {
  auto resultF = result.toType(torch::kFloat32);
  auto goldenF = golden.toType(torch::kFloat32);
  auto diff = (resultF - goldenF).abs();
  auto threshold = atol + rtol * goldenF.abs();
  auto mismatch = (diff > threshold).sum().item<int64_t>();
  auto total = result.numel();

  if (mismatch > 0) {
    double mismatchRate =
        static_cast<double>(mismatch) / static_cast<double>(total);
    auto maxDiff = diff.max().item<float>();
    printf("Mismatches: %ld / %ld (%.4f%%), maxDiff=%.6f\n", mismatch, total,
           mismatchRate * 100.0, maxDiff);
    return mismatchRate < 0.001; // Allow up to 0.1% mismatches for BF16
  }
  return true;
}

// ---------------------------------------------------------------------------
// Core test function: runs RMS Norm on CUDA and compares with PyTorch golden
// ---------------------------------------------------------------------------
static bool testRmsNormCUDA(int32_t N, int32_t H, int32_t W,
                            int32_t warmup_time = 5, int32_t repeat_time = 10) {
  torch::manual_seed(0);
  printf("[rmsNormCUDATest:] shape: (%d, %d, %d).\n", N, H, W);

  // Generate random input and weight using PyTorch
  torch::Tensor torchInput = torch::randn({N, H, W}, torch::kBFloat16);
  torch::Tensor torchWeight = torch::randn({W}, torch::kFloat32);

  const int64_t input_elems = static_cast<int64_t>(N) * H * W;
  const int64_t input_bytes = input_elems * sizeof(__nv_bfloat16);
  const int64_t weight_bytes = static_cast<int64_t>(W) * sizeof(float);

  // Allocate device memory
  void *d_input = nullptr, *d_output = nullptr, *d_weight = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_weight, weight_bytes));

  // Copy input and weight to device
  CUDA_CHECK(cudaMemcpy(d_input, torchInput.contiguous().data_ptr(),
                        input_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, torchWeight.contiguous().data_ptr(),
                        weight_bytes, cudaMemcpyHostToDevice));

  const float eps = 1e-6f;

  // Warm up
  for (int iter = 0; iter < warmup_time; iter++) {
    launchRmsNormCUDAKernel(static_cast<__nv_bfloat16 *>(d_output),
                            static_cast<const __nv_bfloat16 *>(d_input),
                            static_cast<const float *>(d_weight), N, H, W, eps);
  }

  // Benchmark
  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int iter = 0; iter < repeat_time; iter++) {
    CUDA_CHECK(cudaEventRecord(start));
    launchRmsNormCUDAKernel(static_cast<__nv_bfloat16 *>(d_output),
                            static_cast<const __nv_bfloat16 *>(d_input),
                            static_cast<const float *>(d_weight), N, H, W, eps);
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msecOnce = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecOnce, start, stop));
    msecTotal += msecOnce;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  float msecPerIter = msecTotal / repeat_time;
  printf("kernel avg time: %.4f ms.\n", msecPerIter);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Copy result back to host via torch tensor
  torch::Tensor resultTensor = torch::empty({N, H, W}, torch::kBFloat16);
  CUDA_CHECK(cudaMemcpy(resultTensor.contiguous().data_ptr(), d_output,
                        input_bytes, cudaMemcpyDeviceToHost));

  // Compute golden reference using PyTorch:
  // RMS = rsqrt(mean(x^2, dim=-1, keepdim=True) + eps)
  // golden = x * RMS * weight
  torch::Tensor inputF32 = torchInput.toType(torch::kFloat32);
  torch::Tensor RMS =
      inputF32.square().mean({-1}, true).add(torch::Scalar(eps)).rsqrt();
  torch::Tensor goldenTensor =
      (torchInput * RMS * torchWeight).toType(torch::kBFloat16);

  // Compare
  bool pass = compareBF16Tensors(resultTensor, goldenTensor, 0.016f, 1e-3f);

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_weight);

  return pass;
}

// ===============================================================
// Test Suites
// ===============================================================

TEST(brRmsNormCUDAKernelTest, sanity) {
  EXPECT_TRUE(testRmsNormCUDA(1, 1, 4096));
  EXPECT_TRUE(testRmsNormCUDA(1, 32, 4096));
  EXPECT_TRUE(testRmsNormCUDA(1, 64, 4096));
  EXPECT_TRUE(testRmsNormCUDA(1, 128, 4096));
  EXPECT_TRUE(testRmsNormCUDA(1, 256, 4096));

  EXPECT_TRUE(testRmsNormCUDA(8, 256, 4096));
  EXPECT_TRUE(testRmsNormCUDA(1, 717, 5120));
  EXPECT_TRUE(testRmsNormCUDA(1, 4774, 4096));
  EXPECT_TRUE(testRmsNormCUDA(2, 1536, 8192));
  EXPECT_TRUE(testRmsNormCUDA(1, 4096, 7168));
  EXPECT_TRUE(testRmsNormCUDA(1, 4212, 7168));
}

TEST(brRmsNormCUDAKernelTest, perf) {
  const std::vector<int> hidden_sizes = {4096, 5120, 6144, 7168, 8192};
  const std::vector<int> seq_lens_full = {
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const std::vector<int> seq_lens_large = {256, 512, 1024, 2048, 4096, 8192};

  for (int batch : {1, 2, 3, 4}) {
    const auto &seq_lens = (batch == 1) ? seq_lens_full : seq_lens_large;
    for (int hs : hidden_sizes) {
      for (int sl : seq_lens) {
        EXPECT_TRUE(testRmsNormCUDA(batch, sl, hs));
      }
    }
  }
}

TEST(brRmsNormCUDAKernelTest, regression) {
  // Small hidden sizes
  EXPECT_TRUE(testRmsNormCUDA(1, 767, 6144));
  EXPECT_TRUE(testRmsNormCUDA(1, 3800, 5120));
  EXPECT_TRUE(testRmsNormCUDA(2, 3328, 8192));
  EXPECT_TRUE(testRmsNormCUDA(3, 512, 4096));

  // Medium hidden sizes
  EXPECT_TRUE(testRmsNormCUDA(1, 787, 8192));
  EXPECT_TRUE(testRmsNormCUDA(1, 3200, 5120));
  EXPECT_TRUE(testRmsNormCUDA(1, 3328, 4096));
  EXPECT_TRUE(testRmsNormCUDA(3, 2048, 6144));

  // Large hidden sizes
  EXPECT_TRUE(testRmsNormCUDA(1, 717, 5120));
  EXPECT_TRUE(testRmsNormCUDA(1, 4774, 4096));
  EXPECT_TRUE(testRmsNormCUDA(2, 1536, 8192));
  EXPECT_TRUE(testRmsNormCUDA(3, 2048, 6144));
}
