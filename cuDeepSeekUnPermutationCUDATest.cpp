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

#include "cuDeepSeekUnPermutationCUDA.h"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

// ================================================================
// CUDA Macro Checking Macro
// ================================================================
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
             cudaGetErrorString(err));                                         \
      return std::make_tuple(false, 99999.0f);                                 \
    }                                                                          \
  } while (0)

// ================================================================
// Compare BF16 results against golden
// Returns true if mismatch rate is below threshold
// ================================================================
static bool compareBF16Vectors(const std::vector<__nv_bfloat16> &result,
                               const std::vector<__nv_bfloat16> &golden,
                               float rtol = 0.016f, float atol = 1e-2f) {
  int64_t mismatch = 0;
  float max_diff = 0.0f;
  float max_ref = 0.0f;

  for (size_t i = 0; i < result.size(); i++) {
    const float result_value = __bfloat162float(result[i]);
    const float golden_value = __bfloat162float(golden[i]);
    const float diff = std::abs(result_value - golden_value);
    const float ref = std::abs(golden_value);
    const float threshold = atol + rtol * ref;

    if (diff > threshold) {
      mismatch++;
    }
    max_diff = std::max(max_diff, diff);
    max_ref = std::max(max_ref, ref);
  }

  if (mismatch > 0) {
    double rate = static_cast<double>(mismatch) / result.size();
    printf(" Mismatches: %ld / %ld (%.4f%%), maxDiff=%.6f, maxRef=%.6f\n",
           mismatch, static_cast<int64_t>(result.size()), rate * 100.0,
           max_diff, max_ref);
    return rate < 0.001;
  }
  return true;
}

struct TokenSlot {
  int32_t token_id;
  int32_t topk_slot;
};

// ==============================================================================
// Core test function: DeepSeek UnPermutation on CUDA
//
// This function:
// 1. Generates MoE routing data (topk expert selection + softmax probs)
// 2. Generates random per-expert hidden states
// 3. Builds gather indices (reverse mapping: output_token -> expert_token)
// 4. Computes golden reference via PyTorch
// 5. Runs the CUDA kernel
// 6. Compares results
// ==============================================================================
static std::tuple<bool, float> testDeepSeekUnPermutationCUDA(
    uint32_t token_num, uint32_t hidden_size, uint32_t expert_num,
    uint32_t topk, int32_t warmup_time = 1, int32_t repeat_time = 1) {

  printf("[UnPermutation CUDA] token_num=%u, hidden_size=%u, expert_num=%u, "
         "topk=%u\n",
         token_num, hidden_size, expert_num, topk);

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  std::vector<int32_t> expert_indices(token_num * topk);
  std::vector<float> probs(token_num * topk);
  std::vector<int32_t> expert_ids(expert_num);
  std::vector<float> logits(expert_num);
  std::iota(expert_ids.begin(), expert_ids.end(), 0);

  for (uint32_t t = 0; t < token_num; t++) {
    for (uint32_t e = 0; e < expert_num; e++) {
      logits[e] = dist(gen);
    }
    std::partial_sort(
        expert_ids.begin(), expert_ids.begin() + topk, expert_ids.end(),
        [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });

    float max_score = logits[expert_ids[0]];
    float sum_exp = 0.0f;
    for (uint32_t k = 0; k < topk; k++) {
      const int32_t expert_id = expert_ids[k];
      expert_indices[t * topk + k] = expert_id;
      probs[t * topk + k] = std::exp(logits[expert_id] - max_score);
      sum_exp += probs[t * topk + k];
    }
    for (uint32_t k = 0; k < topk; k++) {
      probs[t * topk + k] /= sum_exp;
    }
  }

  std::vector<int64_t> tokens_per_expert(expert_num, 0);
  std::vector<std::vector<TokenSlot>> expert_token_lists(expert_num);
  for (uint32_t t = 0; t < token_num; t++) {
    for (uint32_t k = 0; k < topk; k++) {
      const int32_t expert_id = expert_indices[t * topk + k];
      tokens_per_expert[expert_id]++;
      expert_token_lists[expert_id].push_back(
          {static_cast<int32_t>(t), static_cast<int32_t>(k)});
    }
  }

  std::vector<int64_t> expert_offsets(expert_num + 1, 0);
  for (uint32_t e = 0; e < expert_num; e++) {
    expert_offsets[e + 1] = expert_offsets[e] + tokens_per_expert[e];
  }
  const int64_t total_expert_tokens = expert_offsets[expert_num];

  std::vector<int32_t> gather_src(token_num * topk, -1);
  std::vector<float> gather_probs_vec(token_num * topk, 0.0f);
  for (uint32_t e = 0; e < expert_num; e++) {
    const int64_t base_offset = expert_offsets[e];
    for (int32_t i = 0; i < static_cast<int32_t>(expert_token_lists[e].size());
         i++) {
      const int32_t t = expert_token_lists[e][i].token_id;
      const int32_t k = expert_token_lists[e][i].topk_slot;
      gather_src[t * topk + k] = static_cast<int32_t>(base_offset + i);
      gather_probs_vec[t * topk + k] = probs[t * topk + k];
    }
  }

  std::vector<__nv_bfloat16> hidden_states(total_expert_tokens * hidden_size);
  for (auto &value : hidden_states) {
    value = __float2bfloat16(dist(gen));
  }

  std::vector<__nv_bfloat16> golden_output(token_num * hidden_size);
  for (uint32_t t = 0; t < token_num; t++) {
    for (uint32_t h = 0; h < hidden_size; h++) {
      float accum = 0.0f;
      for (uint32_t k = 0; k < topk; k++) {
        const int32_t src_row = gather_src[t * topk + k];
        const float prob = gather_probs_vec[t * topk + k];
        const int64_t hidden_idx =
            static_cast<int64_t>(src_row) * hidden_size + h;
        accum += __bfloat162float(hidden_states[hidden_idx]) * prob;
      }
      golden_output[static_cast<int64_t>(t) * hidden_size + h] =
          __float2bfloat16(accum);
    }
  }

  // ----- 5. Run CUDA kernel -----
  const int64_t hs_bytes =
      total_expert_tokens * hidden_size * sizeof(__nv_bfloat16);
  const int64_t out_bytes =
      (int64_t)token_num * hidden_size * sizeof(__nv_bfloat16);
  const int64_t src_bytes = (int64_t)token_num * topk * sizeof(int32_t);
  const int64_t probs_bytes = (int64_t)token_num * topk * sizeof(float);

  void *d_hidden = nullptr, *d_output = nullptr;
  void *d_gather_src = nullptr, *d_gather_probs = nullptr;

  CUDA_CHECK(cudaMalloc(&d_hidden, hs_bytes));
  CUDA_CHECK(cudaMalloc(&d_output, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_gather_src, src_bytes));
  CUDA_CHECK(cudaMalloc(&d_gather_probs, probs_bytes));

  CUDA_CHECK(cudaMemcpy(d_hidden, hidden_states.data(), hs_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gather_src, gather_src.data(), src_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gather_probs, gather_probs_vec.data(), probs_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, out_bytes));

  // Warm up
  for (int iter = 0; iter < warmup_time; iter++) {
    CUDA_CHECK(launchDeepSeekUnPermutationKernel(
        static_cast<const __nv_bfloat16 *>(d_hidden),
        static_cast<__nv_bfloat16 *>(d_output),
        static_cast<const int32_t *>(d_gather_src),
        static_cast<const float *>(d_gather_probs), hidden_size, topk,
        token_num));
  }

  // Benchmark
  float msecTotal = 0.0f;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int iter = 0; iter < repeat_time; iter++) {
    CUDA_CHECK(cudaMemset(d_output, 0, out_bytes));
    CUDA_CHECK(cudaEventRecord(start, nullptr));
    CUDA_CHECK(launchDeepSeekUnPermutationKernel(
        static_cast<const __nv_bfloat16 *>(d_hidden),
        static_cast<__nv_bfloat16 *>(d_output),
        static_cast<const int32_t *>(d_gather_src),
        static_cast<const float *>(d_gather_probs), hidden_size, topk,
        token_num));
    CUDA_CHECK(cudaEventRecord(stop, nullptr));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msecOnce = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msecOnce, start, stop));
    msecTotal += msecOnce;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  float msecPerIter = msecTotal / repeat_time;
  float bandwidth = ((float)token_num * hidden_size * 2.0f +
                     (float)total_expert_tokens * hidden_size * 2.0f) /
                    (msecPerIter * 1e6f);
  printf(" kernel avg time: %.4f ms, bandwidth: %.2f GB/s\n", msecPerIter,
         bandwidth);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // ======================== 6. Compare ==================================
  std::vector<__nv_bfloat16> result(token_num * hidden_size);
  CUDA_CHECK(
      cudaMemcpy(result.data(), d_output, out_bytes, cudaMemcpyDeviceToHost));

  bool pass = compareBF16Vectors(result, golden_output, 0.016f, 1e-2f);

  // Clean up
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_gather_src));
  CUDA_CHECK(cudaFree(d_gather_probs));

  return std::make_tuple(pass, pass ? msecPerIter : 99999.0f);
}

// ================================================================
// Test Suites
// ================================================================

// -----------------------------------------------------------------
// TopK=8 sanity tests (matching original SUPA test configurations)
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_top8_sanity, general) {
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(8193, 64, 256, 8)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(12, 7168, 256, 8)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(7179, 7168, 256, 8)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(7179, 7168, 256, 8)));
}

// -----------------------------------------------------------------
// TopK=8 with 64 local experts
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_top8_expert64, general) {
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(8193, 64, 64, 8)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(12, 7168, 64, 8)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(7179, 7168, 64, 8)));
}

// -----------------------------------------------------------------
// TopK=2 sanity tests
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_top2_sanity, general) {
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(7179, 7168, 8, 2)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(18197, 7168, 8, 2)));
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(512, 7168, 8, 2)));
}

// -----------------------------------------------------------------
// Edge case tests
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_edge_cases, general) {
  // Single token
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(1, 64, 4, 2)));
  // Small scale
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(16, 128, 4, 2)));
  // hidden_size not multiple of tile size (1024)
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(128, 300, 8, 2)));
  // TopK=1
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(256, 512, 8, 1)));
  // TopK=4
  EXPECT_TRUE(std::get<0>(testDeepSeekUnPermutationCUDA(1024, 1024, 16, 4)));
}

// -----------------------------------------------------------------
// Performance tests with TopK=2
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_perf_top2, general) {
  auto r1 = testDeepSeekUnPermutationCUDA(8192, 4096, 8, 2);
  auto r2 = testDeepSeekUnPermutationCUDA(8 * 8192, 4096, 8, 2);

  EXPECT_TRUE(std::get<0>(r1));
  EXPECT_TRUE(std::get<0>(r2));
  printf("    [Perf] 8192x4096 topk2: %.4f ms\n", std::get<1>(r1));
  printf("    [Perf] 65536x4096 topk2: %.4f ms\n", std::get<1>(r2));
}

// -----------------------------------------------------------------
// Performance tests with TopK=8
// -----------------------------------------------------------------
TEST(brUnPermutationCUDA_perf_top8, general) {
  auto r1 = testDeepSeekUnPermutationCUDA(8192, 4096, 256, 8);
  auto r2 = testDeepSeekUnPermutationCUDA(4096, 7168, 256, 8);

  EXPECT_TRUE(std::get<0>(r1));
  EXPECT_TRUE(std::get<0>(r2));
  printf("    [Perf] 8192x4096 topk8: %.4f ms\n", std::get<1>(r1));
  printf("    [Perf] 4096x7168 topk8: %.4f ms\n", std::get<1>(r2));
}