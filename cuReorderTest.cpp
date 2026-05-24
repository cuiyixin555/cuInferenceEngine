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

#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

// ---------------------------------------------------------------------------
// Print utility for host tensors
// ---------------------------------------------------------------------------
template <typename T> static void printInfo(T *ptr, int N, int H, int W) {
  for (int n = 0; n < N; ++n) {
    for (int h = 0; h < H; ++h) {
      for (int w = 0; w < W; ++w) {
        size_t pos = (size_t)n * H * W + h * W + w;
        float value = float(ptr[pos]);
        printf("%f ", value);
      }
      printf("\n");
    }
    printf("\n\n");
  }
  printf("\n");
}

// ============================================================================
// Kernel 1: Dense FP32 -> Folded BF16 (Reorder / Fold)
//
// Maps a dense [ori_N, ori_H, ori_W] FP32 tensor into a folded
// [fold_N, fold_H, fold_W] BF16 tensor, where:
// - fold_N = ori_N * h_fold_num * w_fold_num
// - fold_H = ceil(ori_H / h_fold_num)
// - fold_W = ceil(ori_W / w_fold_num)
//
// Each thread block handles one fold_N size. Threads cooperatively load
// W-dimension tiles using cub::BlockLoad / cub::BlockStore.
//
// Grid: (foldN)
// Block: (BLOCK_THREADS)
// ==============================================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void ReorderDenseToFoldKernel(
    const float *__restrict__ src,   // [ori_N, ori_H, ori_W]
    __nv_bfloat16 *__restrict__ dst, // [fold_N, fold_H, fold_W]
    int fold_N, int fold_H, int fold_W, int ori_N, int ori_H, int ori_W,
    int h_fold_num, int w_fold_num) {
  constexpr int TILE_W = BLOCK_THREADS * ITEMS_PER_THREAD;
  using BlockStoreT =
      cub::BlockStore<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ typename BlockStoreT::TempStorage store_temp;

  const int n_fold = blockIdx.x;
  if (n_fold >= fold_N)
    return;

  const int total_fold = h_fold_num * w_fold_num;
  const int coord_n = n_fold / total_fold;
  const int fold_idx = n_fold % total_fold;
  const int h_fold_idx = fold_idx % h_fold_num;
  const int w_fold_idx = fold_idx / h_fold_num;

  for (int h = 0; h < fold_H; h++) {
    // Compute original H coordinate
    int coord_h = h_fold_idx * fold_H + h;

    for (int w_tile = 0; w_tile < fold_W; w_tile += TILE_W) {
      int valid = min(TILE_W, fold_W - w_tile);

      __nv_bfloat16 items[ITEMS_PER_THREAD];

// Each thread loads its items from the original dense tensor
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int w_local = threadIdx.x * ITEMS_PER_THREAD + i;
        int coord_w = w_fold_idx * fold_W + w_tile + w_local;

        float val = 0.0f;
        if (w_local < valid && coord_n < ori_N && coord_h < ori_H &&
            coord_w < ori_W) {
          int64_t addr =
              (int64_t)coord_n * ori_H * ori_W + coord_h * ori_W + coord_w;
          val = src[addr];
        }
        items[i] = __float2bfloat16(val);
      }

      // Cooperative block-level store to folded output
      int64_t dst_offset =
          (int64_t)n_fold * fold_H * fold_W + h * fold_W + w_tile;

      if (valid == TILE_W) {
        BlockStoreT(store_temp).Store(dst + dst_offset, items);
      } else {
        BlockStoreT(store_temp).Store(dst + dst_offset, items, valid);
      }
      __syncthreads();
    }
  }
}

// ============================================================================
// Kernel 2: Folded BF16 -> Dense FP32 (De-Reorder / Unfold)
//
// Reverses Kernel 1. Read folded [fold_N, fold_H, fold_W] BF16 and writes
// to dense [ori_N, ori_H, ori_W] FP32.
//
// Grid: (fold_N)
// Block: (BLOCK_THREADS)
// ============================================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void ReorderFoldToDenseKernel(
    const __nv_bfloat16 *__restrict__ src, // [fold_N, fold_H, fold_W]
    float *__restrict__ dst,               // [ori_N, ori_H, ori_W]
    int fold_N, int fold_H, int fold_W, int ori_N, int ori_H, int ori_W,
    int h_fold_num, int w_fold_num) {
  constexpr int TILE_W = BLOCK_THREADS * ITEMS_PER_THREAD;

  using BlockLoadT =
      cub::BlockLoad<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                     cub::BLOCK_LOAD_WARP_TRANSPOSE>;

  __shared__ typename BlockLoadT::TempStorage load_temp;

  const int n_fold = blockIdx.x;
  if (n_fold >= fold_N)
    return;

  const int total_fold = h_fold_num * w_fold_num;
  const int coord_n = n_fold / total_fold;
  const int fold_idx = n_fold % total_fold;
  const int h_fold_idx = fold_idx % h_fold_num;
  const int w_fold_idx = fold_idx / h_fold_num;

  for (int h = 0; h < fold_H; h++) {
    int coord_h = h_fold_idx * fold_H + h;

    for (int w_tile = 0; w_tile < fold_W; w_tile += TILE_W) {
      int valid = min(TILE_W, fold_W - w_tile);

      __nv_bfloat16 items[ITEMS_PER_THREAD];

      // Cooperative block-level load from folded tensor
      int64_t src_offset =
          (int64_t)n_fold * fold_H * fold_W + h * fold_W + w_tile;

      if (valid == TILE_W) {
        BlockLoadT(load_temp).Load(src + src_offset, items);
      } else {
        BlockLoadT(load_temp).Load(src + src_offset, items, valid,
                                   __float2bfloat16(0.0f));
      }
      __syncthreads();

// Each thread writes its items to the original dense position
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int w_local = threadIdx.x * ITEMS_PER_THREAD + i;
        int coord_w = w_fold_idx * fold_W + w_tile + w_local;

        if (w_local < valid && coord_n < ori_N && coord_h < ori_H &&
            coord_w < ori_W) {
          int64_t addr =
              (int64_t)coord_n * ori_H * ori_W + coord_h * ori_W + coord_w;
          dst[addr] = __bfloat162float(items[i]);
        }
      }
    }
  }
}

// ============================================================================
// Kernel 3: Folded BF16 -> Dense FP32, extract only last token per batch
//
// For each batch n, extracts only the row at index (seq_lens[n] - 1) from the
// folded layout, writing a [ori_N, ori_W] output (onw row per batch).
//
// Grid: (fold_N)
// Block: (BLOCK_THREDAS)
// ============================================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void ReorderFoldToDenseLastTokenKernel(
    const __nv_bfloat16 *__restrict__ src, // [fold_N, fold_H, fold_W]
    float *__restrict__ dst,               // [ori_N, ori_W]
    int fold_N, int fold_H, int fold_W, int ori_N, int ori_H, int ori_W,
    int h_fold_num, int w_fold_num, int pad_H, int pad_W,
    const int *__restrict__ seq_lens) // [ori_N]
{
  constexpr int TILE_W = BLOCK_THREADS * ITEMS_PER_THREAD;

  using BlockLoadT =
      cub::BlockLoad<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                     cub::BLOCK_LOAD_WARP_TRANSPOSE>;

  __shared__ typename BlockLoadT::TempStorage load_temp;

  const int n_fold = blockIdx.x;
  if (n_fold >= fold_N)
    return;

  const int total_fold = h_fold_num * w_fold_num;
  const int coord_n = n_fold / total_fold;
  if (coord_n >= ori_N)
    return;

  const int fold_idx = n_fold % total_fold;
  const int h_fold_idx = fold_idx % h_fold_num;
  const int w_fold_idx = fold_idx / h_fold_num;

  const int H_ori_no_pad = ori_H - pad_H;

  const int last_token_h = seq_lens[coord_n] - 1;

  for (int h = 0; h < fold_H; h++) {
    int coord_h = h_fold_idx * fold_H + h;

    // Only process if this row matches the last token
    if (coord_h != last_token_h)
      continue;
    if (coord_h >= H_ori_no_pad)
      continue;

    for (int w_tile = 0; w_tile < fold_W; w_tile += TILE_W) {
      int valid = min(TILE_W, fold_W - w_tile);

      __nv_bfloat16 items[ITEMS_PER_THREAD];

      int64_t src_offset =
          (int64_t)n_fold * fold_H * fold_W + h * fold_W + w_tile;

      if (valid == TILE_W) {
        BlockLoadT(load_temp).Load(src + src_offset, items);
      } else {
        BlockLoadT(load_temp).Load(src + src_offset, items, valid,
                                   __float2bfloat16(0.0f));
      }
      __syncthreads();

// Each thread writes its items to the original dense position
#pragma unroll
      for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int w_local = threadIdx.x * ITEMS_PER_THREAD + i;
        int coord_w = w_fold_idx * fold_W + w_tile + w_local;

        if (w_local < valid && coord_n < ori_N && coord_w < ori_W) {
          int64_t addr = (int64_t)coord_n * ori_W + coord_w;
          dst[addr] = __bfloat162float(items[i]);
        }
      }
    }
  }
}

// ============================================================================
// Kernel 4: Dnese FP32 -> Folded BF16 with multi-die (NUMA-like) reorder
//
// For multi-device scenarios: the H dimension is split among fold_num segments,
// each stored as a separate fold_N entry. The W dimension is split by grid
// blocks.
//
// Grid: (num_w_splits)
// Block: (BLOCK_THREADS)
// ============================================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void ReorderDenseToMultiSegmentKernel(
    const float *__restrict__ src, // [ori_N, ori_H, ori_W]
    __nv_bfloat16
        *__restrict__ dst, // [fold_N, fold_H, fold_W] (row-major BF16)
    int fold_N, int fold_H, int fold_W, int ori_N, int ori_H, int ori_W,
    int fold_num) {
  constexpr int TILE_W = BLOCK_THREADS * ITEMS_PER_THREAD;

  using BlockStoreT =
      cub::BlockStore<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>;

  __shared__ typename BlockStoreT::TempStorage store_temp;

  const int w_split_idx = blockIdx.x;

  for (int n = 0; n < fold_N; n += fold_num) {
    for (int h = 0; h < fold_H; h++) {
      for (int fold_idx = 0; fold_idx < fold_num; fold_idx++) {
        int n_dst = n + fold_idx;
        if (n_dst >= fold_N)
          continue;

        for (int w_tile = 0; w_tile < fold_W; w_tile += TILE_W) {
          int valid = min(TILE_W, fold_W - w_tile);

          __nv_bfloat16 items[ITEMS_PER_THREAD];

#pragma unroll
          for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int w_local = threadIdx.x * ITEMS_PER_THREAD + i;
            int w_global = fold_W * w_split_idx + w_tile + w_local;
            int h_global = fold_H * fold_idx + h;
            int n_global = n / fold_num;

            float val = 0.0f;
            if (w_local < valid && n_global < ori_N && h_global < ori_H &&
                w_global < ori_W) {
              int64_t addr = (int64_t)n_global * ori_H * ori_W +
                             h_global * ori_W + w_global;
              val = src[addr];
            }
            items[i] = __float2bfloat16(val);
          }

          // Store into per-aplit segment of dst
          int64_t dst_offset = (int64_t)w_split_idx * fold_N * fold_H * fold_W +
                               (int64_t)n_dst * fold_H * fold_W + h * fold_W +
                               w_tile;

          if (valid == TILE_W) {
            BlockStoreT(store_temp).Store(dst + dst_offset, items);
          } else {
            BlockStoreT(store_temp).Store(dst + dst_offset, items, valid);
          }
          __syncthreads();
        }
      }
    }
  }
}

// ============================================================================
// Kernel 5: Folded BF16 -> Dense FP32, multi-die (NUMA-like) de-reorder
//
// For multi-device scenarios: the H dimension is split among fold_num segments,
// each stored as a separate fold_N entry. The W dimension is split by grid
// blocks.
//
// Grid: (num_w_splits)
// Block: (BLOCK_THREADS)
// ============================================================================
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void ReorderMultiSegmentToDenseKernel(
    const __nv_bfloat16
        *__restrict__ src,   // [num_w_splits * fold_N, fold_H, fold_W]
    float *__restrict__ dst, // [ori_N, ori_H, ori_W]
    int fold_N, int fold_H, int fold_W, int ori_N, int ori_H, int ori_W,
    int fold_num) {
  constexpr int TILE_W = BLOCK_THREADS * ITEMS_PER_THREAD;

  using BlockLoadT =
      cub::BlockLoad<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                     cub::BLOCK_LOAD_WARP_TRANSPOSE>;

  __shared__ typename BlockLoadT::TempStorage load_temp;

  const int w_split_idx = blockIdx.x;

  for (int n = 0; n < fold_N; n += fold_num) {
    for (int h = 0; h < fold_H; h++) {
      for (int fold_idx = 0; fold_idx < fold_num; fold_idx++) {
        int n_src = n + fold_idx;
        if (n_src >= fold_N)
          continue;

        for (int w_tile = 0; w_tile < fold_W; w_tile += TILE_W) {
          int valid = min(TILE_W, fold_W - w_tile);

          __nv_bfloat16 items[ITEMS_PER_THREAD];

          int64_t src_offset = (int64_t)w_split_idx * fold_N * fold_H * fold_W +
                               (int64_t)n_src * fold_H * fold_W + h * fold_W +
                               w_tile;

          if (valid == TILE_W) {
            BlockLoadT(load_temp).Load(src + src_offset, items);
          } else {
            BlockLoadT(load_temp).Load(src + src_offset, items, valid,
                                       __float2bfloat16(0.0f));
          }
          __syncthreads();

#pragma unroll
          for (int i = 0; i < ITEMS_PER_THREAD; i++) {
            int w_local = threadIdx.x * ITEMS_PER_THREAD + i;
            int w_global = fold_W * w_split_idx + w_tile + w_local;
            int h_global = fold_H * fold_idx + h;
            int n_global = n / fold_num;

            if (w_local < valid && n_global < ori_N && h_global < ori_H &&
                w_global < ori_W) {
              int64_t addr = (int64_t)n_global * ori_H * ori_W +
                             h_global * ori_W + w_global;
              dst[addr] = __bfloat162float(items[i]);
            }
          }
        }
      }
    }
  }
}

// ============================================================================
// Host-side launcher helpers
// ============================================================================

static constexpr int REORDER_BLOCK_THREADS = 128;
static constexpr int REORDER_ITEMS_PER_THREAD = 4;

inline cudaError_t launchDenseToFold(const float *src, __nv_bfloat16 *dst,
                                     int fold_N, int fold_H, int fold_W,
                                     int ori_N, int ori_H, int ori_W,
                                     int h_fold_num, int w_fold_num,
                                     cudaStream_t stream = nullptr) {
  ReorderDenseToFoldKernel<REORDER_BLOCK_THREADS, REORDER_ITEMS_PER_THREAD>
      <<<fold_N, REORDER_BLOCK_THREADS, 0, stream>>>(
          src, dst, fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, h_fold_num,
          w_fold_num);
  return cudaGetLastError();
}

inline cudaError_t launchFoldToDense(const __nv_bfloat16 *src, float *dst,
                                     int fold_N, int fold_H, int fold_W,
                                     int ori_N, int ori_H, int ori_W,
                                     int h_fold_num, int w_fold_num,
                                     cudaStream_t stream = nullptr) {
  ReorderFoldToDenseKernel<REORDER_BLOCK_THREADS, REORDER_ITEMS_PER_THREAD>
      <<<fold_N, REORDER_BLOCK_THREADS, 0, stream>>>(
          src, dst, fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, h_fold_num,
          w_fold_num);
  return cudaGetLastError();
}

inline cudaError_t launchFoldToDenseLastToken(
    const __nv_bfloat16 *src, float *dst, int fold_N, int fold_H, int fold_W,
    int ori_N, int ori_H, int ori_W, int h_fold_num, int w_fold_num, int pad_H,
    int pad_W, const int *seq_lens, cudaStream_t stream = nullptr) {
  ReorderFoldToDenseLastTokenKernel<REORDER_BLOCK_THREADS,
                                    REORDER_ITEMS_PER_THREAD>
      <<<fold_N, REORDER_BLOCK_THREADS, 0, stream>>>(
          src, dst, fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, h_fold_num,
          w_fold_num, pad_H, pad_W, seq_lens);
  return cudaGetLastError();
}

inline cudaError_t launchDenseToMultiSegment(const float *src,
                                             __nv_bfloat16 *dst, int fold_N,
                                             int fold_H, int fold_W, int ori_N,
                                             int ori_H, int ori_W, int fold_num,
                                             int num_w_splits,
                                             cudaStream_t stream = nullptr) {
  ReorderDenseToMultiSegmentKernel<REORDER_BLOCK_THREADS,
                                   REORDER_ITEMS_PER_THREAD>
      <<<num_w_splits, REORDER_BLOCK_THREADS, 0, stream>>>(
          src, dst, fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, fold_num);
  return cudaGetLastError();
}

inline cudaError_t launchMultiSegmentToDense(const __nv_bfloat16 *src,
                                             float *dst, int fold_N, int fold_H,
                                             int fold_W, int ori_N, int ori_H,
                                             int ori_W, int fold_num,
                                             int num_w_splits,
                                             cudaStream_t stream = nullptr) {
  ReorderMultiSegmentToDenseKernel<REORDER_BLOCK_THREADS,
                                   REORDER_ITEMS_PER_THREAD>
      <<<num_w_splits, REORDER_BLOCK_THREADS, 0, stream>>>(
          src, dst, fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, fold_num);
  return cudaGetLastError();
}

// ============================================================================
// Test: Dense -> Fold -> Dense round-trip (UMA-style)
//
// Corresponds to brReorderTest.cpp::test_uma_reorder()
// Verifies that folding then unfolding a tensor gives back the original data.
// ============================================================================
static bool test_uma_reorder_cuda() {
  int ori_N = 1, ori_H = 2, ori_W = 16;
  int h_fold_num = 1, w_fold_num = 1;

  int fold_N = ori_N * h_fold_num * w_fold_num;
  int fold_H = ori_H / h_fold_num;
  int fold_W = ori_W / w_fold_num;

  const size_t ori_elems = (size_t)ori_N * ori_H * ori_W;
  const size_t fold_elems = (size_t)fold_N * fold_H * fold_W;

  // Host data
  std::vector<float> host_data(ori_elems);
  for (int n = 0; n < ori_N; n++) {
    for (int h = 0; h < ori_H; h++) {
      for (int w = 0; w < ori_W; w++) {
        host_data[n * ori_H * ori_W + h * ori_W + w] = (float)w;
      }
    }
  }

  printf("[UMA Reorder] ori: (%d, %d, %d) -> fold: (%d, %d, %d)\n", ori_N,
         ori_H, ori_W, fold_N, fold_H, fold_W);
  printInfo(host_data.data(), ori_N, ori_H, ori_W);

  // Device memory
  float *d_dense = nullptr;
  __nv_bfloat16 *d_fold = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dense, ori_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fold, fold_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(d_dense, host_data.data(), ori_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Dense -> Fold
  CUDA_CHECK(launchDenseToFold(d_dense, d_fold, fold_N, fold_H, fold_W, ori_N,
                               ori_H, ori_W, h_fold_num, w_fold_num));
  CUDA_CHECK(cudaGetLastError());

  // Fold -> Dense (clear dense first)
  CUDA_CHECK(cudaMemset(d_dense, 0, ori_elems * sizeof(float)));
  CUDA_CHECK(launchFoldToDense(d_fold, d_dense, fold_N, fold_H, fold_W, ori_N,
                               ori_H, ori_W, h_fold_num, w_fold_num));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Verify round-trip
  std::vector<float> result(ori_elems);
  CUDA_CHECK(cudaMemcpy(result.data(), d_dense, ori_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("[UMA Reorder] round-trip result:\n");
  printInfo(result.data(), ori_N, ori_H, ori_W);

  bool pass = true;
  for (size_t i = 0; i < ori_elems; i++) {
    // BF16 round-trip may lose precision; check with tolerance
    if (std::fabs(result[i] - host_data[i]) > 0.1f) {
      printf("Mismatch at index %zu: %f vs %f\n", i, result[i], host_data[i]);
      pass = false;
    }
  }

  cudaFree(d_dense);
  cudaFree(d_fold);
  return pass;
}

// ============================================================================
// Test: Dense -> Fold -> Dense round-trip (NUMA-style)
//
// Corresponds to brReorderTest.cpp::test_numa_reorder()
// Verifies that folding then unfolding a tensor gives back the original data.
// ============================================================================
static bool test_fold_reorder_cuda() {
  int ori_N = 1, ori_H = 8, ori_W = 32;
  int h_fold_num = 1, w_fold_num = 2;

  int fold_N = ori_N * h_fold_num * w_fold_num;
  int fold_H = ori_H / h_fold_num;
  int fold_W = ori_W / w_fold_num;

  const size_t ori_elems = (size_t)ori_N * ori_H * ori_W;
  const size_t fold_elems = (size_t)fold_N * fold_H * fold_W;

  // Host data
  std::vector<float> host_data(ori_elems);
  int v = 1;
  for (int n = 0; n < ori_N; n++) {
    for (int h = 0; h < ori_H; h++) {
      for (int w = 0; w < ori_W; w++) {
        host_data[n * ori_H * ori_W + h * ori_W + w] = (float)v;
      }
      v++;
    }
  }

  printf("[Fold Reorder] ori: (%d, %d, %d) -> fold: (%d, %d, %d) h_fold=%d "
         "w_fold=%d\n",
         ori_N, ori_H, ori_W, fold_N, fold_H, fold_W, h_fold_num, w_fold_num);
  printInfo(host_data.data(), ori_N, ori_H, ori_W);

  float *d_dense = nullptr;
  __nv_bfloat16 *d_fold = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dense, ori_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fold, fold_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(d_dense, host_data.data(), ori_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(launchDenseToFold(d_dense, d_fold, fold_N, fold_H, fold_W, ori_N,
                               ori_H, ori_W, h_fold_num, w_fold_num));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Read back folded data for inspection
  std::vector<__nv_bfloat16> fold_host(fold_elems);
  CUDA_CHECK(cudaMemcpy(fold_host.data(), d_fold,
                        fold_elems * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  printf("[Fold Reorder] folded data:\n");
  std::vector<float> fold_float(fold_elems);
  for (size_t i = 0; i < fold_elems; i++) {
    fold_float[i] = __bfloat162float(fold_host[i]);
  }
  printInfo(fold_float.data(), fold_N, fold_H, fold_W);

  // Fold -> Dense
  CUDA_CHECK(cudaMemset(d_dense, 0, ori_elems * sizeof(float)));
  CUDA_CHECK(launchFoldToDense(d_fold, d_dense, fold_N, fold_H, fold_W, ori_N,
                               ori_H, ori_W, h_fold_num, w_fold_num));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(ori_elems);
  CUDA_CHECK(cudaMemcpy(result.data(), d_dense, ori_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("[Fold Reorder] round-trip result:\n");
  printInfo(result.data(), ori_N, ori_H, ori_W);

  bool pass = true;
  for (size_t i = 0; i < ori_elems; i++) {
    if (std::fabs(result[i] - host_data[i]) > 0.1f) {
      printf("Mismatch at index %zu: %f vs %f\n", i, result[i], host_data[i]);
      pass = false;
    }
  }

  cudaFree(d_dense);
  cudaFree(d_fold);
  return pass;
}

// ============================================================================
// Test: Dense -> Fold, then extract last token per batch
//
// Corresponds to brReorderTest.cpp::test_dump_uma()
// ============================================================================
static bool test_last_token_reorder_cuda() {
  int ori_N = 2, ori_H = 32, ori_W = 8;
  int h_fold_num = 4, w_fold_num = 1;

  int fold_N = ori_N * h_fold_num * w_fold_num;
  int fold_H = ori_H / h_fold_num;
  int fold_W = ori_W / w_fold_num;

  const size_t fold_elems = (size_t)fold_N * fold_H * fold_W;
  const size_t out_elems = (size_t)ori_N * ori_W;
  int pad_H = 0;
  int pad_W = 0;

  // Build folded input data (simulating a folded tensor from earlier stage)
  std::vector<__nv_bfloat16> host_fold(fold_elems);
  int v = 1;
  for (size_t i = 0; i < fold_elems; i++) {
    host_fold[i] = __float2bfloat16((float)v);
    v++;
  }

  printf("[Last Token Reorder] fold: (%d, %d, %d), ori: (%d, %d, %d), padH=%d, "
         "padW=%d\n",
         fold_N, fold_H, fold_W, ori_N, ori_H, ori_W, pad_H, pad_W);

  // Sequence lengths per batch
  std::vector<int> seq_lens = {3, 6};

  __nv_bfloat16 *d_fold = nullptr;
  float *d_out = nullptr;
  int *d_seq_lens = nullptr;

  CUDA_CHECK(cudaMalloc(&d_fold, fold_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_seq_lens, ori_N * sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_fold, host_fold.data(),
                        fold_elems * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, out_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_seq_lens, seq_lens.data(), ori_N * sizeof(int),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(launchFoldToDenseLastToken(d_fold, d_out, fold_N, fold_H, fold_W,
                                        ori_N, ori_H, ori_W, h_fold_num,
                                        w_fold_num, pad_H, pad_W, d_seq_lens));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(out_elems);
  CUDA_CHECK(cudaMemcpy(result.data(), d_out, out_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("[Last Token Reorder] last token result:\n");
  printInfo(result.data(), ori_N, 1, ori_W);

  // Verify: for batch 0, last token is at h=2 (seq_lens=3, 0-indexed row 2)
  // For batch 1, last token is at h=5 (seq_len=6, 0-indexed row 5)
  bool pass = true;
  for (size_t n = 0; n < ori_N; n++) {
    int last_h = seq_lens[n] - 1;
    for (int w = 0; w < ori_W; w++) {
      // Compute expected value from the folded data
      // The fold index for this (n, last_h, w)
      int h_fold_idx = last_h / fold_H;
      int h_in_fold = last_h % fold_H;
      int w_fold_idx = w / fold_W;
      int w_in_fold = w % fold_W;
      int n_fold_idx =
          n * h_fold_num * w_fold_num + w_fold_idx * h_fold_num + h_fold_idx;

      size_t fold_pos =
          (size_t)n_fold_idx * fold_H * fold_W + h_in_fold * fold_W + w_in_fold;
      float expected = __bfloat162float(host_fold[fold_pos]);
      float actual = result[n * ori_W + w];

      if (w < ori_W - pad_W) {
        if (std::fabs(actual - expected) > 0.5f) {
          printf("Mismatch at batch %zu, token %d: %f vs %f\n", n, w, actual,
                 expected);
          pass = false;
        }
      }
    }
  }

  cudaFree(d_fold);
  cudaFree(d_out);
  cudaFree(d_seq_lens);
  return pass;
}

// ============================================================================
// Test: Dense -> Multi-segment -> Dense round-trip (NUMA like)
//
// Corresponds to brReorderTest.cpp::test_weight_numa_reorder()
// Verifies that multi-segment folding then unfolding a tensor gives back the
// original data.
// ============================================================================
static bool test_multi_segment_reorder_cuda() {
  int ori_N = 1, ori_H = 256, ori_W = 64;
  int fold_N = 4, fold_H = 64, fold_W = 4;
  int fold_num = 4;
  int num_w_splits = 16;

  const size_t ori_elems = (size_t)ori_N * ori_H * ori_W;
  const size_t total_fold_elems =
      (size_t)num_w_splits * fold_N * fold_H * fold_W;

  std::vector<float> host_data(ori_elems);
  for (int n = 0; n < ori_N; n++) {
    for (int h = 0; h < ori_H; h++) {
      for (int w = 0; w < ori_W; w++) {
        host_data[n * ori_H * ori_W + h * ori_W + w] = (float)w;
      }
    }
  }

  printf(
      "[Multi-segment Reorder] ori: (%d, %d, %d) -> fold: (%d, %d, %d) * %d\n",
      ori_N, ori_H, ori_W, fold_N, fold_H, fold_W, num_w_splits);

  float *d_dense = nullptr;
  __nv_bfloat16 *d_fold = nullptr;
  CUDA_CHECK(cudaMalloc(&d_dense, ori_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fold, total_fold_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemcpy(d_dense, host_data.data(), ori_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Dense -> Multi-segment fold
  CUDA_CHECK(launchDenseToMultiSegment(d_dense, d_fold, fold_N, fold_H, fold_W,
                                       ori_N, ori_H, ori_W, fold_num,
                                       num_w_splits));
  CUDA_CHECK(cudaDeviceSynchronize());

  // Multi-segment fold -> Dense
  CUDA_CHECK(cudaMemset(d_dense, 0, ori_elems * sizeof(float)));
  CUDA_CHECK(launchMultiSegmentToDense(d_fold, d_dense, fold_N, fold_H, fold_W,
                                       ori_N, ori_H, ori_W, fold_num,
                                       num_w_splits));
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> result(ori_elems);
  CUDA_CHECK(cudaMemcpy(result.data(), d_dense, ori_elems * sizeof(float),
                        cudaMemcpyDeviceToHost));

  printf("[Multi-segment Reorder] round-trip result:\n");
  for (int w = 0; w < std::min(ori_W, 16); w++) {
    printf("%.0f ", result[w]);
  }
  printf("...\n");

  bool pass = true;
  for (size_t i = 0; i < ori_elems; i++) {
    if (std::fabs(result[i] - host_data[i]) > 0.1f) {
      printf("Mismatch at index %zu: %f vs %f\n", i, result[i], host_data[i]);
      pass = false;
    }
  }

  cudaFree(d_dense);
  cudaFree(d_fold);
  return pass;
}

// ============================================================================
// Test: Larger scale round-trip with various fold configs
// ============================================================================
static bool test_large_reorder_cuda() {
  struct TestConfig {
    int ori_N, ori_H, ori_W;
    int h_fold_num, w_fold_num;
  };

  std::vector<TestConfig> configs = {
      {1, 128, 256, 1, 1}, {1, 128, 256, 2, 1}, {1, 128, 256, 1, 2},
      {1, 128, 256, 2, 2}, {2, 64, 128, 2, 2},  {4, 32, 512, 4, 1},
      {1, 1024, 64, 4, 2},
  };

  bool all_pass = true;
  for (auto cfg : configs) {
    int fold_N = cfg.ori_N * cfg.h_fold_num * cfg.w_fold_num;
    int fold_H = cfg.ori_H / cfg.h_fold_num;
    int fold_W = cfg.ori_W / cfg.w_fold_num;

    const size_t ori_elems = (size_t)cfg.ori_N * cfg.ori_H * cfg.ori_W;
    const size_t fold_elems = (size_t)fold_N * fold_H * fold_W;

    printf("[Large Reorder] ori: (%d, %d, %d) fold: (%d, %d, %d) h_fold=%d "
           "w_fold=%d ... ",
           cfg.ori_N, cfg.ori_H, cfg.ori_W, fold_N, fold_H, fold_W,
           cfg.h_fold_num, cfg.w_fold_num);

    std::vector<float> host_data(ori_elems);
    for (size_t i = 0; i < ori_elems; i++) {
      host_data[i] = (float)(i % 1000) * 0.1f;
    }

    float *d_dense = nullptr;
    __nv_bfloat16 *d_fold = nullptr;
    cudaMalloc(&d_dense, ori_elems * sizeof(float));
    cudaMalloc(&d_fold, fold_elems * sizeof(__nv_bfloat16));
    cudaMemcpy(d_dense, host_data.data(), ori_elems * sizeof(float),
               cudaMemcpyHostToDevice);

    launchDenseToFold(d_dense, d_fold, fold_N, fold_H, fold_W, cfg.ori_N,
                      cfg.ori_H, cfg.ori_W, cfg.h_fold_num, cfg.w_fold_num);
    cudaDeviceSynchronize();

    cudaMemset(d_dense, 0, ori_elems * sizeof(float));
    launchFoldToDense(d_fold, d_dense, fold_N, fold_H, fold_W, cfg.ori_N,
                      cfg.ori_H, cfg.ori_W, cfg.h_fold_num, cfg.w_fold_num);
    cudaDeviceSynchronize();

    std::vector<float> result(ori_elems);
    cudaMemcpy(result.data(), d_dense, ori_elems * sizeof(float),
               cudaMemcpyDeviceToHost);

    bool pass = true;
    int mismatch_count = 0;
    for (size_t i = 0; i < ori_elems; i++) {
      if (std::fabs(result[i] - host_data[i]) > 0.25f) {
        if (mismatch_count < 5) {
          printf("Mismatch at index %zu: %f vs %f\n", i, result[i],
                 host_data[i]);
          mismatch_count++;
          pass = false;
        }
      }
    }
    printf("%s (mismatches: %d/%zu)\n", pass ? "PASS" : "FAIL", mismatch_count,
           ori_elems);
    all_pass = all_pass && pass;

    cudaFree(d_dense);
    cudaFree(d_fold);
  }

  return all_pass;
}

// ============================================================================
// GTest Registrations
// ============================================================================

TEST(ReorderCUDATest, uma_roundtrip) { EXPECT_TRUE(test_uma_reorder_cuda()); }

TEST(ReorderCUDATest, fold_roundtrip) { EXPECT_TRUE(test_fold_reorder_cuda()); }

TEST(ReorderCUDATest, last_token_extract) {
  EXPECT_TRUE(test_last_token_reorder_cuda());
}

TEST(ReorderCUDATest, multi_segment_roundtrip) {
  EXPECT_TRUE(test_multi_segment_reorder_cuda());
}

TEST(ReorderCUDATest, large_roundtrip) {
  EXPECT_TRUE(test_large_reorder_cuda());
}
