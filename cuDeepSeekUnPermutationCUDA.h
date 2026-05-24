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

#pragma once

#include <algorithm>
#include <cstdint>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ================================================================================
// DeepSeek Unpermutation CUDA Kernel using CCCL (CUB) block-level cooperative
// APIs
//
// Algorithm (MoE UnPermutation / Gather-Accumulate):
//   For each output token t:
//     output[t, :] = sum_{k=0}^{topk-1} hidden_states[gather_src[t*topk+k], :]
//     * gather_probs[t*topk+k]
//
// Implementation:
//   - cub::BlockLoad    : cooperative block-level load from global memory
//   - cub::BlockStore   : cooperative block-level store to global memory
//   - Gather pattern: each thread block processes one output token's tile of
//   hidden_size
//   - Float32 accumulation for precision, final output in BF16
// ================================================================================

// ---------------------------------------------------------------------------------
// Main kernel: Gather-Accumulate pattern
// Each thread block processes one output token over a tile of the hidden
// dimension. For each topk expert contributing to this output token, the block
// cooperatively loads the expert's hidden states via cub::BlockLoad, multiplies
// by routing probability, and accumulates in FP32. The final result is
// converted to BF16 and written out via cub::BlockStore.
//
// Grid: (ceil(hidden size / TILE_SIZE) * token num)
// bLOCK: (BLOCK_THREADS)
// ---------------------------------------------------------------------------------
template <int BLOCK_THREADS, int ITEMS_PER_THREAD, int MAX_TOPK>
__global__ void DeepSeekUnPermutationKernel(
    const __nv_bfloat16
        *__restrict__ hidden_states,    // [total_expert_tokens, hidden_size]
    __nv_bfloat16 *__restrict__ output, // [token_num, hidden_size]
    const int32_t *__restrict__ gather_src, // [token_num * topk] -> expert
                                            // token row index (-1 = invalid)
    const float
        *__restrict__ gather_probs, // [token_num * topk] -> routing probability
    int32_t hidden_size, int32_t topk, int32_t token_num) {
  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
  const int tiles_per_token = (hidden_size + TILE_SIZE - 1) / TILE_SIZE;
  const int token_idx = blockIdx.x / tiles_per_token;
  const int tile_idx = blockIdx.x - token_idx * tiles_per_token;
  const int tile_start = tile_idx * TILE_SIZE;

  if (token_idx >= token_num || tile_start >= hidden_size)
    return;

  // CUB block-level cooperative load/store types
  using BlockLoadBF16 =
      cub::BlockLoad<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                     cub::BLOCK_LOAD_WARP_TRANSPOSE>;

  using BlockStoreBF16 =
      cub::BlockStore<__nv_bfloat16, BLOCK_THREADS, ITEMS_PER_THREAD,
                      cub::BLOCK_STORE_WARP_TRANSPOSE>;

  // Shared memory: union of CUB temp storages (only one active at a time)
  __shared__ union {
    typename BlockLoadBF16::TempStorage load;
    typename BlockStoreBF16::TempStorage store;
  } temp_storage;

  const int valid_items = min(TILE_SIZE, hidden_size - tile_start);
  const int gather_base = token_idx * topk;

  // Preload gather_src and gather_probs for this token into registers
  int32_t src_rows[MAX_TOPK];
  float probs[MAX_TOPK];
#pragma unroll
  for (int k = 0; k < MAX_TOPK; k++) {
    if (k < topk) {
      src_rows[k] = gather_src[gather_base + k];
      probs[k] = gather_probs[gather_base + k];
    } else {
      src_rows[k] = -1;
      probs[k] = 0.0f;
    }
  }

  // FP32 accumulator for precision
  float accum[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    accum[i] = 0.0f;
  }

  // Gather from all topk contributing expert tokens
  for (int k = 0; k < topk; k++) {
    const int32_t src_row = src_rows[k];
    const float prob = probs[k];

    if (src_row < 0)
      continue;

    __nv_bfloat16 items[ITEMS_PER_THREAD];
    const __nv_bfloat16 *src_ptr = hidden_states +
                                   static_cast<int64_t>(src_row) * hidden_size +
                                   tile_start;

    // Cooperative block-level load: all threads collectively load a tile
    if (valid_items == TILE_SIZE) {
      BlockLoadBF16(temp_storage.load).Load(src_ptr, items);
    } else {
      BlockLoadBF16(temp_storage.load)
          .Load(src_ptr, items, valid_items, __float2bfloat16(0.0f));
    }
    __syncthreads();

// Accumulate: output += hidden_states * prob
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      accum[i] += __bfloat162float(items[i]) * prob;
    }
    __syncthreads();
  }

  // Convert FP32 accumulator back to BF16
  __nv_bfloat16 out_items[ITEMS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    out_items[i] = __float2bfloat16(accum[i]);
  }
  __syncthreads();

  // Cooperative block-level store: all threads collectively write the tile
  __nv_bfloat16 *out_ptr =
      output + static_cast<int64_t>(token_idx) * hidden_size + tile_start;

  if (valid_items == TILE_SIZE) {
    BlockStoreBF16(temp_storage.store).Store(out_ptr, out_items);
  } else {
    BlockStoreBF16(temp_storage.store).Store(out_ptr, out_items, valid_items);
  }
}

// -----------------------------------------------------------------------------
// Host-side launcher
// -----------------------------------------------------------------------------
inline cudaError_t launchDeepSeekUnPermutationKernel(
    const __nv_bfloat16 *hidden_states, __nv_bfloat16 *output,
    const int32_t *gather_src, const float *gather_probs, int32_t hidden_size,
    int32_t topk, int32_t token_num, cudaStream_t stream = nullptr) {
  if (token_num <= 0 || hidden_size <= 0 || topk <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

  const int grid_x = (hidden_size + TILE_SIZE - 1) / TILE_SIZE;
  const int64_t total_blocks = static_cast<int64_t>(grid_x) * token_num;
  if (total_blocks > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid(static_cast<uint32_t>(total_blocks));

  // Dispatch based on topk to allow compile-time unrolling
  switch (topk) {
  case 1:
    DeepSeekUnPermutationKernel<BLOCK_THREADS, ITEMS_PER_THREAD, 1>
        <<<grid, dim3(BLOCK_THREADS), 0, stream>>>(
            hidden_states, output, gather_src, gather_probs, hidden_size, topk,
            token_num);
    break;
  case 2:
    DeepSeekUnPermutationKernel<BLOCK_THREADS, ITEMS_PER_THREAD, 2>
        <<<grid, dim3(BLOCK_THREADS), 0, stream>>>(
            hidden_states, output, gather_src, gather_probs, hidden_size, topk,
            token_num);
    break;
  case 4:
    DeepSeekUnPermutationKernel<BLOCK_THREADS, ITEMS_PER_THREAD, 4>
        <<<grid, dim3(BLOCK_THREADS), 0, stream>>>(
            hidden_states, output, gather_src, gather_probs, hidden_size, topk,
            token_num);
    break;
  case 8:
    DeepSeekUnPermutationKernel<BLOCK_THREADS, ITEMS_PER_THREAD, 8>
        <<<grid, dim3(BLOCK_THREADS), 0, stream>>>(
            hidden_states, output, gather_src, gather_probs, hidden_size, topk,
            token_num);
    break;
  default:
    // Fallback: use MAX_TOPK = 16 for any topk <= 16
    if (topk <= 16) {
      DeepSeekUnPermutationKernel<BLOCK_THREADS, ITEMS_PER_THREAD, 16>
          <<<grid, dim3(BLOCK_THREADS), 0, stream>>>(
              hidden_states, output, gather_src, gather_probs, hidden_size,
              topk, token_num);
    } else {
      return cudaErrorInvalidValue;
    }
    break;
  }

  return cudaGetLastError();
}
