// MIT License

// Copyright (c) 2026 CUI Xin (崔 欣)

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void RmsNormCUDAKernel(__nv_bfloat16 *__restrict__ output, const __nv_bfloat16 *__restrict__ input, const float *__restrict__ weight, const int NH, const int W, const float eps)
{
    // Implementation of the RMSNorm CUDA kernel with BlockLoad, BlockStore, and BlockReduce.
    // This kernel computes the RMS normalization for each row of the input matrix.
    // The weight is read from global memory for each tile (non-cached variant).
    // The kernel is designed to handle arbitrary hidden sizes (W) and batch sizes (N*H).

    constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    // CUB block-level cooperative primitive types
    using BlockLoadBF16 = cub::BlockLoad<__nv_bfloat16, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreBF16 = cub::BlockStore<__nv_bfloat16, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockLoadF32 = cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockReduceF32 = cub::BlockReduce<float, BLOCK_SIZE>;

    // Shared memory: union of CUB temp storage (only one active at a time)
    __shared__ union
    {
        typename BlockLoadBF16::TempStorage load_bf16;
        typename BlockStoreBF16::TempStorage store_bf16;
        typename BlockLoadF32::TempStorage load_f32;
        typename BlockReduceF32::TempStorage reduce;
    } temp_storage;

    __shared__ float s_inv_rms;

    const int row = blockIdx.x;
    if (row >= NH)
        return;

    const __nv_bfloat16 *row_in = input + static_cast<int64_t>(row) * W;
    __nv_bfloat16 *row_out = output + static_cast<int64_t>(row) * W;

    // ====================== Pass 1: Sum of Squares ======================
    // Tile over W, each tile loads TILE_SIZE elements cooperatively via BlockLoad.
    float thread_ssq = 0.0f;

    for (int tile_off = 0; tile_off < W; tile_off += TILE_SIZE)
    {
        __nv_bfloat16 items[ITEMS_PER_THREAD];
        const int valid = (W - tile_off < TILE_SIZE) ? (W - tile_off) : TILE_SIZE;

        // Cooperative load of input tile (BF16)
        BlockLoadBF16(temp_storage.load_bf16).Load(row_in + tile_off, items, valid, __float2bfloat16(0.0f));
        __syncthreads();

// Compute sum of squares for the tile
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            float v = __bfloat162float(items[i]);
            thread_ssq += v * v;
        }
        __syncthreads();
    }

    // Cooperative block-level reduction: sum per-thread SSQ across the block
    float block_ssq = BlockReduceF32(temp_storage.reduce).Sum(thread_ssq);
    __syncthreads();

    // Thread 0 computes the inverse RMS and broadcasts to the block
    if (threadIdx.x == 0)
    {
        s_inv_rms = rsqrtf(block_ssq / static_cast<float>(W) + eps);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // ====================== Pass 2: Normalize and Scale ======================
    // Tile over W again, load input and weight, compute output.
    for (int tile_off = 0; tile_off < W; tile_off += TILE_SIZE)
    {
        __nv_bfloat16 in_items[ITEMS_PER_THREAD];
        float w_items[ITEMS_PER_THREAD];
        const int valid = (W - tile_off < TILE_SIZE) ? (W - tile_off) : TILE_SIZE;

        // Cooperative load of input tile (BF16)
        BlockLoadBF16(temp_storage.load_bf16).Load(row_in + tile_off, in_items, valid, __float2bfloat16(0.0f));
        __syncthreads();

        // Cooperative load of weight tile (F32)
        BlockLoadF32(temp_storage.load_f32).Load(weight + tile_off, w_items, valid, 0.0f);
        __syncthreads();

        // Compute normalized output: x * inv_rms * weight, convert FP32 -> BF16
        __nv_bfloat16 out_items[ITEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            float v = __bfloat162float(in_items[i]);
            out_items[i] = __float2bfloat16(v * inv_rms * w_items[i]);
        }

        // Cooperative store of output tile (BF16)
        BlockStoreBF16(temp_storage.store_bf16).Store(row_out + tile_off, out_items, valid);
        __syncthreads();
    }
}

// -------------------------------------------------------------------------------
// Optimized variant: weight is cached in shared memory to reduce global reads.
// Suitable when N * H is large (many blocks reuse the same weight vector).
// Weight must fit in shared memory: W * sizeof(float) <= avaiable smem.
// -------------------------------------------------------------------------------
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void
RmsNormCUDAKernel_CachedWeight(__nv_bfloat16 *__restrict__ output, const __nv_bfloat16 *__restrict__ input, const float *__restrict__ weight, int32_t NH, int32_t W, float eps)
{
    constexpr int TILE_SIZE = BLOCK_SIZE * ITEMS_PER_THREAD;

    using BlockLoadBF16 = cub::BlockLoad<__nv_bfloat16, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreBF16 = cub::BlockStore<__nv_bfloat16, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockReduceF32 = cub::BlockReduce<float, BLOCK_SIZE>;

    __shared__ union
    {
        typename BlockLoadBF16::TempStorage load_bf16;
        typename BlockStoreBF16::TempStorage store_bf16;
        typename BlockReduceF32::TempStorage reduce;
    } temp_storage;

    // Dynamic shared memory for weight cache
    extern __shared__ float smem_weight[];

    __shared__ float s_inv_rms;

    // Cooperative weight loading into shared memory
    for (int i = threadIdx.x; i < W; i += BLOCK_SIZE)
    {
        smem_weight[i] = weight[i];
    }
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= NH)
        return;

    const __nv_bfloat16 *row_in = input + static_cast<int64_t>(row) * W;
    __nv_bfloat16 *row_out = output + static_cast<int64_t>(row) * W;

    // ====================== Pass 1: Sum of Squares ======================
    float thread_ssq = 0.0f;

    for (int tile_off = 0; tile_off < W; tile_off += TILE_SIZE)
    {
        __nv_bfloat16 items[ITEMS_PER_THREAD];
        const int valid = (W - tile_off < TILE_SIZE) ? (W - tile_off) : TILE_SIZE;

        BlockLoadBF16(temp_storage.load_bf16).Load(row_in + tile_off, items, valid, __float2bfloat16(0.0f));
        __syncthreads();

#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            float v = __bfloat162float(items[i]);
            thread_ssq += v * v;
        }
        __syncthreads();
    }

    float block_ssq = BlockReduceF32(temp_storage.reduce).Sum(thread_ssq);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        s_inv_rms = rsqrtf(block_ssq / static_cast<float>(W) + eps);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;

    // ====================== Pass 2: Normalize with Cached Weight ======================
    for (int tile_off = 0; tile_off < W; tile_off += TILE_SIZE)
    {
        __nv_bfloat16 in_items[ITEMS_PER_THREAD];
        const int valid = (W - tile_off < TILE_SIZE) ? (W - tile_off) : TILE_SIZE;

        BlockLoadBF16(temp_storage.load_bf16).Load(row_in + tile_off, in_items, valid, __float2bfloat16(0.0f));
        __syncthreads();

        // Read weights from shared memory cache (blocked arrangement matches BlockLoad)
        __nv_bfloat16 out_items[ITEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; i++)
        {
            int global_idx = tile_off + threadIdx.x * ITEMS_PER_THREAD + i;
            float w = (global_idx < W) ? smem_weight[global_idx] : 0.0f;
            float v = __bfloat162float(in_items[i]);
            out_items[i] = __float2bfloat16(v * inv_rms * w);
        }

        BlockStoreBF16(temp_storage.store_bf16).Store(row_out + tile_off, out_items, valid);
        __syncthreads();
    }
}

// -------------------------------------------------------------------------------
// Launch wrapper: selects kernel variant and ITEMS_PER_THREAD based on W.
// -------------------------------------------------------------------------------
inline cudaError_t launchRmsNormCUDAKernel(__nv_bfloat16 *output, const __nv_bfloat16 *input, const float *weight, int32_t N, int32_t H, int32_t W, float eps, cudaStream_t stream = nullptr)
{
    const int NH = N * H;
    constexpr int BLOCK_SIZE = 256;

    // Use cached-weight variant for large batch (many rows reuse same weight)
    // and when weight fits in shared memory (W <= 8192 -> 32KB)
    const bool use_cached = (NH > 256) && (W <= 8192);

    if (use_cached)
    {
        const int smem_bytes = W * sizeof(float);
        if (W <= BLOCK_SIZE * 8)
        {
            constexpr int IPT = 8;
            RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W, eps);
        }
        else if (W <= BLOCK_SIZE * 16)
        {
            constexpr int IPT = 16;
            RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W, eps);
        }
        else
        {
            constexpr int IPT = 16;
            RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W, eps);
        }
    }
    else
    {
        if (W <= BLOCK_SIZE * 8)
        {
            constexpr int IPT = 8;
            RmsNormCUDAKernel<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
        }
        else if (W <= BLOCK_SIZE * 16)
        {
            constexpr int IPT = 16;
            RmsNormCUDAKernel<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
        }
        else
        {
            constexpr int IPT = 16;
            RmsNormCUDAKernel<BLOCK_SIZE, IPT><<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
        }
    }

    return cudaGetLastError();
}