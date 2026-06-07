#include "cuRmsNormCUDAKernel.h"

cudaError_t launchRmsNormCUDAKernel(__nv_bfloat16 *output,
                                    const __nv_bfloat16 *input,
                                    const float *weight, int32_t N,
                                    int32_t H, int32_t W, float eps,
                                    cudaStream_t stream) {
  const int NH = N * H;
  constexpr int BLOCK_SIZE = 256;

  const bool use_cached = (NH > 256) && (W <= 8192);

  if (use_cached) {
    const int smem_bytes = W * sizeof(float);
    if (W <= BLOCK_SIZE * 8) {
      constexpr int IPT = 8;
      RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W,
                                                   eps);
    } else if (W <= BLOCK_SIZE * 16) {
      constexpr int IPT = 16;
      RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W,
                                                   eps);
    } else {
      constexpr int IPT = 16;
      RmsNormCUDAKernel_CachedWeight<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, smem_bytes, stream>>>(output, input, weight, NH, W,
                                                   eps);
    }
  } else {
    if (W <= BLOCK_SIZE * 8) {
      constexpr int IPT = 8;
      RmsNormCUDAKernel<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
    } else if (W <= BLOCK_SIZE * 16) {
      constexpr int IPT = 16;
      RmsNormCUDAKernel<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
    } else {
      constexpr int IPT = 16;
      RmsNormCUDAKernel<BLOCK_SIZE, IPT>
          <<<NH, BLOCK_SIZE, 0, stream>>>(output, input, weight, NH, W, eps);
    }
  }

  return cudaGetLastError();
}
