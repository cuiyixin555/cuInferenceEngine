#include "cuElementwiseCUDAKernel_launch.h"
#include "elementwise_kernels.cuh"

namespace {

constexpr int kTile = 256;

inline dim3 elementwiseGrid(int N) {
  return dim3((N + kTile - 1) / kTile);
}

} // namespace

cudaError_t launchElementwiseAddF32(float *a, float *b, float *c, int N,
                                    cudaStream_t stream) {
  elementwise_add_f32_kernel<<<elementwiseGrid(N), kTile, 0, stream>>>(a, b, c,
                                                                         N);
  return cudaGetLastError();
}

cudaError_t launchElementwiseAddF32x4(float *a, float *b, float *c, int N,
                                      cudaStream_t stream) {
  elementwise_add_f32x4_kernel<<<elementwiseGrid(N), kTile / 4, 0, stream>>>(
      a, b, c, N);
  return cudaGetLastError();
}

cudaError_t launchElementwiseAddF16(half *a, half *b, half *c, int N,
                                    cudaStream_t stream) {
  elementwise_add_f16_kernel<<<elementwiseGrid(N), kTile, 0, stream>>>(a, b, c,
                                                                         N);
  return cudaGetLastError();
}

cudaError_t launchElementwiseAddF16x2(half *a, half *b, half *c, int N,
                                      cudaStream_t stream) {
  elementwise_add_f16x2_kernel<<<elementwiseGrid(N), kTile / 2, 0, stream>>>(
      a, b, c, N);
  return cudaGetLastError();
}

cudaError_t launchElementwiseAddF16x8(half *a, half *b, half *c, int N,
                                      cudaStream_t stream) {
  elementwise_add_f16x8_kernel<<<elementwiseGrid(N), kTile / 8, 0, stream>>>(
      a, b, c, N);
  return cudaGetLastError();
}

cudaError_t launchElementwiseAddF16x8Pack(half *a, half *b, half *c, int N,
                                          cudaStream_t stream) {
  elementwise_add_f16x8_pack_kernel<<<elementwiseGrid(N), kTile / 8, 0, stream>>>(
      a, b, c, N);
  return cudaGetLastError();
}
