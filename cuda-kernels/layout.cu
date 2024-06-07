#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void transpose(const size_t numel, const uint32_t d1, const uint32_t d2, const uint32_t d_i, const uint32_t d_j, const uint32_t d_k, const T * src, T * dst) {
    const size_t dst_idx = blockIdx.x * blockDim.x + threadIdx.x;


    // The implementation below is very slow as it computes lots of divisions and multiplications.
    // TODO: Replace it with an optimized implementation and/or process data by blocks.
    size_t dst_idx2 = dst_idx;
    const size_t i = dst_idx2 / (d2 * d_j * d1 * d_k);
    dst_idx2 -= i * d2 * d_j * d1 * d_k;
    const size_t a2 = dst_idx2 / (d_j * d1 * d_k);
    dst_idx2 -= a2 * d_j * d1 * d_k;
    const size_t j = dst_idx2 / (d1 * d_k);
    dst_idx2 -= j * d1 * d_k;
    const size_t a1 = dst_idx2 / d_k;
    dst_idx2 -= a1 * d_k;
    const size_t k = dst_idx2;
    const size_t src_idx = i * d1 * d_j * d2 * d_k + a1 * d_j * d2 * d_k + j * d2 * d_k + a2 * d_k + k;

    dst[dst_idx] = src[src_idx];
}

#define OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void transpose_##RUST_NAME ( \
      const size_t numel, \
      const uint32_t d1, \
      const uint32_t d2, \
      const uint32_t d_i, \
      const uint32_t d_j, \
      const uint32_t d_k, \
      const TYPENAME *src, \
      TYPENAME *dst) { \
    transpose<TYPENAME>(numel, d1, d2, d_i, d_j, d_k, src, dst); \
  } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)


