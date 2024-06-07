#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template <typename T>
__device__ void scale(const size_t numel, T * dst, const T v) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] *= v;
}

template <typename T>
__device__ void add_assign(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] += src[idx];
}

template <typename T>
__device__ void mul_assign(const size_t numel, const T * src, T * dst) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    dst[idx] *= src[idx];
}

#define OPS(TYPENAME, RUST_NAME) \
  extern "C" __global__ void scale_##RUST_NAME ( \
      const size_t numel, \
      TYPENAME *dst, \
      const TYPENAME v) { \
    scale<TYPENAME>(numel, dst, v); \
  } \
  extern "C" __global__ void add_assign_##RUST_NAME ( \
      const size_t numel, \
      const TYPENAME *src, \
      TYPENAME *dst) { \
    add_assign<TYPENAME>(numel, src, dst); \
  } \
  extern "C" __global__ void mul_assign_##RUST_NAME ( \
      const size_t numel, \
      const TYPENAME *src, \
      TYPENAME *dst) { \
    mul_assign<TYPENAME>(numel, src, dst); \
  } \

#if __CUDA_ARCH__ >= 800
OPS(__nv_bfloat16, bf16)
#endif

#if __CUDA_ARCH__ >= 530
OPS(__half, f16)
#endif

OPS(float, f32)
OPS(double, f64)

