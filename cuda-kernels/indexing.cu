#include "cuda_fp16.h"
#include "cuda_bf16.h"
#include<stdint.h>

template<typename T, typename I>
__device__ void index_select(
    const int32_t numel,
    const int32_t dim,
    const I *ids,
    const T *src,
    T *dst
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= numel || j >= dim) {
      return;
    }
    dst[i * dim + j] = src[ids[i] * dim + j];
}

#define IS_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const int32_t numel,  \
    const int32_t dim, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *src, \
    TYPENAME *dst \
) { index_select(numel, dim, ids, src, dst); } \

#if __CUDA_ARCH__ >= 800
IS_OP(__nv_bfloat16, uint32_t, is_u32_bf16);
#endif
#if __CUDA_ARCH__ >= 530
IS_OP(__half, uint32_t, is_u32_f16);
#endif

IS_OP(float, uint32_t, is_u32_f32);
