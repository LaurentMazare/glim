#include<stdint.h>

template<typename T, typename I>
__device__ void index_select(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    bool b = is_contiguous(num_dims, dims, strides);
    for (unsigned int dst_i = blockIdx.x * blockDim.x + threadIdx.x; dst_i < numel; dst_i += blockDim.x * gridDim.x) {
          unsigned int left_i = dst_i / (ids_dim_size * right_size);
          unsigned int id_i = dst_i / right_size % ids_dim_size;
          unsigned int right_i = dst_i % right_size;
          unsigned int src_i = left_i * (src_dim_size * right_size) + ids[id_i] * right_size + right_i;
          unsigned strided_i = b ? src_i : get_strided_index(src_i, num_dims, dims, strides);
          out[dst_i] = inp[strided_i];
    }
}
