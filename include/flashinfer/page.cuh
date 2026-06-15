/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_PAGE_CUH_
#define FLASHINFER_PAGE_CUH_

#include <driver_types.h>

#include <vector>

#include "exception.h"
#include "fastdiv.cuh"
#include "layout.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

/*!
 * \brief Paged key-value cache
 * \tparam layout The layout of last 3 dimensions in KV-Cache.
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 */
template <typename DType, typename IdType>
struct paged_kv_t {
  uint_fastdiv page_size;
  uint32_t num_heads;
  uint32_t head_dim;
  uint32_t batch_size;
  uint32_t stride_page;
  uint32_t stride_n;
  uint32_t stride_h;

  // Internal layout:
  // [max_num_pages, num_heads, page_size, head_dim] if layout == HND
  // [max_num_pages, page_size, num_heads, head_dim] if layout == NHD
  DType* k_data;
  DType* v_data;
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  __host__ __device__ __forceinline__ paged_kv_t()
      : num_heads(0),
        page_size(),
        head_dim(0),
        batch_size(0),
        stride_page(0),
        stride_n(0),
        stride_h(0),
        k_data(nullptr),
        v_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged key-value cache
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache should be contiguous
   * \param v_data The start pointer of value cache, v_cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* k_data,
                                      DType* v_data, IdType* indices, IdType* indptr,
                                      IdType* last_page_len, IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = num_heads * page_size * head_dim;
    this->k_data = k_data;
    this->v_data = v_data;
    stride_n = layout == QKVLayout::kHND ? head_dim : num_heads * head_dim;
    stride_h = layout == QKVLayout::kHND ? page_size * head_dim : head_dim;
  }

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param num_heads The number of heads
   * \param page_size The size of each page
   * \param head_dim The dimension of each head
   * \param batch_size The batch size
   * \param layout The layout of last 3 dimensions in KV-Cache.
   * \param k_data The start pointer of key cache, k_cache doesn't have to be contiguous
   * \param v_data The start pointer of value cache, v_cache doesn't have to be contiguous
   * \param kv_strides custom strides of each dimensions of k_data and v_data
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_t(uint32_t num_heads, uint32_t page_size, uint32_t head_dim,
                                      uint32_t batch_size, QKVLayout layout, DType* k_data,
                                      DType* v_data, const int64_t* kv_strides, IdType* indices,
                                      IdType* indptr, IdType* last_page_len,
                                      IdType* rope_pos_offset = nullptr)
      : num_heads(num_heads),
        page_size(page_size),
        head_dim(head_dim),
        batch_size(batch_size),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page = kv_strides[0];
    this->k_data = k_data;
    this->v_data = v_data;
    stride_n = layout == QKVLayout::kHND ? kv_strides[2] : kv_strides[1];
    stride_h = layout == QKVLayout::kHND ? kv_strides[1] : kv_strides[2];
  }

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  /*!
   * \brief Compute the offset of element in the allocated buffer.
   * \param page_idx The page index
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_elem_offset(size_t page_idx, size_t head_idx,
                                                             size_t entry_idx,
                                                             size_t feat_idx) const {
    return page_idx * stride_page + head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  /*!
   * \brief Compute the offset of element inside the page.
   * \param head_idx The head index
   * \param entry_idx The page entry index
   * \param feat_idx The feature index
   */
  __host__ __device__ __forceinline__ size_t get_elem_offset_in_page(size_t head_idx,
                                                                     size_t entry_idx,
                                                                     size_t feat_idx) const {
    return head_idx * stride_h + entry_idx * stride_n + feat_idx;
  }

  __device__ __forceinline__ DType* get_k_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return k_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  __device__ __forceinline__ size_t protective_get_kv_offset(IdType page_iter, uint32_t head_idx,
                                                             uint32_t entry_idx, uint32_t feat_idx,
                                                             IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __device__ __forceinline__ DType* protective_get_k_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return k_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }

  __device__ __forceinline__ DType* get_v_ptr(IdType page_iter, uint32_t head_idx,
                                              uint32_t entry_idx, uint32_t feat_idx) const {
    return v_data + get_elem_offset(__ldg(indices + page_iter), head_idx, entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* protective_get_v_ptr(IdType page_iter, uint32_t head_idx,
                                                         uint32_t entry_idx, uint32_t feat_idx,
                                                         IdType last_indptr) const {
    return v_data + protective_get_kv_offset(page_iter, head_idx, entry_idx, feat_idx, last_indptr);
  }
};

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the decode phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheDecodeKernel(paged_kv_t<DType, IdType> paged_kv,
                                               DType* __restrict__ key, DType* __restrict__ value) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t batch_idx = blockIdx.x;
  uint32_t head_idx = ty;

  uint32_t seq_len =
      (paged_kv.indptr[batch_idx + 1] - paged_kv.indptr[batch_idx] - 1) * paged_kv.page_size +
      paged_kv.last_page_len[batch_idx];

  uint32_t page_iter = paged_kv.indptr[batch_idx] + (seq_len - 1) / paged_kv.page_size;
  uint32_t entry_idx = (seq_len - 1) % paged_kv.page_size;

  DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
  vec_t<DType, vec_size>::memcpy(
      k_ptr, key + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);

  vec_t<DType, vec_size>::memcpy(
      v_ptr, value + (batch_idx * num_heads + head_idx) * head_dim + tx * vec_size);
}

/*!
 * \brief CUDA kernel to append new keys/values to the paged key-value cache in the prefill phase
 * \tparam head_dim The dimension of each head
 * \tparam vec_size The vector size used in the kernel
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param batch_indices The batch indices of elements to be appended
 * \param positions The positions of elements to be appended
 */
template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void AppendPagedKVCacheKernel(paged_kv_t<DType, IdType> paged_kv,
                                         DType* __restrict__ append_key,
                                         DType* __restrict__ append_value,
                                         IdType* __restrict__ batch_indices,
                                         IdType* __restrict__ positions, uint32_t nnz,
                                         size_t append_k_stride_n, size_t append_k_stride_h,
                                         size_t append_v_stride_n, size_t append_v_stride_h) {
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t num_heads = paged_kv.num_heads;
  uint32_t head_idx = ty;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(paged_kv.indptr[batch_indices[i]] * paged_kv.page_size + positions[i],
                              page_iter, entry_idx);
    DType* k_ptr = paged_kv.get_k_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    DType* v_ptr = paged_kv.get_v_ptr(page_iter, head_idx, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        k_ptr, append_key + i * append_k_stride_n + head_idx * append_k_stride_h + tx * vec_size);
    vec_t<DType, vec_size>::memcpy(
        v_ptr, append_value + i * append_v_stride_n + head_idx * append_v_stride_h + tx * vec_size);
  }
}

/*!
 * \brief Append new keys/values to the paged key-value cache in the decode phase
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCacheDecode(paged_kv_t<DType, IdType> paged_kv, DType* key, DType* value,
                                     cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t batch_size = paged_kv.batch_size;
  uint32_t num_heads = paged_kv.num_heads;
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    // NOTE(Zihao): could be slow for small batch size, will optimize later
    dim3 nblks(batch_size);
    dim3 nthrs(bdx, bdy);
    auto kernel = AppendPagedKVCacheDecodeKernel<HEAD_DIM, vec_size, DType, IdType>;
    void* args[] = {(void*)&paged_kv, (void*)&key, (void*)&value};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

/*!
 * \brief Append new keys/values to the paged key-value cache
 * \tparam layout The layout of last 3 dimension in KV-Cache
 * \tparam DType The data type of the key-value cache
 * \tparam IdType The index data type of the kv-cache
 * \param paged_kv The paged key-value cache
 * \param key The key to be appended
 * \param value The value to be appended
 * \param append_indptr The indptr array of the appended ragged tensor
 * \param stream The CUDA stream to execute kernels.
 * \return status Indicates whether CUDA calls are successful
 */
template <typename DType, typename IdType>
cudaError_t AppendPagedKVCache(paged_kv_t<DType, IdType> paged_kv, DType* append_key,
                               DType* append_value, IdType* batch_indices, IdType* positions,
                               uint32_t nnz, size_t append_k_stride_n, size_t append_k_stride_h,
                               size_t append_v_stride_n, size_t append_v_stride_h,
                               cudaStream_t stream = nullptr) {
  uint32_t head_dim = paged_kv.head_dim;
  uint32_t num_heads = paged_kv.num_heads;
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = AppendPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                       num_threads, smem_size));
    num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(int(nnz), num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    void* args[] = {(void*)&paged_kv,          (void*)&append_key,        (void*)&append_value,
                    (void*)&batch_indices,     (void*)&positions,         (void*)&nnz,
                    (void*)&append_k_stride_n, (void*)&append_k_stride_h, (void*)&append_v_stride_n,
                    (void*)&append_v_stride_h};
    FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template <typename DType, typename IdType>
struct paged_kv_mla_t {
  uint_fastdiv page_size;
  uint32_t head_dim_ckv;
  uint32_t head_dim_kpe;
  uint32_t batch_size;
  uint32_t stride_page_ckv;
  uint32_t stride_page_kpe;
  uint32_t stride_n_ckv;
  uint32_t stride_n_kpe;

  // Internal layout:
  // [max_num_pages, page_size, head_dim]
  DType* ckv_data;
  DType* kpe_data;
  IdType* indices;

  // [batch_size + 1] The page indptr array, with the first element 0, the last element nnz_pages
  IdType* indptr;
  // [batch_size] The offset of the last page for each request in the batch
  IdType* last_page_len;
  // [batch_size] The start position of each request in the batch.
  IdType* rope_pos_offset;

  /*!
   * \brief Construct an empty paged key-value cache
   */
  __host__ __device__ __forceinline__ paged_kv_mla_t()
      : head_dim_ckv(0),
        head_dim_kpe(0),
        batch_size(0),
        stride_page_ckv(0),
        stride_page_kpe(0),
        stride_n_ckv(0),
        stride_n_kpe(0),
        ckv_data(nullptr),
        kpe_data(nullptr),
        indices(nullptr),
        indptr(nullptr),
        last_page_len(nullptr),
        rope_pos_offset(nullptr) {}

  /*!
   * \brief Construct a paged mla kv cache
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_mla_t(uint32_t page_size, uint32_t head_dim_compressed_kv,
                                          uint32_t head_dim_kpe, uint32_t batch_size,
                                          DType* compressed_kv_data, DType* kpe_data,
                                          IdType* indices, IdType* indptr, IdType* last_page_len,
                                          IdType* rope_pos_offset = nullptr)
      : page_size(page_size),
        head_dim_ckv(head_dim_compressed_kv),
        head_dim_kpe(head_dim_kpe),
        batch_size(batch_size),
        ckv_data(compressed_kv_data),
        kpe_data(kpe_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page_ckv = page_size * head_dim_ckv;
    stride_n_ckv = head_dim_ckv;
    stride_page_kpe = page_size * head_dim_kpe;
    stride_n_kpe = head_dim_kpe;
  }

  /*!
   * \brief Construct a paged key-value cache with custom kv-cache strides
   * \param page_size The size of each page
   * \param head_dim_compressed_kv The dimension of compressed-kv
   * \param head_dim_kpe The dimension of k-pe
   * \param batch_size The batch size
   * \param compressed_kv_data The start pointer of compressed-kv cache, cache should be contiguous
   * \param compressed_kv_strides custom strides of each dimensions of compressed-kv cache
   * \param kpe_data The start pointer of k-pe cache, cache should be contiguous
   * \param kpe_strides custom strides of each dimensions of k-pe cache
   * \param indices The page indices array
   * \param indptr The page indptr array
   * \param last_page_len The offset of the last page for each request in the batch
   * \param rope_pos_offset The start position of each request in the batch.
   */
  __host__ __forceinline__ paged_kv_mla_t(uint32_t page_size, uint32_t head_dim_compressed_kv,
                                          uint32_t head_dim_kpe, uint32_t batch_size,
                                          DType* compressed_kv_data,
                                          const int64_t* compressed_kv_strides, DType* kpe_data,
                                          const int64_t* kpe_strides, IdType* indices,
                                          IdType* indptr, IdType* last_page_len,
                                          IdType* rope_pos_offset = nullptr)
      : page_size(page_size),
        head_dim_ckv(head_dim_compressed_kv),
        head_dim_kpe(head_dim_kpe),
        batch_size(batch_size),
        ckv_data(compressed_kv_data),
        kpe_data(kpe_data),
        indices(indices),
        indptr(indptr),
        last_page_len(last_page_len),
        rope_pos_offset(rope_pos_offset) {
    stride_page_ckv = compressed_kv_strides[0];
    stride_n_ckv = compressed_kv_strides[1];
    stride_page_kpe = kpe_strides[0];
    stride_n_kpe = kpe_strides[1];
  }

  __host__ __device__ __forceinline__ uint32_t get_length(uint32_t batch_idx) const {
    if (indptr[batch_idx + 1] == indptr[batch_idx]) {
      return 0;
    }
    return (indptr[batch_idx + 1] - indptr[batch_idx] - 1) * page_size + last_page_len[batch_idx];
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_ckv(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_ckv + entry_idx * stride_n_ckv + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_ckv(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_ckv(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __host__ __device__ __forceinline__ size_t get_elem_offset_kpe(size_t page_idx, size_t entry_idx,
                                                                 size_t feat_idx) const {
    return page_idx * stride_page_kpe + entry_idx * stride_n_kpe + feat_idx;
  }

  __device__ __forceinline__ size_t protective_get_offset_kpe(IdType page_iter, uint32_t entry_idx,
                                                              uint32_t feat_idx,
                                                              IdType last_indptr) const {
    if (page_iter < last_indptr) {
      return get_elem_offset_kpe(__ldg(indices + page_iter), entry_idx, feat_idx);
    } else {
      return 0;
    }
  }

  __device__ __forceinline__ DType* get_ckv_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return ckv_data + get_elem_offset_ckv(__ldg(indices + page_idx), entry_idx, feat_idx);
  }

  __device__ __forceinline__ DType* get_kpe_ptr(size_t page_idx, size_t entry_idx,
                                                size_t feat_idx) const {
    return kpe_data + get_elem_offset_kpe(__ldg(indices + page_idx), entry_idx, feat_idx);
  }
};

template <uint32_t head_dim_ckv, uint32_t head_dim_kpe, uint32_t vec_size, typename DType,
          typename IdType>
__global__ void AppendPagedKVMlaCacheKernel(paged_kv_mla_t<DType, IdType> paged_kv_mla,
                                            DType* __restrict__ append_ckv,
                                            DType* __restrict__ append_kpe,
                                            IdType* __restrict__ batch_indices,
                                            IdType* __restrict__ positions, uint32_t nnz,
                                            size_t append_ckv_stride_n,
                                            size_t append_kpe_stride_n) {
  uint32_t tx = threadIdx.x;
  uint32_t cta_id = blockIdx.x;
  uint32_t num_ctas = gridDim.x;

#pragma unroll 4
  for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
    uint32_t page_iter, entry_idx;
    paged_kv_mla.page_size.divmod(
        paged_kv_mla.indptr[batch_indices[i]] * paged_kv_mla.page_size + positions[i], page_iter,
        entry_idx);
    DType* ckv_ptr = paged_kv_mla.get_ckv_ptr(page_iter, entry_idx, tx * vec_size);
    vec_t<DType, vec_size>::memcpy(ckv_ptr, append_ckv + i * append_ckv_stride_n + tx * vec_size);

    if (tx * vec_size < head_dim_kpe) {
      DType* kpe_ptr = paged_kv_mla.get_kpe_ptr(page_iter, entry_idx, tx * vec_size);
      vec_t<DType, vec_size>::memcpy(kpe_ptr, append_kpe + i * append_kpe_stride_n + tx * vec_size);
    }
  }
}

template <typename DType, typename IdType>
cudaError_t AppendPagedKVMlaCache(paged_kv_mla_t<DType, IdType> paged_kv, DType* append_ckv,
                                  DType* append_kpe, IdType* batch_indices, IdType* positions,
                                  uint32_t nnz, size_t append_ckv_stride_n,
                                  size_t append_kpe_stride_n, cudaStream_t stream = nullptr) {
  int dev_id = 0;
  int num_sms = 0;
  int num_blocks_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  uint32_t head_dim_ckv = paged_kv.head_dim_ckv;
  uint32_t head_dim_kpe = paged_kv.head_dim_kpe;
  constexpr uint32_t HEAD_CKV_DIM = 512;
  constexpr uint32_t HEAD_KPE_DIM = 64;
  FLASHINFER_CHECK(head_dim_ckv == HEAD_CKV_DIM, "head_dim_ckv must be equal to 512");
  FLASHINFER_CHECK(head_dim_kpe == HEAD_KPE_DIM, "head_dim_kpe must be equal to 64");
  constexpr uint32_t vec_size = 2;

  uint32_t bdx = HEAD_CKV_DIM / vec_size;
  uint32_t num_threads = bdx;
  uint32_t smem_size = 0;
  auto kernel = AppendPagedKVMlaCacheKernel<HEAD_CKV_DIM, HEAD_KPE_DIM, vec_size, DType, IdType>;
  FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                                     num_threads, smem_size));
  num_blocks_per_sm = min(num_blocks_per_sm, ceil_div(int(nnz), num_sms));
  dim3 nblks(num_blocks_per_sm * num_sms);
  dim3 nthrs(bdx);
  void* args[] = {(void*)&paged_kv,
                  (void*)&append_ckv,
                  (void*)&append_kpe,
                  (void*)&batch_indices,
                  (void*)&positions,
                  (void*)&nnz,
                  (void*)&append_ckv_stride_n,
                  (void*)&append_kpe_stride_n};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  return cudaSuccess;
}

/*!
 * \brief CUDA kernel to quantize key/value per-token-head to FP8 and write to paged/ragged cache
 * \tparam DTypeIn The input data type (float16/bfloat16)
 * \tparam DTypeCache The FP8 cache data type (__nv_fp8_e4m3/__nv_fp8_e5m2)
 * \tparam HEAD_DIM The dimension of each key head
 * \tparam HEAD_DIM_V The dimension of each value head
 * \tparam HEAD_SIZE_PADDED next_power_of_2(max(HEAD_DIM, HEAD_DIM_V))
 * \param key The key tensor [num_tokens, num_kv_heads, head_dim] (contiguous)
 * \param value The value tensor [num_tokens, num_kv_heads, head_dim_v] (contiguous)
 * \param k_cache The key cache (FP8)
 * \param v_cache The value cache (FP8)
 * \param k_scale_cache The key scale cache (float32)
 * \param v_scale_cache The value scale cache (float32)
 * \param slot_mapping The slot mapping [num_tokens]
 * \param num_tokens Number of tokens
 * \param num_kv_heads Number of KV heads
 * \param stride_kc_n Stride in elements over slot dimension of k_cache
 * \param stride_kc_h Stride in elements over head dimension of k_cache
 * \param stride_vc_n Stride in elements over slot dimension of v_cache
 * \param stride_vc_h Stride in elements over head dimension of v_cache
 * \param stride_ks_h Stride in elements over head dimension of k_scale_cache
 * \param stride_vs_h Stride in elements over head dimension of v_scale_cache
 * Note: Follows FlashInfer convention of stride_n + stride_h. The last dimension
 * (head_dim) is always contiguous (stride=1). Scale cache slot stride is always
 * num_kv_heads (contiguous). Scale cache is NOT assumed to share storage with cache.
 */
template <typename DTypeIn, typename DTypeCache, uint32_t HEAD_DIM, uint32_t HEAD_DIM_V,
          uint32_t HEAD_SIZE_PADDED>
__global__ void ReshapeAndCacheFlashPerTokenHeadKernel(
    const DTypeIn* __restrict__ key, const DTypeIn* __restrict__ value,
    DTypeCache* __restrict__ k_cache, DTypeCache* __restrict__ v_cache,
    float* __restrict__ k_scale_cache, float* __restrict__ v_scale_cache,
    const int32_t* __restrict__ slot_mapping, uint32_t num_tokens, uint32_t num_kv_heads,
    int64_t stride_kc_n, int64_t stride_kc_h, int64_t stride_vc_n, int64_t stride_vc_h,
    int64_t stride_ks_h, int64_t stride_vs_h) {
  uint32_t tok = blockIdx.x;
  uint32_t head = blockIdx.y;

  if (tok >= num_tokens || head >= num_kv_heads) {
    return;
  }

  int32_t slot = slot_mapping[tok];
  if (slot < 0) {
    return;
  }

  // Derive QUANT_MAX from DTypeCache at compile time
  constexpr float QUANT_MAX = finfo<DTypeCache>::max;

  // Compute scale by finding absmax (key/value are contiguous)
  float absmax_k = 0.f, absmax_v = 0.f;
  uint32_t k_base = tok * num_kv_heads * HEAD_DIM + head * HEAD_DIM;
  uint32_t v_base = tok * num_kv_heads * HEAD_DIM_V + head * HEAD_DIM_V;
  for (uint32_t i = threadIdx.x; i < HEAD_SIZE_PADDED; i += blockDim.x) {
    if (i < HEAD_DIM) {
      float val = static_cast<float>(key[k_base + i]);
      float aval = fabsf(val);
      if (aval > absmax_k) {
        absmax_k = aval;
      }
    }
    if (i < HEAD_DIM_V) {
      float val = static_cast<float>(value[v_base + i]);
      float aval = fabsf(val);
      if (aval > absmax_v) {
        absmax_v = aval;
      }
    }
  }

  // Warp-level reduction for absmax
  float warp_max_k = absmax_k;
  float warp_max_v = absmax_v;
  for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
    float val_k = __shfl_down_sync(0xFFFFFFFF, warp_max_k, mask);
    float val_v = __shfl_down_sync(0xFFFFFFFF, warp_max_v, mask);
    if (val_k > warp_max_k) warp_max_k = val_k;
    if (val_v > warp_max_v) warp_max_v = val_v;
  }

  // Cross-warp reduction: each warp's lane 0 writes its warp max to shared memory,
  // then all threads read and reduce
  extern __shared__ char smem_buf[];
  float* smem_max_k = reinterpret_cast<float*>(smem_buf);
  float* smem_max_v = smem_max_k + (blockDim.x + warpSize - 1) / warpSize;
  uint32_t warp_id = threadIdx.x / warpSize;
  if (threadIdx.x % warpSize == 0) {
    smem_max_k[warp_id] = warp_max_k;
    smem_max_v[warp_id] = warp_max_v;
  }
  uint32_t num_warps = (blockDim.x + warpSize - 1) / warpSize;
  __syncthreads();
  absmax_k = smem_max_k[0];
  absmax_v = smem_max_v[0];
  for (uint32_t w = 1; w < num_warps; w++) {
    if (smem_max_k[w] > absmax_k) absmax_k = smem_max_k[w];
    if (smem_max_v[w] > absmax_v) absmax_v = smem_max_v[w];
  }

  // Compute scale
  float k_scale = fmaxf(absmax_k / QUANT_MAX, 1e-6f);
  float v_scale = fmaxf(absmax_v / QUANT_MAX, 1e-6f);

  // Compute reciprocal scale for quantization
  float k_rcp_scale = 1.0f / k_scale;
  float v_rcp_scale = 1.0f / v_scale;

  // Compute cache offsets using stride_n + stride_h convention.
  int64_t k_cache_offset = static_cast<int64_t>(slot) * stride_kc_n + head * stride_kc_h;
  int64_t v_cache_offset = static_cast<int64_t>(slot) * stride_vc_n + head * stride_vc_h;
  int64_t k_scale_offset = static_cast<int64_t>(slot) * num_kv_heads + head * stride_ks_h;
  int64_t v_scale_offset = static_cast<int64_t>(slot) * num_kv_heads + head * stride_vs_h;

  // Quantize and write key
  for (uint32_t i = threadIdx.x; i < HEAD_SIZE_PADDED; i += blockDim.x) {
    if (i < HEAD_DIM) {
      float val = static_cast<float>(key[k_base + i]);
      float qval = fminf(fmaxf(val * k_rcp_scale, -QUANT_MAX), QUANT_MAX);
      k_cache[k_cache_offset + i] = static_cast<DTypeCache>(qval);
    }
  }

  // Quantize and write value
  for (uint32_t i = threadIdx.x; i < HEAD_SIZE_PADDED; i += blockDim.x) {
    if (i < HEAD_DIM_V) {
      float val = static_cast<float>(value[v_base + i]);
      float qval = fminf(fmaxf(val * v_rcp_scale, -QUANT_MAX), QUANT_MAX);
      v_cache[v_cache_offset + i] = static_cast<DTypeCache>(qval);
    }
  }

  // Write scales (one per warp, use lane 0)
  if (threadIdx.x == 0) {
    k_scale_cache[k_scale_offset] = k_scale;
    v_scale_cache[v_scale_offset] = v_scale;
  }
}

/*!
 * \brief Launch ReshapeAndCacheFlashPerTokenHead kernel
 * \tparam DTypeIn The input data type
 * \tparam DTypeCache The FP8 cache data type
 * \param key The key tensor
 * \param value The value tensor
 * \param k_cache The key cache (FP8)
 * \param v_cache The value cache (FP8)
 * \param k_scale_cache The key scale cache (float32)
 * \param v_scale_cache The value scale cache (float32)
 * \param slot_mapping The slot mapping
 * \param num_tokens Number of tokens
 * \param num_kv_heads Number of KV heads
 * \param head_dim Key head dimension
 * \param head_dim_v Value head dimension
 * \param stride_kc_n Stride in elements over slot dimension of k_cache
 * \param stride_kc_h Stride in elements over head dimension of k_cache
 * \param stride_vc_n Stride in elements over slot dimension of v_cache
 * \param stride_vc_h Stride in elements over head dimension of v_cache
 * \param stride_ks_h Stride in elements over head dimension of k_scale_cache
 * \param stride_vs_h Stride in elements over head dimension of v_scale_cache
 * \param stream CUDA stream
 */
template <typename DTypeIn, typename DTypeCache>
cudaError_t ReshapeAndCacheFlashPerTokenHead(
    const DTypeIn* key, const DTypeIn* value, DTypeCache* k_cache, DTypeCache* v_cache,
    float* k_scale_cache, float* v_scale_cache, const int32_t* slot_mapping, uint32_t num_tokens,
    uint32_t num_kv_heads, uint32_t head_dim, uint32_t head_dim_v, int64_t stride_kc_n,
    int64_t stride_kc_h, int64_t stride_vc_n, int64_t stride_vc_h, int64_t stride_ks_h,
    int64_t stride_vs_h, cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    DISPATCH_HEAD_DIM(head_dim_v, HEAD_DIM_V, {
      constexpr uint32_t HEAD_SIZE_PADDED = std::max(HEAD_DIM, HEAD_DIM_V);

      constexpr uint32_t bdx = std::max(32U, std::min(128U, HEAD_SIZE_PADDED));
      dim3 grid(num_tokens, num_kv_heads);
      dim3 block(bdx);

      auto kernel = ReshapeAndCacheFlashPerTokenHeadKernel<DTypeIn, DTypeCache, HEAD_DIM,
                                                           HEAD_DIM_V, HEAD_SIZE_PADDED>;
      void* args[] = {(void*)&key,          (void*)&value,         (void*)&k_cache,
                      (void*)&v_cache,      (void*)&k_scale_cache, (void*)&v_scale_cache,
                      (void*)&slot_mapping, (void*)&num_tokens,    (void*)&num_kv_heads,
                      (void*)&stride_kc_n,  (void*)&stride_kc_h,   (void*)&stride_vc_n,
                      (void*)&stride_vc_h,  (void*)&stride_ks_h,   (void*)&stride_vs_h};
      // Shared memory for cross-warp absmax reduction: 2 floats per warp
      constexpr size_t smem_bytes = ((bdx + 31U) / 32U) * 2 * sizeof(float);
      FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem_bytes, stream));
    });
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLAHSINFER_PAGE_CUH_
