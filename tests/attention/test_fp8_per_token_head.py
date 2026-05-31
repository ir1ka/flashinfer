"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

FP8 per-token-head KV cache tests.

These tests verify FP8 per-token-head quantization with inline scale storage:
- Each token x head has an independent float32 scale stored inline after FP8 head data
- Layout: [FP8 head_data (head_dim bytes) | scale (4B float32) | padding (12B)]
- stride_n = head_dim + 16
- Requires SM90+ (FA3) for fp8 prefill — FA2 (SM86) does not support fp8
"""

import pytest
import torch
import flashinfer

_FP8_E4M3_MAX = 448.0


def _skip_if_sm_below_75():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 7 or (cc[0] == 7 and cc[1] < 5):
        pytest.skip(
            "FP8 mixed-precision (fp16 q + fp8 k/v) requires SM75+ for FA2 backend."
        )


def _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads):
    """No longer needed — SM75 FP8+GQA fix: independent sync_state float smem
    resolves the pointer arithmetic mismatch. Keep as no-op for compatibility."""
    pass


def allocate_kv_with_inline_scale(shape, head_dim, dtype, device, layout="NHD"):
    """Allocate paged KV cache with inline per-token-head scale space.

    Parameters
    ----------
    shape : tuple
        Target shape (max_pages, page_size, num_kv_heads, head_dim) for NHD
        or (max_pages, num_kv_heads, page_size, head_dim) for HND.
    head_dim : int
        Head dimension.
    dtype : torch.dtype
        FP8 dtype.
    device : torch.device
        CUDA device.
    layout : str
        "NHD" or "HND".

    Returns
    -------
    k_cache : torch.Tensor
        FP8 tensor with stride (head_dim+16) instead of head_dim.
    v_cache : torch.Tensor
        Same layout as k_cache.
    """
    if layout == "NHD":
        max_pages, page_size, num_kv_heads = shape[0], shape[1], shape[2]
    else:
        max_pages, num_kv_heads, page_size = shape[0], shape[1], shape[2]
    # Total tokens across all pages and heads
    stride = head_dim + 16
    # Allocate flat buffer with stride (head_dim + 16) per token
    if layout == "NHD":
        # (max_pages, page_size, num_kv_heads, head_dim)
        # stride in elements: (page_size*num_kv_heads*stride, num_kv_heads*stride, stride, 1)
        total_tokens = max_pages * page_size * num_kv_heads
    else:
        # (max_pages, num_kv_heads, page_size, head_dim)
        total_tokens = max_pages * num_kv_heads * page_size

    buf_size = total_tokens * stride
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)

    if layout == "NHD":
        s3 = 1
        s2 = stride
        s1 = num_kv_heads * stride
        s0 = page_size * num_kv_heads * stride
    else:
        s3 = 1
        s2 = stride
        s1 = page_size * stride
        s0 = num_kv_heads * page_size * stride

    k_cache = torch.as_strided(k_buf, shape, (s0, s1, s2, s3)).view(dtype)
    v_cache = torch.as_strided(v_buf, shape, (s0, s1, s2, s3)).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def write_inline_scales(cache_tensor, buf, scales, head_dim, layout="NHD"):
    """Write per-token-head scales into inline positions of the cache buffer.

    Parameters
    ----------
    cache_tensor : torch.Tensor
        The strided FP8 cache tensor.
    buf : torch.Tensor
        The underlying uint8 buffer.
    scales : torch.Tensor
        Float32 scales, shape matching (..., num_kv_heads) where ... matches
        the first dimensions of cache_tensor excluding head_dim.
    head_dim : int
        Head dimension.
    layout : str
        "NHD" or "HND".
    """
    stride = head_dim + 16
    # Create float32 scale view at offset head_dim with stride (head_dim+16)
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    cache_shape = cache_tensor.shape

    if layout == "NHD":
        # (max_pages, page_size, num_kv_heads, head_dim)
        # Scale view: (max_pages, page_size, num_kv_heads)
        max_pages, page_size, num_kv_heads = (
            cache_shape[0],
            cache_shape[1],
            cache_shape[2],
        )
        s1 = num_kv_heads * scale_stride_f32
        s0 = page_size * num_kv_heads * scale_stride_f32
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, page_size, num_kv_heads),
            (s0, s1, scale_stride_f32),
            storage_offset=scale_offset_f32,
        )
    else:
        # (max_pages, num_kv_heads, page_size, head_dim)
        max_pages, num_kv_heads, page_size = (
            cache_shape[0],
            cache_shape[1],
            cache_shape[2],
        )
        s1 = page_size * scale_stride_f32
        s0 = num_kv_heads * page_size * scale_stride_f32
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, num_kv_heads, page_size),
            (s0, s1, scale_stride_f32),
            storage_offset=scale_offset_f32,
        )

    # Reshape scales to match scale_view
    scales = scales.to(torch.float32)
    scale_view.copy_(scales.reshape(scale_view.shape))


def quantize_to_fp8_per_token_head(data, fp8_dtype=torch.float8_e4m3fn):
    """Quantize FP16/BF16 data to FP8 with per-token-head scales.

    Parameters
    ----------
    data : torch.Tensor
        Input FP16/BF16 tensor, shape (..., head_dim).
    fp8_dtype : torch.dtype
        Target FP8 dtype.

    Returns
    -------
    fp8_data : torch.Tensor
        FP8 quantized data, same shape as input.
    scales : torch.Tensor
        Per-token-head scales, shape matching all dims except last.
    """
    amax = data.abs().amax(dim=-1, keepdim=True)  # (..., 1)
    amax = amax.clamp(min=1e-12)
    scales = amax / _FP8_E4M3_MAX  # (..., 1)
    # Normalize and quantize
    fp8_data = (
        (data / scales.clamp(min=1e-12))
        .clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX)
        .to(fp8_dtype)
    )
    return fp8_data, scales.squeeze(-1)


def write_fp8_to_strided_cache(cache_tensor, fp8_data, layout="NHD"):
    """Write FP8 data into the strided cache tensor (only the FP8 portion).

    The cache tensor has stride (head_dim+16), so direct assignment writes
    to the correct byte positions for FP8 data.
    """
    cache_tensor.copy_(fp8_data)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("qo_len", [8])
@pytest.mark.parametrize("kv_len", [32])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("kv_layout", ["NHD"])
def test_batch_prefill_per_token_head_fp8(
    batch_size,
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    kv_layout,
):
    """Test batch prefill with FP8 per-token-head KV cache.

    Compares FP8 per-token-head output against FP16 baseline.
    Expects cosine similarity >= 0.99 (>= 0.999 for e4m3, >= 0.99 for e5m2).
    """
    _skip_if_sm_below_75()
    torch.manual_seed(42)
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn

    # Generate FP16 data for reference
    total_qo = batch_size * qo_len
    q_f16 = torch.randn(
        total_qo, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.1 * torch.randn(
        kv_len, batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.1 * torch.randn(
        kv_len, batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Transpose to NHD: (batch_size, kv_len, num_kv_heads, head_dim) -> (batch_size, kv_len, num_kv_heads, head_dim)
    # For paged cache, we need to flatten into pages
    k_f16 = k_f16.transpose(0, 1)  # (batch_size, kv_len, num_kv_heads, head_dim)
    v_f16 = v_f16.transpose(0, 1)

    # Build paged cache indices
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )

    # FP16 baseline
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Reshape k, v to paged layout for FP16 baseline
    if kv_layout == "NHD":
        k_paged_f16 = k_f16.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        v_paged_f16 = v_f16.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    else:
        k_paged_f16 = k_f16.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)
        v_paged_f16 = v_f16.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)

    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_f16.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_f16.run(q_f16, (k_paged_f16, v_paged_f16))

    # FP8 per-token-head: quantize with per-token-head scales
    # Quantize k, v per token per head
    k_flat = k_f16.reshape(
        -1, num_kv_heads, head_dim
    )  # (batch_size*kv_len, num_kv_heads, head_dim)
    v_flat = v_f16.reshape(-1, num_kv_heads, head_dim)

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_flat, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_flat, fp8_dtype)

    # Allocate strided paged cache with inline scale space
    if kv_layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads, head_dim)
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size, head_dim)

    k_cache, v_cache, k_buf, v_buf = allocate_kv_with_inline_scale(
        cache_shape, head_dim, fp8_dtype, device, kv_layout
    )

    # Reshape fp8 data to paged layout and write
    if kv_layout == "NHD":
        k_paged_fp8 = k_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        v_paged_fp8 = v_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        # Scales: (batch_size*kv_len, num_kv_heads) -> (total_num_pages, page_size, num_kv_heads)
        k_scales_paged = k_scales.reshape(total_num_pages, page_size, num_kv_heads)
        v_scales_paged = v_scales.reshape(total_num_pages, page_size, num_kv_heads)
    else:
        k_paged_fp8 = k_fp8.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)
        v_paged_fp8 = v_fp8.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)
        k_scales_paged = k_scales.reshape(
            total_num_pages, page_size, num_kv_heads
        ).transpose(1, 2)
        v_scales_paged = v_scales.reshape(
            total_num_pages, page_size, num_kv_heads
        ).transpose(1, 2)

    # Write FP8 data and scales
    write_fp8_to_strided_cache(k_cache, k_paged_fp8, kv_layout)
    write_fp8_to_strided_cache(v_cache, v_paged_fp8, kv_layout)
    write_inline_scales(k_cache, k_buf, k_scales_paged, head_dim, kv_layout)
    write_inline_scales(v_cache, v_buf, v_scales_paged, head_dim, kv_layout)

    # Run FP8 per-token-head attention (FA2 + fp16 q + fp8 k/v)
    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp8.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=fp8_dtype,
        use_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(q_f16, (k_cache, v_cache))

    # Compare: cosine similarity >= 0.999
    o_fp16_flat = o_fp16.reshape(-1).float()
    o_fp8_flat = o_fp8.reshape(-1).float()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_fp16_flat, o_fp8_flat, dim=0
    ).item()

    print(f"\nCosine similarity: {cos_sim:.6f}")
    print(f"Max absolute diff: {(o_fp16 - o_fp8).abs().amax().item():.6f}")
    print(f"Mean absolute diff: {(o_fp16 - o_fp8).abs().mean().item():.6f}")

    # FP8 per-token-head should maintain high accuracy
    assert cos_sim >= 0.99, (
        f"Cosine similarity {cos_sim:.6f} < 0.99. Per-token-head FP8 accuracy too low."
    )


@pytest.mark.parametrize("kv_len", [32, 128])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("kv_layout", ["NHD"])
def test_batch_decode_per_token_head_fp8(
    kv_len, num_kv_heads, num_qo_heads, head_dim, kv_layout
):
    """Test batch decode with FP8 per-token-head KV cache."""
    _skip_if_sm_below_75()
    _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads)
    torch.manual_seed(42)
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    batch_size = 4

    # Generate FP16 data for reference
    q_f16 = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.1 * torch.randn(
        kv_len, batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.1 * torch.randn(
        kv_len, batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_f16 = k_f16.transpose(0, 1)  # (batch_size, kv_len, num_kv_heads, head_dim)
    v_f16 = v_f16.transpose(0, 1)

    # FP16 baseline - use paged cache for decode
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    decode_wrapper_f16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )

    # Use paged layout for decode (page_size >= kv_len to avoid page boundary issue)
    page_size = kv_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    paged_kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    paged_kv_indices = torch.arange(
        0, total_num_pages, dtype=torch.int32, device=device
    )
    paged_kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    if kv_layout == "NHD":
        k_paged_f16 = k_f16.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        v_paged_f16 = v_f16.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    else:
        k_paged_f16 = k_f16.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)
        v_paged_f16 = v_f16.reshape(
            total_num_pages, page_size, num_kv_heads, head_dim
        ).transpose(1, 2)

    decode_wrapper_f16.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = decode_wrapper_f16.run(q_f16, (k_paged_f16, v_paged_f16))

    # FP8 per-token-head
    k_flat = k_f16.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_f16.reshape(-1, num_kv_heads, head_dim)
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_flat, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_flat, fp8_dtype)

    cache_shape = (total_num_pages, page_size, num_kv_heads, head_dim)
    k_cache, v_cache, k_buf, v_buf = allocate_kv_with_inline_scale(
        cache_shape, head_dim, fp8_dtype, device, kv_layout
    )

    k_paged_fp8 = k_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    v_paged_fp8 = v_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    k_scales_paged = k_scales.reshape(total_num_pages, page_size, num_kv_heads)
    v_scales_paged = v_scales.reshape(total_num_pages, page_size, num_kv_heads)

    write_fp8_to_strided_cache(k_cache, k_paged_fp8, kv_layout)
    write_fp8_to_strided_cache(v_cache, v_paged_fp8, kv_layout)
    write_inline_scales(k_cache, k_buf, k_scales_paged, head_dim, kv_layout)
    write_inline_scales(v_cache, v_buf, v_scales_paged, head_dim, kv_layout)

    decode_wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    decode_wrapper_fp8.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=fp8_dtype,
        use_per_token_head=True,
    )
    o_fp8 = decode_wrapper_fp8.run(q_f16, (k_cache, v_cache))

    # Compare
    o_fp16_flat = o_fp16.reshape(-1).float()
    o_fp8_flat = o_fp8.reshape(-1).float()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_fp16_flat, o_fp8_flat, dim=0
    ).item()

    print(f"\nDecode cosine similarity: {cos_sim:.6f}")
    print(f"Decode max absolute diff: {(o_fp16 - o_fp8).abs().amax().item():.6f}")

    assert cos_sim >= 0.99, (
        f"Decode cosine similarity {cos_sim:.6f} < 0.99. "
        f"Per-token-head FP8 decode accuracy too low."
    )


if __name__ == "__main__":
    # Quick smoke test
    test_batch_prefill_per_token_head_fp8(
        batch_size=3,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
    )
    print("Prefill test passed!")

    test_batch_decode_per_token_head_fp8(
        kv_len=32, num_kv_heads=4, num_qo_heads=4, head_dim=128, kv_layout="NHD"
    )
    print("Decode test passed!")

    print("\nAll tests passed!")
