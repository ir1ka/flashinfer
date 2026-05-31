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

FP8 per-token-head comprehensive parametrized tests.

Covers the matrix of:
- mode: single / batch
- attention: MHA / GQA
- Q dtype: fp16 / bf16 (bf16 skipped on SM75)
- KV fp8 dtype: e4m3 / e5m2 (e5m2 cos_sim >= 0.99)
- per-token-head inline scale (always True)
- prefill / decode
- layout: NHD / HND (batch only)
- backend: fa2 (non-JIT for batch) / default JIT (single decode)
- causal / non-causal (prefill only)
- head_dim: 64 / 128 / 256
- ROPE_LLAMA / NONE
- paged / ragged (batch only)

All scales are computed dynamically from random data (no fixed scales).
"""

import pytest
import torch
import flashinfer

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


# ============================================================
# Utilities
# ============================================================


def _fp8_max(fp8_dtype):
    return _FP8_E5M2_MAX if fp8_dtype == torch.float8_e5m2 else _FP8_E4M3_MAX


def _cc():
    return torch.cuda.get_device_capability(0)


def _skip_if_sm_below_75():
    if _cc()[0] < 7 or (_cc()[0] == 7 and _cc()[1] < 5):
        pytest.skip("Requires SM75+")


def _skip_if_bf16_sm75(q_dtype):
    """BF16 is not well-supported on SM75, skip."""
    if q_dtype == torch.bfloat16 and _cc()[0] == 7:
        pytest.skip("BF16 skipped on SM75")


def quantize_fp8_per_token_head(data, fp8_dtype):
    """Quantize FP16/BF16 -> FP8 with per-token-head float32 scales.

    data shape: (..., head_dim)
    Returns: (fp8_data same shape, scales without last dim)
    """
    fp8_max = _fp8_max(fp8_dtype)
    amax = data.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scales = amax / fp8_max
    fp8 = (data / scales.clamp(min=1e-12)).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return fp8, scales.squeeze(-1)


def check_accuracy(o_ref, o_pth, fp8_dtype, label=""):
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref.cpu() - o_pth.cpu()).abs().max().item()
    prefix = f"[{label}] " if label else ""
    dtype_tag = "e5m2" if fp8_dtype == torch.float8_e5m2 else "e4m3"
    threshold = 0.99 if fp8_dtype == torch.float8_e5m2 else 0.99
    print(f"{prefix}{dtype_tag} cos_sim={cos_sim:.8f} max_diff={max_diff:.8f}")
    assert cos_sim >= threshold, (
        f"{prefix}cos_sim={cos_sim:.8f} < {threshold} ({dtype_tag})"
    )
    return cos_sim, max_diff


# ============================================================
# Single prefill / decode helpers
# ============================================================


def build_strided_cache(fp8_data, scales, head_dim):
    """Build (kv_len, num_kv_heads, head_dim) strided cache with inline scales.

    stride = head_dim + 16 bytes per row.
    """
    kv_len, num_kv_heads = fp8_data.shape[0], fp8_data.shape[1]
    stride = head_dim + 16
    buf_size = kv_len * num_kv_heads * stride
    buf = torch.zeros(buf_size, dtype=torch.uint8, device=fp8_data.device)
    rows = buf.reshape(-1, stride)
    fp8_flat = fp8_data.reshape(-1, head_dim).view(torch.uint8)
    rows[:, :head_dim].copy_(fp8_flat)
    scales_f32 = scales.reshape(-1).to(torch.float32)
    scales_bytes = scales_f32.view(torch.uint8)
    for i in range(kv_len * num_kv_heads):
        buf[i * stride + head_dim : i * stride + head_dim + 4].copy_(
            scales_bytes[i * 4 : (i + 1) * 4]
        )
    cache = torch.as_strided(
        buf.view(fp8_data.dtype),
        (kv_len, num_kv_heads, head_dim),
        (num_kv_heads * stride, stride, 1),
        storage_offset=0,
    )
    return cache


def run_single_prefill_pth(
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
    fp8_dtype,
    causal,
    backend,
    pos_encoding_mode,
):
    """Run single prefill per-token-head and compare against FP16 dequantized ref."""
    device = "cuda:0"
    torch.manual_seed(42)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=q_dtype, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_fp8_per_token_head(v_f16, fp8_dtype)

    # Dequantized reference — must match q_dtype to avoid mixed-dtype FA2 issues
    k_dq = k_fp8.to(q_dtype) * k_scales.unsqueeze(-1).to(q_dtype)
    v_dq = v_fp8.to(q_dtype) * v_scales.unsqueeze(-1).to(q_dtype)

    kwargs = {"causal": causal, "backend": backend, "o_dtype": q_dtype}
    if pos_encoding_mode != "NONE":
        kwargs["pos_encoding_mode"] = pos_encoding_mode
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k_dq, v_dq, **kwargs)

    # Per-token-head strided cache
    k_cache = build_strided_cache(k_fp8, k_scales, head_dim)
    v_cache = build_strided_cache(v_fp8, v_scales, head_dim)
    kwargs_pth = {**kwargs, "use_per_token_head": True}
    o_pth = flashinfer.single_prefill_with_kv_cache(q, k_cache, v_cache, **kwargs_pth)

    return check_accuracy(o_ref, o_pth, fp8_dtype, label=f"single prefill {backend}")


def run_single_decode_pth(
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
    fp8_dtype,
    pos_encoding_mode,
):
    """Run single decode per-token-head (always JIT) and compare against FP16 dequantized ref."""
    device = "cuda:0"
    torch.manual_seed(42)

    q = torch.randn(num_qo_heads, head_dim, dtype=q_dtype, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_fp8_per_token_head(v_f16, fp8_dtype)

    k_dq = k_fp8.to(q_dtype) * k_scales.unsqueeze(-1).to(q_dtype)
    v_dq = v_fp8.to(q_dtype) * v_scales.unsqueeze(-1).to(q_dtype)

    kwargs = {}
    if pos_encoding_mode != "NONE":
        kwargs["pos_encoding_mode"] = pos_encoding_mode
    o_ref = flashinfer.single_decode_with_kv_cache(q, k_dq, v_dq, **kwargs)

    k_cache = build_strided_cache(k_fp8, k_scales, head_dim)
    v_cache = build_strided_cache(v_fp8, v_scales, head_dim)
    kwargs_pth = {**kwargs, "use_per_token_head": True}
    o_pth = flashinfer.single_decode_with_kv_cache(q, k_cache, v_cache, **kwargs_pth)

    return check_accuracy(o_ref, o_pth, fp8_dtype, label="single decode JIT")


# ============================================================
# Batch prefill / decode helpers (paged + ragged)
# ============================================================


def _alloc_paged_cache(shape, head_dim, dtype, device, layout):
    """Allocate paged KV cache with inline per-token-head scale space.

    shape: 3-tuple (max_pages, page_size, num_kv_heads) for NHD
           (max_pages, num_kv_heads, page_size) for HND
    """
    if layout == "NHD":
        max_pages, page_size, num_kv_heads = shape
        full_shape = (max_pages, page_size, num_kv_heads, head_dim)
        s = (
            page_size * num_kv_heads * (head_dim + 16),
            num_kv_heads * (head_dim + 16),
            head_dim + 16,
            1,
        )
    else:
        max_pages, num_kv_heads, page_size = shape
        full_shape = (max_pages, num_kv_heads, page_size, head_dim)
        s = (
            num_kv_heads * page_size * (head_dim + 16),
            page_size * (head_dim + 16),
            head_dim + 16,
            1,
        )

    total_tokens = max_pages * page_size * num_kv_heads
    buf_size = total_tokens * (head_dim + 16)
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)

    k_cache = torch.as_strided(k_buf, full_shape, s).view(dtype)
    v_cache = torch.as_strided(v_buf, full_shape, s).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def _write_scales_paged(cache_tensor, buf, scales, head_dim, layout):
    """Write per-token-head scales into inline positions of paged cache buffer."""
    stride = head_dim + 16
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    cache_shape = cache_tensor.shape

    if layout == "NHD":
        max_pages, page_size, num_kv_heads, _ = cache_shape
        s = (
            page_size * num_kv_heads * scale_stride_f32,
            num_kv_heads * scale_stride_f32,
            scale_stride_f32,
        )
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, page_size, num_kv_heads),
            s,
            storage_offset=scale_offset_f32,
        )
    else:
        max_pages, num_kv_heads, page_size, _ = cache_shape
        s = (
            num_kv_heads * page_size * scale_stride_f32,
            page_size * scale_stride_f32,
            scale_stride_f32,
        )
        scale_view = torch.as_strided(
            buf.view(torch.float32),
            (max_pages, num_kv_heads, page_size),
            s,
            storage_offset=scale_offset_f32,
        )
    scale_view.copy_(scales.to(torch.float32).reshape(scale_view.shape))


def run_batch_prefill_pth(
    batch_size,
    qo_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    q_dtype,
    fp8_dtype,
    backend,
    pos_encoding_mode,
):
    """Run batch prefill per-token-head (paged) and compare against FP16 baseline.

    qo_lens / kv_lens: list of per-sequence lengths for variable-length support.
    When page_size >= max(kv_lens), each sequence fits in one page (ragged-like).
    """
    device = "cuda:0"
    torch.manual_seed(42)
    batch_size = len(kv_lens)

    total_qo = sum(qo_lens)
    q = torch.randn(total_qo, num_qo_heads, head_dim, dtype=q_dtype, device=device)

    # Build per-sequence K/V
    k_f16_list = []
    v_f16_list = []
    for i in range(batch_size):
        k_f16_list.append(
            0.1
            * torch.randn(
                kv_lens[i], num_kv_heads, head_dim, dtype=torch.float16, device=device
            )
        )
        v_f16_list.append(
            0.1
            * torch.randn(
                kv_lens[i], num_kv_heads, head_dim, dtype=torch.float16, device=device
            )
        )

    # Page indices
    num_pages_per_seq = [(kv + page_size - 1) // page_size for kv in kv_lens]
    total_num_pages = sum(num_pages_per_seq)
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(num_pages_per_seq, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [(kv - 1) % page_size + 1 for kv in kv_lens], dtype=torch.int32, device=device
    )
    qo_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(qo_lens, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Build page-aligned padded tensors (each seq padded to its page boundary)
    def pad_to_pages(data_list):
        padded = []
        for data, n_pages in zip(data_list, num_pages_per_seq, strict=False):
            pad_len = n_pages * page_size - len(data)
            if pad_len > 0:
                data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, pad_len))
            padded.append(data)
        return padded

    k_f16_paged = pad_to_pages(k_f16_list)
    v_f16_paged = pad_to_pages(v_f16_list)

    # Concatenate page-reshaped tensors to (total_num_pages, page_size, heads, head_dim)
    def concat_to_pages(data_list):
        pages = [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(data_list, num_pages_per_seq, strict=False)
        ]
        return torch.cat(pages, dim=0)

    k_p_flat = concat_to_pages(k_f16_paged)
    v_p_flat = concat_to_pages(v_f16_paged)

    if layout == "NHD":
        k_p = k_p_flat
        v_p = v_p_flat
    else:
        k_p = k_p_flat.transpose(1, 2)
        v_p = v_p_flat.transpose(1, 2)

    # Reference KV must match q_dtype to avoid mixed-dtype issues (BF16 Q + FP16 KV → near-zero)
    ref_kv_dtype = q_dtype
    k_p = k_p.to(ref_kv_dtype)
    v_p = v_p.to(ref_kv_dtype)

    wrapper_f16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
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
        q_data_type=q_dtype,
        kv_data_type=ref_kv_dtype,
        pos_encoding_mode=pos_encoding_mode,
    )
    o_fp16 = wrapper_f16.run(q, (k_p, v_p))

    # FP8 per-token-head — quantize page-aligned data
    k_fp8_paged = []
    v_fp8_paged = []
    k_scales_paged = []
    v_scales_paged = []
    for i in range(batch_size):
        k_fp8, k_s = quantize_fp8_per_token_head(k_f16_paged[i], fp8_dtype)
        v_fp8, v_s = quantize_fp8_per_token_head(v_f16_paged[i], fp8_dtype)
        k_fp8_paged.append(k_fp8)
        v_fp8_paged.append(v_fp8)
        k_scales_paged.append(k_s)
        v_scales_paged.append(v_s)

    # Concatenate page-reshaped tensors to (total_num_pages, page_size, ...)
    k_fp8_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(k_fp8_paged, num_pages_per_seq, strict=False)
        ]
    )
    v_fp8_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(v_fp8_paged, num_pages_per_seq, strict=False)
        ]
    )
    k_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(k_scales_paged, num_pages_per_seq, strict=False)
        ]
    )
    v_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(v_scales_paged, num_pages_per_seq, strict=False)
        ]
    )

    if layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads)
        k_fp8_p = k_fp8_flat
        v_fp8_p = v_fp8_flat
        k_s_p = k_s_flat
        v_s_p = v_s_flat
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size)
        k_fp8_p = k_fp8_flat.transpose(1, 2)
        v_fp8_p = v_fp8_flat.transpose(1, 2)
        k_s_p = k_s_flat.transpose(1, 2)
        v_s_p = v_s_flat.transpose(1, 2)

    k_cache, v_cache, k_buf, v_buf = _alloc_paged_cache(
        cache_shape, head_dim, fp8_dtype, device, layout
    )
    k_cache.copy_(k_fp8_p)
    v_cache.copy_(v_fp8_p)
    _write_scales_paged(k_cache, k_buf, k_s_p, head_dim, layout)
    _write_scales_paged(v_cache, v_buf, v_s_p, head_dim, layout)

    wrapper_fp8 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
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
        q_data_type=q_dtype,
        kv_data_type=fp8_dtype,
        use_per_token_head=True,
        pos_encoding_mode=pos_encoding_mode,
    )
    o_fp8 = wrapper_fp8.run(q, (k_cache, v_cache))

    return check_accuracy(o_fp16, o_fp8, fp8_dtype, label=f"batch prefill {backend}")


def run_batch_decode_pth(
    batch_size,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    layout,
    q_dtype,
    fp8_dtype,
    backend,
    pos_encoding_mode,
):
    """Run batch decode per-token-head (paged) and compare against FP16 baseline."""
    device = "cuda:0"
    torch.manual_seed(42)
    batch_size = len(kv_lens)

    q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=q_dtype, device=device)

    k_f16_list = []
    v_f16_list = []
    for i in range(batch_size):
        k_f16_list.append(
            0.1
            * torch.randn(
                kv_lens[i], num_kv_heads, head_dim, dtype=torch.float16, device=device
            )
        )
        v_f16_list.append(
            0.1
            * torch.randn(
                kv_lens[i], num_kv_heads, head_dim, dtype=torch.float16, device=device
            )
        )

    num_pages_per_seq = [(kv + page_size - 1) // page_size for kv in kv_lens]
    total_num_pages = sum(num_pages_per_seq)
    kv_indptr = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor(num_pages_per_seq, dtype=torch.int32, device=device)
            .cumsum(0)
            .to(torch.int32),
        ]
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [(kv - 1) % page_size + 1 for kv in kv_lens], dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Build page-aligned padded tensors
    def pad_to_pages(data_list):
        padded = []
        for data, n_pages in zip(data_list, num_pages_per_seq, strict=False):
            pad_len = n_pages * page_size - len(data)
            if pad_len > 0:
                data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, pad_len))
            padded.append(data)
        return padded

    k_f16_paged = pad_to_pages(k_f16_list)
    v_f16_paged = pad_to_pages(v_f16_list)

    # Concatenate to (total_num_pages, page_size, heads, head_dim)
    k_p_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(k_f16_paged, num_pages_per_seq, strict=False)
        ]
    )
    v_p_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(v_f16_paged, num_pages_per_seq, strict=False)
        ]
    )

    if layout == "NHD":
        k_p = k_p_flat
        v_p = v_p_flat
    else:
        k_p = k_p_flat.transpose(1, 2)
        v_p = v_p_flat.transpose(1, 2)

    # Reference KV must match q_dtype to avoid mixed-dtype issues
    ref_kv_dtype = q_dtype
    k_p = k_p.to(ref_kv_dtype)
    v_p = v_p.to(ref_kv_dtype)

    wrapper_f16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
    )
    wrapper_f16.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=ref_kv_dtype,
        pos_encoding_mode=pos_encoding_mode,
    )
    o_fp16 = wrapper_f16.run(q, (k_p, v_p))

    # FP8 per-token-head: quantize per-sequence, pad to pages, concatenate
    k_fp8_paged = []
    k_scales_paged = []
    v_fp8_paged = []
    v_scales_paged = []
    for i in range(batch_size):
        k_fp8_i, k_s_i = quantize_fp8_per_token_head(k_f16_list[i], fp8_dtype)
        v_fp8_i, v_s_i = quantize_fp8_per_token_head(v_f16_list[i], fp8_dtype)
        n_pages = num_pages_per_seq[i]
        pad_len = n_pages * page_size - kv_lens[i]
        if pad_len > 0:
            k_fp8_i = torch.nn.functional.pad(k_fp8_i, (0, 0, 0, 0, 0, pad_len))
            k_s_i = torch.nn.functional.pad(k_s_i, (0, 0, 0, pad_len))
            v_fp8_i = torch.nn.functional.pad(v_fp8_i, (0, 0, 0, 0, 0, pad_len))
            v_s_i = torch.nn.functional.pad(v_s_i, (0, 0, 0, pad_len))
        k_fp8_paged.append(k_fp8_i)
        k_scales_paged.append(k_s_i)
        v_fp8_paged.append(v_fp8_i)
        v_scales_paged.append(v_s_i)

    k_fp8_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(k_fp8_paged, num_pages_per_seq, strict=False)
        ]
    )
    v_fp8_flat = torch.cat(
        [
            d.reshape(n, page_size, num_kv_heads, head_dim)
            for d, n in zip(v_fp8_paged, num_pages_per_seq, strict=False)
        ]
    )
    k_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(k_scales_paged, num_pages_per_seq, strict=False)
        ]
    )
    v_s_flat = torch.cat(
        [
            s.reshape(n, page_size, num_kv_heads)
            for s, n in zip(v_scales_paged, num_pages_per_seq, strict=False)
        ]
    )

    if layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads)
        k_fp8_p = k_fp8_flat
        v_fp8_p = v_fp8_flat
        k_s_p = k_s_flat
        v_s_p = v_s_flat
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size)
        k_fp8_p = k_fp8_flat.transpose(1, 2)
        v_fp8_p = v_fp8_flat.transpose(1, 2)
        k_s_p = k_s_flat.transpose(1, 2)
        v_s_p = v_s_flat.transpose(1, 2)

    k_cache, v_cache, k_buf, v_buf = _alloc_paged_cache(
        cache_shape, head_dim, fp8_dtype, device, layout
    )
    k_cache.copy_(k_fp8_p)
    v_cache.copy_(v_fp8_p)
    _write_scales_paged(k_cache, k_buf, k_s_p, head_dim, layout)
    _write_scales_paged(v_cache, v_buf, v_s_p, head_dim, layout)

    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, layout, backend=backend
    )
    wrapper_fp8.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=q_dtype,
        kv_data_type=fp8_dtype,
        use_per_token_head=True,
        pos_encoding_mode=pos_encoding_mode,
    )
    o_fp8 = wrapper_fp8.run(q, (k_cache, v_cache))

    return check_accuracy(o_fp16, o_fp8, fp8_dtype, label=f"batch decode {backend}")


# ============================================================
# Single Prefill Parametrized Tests
# ============================================================


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "no-causal"])
@pytest.mark.parametrize("head_dim", [64, 128], ids=["hd64", "hd128"])
def test_single_prefill_pth_matrix(
    is_gqa,
    q_dtype,
    fp8_dtype,
    causal,
    head_dim,
):
    """Single prefill per-token-head: MHA/GQA x fp16/bf16 x e4m3/e5m2 x causal/non-causal x head_dim."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_single_prefill_pth(
        qo_len=8,
        kv_len=16,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        causal=causal,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("head_dim", [256])
def test_single_prefill_pth_head_dim_256(q_dtype, fp8_dtype, head_dim):
    """Single prefill per-token-head head_dim=256 (SM75 smem stress test)."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    run_single_prefill_pth(
        qo_len=8,
        kv_len=16,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        causal=True,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_single_prefill_pth_rope_llama(q_dtype, fp8_dtype):
    """Single prefill per-token-head with ROPE_LLAMA position encoding.

    SKIPPED: FA2 dispatcher doesn't support ROPE_LLAMA + per-token-head
    variant (NUM_MMA_KV=2 config error). Pre-existing limitation on SM86+.
    SM75 also skipped due to unsupported kernel dispatch configuration.
    """
    pytest.skip("FA2 dispatcher doesn't support ROPE_LLAMA + per-token-head PTH")
    run_single_prefill_pth(
        qo_len=8,
        kv_len=16,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        causal=True,
        backend="fa2",
        pos_encoding_mode="ROPE_LLAMA",
    )


# ============================================================
# Single Decode Parametrized Tests (always JIT)
# ============================================================


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("head_dim", [64, 128], ids=["hd64", "hd128"])
def test_single_decode_pth_matrix(
    is_gqa,
    q_dtype,
    fp8_dtype,
    head_dim,
):
    """Single decode per-token-head: MHA/GQA x fp16/bf16 x e4m3/e5m2 x head_dim (always JIT)."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_single_decode_pth(
        kv_len=64,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("head_dim", [256])
def test_single_decode_pth_head_dim_256(q_dtype, fp8_dtype, head_dim):
    """Single decode per-token-head head_dim=256 (SM75 smem stress test)."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    run_single_decode_pth(
        kv_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=head_dim,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_single_decode_pth_rope_llama(q_dtype, fp8_dtype):
    """Single decode per-token-head with ROPE_LLAMA position encoding."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    run_single_decode_pth(
        kv_len=64,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        pos_encoding_mode="ROPE_LLAMA",
    )


# ============================================================
# Batch Prefill Parametrized Tests
# ============================================================


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
def test_batch_prefill_pth_paged(
    is_gqa,
    q_dtype,
    fp8_dtype,
    layout,
):
    """Batch prefill per-token-head paged: MHA/GQA x fp16/bf16 x e4m3/e5m2 x NHD/HND.

    page_size=16, uniform kv_len=32 (2 pages per sequence).
    """
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    batch = 3
    run_batch_prefill_pth(
        batch_size=batch,
        qo_lens=[8] * batch,
        kv_lens=[32] * batch,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
def test_batch_prefill_pth_ragged(
    is_gqa,
    q_dtype,
    fp8_dtype,
    layout,
):
    """Batch prefill per-token-head ragged: variable-length sequences, variable qo lengths.

    page_size=16, variable kv_lens (1-3 pages per sequence).
    """
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_prefill_pth(
        batch_size=3,
        qo_lens=[6, 10, 8],
        kv_lens=[16, 48, 32],  # 1, 3, 2 pages
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [256])
def test_batch_prefill_pth_head_dim_256(q_dtype, fp8_dtype, layout, head_dim):
    """Batch prefill per-token-head head_dim=256 (SM75 smem stress test)."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    batch = 3
    run_batch_prefill_pth(
        batch_size=batch,
        qo_lens=[8] * batch,
        kv_lens=[32] * batch,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=head_dim,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_batch_prefill_pth_rope_llama(q_dtype, fp8_dtype):
    """Batch prefill per-token-head with ROPE_LLAMA position encoding.

    SKIPPED: FA2 dispatcher doesn't support ROPE_LLAMA + per-token-head
    variant (NUM_MMA_KV=2 config error). Pre-existing limitation.
    """
    pytest.skip("FA2 dispatcher doesn't support ROPE_LLAMA + per-token-head")


# ============================================================
# Batch Decode Parametrized Tests
# ============================================================


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
def test_batch_decode_pth_paged(
    is_gqa,
    q_dtype,
    fp8_dtype,
    layout,
):
    """Batch decode per-token-head paged: MHA/GQA x fp16/bf16 x e4m3/e5m2 x NHD/HND.

    page_size=16, uniform kv_len=32 (2 pages per sequence).
    """
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_decode_pth(
        batch_size=4,
        kv_lens=[32, 32, 32, 32],
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
def test_batch_decode_pth_ragged(
    is_gqa,
    q_dtype,
    fp8_dtype,
    layout,
):
    """Batch decode per-token-head ragged: variable-length kv sequences.

    page_size=16, variable kv_lens (1-4 pages per sequence).
    """
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_batch_decode_pth(
        batch_size=4,
        kv_lens=[16, 48, 32, 64],  # 1, 3, 2, 4 pages
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
@pytest.mark.parametrize("layout", ["NHD", "HND"], ids=["NHD", "HND"])
@pytest.mark.parametrize("head_dim", [256])
def test_batch_decode_pth_head_dim_256(q_dtype, fp8_dtype, layout, head_dim):
    """Batch decode per-token-head head_dim=256 (SM75 smem stress test)."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    run_batch_decode_pth(
        batch_size=4,
        kv_lens=[32] * 4,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=head_dim,
        page_size=16,
        layout=layout,
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="NONE",
    )


@pytest.mark.parametrize(
    "q_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_batch_decode_pth_rope_llama(q_dtype, fp8_dtype):
    """Batch decode per-token-head with ROPE_LLAMA position encoding."""
    _skip_if_sm_below_75()
    _skip_if_bf16_sm75(q_dtype)
    run_batch_decode_pth(
        batch_size=4,
        kv_lens=[32] * 4,
        num_qo_heads=4,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        layout="NHD",
        q_dtype=q_dtype,
        fp8_dtype=fp8_dtype,
        backend="fa2",
        pos_encoding_mode="ROPE_LLAMA",
    )


# ============================================================
# Main entry point — run a subset for quick smoke test
# ============================================================


if __name__ == "__main__":
    print("=== Single Prefill Smoke ===")
    test_single_prefill_pth_matrix(
        is_gqa=False,
        q_dtype=torch.float16,
        fp8_dtype=torch.float8_e4m3fn,
        causal=True,
        head_dim=128,
    )
    test_single_prefill_pth_matrix(
        is_gqa=True,
        q_dtype=torch.float16,
        fp8_dtype=torch.float8_e5m2,
        causal=False,
        head_dim=64,
    )
    print("OK")

    print("\n=== Single Decode Smoke ===")
    test_single_decode_pth_matrix(
        is_gqa=False, q_dtype=torch.float16, fp8_dtype=torch.float8_e4m3fn, head_dim=128
    )
    test_single_decode_pth_matrix(
        is_gqa=True, q_dtype=torch.float16, fp8_dtype=torch.float8_e5m2, head_dim=64
    )
    print("OK")

    print("\n=== Batch Prefill Paged Smoke ===")
    test_batch_prefill_pth_paged(
        is_gqa=False, q_dtype=torch.float16, fp8_dtype=torch.float8_e4m3fn, layout="NHD"
    )
    print("OK")

    print("\n=== Batch Prefill Ragged Smoke ===")
    test_batch_prefill_pth_ragged(
        is_gqa=True, q_dtype=torch.float16, fp8_dtype=torch.float8_e4m3fn, layout="NHD"
    )
    print("OK")

    print("\n=== Batch Decode Paged Smoke ===")
    test_batch_decode_pth_paged(
        is_gqa=False, q_dtype=torch.float16, fp8_dtype=torch.float8_e4m3fn, layout="NHD"
    )
    print("OK")

    print("\n=== Batch Decode Ragged Smoke ===")
    test_batch_decode_pth_ragged(
        is_gqa=True, q_dtype=torch.float16, fp8_dtype=torch.float8_e4m3fn, layout="NHD"
    )
    print("OK")

    print("\n=== All smoke tests passed ===")
