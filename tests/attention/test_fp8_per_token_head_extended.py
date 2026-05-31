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

FP8 per-token-head KV cache extended tests.

Covers additional parameter combinations not tested in the base test suite:
- head_dim variants (64, 256)
- kv_layout HND
- page_size variants (1, 8)
- float8_e5m2 dtype
- pos_encoding_mode ROPE_LLAMA
- large kv_len / large batch_size
- MHA vs GQA
- non-causal prefill
- long sequence prefill
"""

import pytest
import torch
import flashinfer

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def _get_fp8_max(fp8_dtype):
    if fp8_dtype == torch.float8_e5m2:
        return _FP8_E5M2_MAX
    return _FP8_E4M3_MAX


def check_fp8_accuracy(o_ref, o_pth, fp8_dtype, label=""):
    """Compare FP8 per-token-head output against FP16 dequantized reference.

    Thresholds:
    - e4m3: cos_sim >= 0.99
    - e5m2: cos_sim >= 0.99 (wider dynamic range, coarser quantization)
    Batch routines with larger problem sizes typically achieve >= 0.999.
    """
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref.cpu() - o_pth.cpu()).abs().max().item()

    is_e5m2 = fp8_dtype == torch.float8_e5m2
    dtype_tag = "e5m2" if is_e5m2 else "e4m3"
    prefix = f"[{label}] " if label else ""

    print(f"{prefix}{dtype_tag} cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}")

    assert cos_sim >= 0.99, f"{prefix}cos_sim={cos_sim:.8f} < 0.99 ({dtype_tag})"

    return cos_sim, max_diff


def _skip_if_sm_below_75():
    cc = torch.cuda.get_device_capability(0)
    if cc[0] < 7 or (cc[0] == 7 and cc[1] < 5):
        pytest.skip("FP8 mixed-precision requires SM75+ for FA2 backend.")


def _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads):
    """No longer needed — SM75 FP8+GQA fix: independent sync_state float smem
    resolves the pointer arithmetic mismatch. Keep as no-op for compatibility."""
    pass


def quantize_to_fp8_per_token_head(data, fp8_dtype=torch.float8_e4m3fn):
    amax = data.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=1e-12)
    fp8_max = _get_fp8_max(fp8_dtype)
    scales = amax / fp8_max
    fp8_data = (
        (data / scales.clamp(min=1e-12)).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    )
    return fp8_data, scales.squeeze(-1)


def quantize_to_fp8_per_token_head_3d(data, fp8_dtype=torch.float8_e4m3fn):
    """Quantize (batch, seq, heads, head_dim) -> scales per (batch, seq, heads)."""
    amax = data.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=1e-12)
    fp8_max = _get_fp8_max(fp8_dtype)
    scales = amax / fp8_max
    fp8_data = (
        (data / scales.clamp(min=1e-12)).clamp(min=-fp8_max, max=fp8_max).to(fp8_dtype)
    )
    return fp8_data, scales.squeeze(-1)


def strided_cache_from_fp8(fp8_data, scales, head_dim):
    """Build strided cache with inline scales for single prefill/decode."""
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


# ============================================================
# Single Prefill Extended Tests
# ============================================================


def test_single_prefill_per_token_head_mha():
    """Single prefill per_token_head with MHA (num_qo == num_kv)."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    head_dim = 128
    qo_len, kv_len = 8, 16
    num_heads = 4

    torch.manual_seed(42)
    q_f16 = torch.randn(qo_len, num_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q_f16, k_dq, v_dq, causal=True, backend="fa2"
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_sim, max_diff = check_fp8_accuracy(
        o_ref, o_pth, torch.float8_e4m3fn, label="single prefill MHA"
    )


def test_single_prefill_per_token_head_non_causal():
    """Single prefill per_token_head with causal=False."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    head_dim = 128
    qo_len, kv_len = 8, 16
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(77)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q_f16, k_dq, v_dq, causal=False, backend="fa2"
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=False,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single prefill non-causal] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


@pytest.mark.parametrize("head_dim", [64, 256])
def test_single_prefill_per_token_head_head_dim_variant(head_dim):
    """Single prefill per_token_head with different head dimensions."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    qo_len, kv_len = 8, 16
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(42)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q_f16, k_dq, v_dq, causal=True, backend="fa2"
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single prefill head_dim={head_dim}] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_single_prefill_per_token_head_e5m2():
    """Single prefill per_token_head with float8_e5m2 dtype."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    fp8_dtype = torch.float8_e5m2
    head_dim = 128
    qo_len, kv_len = 8, 16
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(42)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q_f16, k_dq, v_dq, causal=True, backend="fa2"
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single prefill e5m2] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_single_prefill_per_token_head_long_seq():
    """Single prefill per_token_head with long sequence."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    head_dim = 128
    qo_len, kv_len = 64, 1024
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(42)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_prefill_with_kv_cache(
        q_f16, k_dq, v_dq, causal=True, backend="fa2"
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single prefill long_seq kv_len={kv_len}] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


# ============================================================
# Single Decode Extended Tests
# ============================================================


@pytest.mark.parametrize("head_dim", [64, 256])
def test_single_decode_per_token_head_head_dim_variant(head_dim):
    """Single decode per_token_head with different head dimensions."""
    _skip_if_sm_below_75()
    fp8_dtype = torch.float8_e4m3fn
    kv_len = 64
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads)

    device = "cuda:0"
    torch.manual_seed(42)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_decode_with_kv_cache(q_f16, k_dq, v_dq)
    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        use_per_token_head=True,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single decode head_dim={head_dim}] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_single_decode_per_token_head_e5m2():
    """Single decode per_token_head with float8_e5m2 dtype."""
    _skip_if_sm_below_75()
    fp8_dtype = torch.float8_e5m2
    head_dim = 128
    kv_len = 64
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads)

    device = "cuda:0"
    torch.manual_seed(42)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_decode_with_kv_cache(q_f16, k_dq, v_dq)
    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        use_per_token_head=True,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single decode e5m2] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_single_decode_per_token_head_rope_llama():
    """Single decode per_token_head with ROPE_LLAMA position encoding."""
    _skip_if_sm_below_75()
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 64
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads)

    device = "cuda:0"
    torch.manual_seed(42)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_dq,
        v_dq,
        pos_encoding_mode="ROPE_LLAMA",
    )
    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        pos_encoding_mode="ROPE_LLAMA",
        use_per_token_head=True,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single decode ROPE_LLAMA] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_single_decode_per_token_head_long_seq():
    """Single decode per_token_head with long sequence."""
    _skip_if_sm_below_75()
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 1024
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_fp8_gqa(num_qo_heads, num_kv_heads)

    device = "cuda:0"
    torch.manual_seed(42)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_decode_with_kv_cache(q_f16, k_dq, v_dq)
    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        use_per_token_head=True,
    )

    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()
    print(f"[single decode long_seq kv_len={kv_len}] cos_sim: {cos_sim:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


# ============================================================
# Batch Prefill Extended Tests
# ============================================================


def _allocate_paged_kv_with_inline_scale(shape, head_dim, dtype, device, layout="NHD"):
    """Allocate paged KV cache with inline per-token-head scale space."""
    if layout == "NHD":
        max_pages, page_size, num_kv_heads = shape[0], shape[1], shape[2]
    else:
        max_pages, num_kv_heads, page_size = shape[0], shape[1], shape[2]
    stride = head_dim + 16
    total_tokens = max_pages * page_size * num_kv_heads
    buf_size = total_tokens * stride
    k_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)
    v_buf = torch.empty(buf_size, dtype=torch.uint8, device=device)

    if layout == "NHD":
        s = (page_size * num_kv_heads * stride, num_kv_heads * stride, stride, 1)
    else:
        s = (num_kv_heads * page_size * stride, page_size * stride, stride, 1)

    k_cache = torch.as_strided(k_buf, shape, s).view(dtype)
    v_cache = torch.as_strided(v_buf, shape, s).view(dtype)
    return k_cache, v_cache, k_buf, v_buf


def _write_inline_scales_paged(cache_tensor, buf, scales, head_dim, layout="NHD"):
    """Write per-token-head scales into inline positions of the paged cache buffer."""
    stride = head_dim + 16
    scale_stride_f32 = stride // 4
    scale_offset_f32 = head_dim // 4
    cache_shape = cache_tensor.shape

    if layout == "NHD":
        max_pages, page_size, num_kv_heads = (
            cache_shape[0],
            cache_shape[1],
            cache_shape[2],
        )
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
        max_pages, num_kv_heads, page_size = (
            cache_shape[0],
            cache_shape[1],
            cache_shape[2],
        )
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


def _run_batch_prefill_per_token_head(
    batch_size,
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    kv_layout,
    fp8_dtype,
):
    """Helper: run batch prefill per_token_head and return (cos_sim, max_diff)."""
    device = "cuda:0"
    torch.manual_seed(42)

    total_qo = batch_size * qo_len
    q_f16 = torch.randn(
        total_qo, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

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

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # FP16 baseline
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

    # FP8 per-token-head
    k_flat = k_f16.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_f16.reshape(-1, num_kv_heads, head_dim)
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_flat, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_flat, fp8_dtype)

    if kv_layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads, head_dim)
        k_paged_fp8 = k_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        v_paged_fp8 = v_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        k_scales_paged = k_scales.reshape(total_num_pages, page_size, num_kv_heads)
        v_scales_paged = v_scales.reshape(total_num_pages, page_size, num_kv_heads)
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size, head_dim)
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

    k_cache, v_cache, k_buf, v_buf = _allocate_paged_kv_with_inline_scale(
        cache_shape, head_dim, fp8_dtype, device, kv_layout
    )
    k_cache.copy_(k_paged_fp8)
    v_cache.copy_(v_paged_fp8)
    _write_inline_scales_paged(k_cache, k_buf, k_scales_paged, head_dim, kv_layout)
    _write_inline_scales_paged(v_cache, v_buf, v_scales_paged, head_dim, kv_layout)

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

    o_fp16_flat = o_fp16.reshape(-1).float()
    o_fp8_flat = o_fp8.reshape(-1).float()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_fp16_flat, o_fp8_flat, dim=0
    ).item()
    max_diff = (o_fp16 - o_fp8).abs().amax().item()
    return cos_sim, max_diff


@pytest.mark.parametrize("kv_layout", ["HND"])
def test_batch_prefill_per_token_head_hnd(kv_layout):
    """Batch prefill per_token_head with HND layout."""
    _skip_if_sm_below_75()
    cos_sim, max_diff = _run_batch_prefill_per_token_head(
        batch_size=3,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        page_size=16,
        kv_layout=kv_layout,
        fp8_dtype=torch.float8_e4m3fn,
    )
    print(
        f"[batch prefill {kv_layout}] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


@pytest.mark.parametrize("page_size", [1, 8])
def test_batch_prefill_per_token_head_page_size_variant(page_size):
    """Batch prefill per_token_head with different page sizes."""
    _skip_if_sm_below_75()
    cos_sim, max_diff = _run_batch_prefill_per_token_head(
        batch_size=3,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        page_size=page_size,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
    )
    print(
        f"[batch prefill page_size={page_size}] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_prefill_per_token_head_head_dim_256():
    """Batch prefill per_token_head with head_dim=256."""
    _skip_if_sm_below_75()
    cos_sim, max_diff = _run_batch_prefill_per_token_head(
        batch_size=3,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=256,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
    )
    print(
        f"[batch prefill head_dim=256] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_prefill_per_token_head_e5m2():
    """Batch prefill per_token_head with float8_e5m2 dtype."""
    _skip_if_sm_below_75()
    cos_sim, max_diff = _run_batch_prefill_per_token_head(
        batch_size=3,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e5m2,
    )
    print(f"[batch prefill e5m2] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_prefill_per_token_head_large_batch():
    """Batch prefill per_token_head with large batch size."""
    _skip_if_sm_below_75()
    cos_sim, max_diff = _run_batch_prefill_per_token_head(
        batch_size=12,
        qo_len=8,
        kv_len=32,
        num_kv_heads=4,
        num_qo_heads=4,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
    )
    print(
        f"[batch prefill large_batch] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


# ============================================================
# Batch Decode Extended Tests
# ============================================================


def _run_batch_decode_per_token_head(
    batch_size,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    page_size,
    kv_layout,
    fp8_dtype,
    pos_encoding_mode,
):
    """Helper: run batch decode per_token_head and return (cos_sim, max_diff)."""
    device = "cuda:0"
    torch.manual_seed(42)

    q_f16 = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.1 * torch.randn(
        batch_size, kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

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

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # FP16 baseline
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

    decode_wrapper_f16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    decode_wrapper_f16.plan(
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = decode_wrapper_f16.run(q_f16, (k_paged_f16, v_paged_f16))

    # FP8 per-token-head
    k_flat = k_f16.reshape(-1, num_kv_heads, head_dim)
    v_flat = v_f16.reshape(-1, num_kv_heads, head_dim)
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_flat, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_flat, fp8_dtype)

    if kv_layout == "NHD":
        cache_shape = (total_num_pages, page_size, num_kv_heads, head_dim)
        k_paged_fp8 = k_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        v_paged_fp8 = v_fp8.reshape(total_num_pages, page_size, num_kv_heads, head_dim)
        k_scales_paged = k_scales.reshape(total_num_pages, page_size, num_kv_heads)
        v_scales_paged = v_scales.reshape(total_num_pages, page_size, num_kv_heads)
    else:
        cache_shape = (total_num_pages, num_kv_heads, page_size, head_dim)
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

    k_cache, v_cache, k_buf, v_buf = _allocate_paged_kv_with_inline_scale(
        cache_shape, head_dim, fp8_dtype, device, kv_layout
    )
    k_cache.copy_(k_paged_fp8)
    v_cache.copy_(v_paged_fp8)
    _write_inline_scales_paged(k_cache, k_buf, k_scales_paged, head_dim, kv_layout)
    _write_inline_scales_paged(v_cache, v_buf, v_scales_paged, head_dim, kv_layout)

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
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=fp8_dtype,
        use_per_token_head=True,
    )
    o_fp8 = decode_wrapper_fp8.run(q_f16, (k_cache, v_cache))

    o_fp16_flat = o_fp16.reshape(-1).float()
    o_fp8_flat = o_fp8.reshape(-1).float()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_fp16_flat, o_fp8_flat, dim=0
    ).item()
    max_diff = (o_fp16 - o_fp8).abs().amax().item()
    return cos_sim, max_diff


@pytest.mark.parametrize("kv_layout", ["HND"])
def test_batch_decode_per_token_head_hnd(kv_layout):
    """Batch decode per_token_head with HND layout."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=16,
        kv_layout=kv_layout,
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
    )
    print(
        f"[batch decode {kv_layout}] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


@pytest.mark.parametrize("page_size", [1, 8])
def test_batch_decode_per_token_head_page_size_variant(page_size):
    """Batch decode per_token_head with different page sizes."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=page_size,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
    )
    print(
        f"[batch decode page_size={page_size}] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_decode_per_token_head_head_dim_256():
    """Batch decode per_token_head with head_dim=256."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=256,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
    )
    print(
        f"[batch decode head_dim=256] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_decode_per_token_head_e5m2():
    """Batch decode per_token_head with float8_e5m2 dtype."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e5m2,
        pos_encoding_mode="NONE",
    )
    print(f"[batch decode e5m2] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_decode_per_token_head_rope_llama():
    """Batch decode per_token_head with ROPE_LLAMA position encoding."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="ROPE_LLAMA",
    )
    print(f"[batch decode ROPE_LLAMA] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}")
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_decode_per_token_head_large_kv():
    """Batch decode per_token_head with large kv_len."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=4,
        kv_len=512,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
    )
    print(
        f"[batch decode large_kv kv_len=512] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


def test_batch_decode_per_token_head_large_batch():
    """Batch decode per_token_head with large batch size."""
    _skip_if_sm_below_75()
    num_qo_heads, num_kv_heads = 4, 4
    cos_sim, max_diff = _run_batch_decode_per_token_head(
        batch_size=12,
        kv_len=32,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=128,
        page_size=16,
        kv_layout="NHD",
        fp8_dtype=torch.float8_e4m3fn,
        pos_encoding_mode="NONE",
    )
    print(
        f"[batch decode large_batch] cos_sim: {cos_sim:.8f}, max_diff: {max_diff:.8f}"
    )
    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"


if __name__ == "__main__":
    print("=== Single Prefill Extended ===")
    test_single_prefill_per_token_head_mha()
    print("test_single_prefill_per_token_head_mha passed!")

    test_single_prefill_per_token_head_non_causal()
    print("test_single_prefill_per_token_head_non_causal passed!")

    test_single_prefill_per_token_head_head_dim_variant(64)
    print("test_single_prefill_per_token_head_head_dim_variant(64) passed!")

    test_single_prefill_per_token_head_head_dim_variant(256)
    print("test_single_prefill_per_token_head_head_dim_variant(256) passed!")

    test_single_prefill_per_token_head_e5m2()
    print("test_single_prefill_per_token_head_e5m2 passed!")

    test_single_prefill_per_token_head_long_seq()
    print("test_single_prefill_per_token_head_long_seq passed!")

    print("\n=== Single Decode Extended ===")
    test_single_decode_per_token_head_head_dim_variant(64)
    print("test_single_decode_per_token_head_head_dim_variant(64) passed!")

    test_single_decode_per_token_head_head_dim_variant(256)
    print("test_single_decode_per_token_head_head_dim_variant(256) passed!")

    test_single_decode_per_token_head_e5m2()
    print("test_single_decode_per_token_head_e5m2 passed!")

    test_single_decode_per_token_head_rope_llama()
    print("test_single_decode_per_token_head_rope_llama passed!")

    test_single_decode_per_token_head_long_seq()
    print("test_single_decode_per_token_head_long_seq passed!")

    print("\n=== Batch Prefill Extended ===")
    test_batch_prefill_per_token_head_hnd("HND")
    print("test_batch_prefill_per_token_head_hnd passed!")

    test_batch_prefill_per_token_head_page_size_variant(1)
    print("test_batch_prefill_per_token_head_page_size_variant(1) passed!")

    test_batch_prefill_per_token_head_page_size_variant(8)
    print("test_batch_prefill_per_token_head_page_size_variant(8) passed!")

    test_batch_prefill_per_token_head_head_dim_256()
    print("test_batch_prefill_per_token_head_head_dim_256 passed!")

    test_batch_prefill_per_token_head_e5m2()
    print("test_batch_prefill_per_token_head_e5m2 passed!")

    test_batch_prefill_per_token_head_large_batch()
    print("test_batch_prefill_per_token_head_large_batch passed!")

    print("\n=== Batch Decode Extended ===")
    test_batch_decode_per_token_head_hnd("HND")
    print("test_batch_decode_per_token_head_hnd passed!")

    test_batch_decode_per_token_head_page_size_variant(1)
    print("test_batch_decode_per_token_head_page_size_variant(1) passed!")

    test_batch_decode_per_token_head_page_size_variant(8)
    print("test_batch_decode_per_token_head_page_size_variant(8) passed!")

    test_batch_decode_per_token_head_head_dim_256()
    print("test_batch_decode_per_token_head_head_dim_256 passed!")

    test_batch_decode_per_token_head_e5m2()
    print("test_batch_decode_per_token_head_e5m2 passed!")

    test_batch_decode_per_token_head_rope_llama()
    print("test_batch_decode_per_token_head_rope_llama passed!")

    test_batch_decode_per_token_head_large_kv()
    print("test_batch_decode_per_token_head_large_kv passed!")

    test_batch_decode_per_token_head_large_batch()
    print("test_batch_decode_per_token_head_large_batch passed!")

    print("\n=== All extended tests passed! ===")
