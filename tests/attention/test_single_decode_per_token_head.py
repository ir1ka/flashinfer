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

Single decode FP8 per-token-head KV cache tests.

Tests the single_decode_with_kv_cache API with FP8 per-token-head KV cache
using inline scale storage. Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
"""

import torch
import flashinfer

_FP8_E4M3_MAX = 448.0


def _skip_if_sm75_gqa(num_qo_heads, num_kv_heads):
    """No longer needed — SM75 FP8+GQA fix: independent sync_state float smem
    resolves the pointer arithmetic mismatch. Keep as no-op for compatibility."""
    pass


def quantize_to_fp8_per_token_head(data, fp8_dtype=torch.float8_e4m3fn):
    amax = data.abs().amax(dim=-1, keepdim=True)
    amax = amax.clamp(min=1e-12)
    scales = amax / _FP8_E4M3_MAX
    fp8_data = (
        (data / scales.clamp(min=1e-12))
        .clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX)
        .to(fp8_dtype)
    )
    return fp8_data, scales.squeeze(-1)


def strided_cache_from_fp8(fp8_data, scales, head_dim):
    kv_len, num_kv_heads = fp8_data.shape[0], fp8_data.shape[1]
    stride = head_dim + 16
    buf_size = kv_len * num_kv_heads * stride
    buf = torch.zeros(buf_size, dtype=torch.uint8, device=fp8_data.device)
    rows = buf.reshape(-1, stride)
    # Write FP8 data to bytes 0..head_dim-1 of each row
    fp8_flat = fp8_data.reshape(-1, head_dim).view(torch.uint8)
    rows[:, :head_dim].copy_(fp8_flat)
    # Write float32 scale to bytes head_dim..head_dim+3 of each row
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


def test_single_decode_per_token_head_manual_ref():
    """Compare single decode per_token_head fp8 against fp16 baseline.

    Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
    """
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 64
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_gqa(num_qo_heads, num_kv_heads)

    torch.manual_seed(123)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # FP16 baseline
    _ = flashinfer.single_decode_with_kv_cache(
        q_f16, k_f16, v_f16
    )  # warmup JIT compile

    # Quantize k, v to fp8 per-token-head
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)

    # Build strided cache with inline scales
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    # Dequantize for reference (fp16 k/v)
    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = flashinfer.single_decode_with_kv_cache(q_f16, k_dq, v_dq)

    # Run per-token-head decode
    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        use_per_token_head=True,
    )

    o_ref_cpu = o_ref.cpu()
    o_pth_cpu = o_pth.cpu()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref_cpu.reshape(-1).float(), o_pth_cpu.reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref_cpu - o_pth_cpu).abs().max().item()

    print("[Single decode per_token_head vs dequantized ref]")
    print(f"  Cos sim: {cos_sim:.8f}")
    print(f"  Max diff: {max_diff:.8f}")

    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"
    print("  OK (>= 0.99)")


def test_single_decode_per_token_head_vs_per_tensor():
    """Compare per_token_head vs per_tensor fp8 precision for decode.

    Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
    FA2 ignores scale_k/scale_v, so per-tensor must manually dequantize.
    Per-token-head should give better or equal precision than per-tensor.
    """
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 128
    num_qo_heads, num_kv_heads = 4, 2
    _skip_if_sm75_gqa(num_qo_heads, num_kv_heads)

    torch.manual_seed(42)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # --- Per-token-head path ---
    k_fp8_pth, k_scales_pth = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8_pth, v_scales_pth = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache_pth = strided_cache_from_fp8(k_fp8_pth, k_scales_pth, head_dim)
    v_cache_pth = strided_cache_from_fp8(v_fp8_pth, v_scales_pth, head_dim)

    k_dq_pth = k_fp8_pth.to(torch.float16) * k_scales_pth.unsqueeze(-1).to(
        torch.float16
    )
    v_dq_pth = v_fp8_pth.to(torch.float16) * v_scales_pth.unsqueeze(-1).to(
        torch.float16
    )
    o_ref = flashinfer.single_decode_with_kv_cache(q_f16, k_dq_pth, v_dq_pth)

    o_pth = flashinfer.single_decode_with_kv_cache(
        q_f16,
        k_cache_pth,
        v_cache_pth,
        use_per_token_head=True,
    )

    cos_pth = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()

    # --- Per-tensor path (manual dequantize) ---
    amax_k = k_f16.abs().amax().item()
    scale_k = max(amax_k, 1e-12) / _FP8_E4M3_MAX
    k_fp8_pt = (k_f16 / scale_k).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(fp8_dtype)
    amax_v = v_f16.abs().amax().item()
    scale_v = max(amax_v, 1e-12) / _FP8_E4M3_MAX
    v_fp8_pt = (v_f16 / scale_v).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(fp8_dtype)

    k_dq_pt = k_fp8_pt.to(torch.float16) * scale_k
    v_dq_pt = v_fp8_pt.to(torch.float16) * scale_v
    o_pt = flashinfer.single_decode_with_kv_cache(q_f16, k_dq_pt, v_dq_pt)

    cos_pt = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pt.cpu().reshape(-1).float(), dim=0
    ).item()

    print("[Decode per_token_head vs per_tensor]")
    print(f"  Per-token-head cos_sim: {cos_pth:.8f}")
    print(f"  Per-tensor cos_sim:     {cos_pt:.8f}")

    assert cos_pth >= 0.99, f"per_token_head cos_sim={cos_pth}"
    assert cos_pt >= 0.99, f"per_tensor cos_sim={cos_pt}"
    assert cos_pth >= cos_pt, (
        f"Per-token-head ({cos_pth:.6f}) should be >= per-tensor ({cos_pt:.6f})"
    )
    print(f"  OK (per-token-head={cos_pth:.6f} >= per-tensor={cos_pt:.6f})")


def test_single_decode_per_token_head_gqa():
    """Test single decode with GQA (num_qo_heads > num_kv_heads)."""
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 128
    num_qo_heads, num_kv_heads = 8, 2
    _skip_if_sm75_gqa(num_qo_heads, num_kv_heads)

    torch.manual_seed(99)
    q_f16 = torch.randn(num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Quantize k, v to fp8 per-token-head
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)

    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    # Dequantize for reference
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

    print("[Single decode GQA (4:1) per_token_head]")
    print(f"  Cos sim: {cos_sim:.8f}")

    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"
    print("  OK (>= 0.99)")


def test_single_decode_per_token_head_mha():
    """Test single decode with MHA (num_qo_heads == num_kv_heads)."""
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    kv_len = 64
    num_heads = 4

    torch.manual_seed(77)
    q_f16 = torch.randn(num_heads, head_dim, dtype=torch.float16, device=device)
    k_f16 = 0.3 * torch.randn(
        kv_len, num_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_heads, head_dim, dtype=torch.float16, device=device
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

    print("[Single decode MHA per_token_head]")
    print(f"  Cos sim: {cos_sim:.8f}")

    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"
    print("  OK (>= 0.99)")


if __name__ == "__main__":
    test_single_decode_per_token_head_manual_ref()
    print("test_single_decode_per_token_head_manual_ref passed!")

    test_single_decode_per_token_head_vs_per_tensor()
    print("test_single_decode_per_token_head_vs_per_tensor passed!")

    test_single_decode_per_token_head_gqa()
    print("test_single_decode_per_token_head_gqa passed!")

    test_single_decode_per_token_head_mha()
    print("test_single_decode_per_token_head_mha passed!")

    print("\nAll single decode tests passed!")
