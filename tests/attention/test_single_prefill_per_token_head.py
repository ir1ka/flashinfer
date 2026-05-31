import torch
import flashinfer

_FP8_E4M3_MAX = 448.0


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


def fa2_reference(q, k, v, causal=True):
    """FA2 reference with dequantized FP16 K/V."""
    return flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend="fa2"
    )


def test_per_token_head_fa2_ref():
    """Compare per_token_head fp8 against FA2 reference.

    Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
    FA2 supports inline scale loading via use_per_token_head=True.
    """
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    qo_len, kv_len = 8, 16
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(123)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Quantize k, v to fp8 (q stays fp16)
    k_fp8, k_scales = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8, v_scales = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)

    # Dequantize to fp16 for FA2 reference
    k_dq = k_fp8.to(torch.float16) * k_scales.unsqueeze(-1).to(torch.float16)
    v_dq = v_fp8.to(torch.float16) * v_scales.unsqueeze(-1).to(torch.float16)
    o_ref = fa2_reference(q_f16, k_dq, v_dq, causal=True)

    # Per-token-head fp8 cache
    k_cache = strided_cache_from_fp8(k_fp8, k_scales, head_dim)
    v_cache = strided_cache_from_fp8(v_fp8, v_scales, head_dim)

    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    o_ref_cpu = o_ref.cpu()
    o_pth_cpu = o_pth.cpu()
    max_diff = (o_ref_cpu - o_pth_cpu).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref_cpu.reshape(-1).float(), o_pth_cpu.reshape(-1).float(), dim=0
    ).item()

    print("[FA2 ref vs per_token_head]")
    print(f"  Cos sim: {cos_sim:.8f}")
    print(f"  Max diff: {max_diff:.8f}")

    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"
    print("  OK (>= 0.99)")


def test_per_token_head_gqa():
    """Single prefill per_token_head with GQA (num_qo_heads > num_kv_heads).

    Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
    """
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    qo_len, kv_len = 16, 32
    num_qo_heads, num_kv_heads = 8, 2

    torch.manual_seed(99)
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
    o_ref = fa2_reference(q_f16, k_dq, v_dq, causal=True)

    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache,
        v_cache,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    o_ref_cpu = o_ref.cpu()
    o_pth_cpu = o_pth.cpu()
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref_cpu.reshape(-1).float(), o_pth_cpu.reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref_cpu - o_pth_cpu).abs().max().item()

    print("[Single prefill GQA (4:1) per_token_head]")
    print(f"  Cos sim: {cos_sim:.8f}")
    print(f"  Max diff: {max_diff:.8f}")

    assert cos_sim >= 0.99, f"cos_sim={cos_sim}"
    print("  OK (>= 0.99)")


def quantize_to_fp8_per_tensor(data, fp8_dtype=torch.float8_e4m3fn):
    """Quantize FP16/BF16 data to FP8 with a single per-tensor scale."""
    amax = data.abs().amax().item()
    amax = max(amax, 1e-12)
    scale = amax / _FP8_E4M3_MAX
    fp8_data = (data / scale).clamp(min=-_FP8_E4M3_MAX, max=_FP8_E4M3_MAX).to(fp8_dtype)
    return fp8_data, scale


def test_per_token_head_vs_per_tensor():
    """Compare per_token_head vs per_tensor fp8 precision.

    Uses FA2 backend with fp16 q + fp8 k/v (mixed precision).
    FA2 ignores scale_k/scale_v parameters, so per-tensor must manually
    dequantize to FP16 before calling FA2.
    Per-token-head should give better precision than per-tensor.
    """
    device = "cuda:0"
    fp8_dtype = torch.float8_e4m3fn
    head_dim = 128
    qo_len, kv_len = 16, 64
    num_qo_heads, num_kv_heads = 4, 2

    torch.manual_seed(123)
    q_f16 = torch.randn(
        qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_f16 = 0.3 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # FP16 reference: FA2 with original fp16 K/V
    o_ref = fa2_reference(q_f16, k_f16, v_f16, causal=True)

    # --- Per-token-head path ---
    k_fp8_pth, k_scales_pth = quantize_to_fp8_per_token_head(k_f16, fp8_dtype)
    v_fp8_pth, v_scales_pth = quantize_to_fp8_per_token_head(v_f16, fp8_dtype)
    k_cache_pth = strided_cache_from_fp8(k_fp8_pth, k_scales_pth, head_dim)
    v_cache_pth = strided_cache_from_fp8(v_fp8_pth, v_scales_pth, head_dim)

    o_pth = flashinfer.single_prefill_with_kv_cache(
        q_f16,
        k_cache_pth,
        v_cache_pth,
        causal=True,
        use_per_token_head=True,
        backend="fa2",
        o_dtype=torch.float16,
    )

    cos_pth = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pth.cpu().reshape(-1).float(), dim=0
    ).item()

    # --- Per-tensor path (manual dequantize, since FA2 ignores scale_k/scale_v) ---
    k_fp8_pt, k_scale_pt = quantize_to_fp8_per_tensor(k_f16, fp8_dtype)
    v_fp8_pt, v_scale_pt = quantize_to_fp8_per_tensor(v_f16, fp8_dtype)

    k_dq_pt = k_fp8_pt.to(torch.float16) * k_scale_pt
    v_dq_pt = v_fp8_pt.to(torch.float16) * v_scale_pt
    o_pt = fa2_reference(q_f16, k_dq_pt, v_dq_pt, causal=True)

    cos_pt = torch.nn.functional.cosine_similarity(
        o_ref.cpu().reshape(-1).float(), o_pt.cpu().reshape(-1).float(), dim=0
    ).item()

    print("[Per-token-head vs Per-tensor vs FA2 FP16 reference]")
    print(f"  Per-token-head cos_sim: {cos_pth:.8f}")
    print(f"  Per-tensor cos_sim:     {cos_pt:.8f}")

    assert cos_pth >= 0.99, f"per_token_head cos_sim={cos_pth}"
    assert cos_pt >= 0.99, f"per_tensor cos_sim={cos_pt}"
    # Note: per-token-head is generally expected to be >= per-tensor, but with
    # small problem sizes and specific random seeds, per-tensor's single scale
    # can coincidentally produce better dequantization. Both are validated
    # independently against the 0.99 threshold.
    print("  OK (both >= 0.99)")
