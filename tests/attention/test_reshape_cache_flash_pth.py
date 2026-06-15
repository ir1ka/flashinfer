"""
Copyright (c) 2024 by FlashInfer code contributors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for reshape_and_cache_flash_per_token_head (FP8 per-token-head quantize + cache write).
"""

import pytest
import torch
import flashinfer
from tests.utils_fp8 import get_cos_sim_threshold


# ============================================================
# Helpers
# ============================================================


def _cc():
    return torch.cuda.get_device_capability(0)


def _skip_if_sm_below_75():
    if _cc()[0] < 7 or (_cc()[0] == 7 and _cc()[1] < 5):
        pytest.skip("Requires SM75+")


def _skip_if_not_fp16_sm75(dtype: torch.dtype):
    if dtype != torch.float16 and _cc()[0] <= 7:
        pytest.skip(f"{dtype} skipped on SM75")


def check_fp8_accuracy(
    q_ref: torch.Tensor,
    q_act: torch.Tensor,
    s_ref: torch.Tensor,
    s_act: torch.Tensor,
    label: str = "",
):
    """Compare FP8 quantized data + scales between reference and actual."""
    prefix = f"[{label}] " if label else ""
    s_cos = torch.nn.functional.cosine_similarity(
        s_ref.reshape(-1).float(), s_act.reshape(-1).float(), dim=0
    ).item()
    s_max_diff = (s_ref - s_act).abs().max().item()

    q_ref_f32 = q_ref.to(torch.float32)
    q_act_f32 = q_act.to(torch.float32)
    q_cos = torch.nn.functional.cosine_similarity(
        q_ref_f32.reshape(-1), q_act_f32.reshape(-1), dim=0
    ).item()
    q_max_diff = (q_ref_f32 - q_act_f32).abs().max().item()

    threshold = get_cos_sim_threshold(torch.float8_e4m3fn)
    print(
        f"{prefix}scale cos_sim={s_cos:.8f} max_diff={s_max_diff:.8e} | "
        f"quant cos_sim={q_cos:.8f} max_diff={q_max_diff:.8e}"
    )
    assert s_cos >= threshold, f"{prefix}scale cos_sim={s_cos:.8f} < {threshold}"
    assert q_cos >= threshold, f"{prefix}quant cos_sim={q_cos:.8f} < {threshold}"
    return q_cos, s_cos


def _ref_reshape_and_cache_flash_pth(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Pure PyTorch reference: per-token-head FP8 quantize + cache write."""
    fp8_dtype = k_cache.dtype
    quant_max = torch.finfo(fp8_dtype).max

    num_tokens, num_kv_heads, head_dim = key.shape
    head_dim_v = value.shape[2]

    for tok in range(num_tokens):
        slot = int(slot_mapping[tok].item())
        if slot < 0:
            continue
        for h in range(num_kv_heads):
            k_f32 = key[tok, h, :head_dim].to(torch.float32)
            k_amax = k_f32.abs().amax().item()
            k_scale = max(k_amax / quant_max, 1e-6)
            k_q = (k_f32 / k_scale).clamp(-quant_max, quant_max).to(fp8_dtype)
            k_cache[slot, h, :head_dim] = k_q
            k_scale_cache[slot, h] = k_scale

            v_f32 = value[tok, h, :head_dim_v].to(torch.float32)
            v_amax = v_f32.abs().amax().item()
            v_scale = max(v_amax / quant_max, 1e-6)
            v_q = (v_f32 / v_scale).clamp(-quant_max, quant_max).to(fp8_dtype)
            v_cache[slot, h, :head_dim_v] = v_q
            v_scale_cache[slot, h] = v_scale


# ============================================================
# Ragged layout
# ============================================================


def run_reshape_cache_pth_ragged(
    num_tokens,
    num_kv_heads,
    head_dim,
    head_dim_v,
    total_seq_len,
    dtype,
    fp8_dtype,
    slot_offsets=None,
):
    """Test reshape_and_cache_flash_per_token_head with ragged layout."""
    device = "cuda"
    if slot_offsets is None:
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
    else:
        slot_mapping = torch.tensor(slot_offsets, dtype=torch.int32, device=device)

    key = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    value = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim_v, dtype=dtype, device=device
    )

    k_cache_ref = torch.zeros(
        total_seq_len, num_kv_heads, head_dim, dtype=fp8_dtype, device=device
    )
    v_cache_ref = torch.zeros(
        total_seq_len, num_kv_heads, head_dim_v, dtype=fp8_dtype, device=device
    )
    k_scale_ref = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )
    v_scale_ref = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )
    _ref_reshape_and_cache_flash_pth(
        key,
        value,
        k_cache_ref,
        v_cache_ref,
        k_scale_ref,
        v_scale_ref,
        slot_mapping,
    )

    k_cache = torch.zeros_like(k_cache_ref)
    v_cache = torch.zeros_like(v_cache_ref)
    k_scale = torch.zeros_like(k_scale_ref)
    v_scale = torch.zeros_like(v_scale_ref)
    flashinfer.reshape_and_cache_flash_per_token_head(
        key,
        value,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping,
    )

    check_fp8_accuracy(
        k_cache_ref,
        k_cache,
        k_scale_ref,
        k_scale,
        label=f"ragged-k d={head_dim} dv={head_dim_v}",
    )
    check_fp8_accuracy(
        v_cache_ref,
        v_cache,
        v_scale_ref,
        v_scale,
        label=f"ragged-v d={head_dim} dv={head_dim_v}",
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_reshape_cache_pth_ragged(head_dim, dtype, fp8_dtype):
    """Ragged layout with various head_dims and dtypes."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_ragged(
        num_tokens=16,
        num_kv_heads=4,
        head_dim=head_dim,
        head_dim_v=head_dim,
        total_seq_len=64,
        dtype=dtype,
        fp8_dtype=fp8_dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_ragged_gqa(dtype):
    """Ragged layout with GQA."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_ragged(
        num_tokens=8,
        num_kv_heads=2,
        head_dim=128,
        head_dim_v=128,
        total_seq_len=32,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_ragged_scattered(dtype):
    """Ragged layout with non-contiguous slot mapping."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_ragged(
        num_tokens=8,
        num_kv_heads=4,
        head_dim=128,
        head_dim_v=128,
        total_seq_len=64,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
        slot_offsets=[10, 30, 5, 45, 20, 55, 15, 50],
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_ragged_head_dim_v_diff(dtype):
    """Ragged layout with head_dim_v != head_dim."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_ragged(
        num_tokens=8,
        num_kv_heads=4,
        head_dim=128,
        head_dim_v=64,
        total_seq_len=32,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
    )


# ============================================================
# Paged layout
# ============================================================


def run_reshape_cache_pth_paged(
    num_tokens,
    num_kv_heads,
    head_dim,
    head_dim_v,
    page_size,
    max_num_pages,
    dtype,
    fp8_dtype,
    slot_offsets=None,
):
    """Test reshape_and_cache_flash_per_token_head with paged layout."""
    device = "cuda"
    if slot_offsets is None:
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)
    else:
        slot_mapping = torch.tensor(slot_offsets, dtype=torch.int32, device=device)

    key = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    value = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim_v, dtype=dtype, device=device
    )

    k_cache_ref = torch.zeros(
        max_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=fp8_dtype,
        device=device,
    )
    v_cache_ref = torch.zeros(
        max_num_pages,
        page_size,
        num_kv_heads,
        head_dim_v,
        dtype=fp8_dtype,
        device=device,
    )
    k_scale_ref = torch.zeros(
        max_num_pages, page_size, num_kv_heads, dtype=torch.float32, device=device
    )
    v_scale_ref = torch.zeros(
        max_num_pages, page_size, num_kv_heads, dtype=torch.float32, device=device
    )

    # Flatten to ragged-style for reference
    k_cache_ref_3d = k_cache_ref.reshape(-1, num_kv_heads, head_dim)
    v_cache_ref_3d = v_cache_ref.reshape(-1, num_kv_heads, head_dim_v)
    k_scale_ref_2d = k_scale_ref.reshape(-1, num_kv_heads)
    v_scale_ref_2d = v_scale_ref.reshape(-1, num_kv_heads)
    _ref_reshape_and_cache_flash_pth(
        key,
        value,
        k_cache_ref_3d,
        v_cache_ref_3d,
        k_scale_ref_2d,
        v_scale_ref_2d,
        slot_mapping,
    )

    k_cache = torch.zeros_like(k_cache_ref)
    v_cache = torch.zeros_like(v_cache_ref)
    k_scale = torch.zeros_like(k_scale_ref)
    v_scale = torch.zeros_like(v_scale_ref)
    flashinfer.reshape_and_cache_flash_per_token_head(
        key,
        value,
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        slot_mapping,
    )

    check_fp8_accuracy(
        k_cache_ref,
        k_cache,
        k_scale_ref,
        k_scale,
        label=f"paged-k d={head_dim} dv={head_dim_v}",
    )
    check_fp8_accuracy(
        v_cache_ref,
        v_cache,
        v_scale_ref,
        v_scale,
        label=f"paged-v d={head_dim} dv={head_dim_v}",
    )


@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3", "e5m2"]
)
def test_reshape_cache_pth_paged(head_dim, dtype, fp8_dtype):
    """Paged layout with various head_dims and dtypes."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_paged(
        num_tokens=16,
        num_kv_heads=4,
        head_dim=head_dim,
        head_dim_v=head_dim,
        page_size=8,
        max_num_pages=16,
        dtype=dtype,
        fp8_dtype=fp8_dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_paged_gqa(dtype):
    """Paged layout with GQA."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_paged(
        num_tokens=8,
        num_kv_heads=2,
        head_dim=128,
        head_dim_v=128,
        page_size=8,
        max_num_pages=16,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_paged_scattered(dtype):
    """Paged layout with scattered slot positions."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_paged(
        num_tokens=8,
        num_kv_heads=4,
        head_dim=128,
        head_dim_v=128,
        page_size=8,
        max_num_pages=16,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
        slot_offsets=[10, 30, 5, 45, 20, 55, 15, 50],
    )


# ============================================================
# Shared storage (non-contiguous stride) tests
# ============================================================


def run_reshape_cache_pth_strided_ragged(
    num_tokens,
    num_kv_heads,
    head_dim,
    total_seq_len,
    dtype,
    fp8_dtype,
):
    """Test ragged layout with non-contiguous strides (as_strided cache)."""
    device = "cuda"
    slot_mapping = torch.arange(num_tokens, dtype=torch.int32, device=device)

    key = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    value = 0.3 * torch.randn(
        num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    # Build non-contiguous cache: head_dim stride is larger than head_dim
    stride = head_dim + 16
    k_fp8_buf = torch.zeros(
        total_seq_len, num_kv_heads, stride, dtype=torch.uint8, device=device
    )
    v_fp8_buf = torch.zeros(
        total_seq_len, num_kv_heads, stride, dtype=torch.uint8, device=device
    )
    k_scale_cache = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )
    v_scale_cache = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )

    k_cache = torch.as_strided(
        k_fp8_buf.view(fp8_dtype),
        (total_seq_len, num_kv_heads, head_dim),
        (num_kv_heads * stride, stride, 1),
        storage_offset=0,
    )
    v_cache = torch.as_strided(
        v_fp8_buf.view(fp8_dtype),
        (total_seq_len, num_kv_heads, head_dim),
        (num_kv_heads * stride, stride, 1),
        storage_offset=0,
    )

    # Reference on contiguous copies
    k_contig = torch.zeros(
        total_seq_len, num_kv_heads, head_dim, dtype=fp8_dtype, device=device
    )
    v_contig = torch.zeros(
        total_seq_len, num_kv_heads, head_dim, dtype=fp8_dtype, device=device
    )
    ks_contig = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )
    vs_contig = torch.zeros(
        total_seq_len, num_kv_heads, dtype=torch.float32, device=device
    )
    _ref_reshape_and_cache_flash_pth(
        key,
        value,
        k_contig,
        v_contig,
        ks_contig,
        vs_contig,
        slot_mapping,
    )

    flashinfer.reshape_and_cache_flash_per_token_head(
        key,
        value,
        k_cache,
        v_cache,
        k_scale_cache,
        v_scale_cache,
        slot_mapping,
    )

    check_fp8_accuracy(k_contig, k_cache, ks_contig, k_scale_cache, label="strided-k")
    check_fp8_accuracy(v_contig, v_cache, vs_contig, v_scale_cache, label="strided-v")


@pytest.mark.parametrize("head_dim", [64, 128], ids=["hd64", "hd128"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_reshape_cache_pth_strided_ragged(head_dim, dtype):
    """Ragged layout with non-contiguous as_strided cache."""
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    run_reshape_cache_pth_strided_ragged(
        num_tokens=16,
        num_kv_heads=4,
        head_dim=head_dim,
        total_seq_len=64,
        dtype=dtype,
        fp8_dtype=torch.float8_e4m3fn,
    )


# ============================================================
# Edge case tests
# ============================================================


def test_reshape_cache_pth_single_token():
    """Test with single token."""
    _skip_if_sm_below_75()
    run_reshape_cache_pth_ragged(
        num_tokens=1,
        num_kv_heads=4,
        head_dim=128,
        head_dim_v=128,
        total_seq_len=16,
        dtype=torch.bfloat16,
        fp8_dtype=torch.float8_e4m3fn,
        slot_offsets=[5],
    )


# ============================================================
# Smoke tests
# ============================================================


if __name__ == "__main__":
    dtypes = [torch.float16]
    if _cc()[0] > 7:
        dtypes.append(torch.bfloat16)

    for dtype in dtypes:
        fp8_dtype = torch.float8_e4m3fn

        # Ragged basic
        test_reshape_cache_pth_ragged(head_dim=128, dtype=dtype, fp8_dtype=fp8_dtype)
        print(f"ragged {dtype}/{fp8_dtype} smoke passed")

        # Ragged GQA
        test_reshape_cache_pth_ragged_gqa(dtype=dtype)
        print(f"ragged GQA {dtype} smoke passed")

        # Ragged scattered slots
        test_reshape_cache_pth_ragged_scattered(dtype=dtype)
        print(f"ragged scattered {dtype} smoke passed")

        # Ragged head_dim_v != head_dim
        test_reshape_cache_pth_ragged_head_dim_v_diff(dtype=dtype)
        print(f"ragged head_dim_v_diff {dtype} smoke passed")

        # Paged basic
        test_reshape_cache_pth_paged(head_dim=128, dtype=dtype, fp8_dtype=fp8_dtype)
        print(f"paged {dtype}/{fp8_dtype} smoke passed")

        # Paged GQA
        test_reshape_cache_pth_paged_gqa(dtype=dtype)
        print(f"paged GQA {dtype} smoke passed")

        # Paged scattered slots
        test_reshape_cache_pth_paged_scattered(dtype=dtype)
        print(f"paged scattered {dtype} smoke passed")

        # Strided ragged
        test_reshape_cache_pth_strided_ragged(head_dim=128, dtype=dtype)
        print(f"strided ragged {dtype} smoke passed")

    # Edge cases
    test_reshape_cache_pth_single_token()
    print("single token smoke passed")

    print("\nAll reshape_and_cache_flash_per_token_head smoke tests passed")
