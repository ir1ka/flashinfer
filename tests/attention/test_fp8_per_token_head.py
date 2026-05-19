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
"""

import pytest
import torch

import flashinfer


def quantize_fp8_per_token_head(
    data: torch.Tensor, dtype=torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize data to FP8 with per-token-head scales.

    data: [*, num_heads, head_dim] -> (fp8_data, scales)
    scales shape: [*, num_heads], dtype: float32
    Dequant verification: (fp8_data * scales[..., None]).close(data)
    """
    finfo = torch.finfo(dtype)
    absmax = data.abs().amax(dim=-1, keepdim=True)  # [*, num_heads, 1]
    scales = (absmax / finfo.max).clamp(min=1e-12).squeeze(-1)  # [*, num_heads]
    scales = scales.to(torch.float32)  # scales must be float32 for kernel
    fp8_data = (
        (data / scales.unsqueeze(-1)).clamp(finfo.min, finfo.max).to(dtype)
    )  # [*, num_heads, head_dim]
    return fp8_data, scales


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("qo_len", [7, 19])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_fp8_per_token_head_k_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch prefill with FP8 per-token-head K scale only."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    # Build FP16 paged KV
    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV (both K and V quantized)
    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    # Scale tensors: [batch_size, kv_len, num_kv_heads] (3D for stride_batch support)
    k_scales_3d = k_scales  # already [batch_size, kv_len, num_kv_heads]
    v_scales_3d = v_scales  # already [batch_size, kv_len, num_kv_heads]

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    # Reference: FP16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
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
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined, qo_indptr)

    # Test: per-token-head K and V scales (both K and V are FP8 quantized)
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
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        qo_indptr,
        k_scale_per_token_head=k_scales_3d,
        v_scale_per_token_head=v_scales_3d,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("qo_len", [7, 19])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_fp8_per_token_head_v_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch prefill with FP8 per-token-head V scale only.

    K uses per-tensor scale folded into sm_scale (standard FP8 approach).
    V uses per-token-head scale.
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    # Build FP16 paged KV
    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV (both K and V quantized)
    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    # Per-tensor K scale (folded into sm_scale)
    k_scale_per_tensor = (
        k_fp16_raw.abs().amax().item() / torch.finfo(torch.float8_e4m3fn).max
    )

    v_scales_3d = v_scales  # already [batch_size, kv_len, num_kv_heads]

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    # Reference: FP16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
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
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined, qo_indptr)

    # Test: per-tensor K scale + per-token-head V scale
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
        kv_data_type=torch.float8_e4m3fn,
        use_v_scale_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        qo_indptr,
        k_scale=k_scale_per_tensor,
        v_scale_per_token_head=v_scales_3d,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("qo_len", [7, 19])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_fp8_per_token_head_kv_scale(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch prefill with FP8 per-token-head K+V scales."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
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
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined, qo_indptr)

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
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    k_scales_3d = k_scales  # already [batch_size, kv_len, num_kv_heads]
    v_scales_3d = v_scales  # already [batch_size, kv_len, num_kv_heads]
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        qo_indptr,
        k_scale_per_token_head=k_scales_3d,
        v_scale_per_token_head=v_scales_3d,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_fp8_per_token_head_k_scale(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch decode with FP8 per-token-head K scale only."""
    if num_qo_heads > num_kv_heads:
        pytest.skip("FP8 decode GQA not supported on SM75")
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV (both K and V quantized)
    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_fp16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined)

    # Both K and V are FP8, so both need per-token-head scales for correct dequantization
    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp8.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    # Scale tensor shape: [batch_size, kv_len, num_kv_heads]
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        k_scale_per_token_head=k_scales,
        v_scale_per_token_head=v_scales,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_fp8_per_token_head_v_scale(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch decode with FP8 per-token-head V scale only."""
    if num_qo_heads > num_kv_heads:
        pytest.skip("FP8 decode GQA not supported on SM75")
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV (both K and V quantized)
    k_fp8_reshaped, _k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    # Per-tensor K scale (folded into sm_scale)
    k_scale_per_tensor = (
        k_fp16_raw.abs().amax().item() / torch.finfo(torch.float8_e4m3fn).max
    )

    indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_fp16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined)

    # Per-tensor K scale + per-token-head V scale
    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp8.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_v_scale_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(
        q, kv_fp8_combined, k_scale=k_scale_per_tensor, v_scale_per_token_head=v_scales
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 8])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_decode_fp8_per_token_head_kv_scale(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch decode with FP8 per-token-head K+V scales."""
    if num_qo_heads > num_kv_heads:
        pytest.skip("FP8 decode GQA not supported on SM75")
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_padded = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_padded = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    else:
        k_fp8_padded, v_fp8_padded = k_fp8_reshaped, v_fp8_reshaped
    k_fp8_paged = (
        k_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_padded.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper_fp16 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined)

    wrapper_fp8 = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp8.plan(
        indptr,
        indices,
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        k_scale_per_token_head=k_scales,
        v_scale_per_token_head=v_scales,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


def test_backend_guard_prefill():
    """Test that non-fa2 backends raise when per-token-head scales are used."""
    try:
        wrapper_cls = flashinfer.BatchPrefillWithPagedKVCacheWrapper
        workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device=torch.device("cuda")
        )
        wrapper_cls(workspace_buffer, "NHD", backend="cudnn")
    except Exception:
        pytest.skip("cudnn backend not available")

    torch.manual_seed(42)
    device = torch.device("cuda")
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    scales = torch.randn(32, 4, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="fp8_per_token_head"):
        wrapper.run(
            torch.randn(4, 4, 128, dtype=torch.float16, device=device),
            (
                torch.randn(32, 4, 128, device=device).to(torch.float8_e4m3fn),
                torch.randn(32, 4, 128, device=device).to(torch.float8_e4m3fn),
            ),
            torch.tensor([0, 4], dtype=torch.int32, device=device),
            k_scale_per_token_head=scales,
        )


def test_backend_guard_decode():
    """Test that non-fa2 backends raise when per-token-head scales are used."""
    try:
        wrapper_cls = flashinfer.BatchDecodeWithPagedKVCacheWrapper
        workspace_buffer = torch.empty(
            32 * 1024 * 1024, dtype=torch.int8, device=torch.device("cuda")
        )
        wrapper_cls(workspace_buffer, "NHD", backend="cudnn")
    except Exception:
        pytest.skip("cudnn backend not available")

    torch.manual_seed(42)
    device = torch.device("cuda")
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="cudnn"
    )
    scales = torch.randn(32, 4, dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="fp8_per_token_head"):
        wrapper.run(
            torch.randn(4, 4, 128, dtype=torch.float16, device=device),
            (
                torch.randn(32, 4, 128, device=device).to(torch.float8_e4m3fn),
                torch.randn(32, 4, 128, device=device).to(torch.float8_e4m3fn),
            ),
            k_scale_per_token_head=scales,
        )


@pytest.mark.parametrize("qo_len", [16, 32])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
def test_single_prefill_fp8_per_token_head(
    qo_len,
    kv_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test single prefill with FP8 per-token-head K and V scales."""
    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda"))[0] < 8:
        pytest.skip("single_prefill requires SM80+")

    torch.manual_seed(42)
    device = torch.device("cuda")

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_fp16 = 0.05 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16 = 0.05 * torch.randn(
        kv_len, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Reference: FP16
    o_fp16 = flashinfer.single_prefill_with_kv_cache(q, k_fp16, v_fp16, causal=True)

    # FP8 with per-token-head scales
    k_fp8, k_scales = quantize_fp8_per_token_head(k_fp16)
    v_fp8, v_scales = quantize_fp8_per_token_head(v_fp16)

    o_fp8 = flashinfer.single_prefill_with_kv_cache(
        q,
        k_fp8,
        v_fp8,
        causal=True,
        k_scale_per_token_head=k_scales,
        v_scale_per_token_head=v_scales,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3, 5])
@pytest.mark.parametrize("qo_len", [7, 19])
@pytest.mark.parametrize("kv_len", [32, 64])
@pytest.mark.parametrize("page_size", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_fp8_per_token_head_multi_request(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test batch prefill with FP8 per-token-head scales across multiple requests.

    This specifically validates that request_idx is correctly used for per-request
    scale tensor indexing (persistent.cuh request_idx=work_idx fix).
    """
    torch.manual_seed(42)
    device = torch.device("cuda")
    kv_layout = "NHD"

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    pad_size = num_pages_per_seq * page_size - kv_len

    # Build FP16 paged KV
    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    if pad_size > 0:
        k_reshaped = torch.cat(
            [
                k_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_reshaped = torch.cat(
            [
                v_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float16,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV
    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    if pad_size > 0:
        k_fp8_reshaped = torch.cat(
            [
                k_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
        v_fp8_reshaped = torch.cat(
            [
                v_fp8_reshaped,
                torch.zeros(
                    batch_size,
                    pad_size,
                    num_kv_heads,
                    head_dim,
                    dtype=torch.float8_e4m3fn,
                    device=device,
                ),
            ],
            dim=1,
        )
    k_fp8_paged = (
        k_fp8_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    k_scales_3d = k_scales
    v_scales_3d = v_scales

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=device)

    # Reference: FP16
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa2"
    )
    wrapper_fp16.plan(
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
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined, qo_indptr)

    # Test: FP8 per-token-head
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
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    o_fp8 = wrapper_fp8.run(
        q,
        kv_fp8_combined,
        qo_indptr,
        k_scale_per_token_head=k_scales_3d,
        v_scale_per_token_head=v_scales_3d,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)


@pytest.mark.parametrize("batch_size", [3])
@pytest.mark.parametrize("kv_len", [32])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_attention_fp8_per_token_head(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Test BatchAttention with FP8 per-token-head K and V scales.

    SM75 has pre-existing compilation issues with BatchAttention, so this test
    skips on SM75.
    """
    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 8:
        pytest.skip("BatchAttention has pre-existing SM75 compilation issues")

    torch.manual_seed(42)
    device = torch.device("cuda")

    qo_len = kv_len
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size

    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_fp16_raw = 0.05 * torch.randn(
        kv_len * batch_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )

    # Build FP16 paged KV
    k_reshaped = k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    v_reshaped = v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    k_paged_f16 = (
        k_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_paged_f16 = (
        v_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp16_combined = torch.stack([k_paged_f16, v_paged_f16], dim=1)

    # Build FP8 paged KV
    k_fp8_reshaped, k_scales = quantize_fp8_per_token_head(
        k_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    v_fp8_reshaped, v_scales = quantize_fp8_per_token_head(
        v_fp16_raw.view(batch_size, kv_len, num_kv_heads, head_dim)
    )
    k_fp8_paged = (
        k_fp8_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    v_fp8_paged = (
        v_fp8_reshaped.view(
            batch_size, num_pages_per_seq, page_size, num_kv_heads, head_dim
        )
        .transpose(0, 1)
        .reshape(total_num_pages, page_size, num_kv_heads, head_dim)
    )
    kv_fp8_combined = torch.stack([k_fp8_paged, v_fp8_paged], dim=1)

    k_scales_3d = k_scales
    v_scales_3d = v_scales

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, dtype=torch.int32, device=device)
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    # Reference: FP16 via BatchPrefillWithPagedKVCacheWrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper_fp16 = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device
    )
    wrapper_fp16.plan(
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
        causal=True,
    )
    o_fp16 = wrapper_fp16.run(q, kv_fp16_combined, qo_indptr)

    # Test: BatchAttention FP8 per-token-head
    batch_attention = flashinfer.BatchAttention(kv_layout="NHD")
    batch_attention.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float8_e4m3fn,
        use_k_scale_per_token_head=True,
        use_v_scale_per_token_head=True,
    )
    o_fp8, _ = batch_attention.run(
        q,
        kv_fp8_combined,
        k_scale_per_token_head=k_scales_3d,
        v_scale_per_token_head=v_scales_3d,
    )

    torch.testing.assert_close(o_fp16, o_fp8, atol=2e-2, rtol=5e-2)
