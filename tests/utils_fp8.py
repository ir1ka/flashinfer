import torch


def get_cos_sim_threshold(dtype: torch.dtype, mode: str = "prefill") -> float:
    """Return cosine similarity threshold for FP8 per-token-head accuracy check."""
    return 0.999


def to_float8(
    x: torch.Tensor, dtype=torch.float8_e4m3fn
) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = (
        torch.maximum(min_val.abs(), max_val.abs()).to(torch.float32).clamp(min=1e-12)
    )
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


def to_float8_per_token_head(x: torch.Tensor, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    amax = x.abs().amax(dim=-1, keepdim=True).to(torch.float32).clamp(min=1e-12)
    scales = finfo.max / amax
    x_scl_sat = (x * scales).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scales.squeeze(-1).reciprocal()
