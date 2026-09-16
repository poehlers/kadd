"""Monkey-patches torchvision's ShiftedWindowAttention3d to use
F.scaled_dot_product_attention (SDPA) instead of a manual
matmul -> add-bias -> softmax -> matmul sequence.

Why: profiled the REAL model (kadd/classify/model.py's swin3d encoder,
window_size=[8,7,7]=392 tokens/window) locally with torch.profiler and
found attention ops (bmm + softmax + masked_fill) are ~42% of total
forward compute -- the single largest category, bigger than all
Linear/MLP layers combined (~17%). This contradicted the initial
assumption that Swin's windowed attention (small N per window) would make
attention a minor cost -- 392 tokens/window turns out not to be small
enough for that to hold. SDPA fuses the QK^T -> bias/mask add -> softmax
-> @V sequence into one IO-aware kernel, avoiding materializing the full
N x N score matrix in HBM -- on CUDA this dispatches to FlashAttention
(Ampere+) or PyTorch's memory-efficient attention backend (Turing, e.g.
this project's Tesla T4s -- FlashAttention's own CUDA kernels require
Ampere+, so T4 gets the xFormers-derived memory-efficient backend
instead; still a fused, IO-aware kernel in the same category of win, just
not literally "FlashAttention" by name on this specific hardware).

Mathematically EXACT, not an approximation: same scale (head_dim**-0.5,
matches SDPA's default), same additive relative-position-bias + shift-
mask (combined into SDPA's attn_mask), same softmax, same weighted sum.
Verified via a direct forward-pass equivalence test (patched vs.
unpatched output on identical weights/input) in tests/test_smoke.py, not
just assumed from the math. Doesn't touch any nn.Module structure or
parameter name (qkv/proj/relative_position_bias_table are untouched --
only the FUNCTIONAL computation path used inside forward changes), so
existing checkpoints (e.g. data/resume_checkpoint.pth) remain fully
load-compatible.

All window-partition/pad/shift/reverse-window plumbing below is an
unmodified copy of torchvision's own shifted_window_attention_3d (only
the "multi-head attention" block in the middle is replaced) -- kept
verbatim rather than refactored, so a future torchvision version bump is
easy to diff against.
"""
import torchvision.models.video.swin_transformer as _st
import torch.nn.functional as F
from torch import Tensor

_PATCHED = False


def _sdpa_shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: list,
    num_heads: int,
    shift_size: list,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias=None,
    proj_bias=None,
    training: bool = True,
) -> Tensor:
    b, t, h, w, c = input.shape
    pad_size = _st._compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    if sum(shift_size) > 0:
        x = x.roll(shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0], window_size[0],
        padded_size[1] // window_size[1], window_size[1],
        padded_size[2] // window_size[2], window_size[2],
        c,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )

    n = x.size(1)
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), n, 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    # each (B*nW, nH, N, head_dim), deliberately UNSCALED -- SDPA applies the default scale
    # (head_dim ** -0.5) internally, exactly matching the original's manual `q * (c//nH)**-0.5`.
    q, k, v = qkv[0], qkv[1], qkv[2]

    combined_bias = relative_position_bias  # (1, nH, N, N)
    if sum(shift_size) > 0:
        attn_mask = _st._compute_attention_mask_3d(
            x, padded_size, tuple(window_size), tuple(shift_size)
        )  # (nW, N, N)
        combined_bias = combined_bias + attn_mask.unsqueeze(1)  # -> (nW, nH, N, N)
        # Tile across the true batch b: matches x's batch-major/window-minor flattening order
        # exactly (b*num_windows above iterates batch-outer, window-inner), same result as the
        # original's view(B,nW,...) + mask.unsqueeze(1).unsqueeze(0) broadcast-then-reshape.
        combined_bias = combined_bias.repeat(b, 1, 1, 1)  # -> (B*nW, nH, N, N)

    attn_out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=combined_bias.to(q.dtype),
        dropout_p=attention_dropout if training else 0.0,
    )
    x = attn_out.transpose(1, 2).reshape(x.size(0), n, c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    x = x.view(
        b,
        padded_size[0] // window_size[0], padded_size[1] // window_size[1], padded_size[2] // window_size[2],
        window_size[0], window_size[1], window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    if sum(shift_size) > 0:
        x = x.roll(shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    x = x[:, :t, :h, :w, :].contiguous()
    return x


def apply_sdpa_patch():
    """Idempotent -- safe to call more than once (e.g. if multiple encoders
    get built in one process, as train_kernel.py's multi-GPU replication
    does internally)."""
    global _PATCHED
    if _PATCHED:
        return
    _st.shifted_window_attention_3d = _sdpa_shifted_window_attention_3d
    _PATCHED = True
