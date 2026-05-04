"""
Stable Python wiring for the TurboQuant MLA encode kernel — the encode-side
mirror of `_mla_decode_helpers.py`. Holds:

  - sys.path setup for the turboquant venv (scipy for compute_uniform_params).
  - Pool layout constants matching `tq_sm100_fmha_mla_tq4bit.hpp`.
  - 4-bit uniform quant params (precomputed from `compute_uniform_params`).
  - Inline-build flags + include paths for `load_inline()`.
  - `make_pool_layout(...)` allocates the empty pool/page-table tensors.
  - `encode_mla_paged_kwargs(...)` packs the long pybind11 keyword-arg list
    so ModelNew.forward stays a one-liner.
  - `encode_kv_inline(enc, kv, sl)` end-to-end helper that runs `enc` (a
    pybind module exposing `encode_mla_paged`) and returns the populated pool.
  - `dequant_pool_to_cache(pool)` reverses the 4-bit quant so KernelBench can
    compare bf16 caches with `atol=1e-2` instead of bit-exact uint8 nibbles.

Imported by `52_mla_encode_b200_template.py`, `52_mla_encode_b200_ref.py`,
and `_gen_encode_ref_outputs.py`.

NOTE: this module deliberately does NOT import `_mla_decode_helpers` and does
NOT load the prebuilt encoder `.so`. The encode .cu has a
`TORCH_LIBRARY_FRAGMENT(turbo_quant, ...)` block that registers
`turbo_quant::encode_mla_paged` to torch.ops globally; loading both the
prebuilt and inline builds in one process double-registers and crashes.
The candidate eval process loads ONLY the inline build via load_inline().
"""
from __future__ import annotations

import math
import os as _os
import pathlib
import sys
import tempfile

import torch

sys.path.insert(0, "/workspace/turboquant/.venv/lib/python3.12/site-packages")
sys.path.insert(0, "/workspace/turboquant")
from turbo_quant.quantize import compute_uniform_params as _compute_uniform_params

# --- pool layout (must match tq_sm100_fmha_mla_tq4bit.hpp) ----------------
D_CKV          = 512
D_KPE          = 64
D_TOTAL        = D_CKV + D_KPE                       # 576
CKV_DATA_OFF   = 0
KPE_DATA_OFF   = CKV_DATA_OFF + D_CKV // 2           # 256
CKV_NORM_OFF   = KPE_DATA_OFF + D_KPE // 2           # 288
KPE_NORM_OFF   = CKV_NORM_OFF + 2                    # 290
BYTES_PER_POS  = ((KPE_NORM_OFF + 2) + 15) & ~15     # 304
PAGE_SIZE      = 128

# --- 4-bit uniform quant params ------------------------------------------
CKV_STEP, CKV_BIAS, _ckv_thresh = _compute_uniform_params(head_dim=D_CKV, num_bits=4)
KPE_STEP, KPE_BIAS, _kpe_thresh = _compute_uniform_params(head_dim=D_KPE, num_bits=4)
CKV_THRESH = torch.tensor(_ckv_thresh, dtype=torch.float32, device="cuda")
KPE_THRESH = torch.tensor(_kpe_thresh, dtype=torch.float32, device="cuda")

# --- inline-build flags ---------------------------------------------------
_os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_TQ_CSRC       = pathlib.Path("/workspace/turboquant/csrc")
_TQ_ENCODE_DIR = _TQ_CSRC / "encode"
_TQ_CUTLASS    = pathlib.Path("/workspace/turboquant/third_party/cutlass")
_TQ_FLASHINFER = pathlib.Path("/workspace/turboquant/third_party/flashinfer/include")

INCLUDE_PATHS = [
    str(_TQ_ENCODE_DIR),                      # quant_utils.cuh + neighbors
    str(_TQ_CUTLASS / "include"),
    str(_TQ_CUTLASS / "tools/util/include"),
    str(_TQ_CSRC / "shared"),
    str(_TQ_CSRC),
    str(_TQ_FLASHINFER),
]

# Encode .cu uses cute SM100 UMMA intrinsics, so target sm_100a explicitly.
# No `-DTQ_NO_TORCH_LIBRARY` here: the candidate process never loads the
# prebuilt encoder, so the inline build's TORCH_LIBRARY_FRAGMENT registration
# of `turbo_quant::encode_mla_paged` is the only one and is safe.
CUDA_FLAGS = [
    "-O3", "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-lineinfo",
]
CXX_FLAGS = ["-std=c++17"]


def materialize_local_headers(headers: dict[str, str]) -> str:
    """Write each (header_name, body) pair to a fresh tempdir and return
    its path. Prepend the result to `extra_include_paths` so the inline
    build's `#include "..."` directives resolve to the agent's edits
    rather than the original turboquant source on disk."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="autocomp_local_hdrs_"))
    for name, content in headers.items():
        f = d / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return str(d)


# --- KPE fused table (Hadamard ⊗ identity rotation; signs baked in) -----
def _identity_kpe_table(K: int, device) -> torch.Tensor:
    """KPE 'codebook' that the encoder treats as identity (CKV path only).

    Same shape/contents the decode-side `_mla_decode_helpers._identity_kpe_table`
    builds — kept here so this module is self-contained.
    """
    P_HALF = D_KPE // 2
    t = torch.zeros(K, P_HALF, 4, dtype=torch.float32, device=device)
    t[..., 0] = 1.0; t[..., 1] = 1.0; t[..., 2] = 1.0; t[..., 3] = -1.0
    return t.reshape(K, D_KPE * 2).contiguous()


# --- pool allocation -----------------------------------------------------
def make_pool_layout(B: int, K: int, device, page_size: int = PAGE_SIZE):
    """Allocate the (empty) pool + page table the encoder will fill in.

    Returns:
        pool        : [num_pages, page_size, BYTES_PER_POS] uint8 (zeroed)
        page_table  : [B, pages_per_seq] int32
        page_size   : may shrink (gcd with K) if K isn't divisible by PAGE_SIZE
        page_stride : page_size * BYTES_PER_POS
        page_indices: [B*K] int32 — per-token page id
        pos_in_page : [B*K] int32 — per-token offset within its page
        kpe_fused_table : [K, D_KPE*2] float32 — identity rotation
    """
    page_size_eff = math.gcd(K, page_size)
    pages_per_seq = K // page_size_eff
    num_pages = B * pages_per_seq
    page_stride = page_size_eff * BYTES_PER_POS

    page_table = torch.arange(num_pages, dtype=torch.int32,
                              device=device).view(B, pages_per_seq)
    pool = torch.zeros(num_pages, page_size_eff, BYTES_PER_POS,
                       dtype=torch.uint8, device=device)

    tok_idx = torch.arange(B * K, device=device)
    bi = tok_idx // K
    ki = tok_idx % K
    page_indices = page_table[bi, ki // page_size_eff].contiguous()
    pos_in_page  = (ki % page_size_eff).to(torch.int32).contiguous()
    fused = _identity_kpe_table(K, device)
    return pool, page_table, page_size_eff, page_stride, page_indices, pos_in_page, fused


# --- pybind11 keyword-arg pack for `encode_mla_paged` --------------------
def encode_mla_paged_kwargs(
    *,
    latent: torch.Tensor,        # [B*K, D_TOTAL] bf16
    pool: torch.Tensor,          # uint8 [num_pages, page_size, BYTES_PER_POS]
    page_indices: torch.Tensor,  # [B*K] int32
    pos_in_page: torch.Tensor,   # [B*K] int32
    page_size: int,
    page_stride: int,
    kpe_fused_table: torch.Tensor,  # [K, D_KPE*2] float32
    seq_lens: torch.Tensor,      # [B] int32
    seq_len: int,                # K (max seq len)
    batch_size: int,             # B
) -> dict:
    """Builds the kwargs dict for `_enc.encode_mla_paged(...)`.

    Q-side RoPE / cu_seqlens are unused for the encode-only autocomp problem
    (we benchmark just the kv encode path), so all those optional tensors are
    None and the related ints are 0.
    """
    return dict(
        latent_cache=latent,
        ckv_thresholds=CKV_THRESH, kpe_thresholds=KPE_THRESH,
        pool=pool, page_indices=page_indices, pos_in_page=pos_in_page,
        tokens_per_page=page_size,
        ckv_data_offset=CKV_DATA_OFF, kpe_data_offset=KPE_DATA_OFF,
        ckv_norm_offset=CKV_NORM_OFF, kpe_norm_offset=KPE_NORM_OFF,
        bytes_per_pos=BYTES_PER_POS, page_stride=page_stride,
        kv_lora_rank=D_CKV, qk_rope_head_dim=D_KPE,
        kpe_fused_table=kpe_fused_table,
        ckv_signs=None,
        seq_lens=seq_lens, seq_len=seq_len, batch_size=batch_size,
        fused_q=None, q_pe=None, num_heads=0, q_rope_table=None,
        cu_q_seqlens=None, cu_kv_seqlens=None,
        paged_kv_indptr=None, paged_kv_last_page_len=None,
        tokens_per_page_for_cuseqlens=0, ng=0, nc=0,
        use_codebook=False,
    )


# --- end-to-end encode helper -------------------------------------------
def encode_kv_inline(enc_module, kv_cache: torch.Tensor,
                     seq_lens: torch.Tensor, page_size: int = PAGE_SIZE):
    """Run `enc_module.encode_mla_paged(...)` on `kv_cache`, return the pool.

    Works for either the prebuilt encoder .so or the agent's inline build —
    both expose the same `encode_mla_paged(...)` symbol with identical
    pybind keyword args.
    """
    B, K, _ = kv_cache.shape
    dev = kv_cache.device
    pool, pt, ps, pstride, pi, pip, fused = make_pool_layout(B, K, dev, page_size)
    latent = kv_cache.reshape(B * K, kv_cache.shape[-1]).contiguous()
    enc_module.encode_mla_paged(**encode_mla_paged_kwargs(
        latent=latent, pool=pool, page_indices=pi, pos_in_page=pip,
        page_size=ps, page_stride=pstride, kpe_fused_table=fused,
        seq_lens=seq_lens.int(), seq_len=K, batch_size=B,
    ))
    torch.cuda.synchronize()
    return pool, pt, ps, pstride


# --- dequant for ref comparison -----------------------------------------
# We compare encoder outputs in DEQUANTIZED bf16 form (not raw uint8 nibbles)
# so KernelBench's `atol=1e-2` absorbs harmless rounding drift in the
# encoder's intermediate fp32 math (e.g. fmaf vs separate mul/add) while
# still flagging real bugs — an off-by-one nibble shifts dequant by
# `step * norm` ~ 0.1, well above atol.
def _dequant_pool_to_cache_eager(pool: torch.Tensor) -> torch.Tensor:
    """Eager-mode dequant. Used as the source for `torch.compile` below
    and as a fallback if compilation fails. Matches the kernel's dequant
    formula exactly:
      dq[i] = norm * (step * nibble_i + bias)
    with `nibble_i = (byte[i//2] >> ((i&1)*4)) & 0xF`.
    """
    NP, PS, _ = pool.shape
    dev = pool.device

    ckv_bytes = pool[..., CKV_DATA_OFF:CKV_DATA_OFF + D_CKV // 2].to(torch.int32)
    kpe_bytes = pool[..., KPE_DATA_OFF:KPE_DATA_OFF + D_KPE // 2].to(torch.int32)

    ckv_lo = (ckv_bytes & 0xF).to(torch.float32)
    ckv_hi = ((ckv_bytes >> 4) & 0xF).to(torch.float32)
    ckv_f  = torch.stack([ckv_lo, ckv_hi], dim=-1).flatten(-2)  # [NP, PS, D_CKV]

    kpe_lo = (kpe_bytes & 0xF).to(torch.float32)
    kpe_hi = ((kpe_bytes >> 4) & 0xF).to(torch.float32)
    kpe_f  = torch.stack([kpe_lo, kpe_hi], dim=-1).flatten(-2)  # [NP, PS, D_KPE]

    # `view(bf16)` requires a contiguous trailing axis of size 2 in bytes.
    ckv_norm = pool[..., CKV_NORM_OFF:CKV_NORM_OFF + 2].contiguous() \
        .view(torch.bfloat16).squeeze(-1).float()  # [NP, PS]
    kpe_norm = pool[..., KPE_NORM_OFF:KPE_NORM_OFF + 2].contiguous() \
        .view(torch.bfloat16).squeeze(-1).float()

    cache = torch.empty(NP, PS, D_TOTAL, dtype=torch.bfloat16, device=dev)
    cache[..., :D_CKV] = (ckv_norm.unsqueeze(-1) * (CKV_STEP * ckv_f + CKV_BIAS)).to(torch.bfloat16)
    cache[..., D_CKV:] = (kpe_norm.unsqueeze(-1) * (KPE_STEP * kpe_f + KPE_BIAS)).to(torch.bfloat16)
    return cache


# torch.compile fuses the ~10 separate kernel launches above into 1-2
# kernels, dropping the dequant cost from ~365us to ~65us on B=8/K=4096.
# That matters because dequant runs INSIDE ModelNew.forward (= the timed
# region), so a slow dequant dilutes the agent's encoder-speedup signal.
# `dynamic=True` lets a single compiled graph handle both the canonical
# (B=8, K=4096) and small-K (B=4, K=512) variants without recompilation.
dequant_pool_to_cache = torch.compile(
    _dequant_pool_to_cache_eager,
    dynamic=True, fullgraph=True,
)
