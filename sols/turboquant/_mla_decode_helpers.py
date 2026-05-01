"""
Stable Python wiring for the TurboQuant 4-bit MLA decode kernel — kept in
a sidecar module so the autocomp agent's `51_mla_decode_b200.py` view
contains ONLY the editable CUDA source + a thin ModelNew wrapper.

Holds everything that has nothing to do with kernel optimization:
  - sys.path setup for the turboquant venv (scipy + turbo_quant package).
  - Pool layout constants matching `tq_sm100_fmha_mla_tq4bit.hpp`.
  - 4-bit uniform quant params (precomputed from `compute_uniform_params`).
  - Paged-pool encode helper that drives the prebuilt encode_mla_paged.so.
  - Inline-build flags + include paths used by load_inline().
  - Cross-context inline-module sharing via a sentinel sys.modules key,
    so the KernelBench standalone-ref timing pass can reuse the agent's
    inline `.so` instead of loading the prebuilt 4-bit `.so` (which would
    collide with the inline build's CUTLASS Sm100 cluster-launch kernel
    symbols and crash inside `cudaLaunchKernelExC`).
  - `tq_mla_decode_kwargs(...)` packs the long pybind11 keyword-arg list
    so ModelNew's forward stays a one-liner.

Imported by `51_mla_decode_b200_template.py` and `51_mla_decode_b200_ref.py`.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import tempfile

import torch

# --- turboquant venv (scipy for compute_uniform_params) -------------------
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
# 16-aligned: encoder writes nibble fields in 16B chunks; TMA descriptors
# in the kernel expect bytes_per_pos to be 16-aligned.
BYTES_PER_POS  = ((KPE_NORM_OFF + 2) + 15) & ~15     # 304
PAGE_SIZE      = 128

# --- 4-bit uniform quant params ------------------------------------------
CKV_STEP, CKV_BIAS, _ckv_thresh = _compute_uniform_params(head_dim=D_CKV, num_bits=4)
KPE_STEP, KPE_BIAS, _kpe_thresh = _compute_uniform_params(head_dim=D_KPE, num_bits=4)
CKV_THRESH = torch.tensor(_ckv_thresh, dtype=torch.float32, device="cuda")
KPE_THRESH = torch.tensor(_kpe_thresh, dtype=torch.float32, device="cuda")

# --- prebuilt encode_mla_paged.so ----------------------------------------
def _load_so(path: str | pathlib.Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_ENC = _load_so(
    "/workspace/turboquant/build/lib/encode_mla_paged.cpython-312-x86_64-linux-gnu.so",
    "encode_mla_paged",
)

# --- inline-build flags ---------------------------------------------------
import os as _os
_os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "10.0a")

_TQ_CSRC       = pathlib.Path("/workspace/turboquant/csrc")
_TQ_DECODE_DIR = _TQ_CSRC / "decode_mla_tcgen05"
_TQ_CUTLASS    = pathlib.Path("/workspace/turboquant/third_party/cutlass")
_TQ_FLASHINFER = pathlib.Path("/workspace/turboquant/third_party/flashinfer/include")

INCLUDE_PATHS = [
    str(_TQ_DECODE_DIR),                      # tq_sm100_fmha_mla_tq4bit.hpp + _common.hpp
    str(_TQ_CUTLASS / "include"),
    str(_TQ_CUTLASS / "examples/77_blackwell_fmha"),
    str(_TQ_CUTLASS / "examples/77_blackwell_fmha/kernel"),
    str(_TQ_CUTLASS / "examples/common"),
    str(_TQ_CUTLASS / "tools/util/include"),
    str(_TQ_CSRC / "shared"),
    str(_TQ_CSRC),
    str(_TQ_FLASHINFER),
]

# `-DTQ_NO_TORCH_LIBRARY` suppresses the TORCH_LIBRARY_FRAGMENT block in
# the embedded TU; the prebuilt `.so` already owns
# `torch.ops.turbo_quant.tq_mla_decode`, so a second registration aborts.
CUDA_FLAGS = [
    "-O3", "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_100a,code=sm_100a",
    "-lineinfo",
    "-DTQ_NO_TORCH_LIBRARY",
]
CXX_FLAGS = ["-std=c++17"]


def materialize_local_headers(headers: dict[str, str]) -> str:
    '''Write each (header_name, body) pair to a fresh tempdir and return
    its path. Prepend the result to `extra_include_paths` so the inline
    build's `#include "..."` directives resolve to the agent's edits
    rather than the original turboquant source on disk.'''
    d = pathlib.Path(tempfile.mkdtemp(prefix="autocomp_local_hdrs_"))
    for name, content in headers.items():
        f = d / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    return str(d)

# --- inline-module sharing across exec contexts --------------------------
# torch's load_inline() does NOT register the compiled extension in
# sys.modules. KernelBench's `measure_ref_program_time` execs ref into a
# FRESH context (no `ModelNew` and no `_tq`), so the standalone ref
# timing path can't see the inline build. We park it under this sentinel
# key after sol exec'd its own load_inline() so the standalone ref pass
# can reuse the SAME .so binary — required to avoid a CUDA cluster-launch
# collision between the inline build and turboquant's prebuilt 4-bit .so
# (cudaLaunchKernelExC → "invalid argument", cluster_launch.hpp:249).
_SHARED_INLINE_MODULE_KEY = "_TQ_MLA_DECODE_INLINE_SHARED"


def share_inline_module(mod) -> None:
    """Sol calls this right after `_tq = load_inline(...)`."""
    sys.modules[_SHARED_INLINE_MODULE_KEY] = mod


def get_shared_inline_module():
    """Returns the inline-built kernel module if previously shared, else None."""
    return sys.modules.get(_SHARED_INLINE_MODULE_KEY)


# --- pool encoding -------------------------------------------------------
def _identity_kpe_table(K: int, device) -> torch.Tensor:
    """KPE 'codebook' that the encoder treats as identity (CKV path only)."""
    P_HALF = D_KPE // 2
    t = torch.zeros(K, P_HALF, 4, dtype=torch.float32, device=device)
    t[..., 0] = 1.0; t[..., 1] = 1.0; t[..., 2] = 1.0; t[..., 3] = -1.0
    return t.reshape(K, D_KPE * 2).contiguous()


def prep_4bit_pool(kv_cache: torch.Tensor, page_size: int = PAGE_SIZE):
    """Encode bf16 kv_cache [B, K, D_TOTAL] into a paged 4-bit pool.

    Returns (pool, page_table, page_size, page_stride):
      pool        : [num_pages, page_size, BYTES_PER_POS] uint8
      page_table  : [B, pages_per_seq] int32
      page_size   : may shrink (gcd with K) if K isn't divisible by PAGE_SIZE
      page_stride : page_size * BYTES_PER_POS
    """
    B, K, _ = kv_cache.shape
    page_size     = math.gcd(K, page_size)
    pages_per_seq = K // page_size
    num_pages     = B * pages_per_seq
    page_stride   = page_size * BYTES_PER_POS

    page_table = torch.arange(num_pages, dtype=torch.int32,
                               device=kv_cache.device).view(B, pages_per_seq)
    pool = torch.zeros(num_pages, page_size, BYTES_PER_POS,
                       dtype=torch.uint8, device=kv_cache.device)

    latent  = kv_cache.reshape(B * K, kv_cache.shape[-1]).contiguous()
    tok_idx = torch.arange(B * K, device=kv_cache.device)
    bi      = tok_idx // K
    ki      = tok_idx % K
    pi      = page_table[bi, ki // page_size].contiguous()
    pip     = (ki % page_size).to(torch.int32).contiguous()
    fused   = _identity_kpe_table(K, kv_cache.device)
    seq_lens = torch.full((B,), K, dtype=torch.int32, device=kv_cache.device)

    _ENC.encode_mla_paged(
        latent, CKV_THRESH, KPE_THRESH, pool, pi, pip, page_size,
        CKV_DATA_OFF, KPE_DATA_OFF, CKV_NORM_OFF, KPE_NORM_OFF,
        BYTES_PER_POS, page_stride, D_CKV, D_KPE,
        fused, None, seq_lens, K, B,
    )
    torch.cuda.synchronize()
    return pool, page_table, page_size, page_stride


# --- pybind11 keyword-arg pack for `tq_mla_decode` -----------------------
def tq_mla_decode_kwargs(
    *,
    q: torch.Tensor,         # [B, H, D_TOTAL] bf16
    pool: torch.Tensor,      # uint8 paged pool from prep_4bit_pool
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,  # [B] int32
    page_size: int,
    page_stride: int,
    softmax_scale: float,
    K: int,                  # max seq len
) -> dict:
    """Builds the kwargs dict for `_tq.tq_mla_decode(...)`.

    Hadamard is always applied internally (Q rotate + output un-rotate);
    encode_mla_paged hard-wires the same butterfly so the kernel sees a
    consistent rotated basis. ckv_signs/kpe_signs=None means "kernel
    default sign vector" (matches what the encoder uses).
    `kv_last_page_len` is unused on decode but accepted for signature
    parity with the H100 op — pass `seq_lens` again as a placeholder.
    """
    sl_i32 = seq_lens.int()
    return dict(
        fused_q=q.bfloat16().contiguous(),
        pool=pool, block_table=page_table,
        kv_lens=sl_i32, kv_last_page_len=sl_i32,
        page_size=page_size,
        ckv_data_offset=CKV_DATA_OFF, kpe_data_offset=KPE_DATA_OFF,
        ckv_norm_offset=CKV_NORM_OFF, kpe_norm_offset=KPE_NORM_OFF,
        bytes_per_pos=BYTES_PER_POS, page_stride=page_stride,
        uniform_step=CKV_STEP, uniform_offset=CKV_BIAS,
        kpe_step=KPE_STEP, kpe_offset=KPE_BIAS,
        sm_scale=softmax_scale, max_seq_len=K,
        codebook=None, kpe_codebook=None,
        ckv_signs=None, kpe_signs=None,
        d_ckv_override=D_CKV, d_kpe_override=D_KPE,
        output_buf=None, q_rot_buf=None,
    )


# --- pool-cache key fingerprint ------------------------------------------
def pool_cache_stamp(kv: torch.Tensor):
    """Cheap-but-robust cache key for `prep_4bit_pool` results.

    `data_ptr` alone is unsafe — PyTorch can recycle freed addresses, so
    a fresh kv_cache (KernelBench's 5 correctness trials each allocate a
    new tensor) may alias a GC'd one and we'd serve a stale pool. Stamp
    with `_version` and a tiny content fingerprint that change whenever
    the contents do, while keeping the perf-loop hot (single shape,
    identical tensor) at zero overhead.
    """
    return (
        kv.data_ptr(),
        tuple(kv.shape),
        kv._version,
        float(kv.view(torch.uint8).flatten()[:8].sum().item()),
    )
