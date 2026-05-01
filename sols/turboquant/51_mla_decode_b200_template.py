"""
TurboQuant 4-bit MLA decode (Blackwell sm100) — autocomp solution shell.

`run_search.py` populates the embed block below before each run with
two values the agent is free to edit:

  * `CUDA_SOURCE`  — the main `.cu` translation unit (with its
                     `#include "..."` directives left intact).
  * `LOCAL_HEADERS` — `{header_name: source}` for every local turboquant
                     header transitively included from CUDA_SOURCE.
                     `materialize_local_headers` writes them to a
                     tempdir before `load_inline` so the directives in
                     CUDA_SOURCE resolve to *these* edits, not the
                     pristine turboquant copies.

After the run, the optimized blobs are split back into per-file `.cu` /
`.hpp` / `.cuh` artifacts under `output/<run>/best_kernel/`.

`tq_mla_decode` is the high-level pybind11 entry point exposed by the
embedded TU; it Hadamard-rotates Q, runs the 4-bit kernel, and
inverse-rotates the output. encode_mla_paged hard-wires the same
Hadamard so the kernel sees a consistent rotated basis.

DO NOT change the pybind11 surface: keep `tq_mla_decode`,
`mla_decode_4bit_ckv_kpe`, and `mla_decode_4bit_ckv_kpe_stacked_v0` all
callable with their current signatures.
"""
import sys
sys.path.insert(0, "/workspace/autocomp/sols/turboquant")
import _mla_decode_helpers as _h

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# __CUDA_SOURCE_BEGIN__
CUDA_SOURCE = ""
LOCAL_HEADERS: dict[str, str] = {}
# __CUDA_SOURCE_END__

_tq = load_inline(
    name="tq_mla_decode_tcgen05_inline",
    cpp_sources=[""], cuda_sources=[CUDA_SOURCE],
    extra_include_paths=[_h.materialize_local_headers(LOCAL_HEADERS), *_h.INCLUDE_PATHS],
    extra_cuda_cflags=_h.CUDA_FLAGS,
    extra_cflags=_h.CXX_FLAGS,
    verbose=False,
)
# Make the inline binary visible to the standalone-ref timing pass; see
# `_mla_decode_helpers.share_inline_module` for why.
_h.share_inline_module(_tq)


class ModelNew(nn.Module):
    def __init__(self, softmax_scale: float):
        super().__init__()
        self.softmax_scale = softmax_scale
        self._pool_cache: dict = {}

    def _get_pool(self, kv_cache: torch.Tensor):
        stamp = _h.pool_cache_stamp(kv_cache)
        if stamp not in self._pool_cache:
            self._pool_cache[stamp] = _h.prep_4bit_pool(kv_cache)
        return self._pool_cache[stamp]

    def forward(self, q, kv_cache, seq_lens):
        pool, page_table, page_size, page_stride = self._get_pool(kv_cache)
        return _tq.tq_mla_decode(**_h.tq_mla_decode_kwargs(
            q=q, pool=pool, page_table=page_table, seq_lens=seq_lens,
            page_size=page_size, page_stride=page_stride,
            softmax_scale=self.softmax_scale, K=int(kv_cache.shape[1]),
        ))
