"""
TurboQuant MLA encode (Blackwell sm100) — autocomp solution shell.

Mirror of `51_mla_decode_b200_template.py` for the encoder. `run_search.py`
populates the embed block below before each run with two values the agent
is free to edit:

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

ModelNew contract:
  * forward(kv_cache, seq_lens) -> bf16 [num_pages, page_size, D_TOTAL]
    Runs the inline-built encoder on `kv_cache`, then dequantizes the
    resulting 4-bit pool back to bf16 cache form. Comparing in dequantized
    bf16 (instead of raw uint8 nibbles) lets KernelBench's `atol=1e-2`
    absorb harmless rounding drift while still catching real correctness
    regressions — an off-by-one nibble shifts dequant by `step * norm`
    ~ 0.1, well above the tolerance.

DO NOT change the pybind11 surface: keep the symbol named
`encode_mla_paged` and accepting the kwargs in
`_mla_encode_helpers.encode_mla_paged_kwargs`.
"""
import sys
sys.path.insert(0, "/workspace/autocomp/sols/turboquant")
import _mla_encode_helpers as _h

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# __CUDA_SOURCE_BEGIN__
CUDA_SOURCE = ""
LOCAL_HEADERS: dict[str, str] = {}
# __CUDA_SOURCE_END__

_enc = load_inline(
    name="encode_mla_paged_inline",
    cpp_sources=[""], cuda_sources=[CUDA_SOURCE],
    extra_include_paths=[_h.materialize_local_headers(LOCAL_HEADERS), *_h.INCLUDE_PATHS],
    extra_cuda_cflags=_h.CUDA_FLAGS,
    extra_cflags=_h.CXX_FLAGS,
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    def forward(self, kv_cache: torch.Tensor, seq_lens: torch.Tensor):
        pool, _pt, _ps, _stride = _h.encode_kv_inline(
            _enc, kv_cache, seq_lens, self.page_size,
        )
        return _h.dequant_pool_to_cache(pool)
