"""
KernelBench reference for problem 52 (MLA encode B200).

Mirror of `51_mla_decode_b200_ref.py`. Returns precomputed dequantized
caches for a panel of seeded (kv_cache, seq_lens) inputs, looked up by
content fingerprint of `kv_cache`.

Same three-failure-mode argument as the decode ref applies here:

  - Symbol collision: loading both the prebuilt encoder `.so` and the
    candidate's inline `.so` in one process double-registers
    `torch.ops.turbo_quant.encode_mla_paged` (TORCH_LIBRARY_FRAGMENT in
    the encode source) and crashes. This ref loads no `.so`.

  - Tautological correctness when ref delegates to ModelNew: a candidate
    with a wrong rotation or wrong norm formula would still pass
    `allclose(ref, cand)` because both compute the same wrong thing.

  - Single-input overfitting: with one saved (input, output) pair the
    candidate could hardcode constants or build a `kv.data_ptr()`-keyed
    lookup. Cycling through 5 variants (4 seeds at canonical shape +
    one smaller-K shape) makes such cheats fail on uncovered trials.

Numerical contract:
  - Each `get_inputs()` call rotates through precomputed variants via a
    module-local counter. Both ref and candidate see the same `kv_cache`
    in any single trial because KB calls `model(*inputs)` and
    `model_new(*inputs)` with the same `inputs` list.
  - Outputs are compared in dequantized bf16 form (not raw uint8 nibbles)
    so KB's `atol=1e-2` absorbs harmless rounding drift in the encoder's
    fp32 math while still flagging real bugs.

If `_mla_encode_ref_outputs.pt` is missing or stale, regenerate via:
    /workspace/.venv/bin/python sols/turboquant/_gen_encode_ref_outputs.py
"""
import pathlib

import torch
import torch.nn as nn

# KernelBench `exec`s this file without setting `__file__`, so resolve our
# directory by hand. The ref always lives at sols/turboquant/ relative to
# the autocomp repo root.
_HERE = pathlib.Path("/workspace/autocomp/sols/turboquant")

_REF_PATH = _HERE / "_mla_encode_ref_outputs.pt"
if not _REF_PATH.exists():
    raise FileNotFoundError(
        f"Missing precomputed reference outputs: {_REF_PATH}.\n"
        f"Run: /workspace/.venv/bin/python {_HERE}/_gen_encode_ref_outputs.py"
    )

_PAYLOAD = torch.load(_REF_PATH, map_location="cuda", weights_only=True)
_VARIANTS: list[dict] = _PAYLOAD["variants"]
_REF_PAGE_SIZE = int(_PAYLOAD["page_size"])

# fingerprint(kv) -> saved dequantized cache (CUDA, ready to return).
_FP_TO_CACHE: dict[int, torch.Tensor] = {
    int(v["fingerprint"]): v["cache"] for v in _VARIANTS
}


def _fingerprint(kv: torch.Tensor) -> int:
    """Must match `fingerprint(...)` in `_gen_encode_ref_outputs.py`."""
    return int(kv.view(torch.uint8).flatten()[:64].sum().item())


# Module-level rotation counter. KernelBench calls `get_inputs` once per
# correctness trial and once again for the perf-timing pass, so cycling
# here naturally covers every variant across the default 5 trials.
_TRIAL_IDX = [0]


class Model(nn.Module):
    """Returns the saved dequantized cache for the variant matching `kv_cache`."""

    def __init__(self, page_size: int):
        super().__init__()
        self.page_size = page_size

    @torch.compiler.disable
    def forward(self, kv_cache, seq_lens):
        fp = _fingerprint(kv_cache)
        cache = _FP_TO_CACHE.get(fp)
        if cache is None:
            raise RuntimeError(
                f"Reference: kv_cache fingerprint {fp} not in precomputed "
                f"variants. Either get_inputs() and the saved variants are "
                f"out of sync, or KernelBench mutated the input. Known "
                f"fingerprints: {sorted(_FP_TO_CACHE)}"
            )
        return cache.to(kv_cache.device, non_blocking=True).clone()


def get_inputs():
    idx = _TRIAL_IDX[0] % len(_VARIANTS)
    _TRIAL_IDX[0] += 1
    v = _VARIANTS[idx]
    return [v["kv"].cuda(non_blocking=True),
            v["sl"].cuda(non_blocking=True)]


def get_init_inputs():
    return [_REF_PAGE_SIZE]
