"""
KernelBench reference for problem 51 (MLA decode B200).

Strategy: PRECOMPUTED 4-BIT GROUND TRUTH OVER MULTIPLE VARIANTS.
The reference returns the saved output of the prebuilt 4-bit kernel run
on a small set of fixed seeded inputs. This sidesteps three failure modes
that crippled earlier ref designs:

  - Symbol collision (cluster_launch.hpp:249 invalid argument) when the
    prebuilt 4-bit `.so` and the candidate's inline-built `.so` are
    loaded together — only the first-loaded one's CUTLASS Sm100 cluster-
    launch state survives. Loading no `.so` here avoids the conflict.

  - Tautological correctness when ref delegates to ModelNew (the previous
    paired-path design): a candidate that re-enabled non-deterministic
    split-K, or one that silently truncated attention to one tile (the
    bug in the previous `split_kv=1` patch on the prebuilt kernel),
    still passed `allclose(ref, cand)` because both computed the same
    thing.

  - Single-input overfitting: if the ref returns the same saved output
    for every trial, an agent could hardcode constants from that input
    or build a `q.data_ptr()`-keyed lookup. Cycling through several
    seeded variants (and one smaller-K shape) makes such cheats fail
    on the trials they don't cover.

Numerical contract:
  - Each `get_inputs()` call rotates through the precomputed variants
    via a module-local counter, so KernelBench's per-trial inputs are
    deterministic AND varied. Both ref and candidate see the same input
    in any single trial because KB calls `model(*inputs)` and
    `model_new(*inputs)` with the same `inputs` list.
  - `Model.forward` looks the saved output up by content fingerprint of
    `q` (sum of first 64 bytes), so the lookup survives any KB-internal
    cloning / dtype conversion of the input list as long as the data
    bytes are preserved.
  - Saved outputs were generated with split_kv=-1, so every K token is
    covered. Kernel-vs-kernel rerun max diff ≤ 2.4e-3 << atol=1e-2.

If `_mla_decode_ref_outputs.pt` is missing or stale, regenerate via:
    /workspace/turboquant/.venv/bin/python sols/turboquant/_gen_ref_outputs.py
"""
import math
import pathlib
import sys

import torch
import torch.nn as nn

# KernelBench `exec`s this file without setting `__file__`, so resolve our
# directory by hand. The ref is always at sols/turboquant/ relative to the
# autocomp repo root.
_HERE = pathlib.Path("/workspace/autocomp/sols/turboquant")
sys.path.insert(0, str(_HERE))
import _mla_decode_helpers as _h

_REF_PATH = _HERE / "_mla_decode_ref_outputs.pt"
if not _REF_PATH.exists():
    raise FileNotFoundError(
        f"Missing precomputed reference outputs: {_REF_PATH}.\n"
        f"Run: /workspace/turboquant/.venv/bin/python {_HERE}/_gen_ref_outputs.py"
    )

_PAYLOAD = torch.load(_REF_PATH, map_location="cuda", weights_only=True)
_VARIANTS: list[dict] = _PAYLOAD["variants"]
_REF_SOFTMAX_SCALE = float(_PAYLOAD["softmax_scale"])

# fingerprint(q) -> saved out_tensor (CUDA, ready to return).
_FP_TO_OUT: dict[int, torch.Tensor] = {
    int(v["fingerprint"]): v["out"] for v in _VARIANTS
}


def _fingerprint(q: torch.Tensor) -> int:
    """Must match `fingerprint(...)` in `_gen_ref_outputs.py`."""
    return int(q.view(torch.uint8).flatten()[:64].sum().item())


# Module-level rotation counter. KernelBench calls `get_inputs` once per
# correctness trial and once again for the perf-timing pass, so cycling
# here naturally covers every variant across the 5 default trials.
_TRIAL_IDX = [0]


class Model(nn.Module):
    """Returns the saved 4-bit kernel output for the variant matching `q`."""

    def __init__(self, softmax_scale: float):
        super().__init__()
        self.softmax_scale = softmax_scale

    @torch.compiler.disable
    def forward(self, q, kv_cache, seq_lens):
        fp = _fingerprint(q)
        out = _FP_TO_OUT.get(fp)
        if out is None:
            raise RuntimeError(
                f"Reference: q fingerprint {fp} not in precomputed variants. "
                f"Either get_inputs() and the saved variants are out of sync, "
                f"or KernelBench mutated the input. Known fingerprints: "
                f"{sorted(_FP_TO_OUT)}"
            )
        return out.to(q.device, non_blocking=True).clone()


def get_inputs():
    idx = _TRIAL_IDX[0] % len(_VARIANTS)
    _TRIAL_IDX[0] += 1
    v = _VARIANTS[idx]
    return [v["q"].cuda(non_blocking=True),
            v["kv"].cuda(non_blocking=True),
            v["sl"].cuda(non_blocking=True)]


def get_init_inputs():
    return [_REF_SOFTMAX_SCALE]
