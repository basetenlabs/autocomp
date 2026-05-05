"""
One-shot generator for precomputed reference outputs of problem 51.

Why a precomputed `.pt`:
  - The ref needs to call the prebuilt 4-bit `.so` to obtain a true ground
    truth, but loading the prebuilt + the candidate's inline `.so` in the
    same process collides on CUTLASS Sm100 cluster-launch state
    (cluster_launch.hpp:249: invalid argument), masking real bugs.
  - Precomputing once, then having the ref `torch.load` the result, keeps
    the eval process clean (no second .so loaded).

Why MULTIPLE variants:
  - KernelBench runs N correctness trials per eval. With a single saved
    (input, output) pair, the ref returns the same output every trial and
    the candidate could hardcode constants, build a lookup keyed on
    `q.data_ptr()`, or otherwise overfit to one specific input.
  - We precompute several (q, kv, sl, out) tuples — multiple seeds at the
    canonical shape (catches distribution-specific shortcuts) plus one
    smaller shape (catches K-prefix bugs like the original split_kv=1
    truncation regression). The ref cycles through them across trials and
    looks the saved output up by content fingerprint.

Numerical contract:
  - Inputs are sampled with explicit `torch.Generator(seed=...)` so every
    run of the eval sees byte-identical tensors regardless of any global
    seed KernelBench sets.
  - Reruns of the prebuilt kernel on these inputs settle into ≤ 2.4e-3
    max abs diff (1 bf16 ULP at ~77% of output positions) after a small
    warmup, well under KernelBench's `precision=bf16` atol=rtol=1e-2.

Run after rebuilding the prebuilt 4-bit `.so` (which lives at
`/workspace/turboquant/build/lib/tq_mla_decode_4bit.*.so`).
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _mla_decode_helpers as _h

# Canonical KernelBench shape (matches the autocomp problem 51 spec).
REF_H = 128
REF_SOFTMAX_SCALE = 1.0 / math.sqrt(_h.D_TOTAL)

# (B, K, seed) tuples to precompute. Multiple seeds at the canonical shape
# stress distribution-specific shortcuts; the smaller-K variant catches
# K-prefix truncation bugs (the original split_kv=1 patch silently dropped
# all K tokens beyond the first 128, and would have passed a single-shape
# correctness check trivially).
VARIANTS = [
    dict(B=8, K=4096, seed=0),
    dict(B=8, K=4096, seed=1),
    dict(B=8, K=4096, seed=2),
    dict(B=8, K=4096, seed=3),
    dict(B=4, K=512,  seed=4),
]

OUT_PATH = _HERE / "_mla_decode_ref_outputs.pt"
PREBUILT_SO = ("/workspace/turboquant/build/lib/"
               "tq_mla_decode_4bit.cpython-312-x86_64-linux-gnu.so")


def make_inputs(B: int, K: int, seed: int, device: torch.device | str = "cuda"):
    """Returns the (q, kv, sl) tensors for one canonical variant."""
    g = torch.Generator(device=device).manual_seed(seed)
    q  = torch.randn(B, REF_H, _h.D_TOTAL, dtype=torch.bfloat16,
                     device=device, generator=g)
    kv = torch.randn(B, K,     _h.D_TOTAL, dtype=torch.bfloat16,
                     device=device, generator=g)
    sl = torch.full((B,), K, dtype=torch.int32, device=device)
    return q, kv, sl


def fingerprint(q: torch.Tensor) -> int:
    """Cheap content hash: sum of the first 64 bytes of q.

    Same in the gen script and the ref so saved outputs can be looked up
    by the candidate-side input. Tensors are bf16 with deterministic
    seeded init, so the first 64 bytes are stable per (B, K, seed).
    """
    return int(q.view(torch.uint8).flatten()[:64].sum().item())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to generate reference outputs")

    print(f"Loading prebuilt 4-bit kernel from {PREBUILT_SO}")
    tq = _h._load_so(PREBUILT_SO, "tq_mla_decode_4bit")

    payload: dict = {
        "softmax_scale": REF_SOFTMAX_SCALE,
        "variants": [],   # list of {B, K, seed, fingerprint, q, kv, sl, out}
    }

    for v in VARIANTS:
        B, K, seed = v["B"], v["K"], v["seed"]
        print(f"\n--- variant B={B} K={K} seed={seed} ---")
        q, kv, sl = make_inputs(B, K, seed, "cuda")
        pool, page_table, ps, page_stride = _h.prep_4bit_pool(kv)

        kw = _h.tq_mla_decode_kwargs(
            q=q, pool=pool, page_table=page_table, seq_lens=sl,
            page_size=ps, page_stride=page_stride,
            softmax_scale=REF_SOFTMAX_SCALE, K=K,
        )
        # Warmup: the very first call after a CUTLASS workspace alloc can
        # drift ~7e-3 from later calls; reruns settle ≤ 2.4e-3.
        for _ in range(3):
            _ = tq.tq_mla_decode(**kw)
        torch.cuda.synchronize()

        out = tq.tq_mla_decode(**kw)
        torch.cuda.synchronize()
        print(f"  out.shape={tuple(out.shape)} max={out.abs().max().item():.4f}")

        rerun_diffs = []
        for _ in range(8):
            out_i = tq.tq_mla_decode(**kw)
            torch.cuda.synchronize()
            rerun_diffs.append((out.float() - out_i.float()).abs().max().item())
        rerun_max = max(rerun_diffs)
        print(f"  rerun max diff over 8 trials: {rerun_max:.5f}")
        assert rerun_max < 5e-3, (
            f"Kernel non-determinism is too large ({rerun_max:.5f}) on "
            f"variant B={B} K={K} seed={seed} — would eat into "
            f"KernelBench's atol=1e-2 budget.")

        fp = fingerprint(q)
        payload["variants"].append(dict(
            B=B, K=K, seed=seed, fingerprint=fp,
            q=q.cpu(), kv=kv.cpu(), sl=sl.cpu(), out=out.cpu(),
        ))

    # Sanity: fingerprints are unique across variants
    fps = [v["fingerprint"] for v in payload["variants"]]
    assert len(set(fps)) == len(fps), f"Fingerprint collision: {fps}"

    torch.save(payload, OUT_PATH)
    print(f"\nSaved {OUT_PATH}  "
          f"({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MiB, "
          f"{len(payload['variants'])} variants)")


if __name__ == "__main__":
    main()
