"""
One-shot generator for precomputed reference outputs of problem 52 (encode).

Mirror of `_gen_ref_outputs.py` (decode). Runs the prebuilt
`encode_mla_paged.so` on a small panel of seeded (B, K) variants and saves
the dequantized pool output for each, so the eval-time ref can `torch.load`
them and avoid loading any `.so` itself (preventing
`torch.ops.turbo_quant.encode_mla_paged` double-registration).

Why MULTIPLE variants:
  - KernelBench runs N correctness trials per eval. With a single saved
    (input, output) pair, the candidate could memoize on
    `kv_cache.data_ptr()` or hardcode constants. Cycling through several
    seeds (catches distribution-specific shortcuts) plus one smaller-K
    shape (catches K-prefix bugs) makes those cheats fail.

Why DEQUANTIZED bf16 instead of raw uint8:
  - Comparing raw uint8 nibbles is bit-exact, which would fail any
    legitimate optimization that changes intermediate rounding (fmaf vs
    independent mul/add, different reduction order, etc.). Comparing
    dequantized bf16 with KB's atol=1e-2 absorbs ULP-scale drift in the
    norm sqrt while still catching real bugs — an off-by-one nibble
    shifts dequant by `step * norm` ~ 0.1, well above the tolerance.

Run after rebuilding `/workspace/turboquant/build/lib/encode_mla_paged.*.so`.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _mla_encode_helpers as _h

# (B, K, seed) tuples to precompute. Mirror of the decode panel so a single
# `kv_cache` distribution covers both encode and decode evals.
VARIANTS = [
    dict(B=8, K=4096, seed=0),
    dict(B=8, K=4096, seed=1),
    dict(B=8, K=4096, seed=2),
    dict(B=8, K=4096, seed=3),
    dict(B=4, K=512,  seed=4),
]

PAGE_SIZE = _h.PAGE_SIZE
OUT_PATH = _HERE / "_mla_encode_ref_outputs.pt"
PREBUILT_SO = ("/workspace/turboquant/build/lib/"
               "encode_mla_paged.cpython-312-x86_64-linux-gnu.so")


def make_inputs(B: int, K: int, seed: int, device: str = "cuda"):
    """Returns the (kv, sl) tensors for one canonical variant."""
    g = torch.Generator(device=device).manual_seed(seed)
    kv = torch.randn(B, K, _h.D_TOTAL, dtype=torch.bfloat16,
                     device=device, generator=g)
    sl = torch.full((B,), K, dtype=torch.int32, device=device)
    return kv, sl


def fingerprint(kv: torch.Tensor) -> int:
    """Cheap content hash: sum of the first 64 bytes of kv. Same in the
    gen script and the ref so saved outputs can be looked up by the
    candidate-side input. Tensors are bf16 with deterministic seeded
    init, so the first 64 bytes are stable per (B, K, seed)."""
    return int(kv.view(torch.uint8).flatten()[:64].sum().item())


def _load_so(path: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required to generate reference outputs")

    print(f"Loading prebuilt encoder from {PREBUILT_SO}")
    # modname MUST match the .so stem — pybind's PyInit_<name> is fixed at
    # compile time to TORCH_EXTENSION_NAME (= "encode_mla_paged" here).
    enc = _load_so(PREBUILT_SO, "encode_mla_paged")

    payload = {"page_size": PAGE_SIZE, "variants": []}

    for v in VARIANTS:
        B, K, seed = v["B"], v["K"], v["seed"]
        print(f"\n--- variant B={B} K={K} seed={seed} ---")
        kv, sl = make_inputs(B, K, seed, "cuda")

        # Warmup runs: first call after CUTLASS workspace alloc can drift
        # vs subsequent ones. Encode is much more deterministic than decode
        # but we keep the same structure for symmetry.
        for _ in range(3):
            pool, _pt, _ps, _stride = _h.encode_kv_inline(enc, kv, sl, PAGE_SIZE)
        cache = _h.dequant_pool_to_cache(pool)
        torch.cuda.synchronize()
        print(f"  cache.shape={tuple(cache.shape)} max={cache.abs().max().item():.4f}")

        rerun_diffs = []
        for _ in range(8):
            p_i, _, _, _ = _h.encode_kv_inline(enc, kv, sl, PAGE_SIZE)
            c_i = _h.dequant_pool_to_cache(p_i)
            torch.cuda.synchronize()
            rerun_diffs.append((cache.float() - c_i.float()).abs().max().item())
        rerun_max = max(rerun_diffs)
        print(f"  rerun max diff over 8 trials: {rerun_max:.5f}")
        assert rerun_max < 5e-3, (
            f"Encode kernel non-determinism too large ({rerun_max:.5f}) on "
            f"B={B} K={K} seed={seed} — would eat into KernelBench's "
            f"atol=1e-2 budget.")

        fp = fingerprint(kv)
        payload["variants"].append(dict(
            B=B, K=K, seed=seed, fingerprint=fp,
            kv=kv.cpu(), sl=sl.cpu(), cache=cache.cpu(),
        ))

    fps = [v["fingerprint"] for v in payload["variants"]]
    assert len(set(fps)) == len(fps), f"Fingerprint collision: {fps}"

    torch.save(payload, OUT_PATH)
    print(f"\nSaved {OUT_PATH}  "
          f"({OUT_PATH.stat().st_size / 1024 / 1024:.1f} MiB, "
          f"{len(payload['variants'])} variants)")


if __name__ == "__main__":
    main()
