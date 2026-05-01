"""Entry point for running Autocomp optimization.

Usage:
    python -m autocomp.search.run_search

Configure the parameters in the `main()` function below.
"""
import pathlib
import re
import random

from autocomp.common import logger
from autocomp.search.search import (
    create_backend_and_agents,
    load_initial_code,
    BeamSearchStrategy,
    ExhaustiveSearchStrategy,
)
from autocomp.search.prob import Prob
from autocomp.hw_config import (
    CudaHardwareConfig,
    GemminiHardwareConfig,
    MetalHardwareConfig,
    SaturnHardwareConfig,
    TrnHardwareConfig,
    TpuHardwareConfig,
)


def main():
    # ------------------------------------------------------------------
    # Target & environment
    # ------------------------------------------------------------------
    backend_name = "kernelbench"           # "gemmini", "trn", "tpu", "jaxbench", "kernelbench", "gpumode", "saturn", "xnnpack", "metal"
    agent_name = "cuda"    # "gemmini", "trn", "cuda", "saturn", "built:<name>", or path
    simulator = None                # "firesim"/"spike" for gemmini, saturn, and xnnpack; "gpumode-local"/"gpumode-cli" for gpumode
    hw_config = CudaHardwareConfig("NVIDIA B200", "2.9.0", "13.1")
    # hw_config = GemminiHardwareConfig(pe_dim=16, spad_size_kb=256, acc_size_kb=64)
    # hw_config = CudaHardwareConfig("NVIDIA L40S", "2.5.0", "12.4")
    # hw_config = TpuHardwareConfig("v6e-1")
    # hw_config = MetalHardwareConfig("M2", "4.0", "apple8", 8)

    prob_type = "kb-level3"              # see README.md or sols/ for available problems
    prob_id = 51                         # KernelBench/level3/51_mla_decode_b200.py

    # ------------------------------------------------------------------
    # Optional: external CUDA kernel round-tripped each run. Set
    # `kernel_cu_path = None` for problems whose sol file is the source of
    # truth (no separate `.cu` to embed).
    # ------------------------------------------------------------------
    sol_file: pathlib.Path | None = pathlib.Path(
        "sols/turboquant/51_mla_decode_b200.py"
    )
    kernel_cu_path: pathlib.Path | None = pathlib.Path(
        "/workspace/turboquant/csrc/decode_mla_tcgen05/"
        "tq_mla_decode_tcgen05_4bit.cu"
    )
    kernel_template: pathlib.Path | None = pathlib.Path(
        "sols/turboquant/51_mla_decode_b200_template.py"
    )
    kernel_local_includes: list[pathlib.Path] = [
        pathlib.Path("/workspace/turboquant/csrc/decode_mla_tcgen05"),
        pathlib.Path("/workspace/turboquant/csrc/shared"),
    ]
    if kernel_cu_path is not None:
        assert kernel_template is not None and sol_file is not None
        _embed_kernel(
            kernel_template, kernel_cu_path, sol_file,
            local_include_dirs=kernel_local_includes,
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    # Format: "provider::model" (openai, anthropic, together, aws, gcp, vllm)
    models = [
#        "anthropic::claude-opus-4-7",
#        "openai::gpt-5.5",
        "baseten::deepseek-ai.DeepSeek-V4-Pro",
        "baseten::moonshotai.Kimi-K2.6",
        "baseten::MiniMaxAI.MiniMax-M2.5",
        "baseten::moonshotai.Kimi-K2.5",
        "baseten::zai-org.GLM-5",
    ]
    code_models = None  # None = same as planning models

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    search_strategy = "beam"
    metric = "latency"
    iterations = 8
    num_plan_candidates = 7
    num_code_candidates = 3
    beam_size = 4
    dropout_menu_options = 0.25
    early_stop_iters = 0            # 0 = disabled
    early_stop_threshold = 1.0
    skip_planning = False           # True = bypass plan phase, generate code directly
    continue_from = ""

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------
    use_edits = False
    reimplement_failed = False

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    translate_iters = 0
    translate_perf_threshold = 15
    translate_drop_original = True
    translate_score = True

    # ------------------------------------------------------------------
    # Built-agent options
    # ------------------------------------------------------------------
    menu_strategy = "one-shot"      # None (static menu) or "one-shot"
    fine_grained_isa = True
    example_rate = 0.25

    # ------------------------------------------------------------------
    # Advanced / rarely changed
    # ------------------------------------------------------------------
    give_score_feedback = 1
    give_util_feedback = 0
    give_hw_feedback = 0
    include_ancestors = False
    plan_icl_examples = False
    code_icl_examples = False
    num_analyses = 0
    num_pairs_to_combine = 0
    num_gen_per_combine = 0
    trigger_exhaustive_threshold = 1
    trigger_exhaustive_iters = 20
    start_exhaustive_iters = 0
    prevent_duplicate_level = 0     # 0: same parent+plan, 1: same parent, 2: any shared ancestor
    random.seed(1111)

    # ------------------------------------------------------------------
    # Sanitize model names for filesystem
    # ------------------------------------------------------------------
    models = [m.replace("/", "_") for m in models]
    if code_models is not None:
        code_models = [m.replace("/", "_") for m in code_models]

    # ------------------------------------------------------------------
    # Build output directory & start logging
    # ------------------------------------------------------------------
    built_menu_strategy_enum = {None: 0, "one-shot": 1}
    clean_agent_name = pathlib.Path(agent_name).name if "/" in agent_name else agent_name
    output_str = f"{clean_agent_name}"
    output_str += f"_{prob_type}_{prob_id}_{search_strategy}_iters{iterations}"
    if simulator is not None:
        output_str += f"_{simulator}"
    hw_desc = hw_config.get_hw_description().replace(" ", "").replace("(", "_").replace(")", "").replace(",", "_")
    output_str += f"_{hw_desc}"
    for model in models:
        output_str += f"_{model[-20:]}"
    if code_models is not None:
        output_str += "_code"
        for model in code_models:
            output_str += f"_{model[-20:]}"
    if dropout_menu_options:
        output_str += f"_do{dropout_menu_options}"
    if search_strategy == "beam":
        if num_analyses:
            output_str += f"_an{num_analyses}"
        output_str += f"_p{num_plan_candidates}_c{num_code_candidates}_b{beam_size}"
    if translate_iters > 0:
        output_str += f"_tr{translate_iters}_{translate_perf_threshold}"
        if translate_drop_original:
            output_str += "_trdrop"
        if translate_score:
            output_str += "_tscore"
    if give_score_feedback:
        output_str += f"_score{give_score_feedback}"
    if give_util_feedback:
        output_str += f"_util{give_util_feedback}"
    if give_hw_feedback:
        output_str += f"_hwfb{give_hw_feedback}"
    if include_ancestors:
        output_str += "_anc1"
    if prevent_duplicate_level:
        output_str += f"_pd{prevent_duplicate_level}"
    if plan_icl_examples:
        output_str += "_picl1"
    if code_icl_examples:
        output_str += "_cicl1"
    if reimplement_failed:
        output_str += "_reimpl1"
    if early_stop_iters > 0:
        output_str += f"_es{early_stop_iters}_{early_stop_threshold}"
    if menu_strategy:
        output_str += f"_ms{built_menu_strategy_enum[menu_strategy]}"
    if fine_grained_isa:
        output_str += "_fgisa1"
    if example_rate > 0:
        output_str += f"_ex{example_rate}"
    if continue_from:
        output_str += "_continued"
    if use_edits:
        output_str += "_edits"
    if skip_planning:
        output_str += "_noplan"
    output_dir = pathlib.Path("output") / output_str
    output_dir.mkdir(parents=True, exist_ok=True)

    import autocomp.common.my_logging
    autocomp.common.my_logging.move_log(output_dir, tag="search")
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Initialize and run
    # ------------------------------------------------------------------
    prob = Prob(prob_type, prob_id, sol_file=sol_file)
    initial_code = load_initial_code(backend_name, prob)
    eval_backend, agent, code_agent = create_backend_and_agents(
        backend_name, agent_name, hw_config, prob, models, code_models,
        menu_strategy=menu_strategy, fine_grained_isa=fine_grained_isa,
        example_rate=example_rate, cache_dir=output_dir,
    )

    common_kwargs = dict(
        output_dir=output_dir, eval_backend=eval_backend, agent=agent,
        orig_code=initial_code, prob=prob, metric=metric, simulator=simulator,
        give_score_feedback=give_score_feedback,
        give_util_feedback=give_util_feedback,
        give_hw_feedback=give_hw_feedback,
        include_ancestors=include_ancestors,
        plan_icl_examples=plan_icl_examples,
        code_icl_examples=code_icl_examples,
        dropout_menu_options=dropout_menu_options,
        prevent_duplicate_level=prevent_duplicate_level,
        translate_iters=translate_iters,
        translate_perf_threshold=translate_perf_threshold,
        translate_drop_original=translate_drop_original,
        translate_score=translate_score,
        code_agent=code_agent,
        early_stop_iters=early_stop_iters,
        early_stop_threshold=early_stop_threshold,
        continue_from=continue_from,
        use_edits=use_edits,
    )

    if search_strategy == "exhaustive":
        optimizer = ExhaustiveSearchStrategy(**common_kwargs)
    elif search_strategy == "beam":
        optimizer = BeamSearchStrategy(
            **common_kwargs,
            num_analyses=num_analyses,
            num_plan_candidates=num_plan_candidates,
            num_code_candidates=num_code_candidates,
            beam_size=beam_size,
            num_pairs_to_combine=num_pairs_to_combine,
            num_gen_per_combine=num_gen_per_combine,
            trigger_exhaustive_threshold=trigger_exhaustive_threshold,
            trigger_exhaustive_iters=trigger_exhaustive_iters,
            start_exhaustive_iters=start_exhaustive_iters,
            reimplement_failed=reimplement_failed,
            skip_planning=skip_planning,
        )
    else:
        raise ValueError(f"Unknown search strategy: {search_strategy}")

    optimizer.optimize(iterations)

    if kernel_cu_path is not None:
        _write_back_best_kernel(output_dir)


# ---------------------------------------------------------------------------
# CUDA-kernel insertion: round-trips a `.cu` file through the search.
# `_embed_kernel` substitutes the kernel into a Python template;
# `_write_back_best_kernel` extracts the agent's best edit back to disk.
# ---------------------------------------------------------------------------

_CUDA_SOURCE_BEGIN = "# __CUDA_SOURCE_BEGIN__"
_CUDA_SOURCE_END   = "# __CUDA_SOURCE_END__"
_INCLUDE_PATTERN   = re.compile(r'#include\s+"([^"]+)"')


def _strip_c_comments(code: str) -> str:
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    code = re.sub(r'//[^\n]*', '', code)
    # Stripped comments leave trailing or whitespace-only lines; flatten
    # those before collapsing blank-line runs.
    code = re.sub(r'[ \t]+\n', '\n', code)
    code = re.sub(r'\n{3,}', '\n\n', code)
    return code.strip()


def _discover_local_headers(
    src: str,
    src_dir: pathlib.Path,
    local_dirs: list[pathlib.Path] | None = None,
) -> dict[str, str]:
    '''Walk `#include "..."` directives transitively and collect their bodies.

    Returns a flat dict keyed by the header name as it appears in the
    `#include` directive (so the embedded sol can write each one to a
    tempdir for the compiler to find). Headers not found in `src_dir` or
    `local_dirs` are left for `extra_include_paths` to resolve.
    '''
    headers: dict[str, str] = {}
    extra = list(local_dirs or [])
    queue: list[tuple[str, pathlib.Path]] = [(src, src_dir)]
    while queue:
        text, owner_dir = queue.pop()
        for m in _INCLUDE_PATTERN.finditer(text):
            header = m.group(1)
            if header in headers:
                continue
            for d in [owner_dir] + extra:
                candidate = (d / header).resolve()
                if candidate.exists():
                    content = candidate.read_text()
                    headers[header] = content
                    queue.append((content, candidate.parent))
                    break
    return headers


def _embed_kernel(
    template_path: pathlib.Path,
    cu_path: pathlib.Path,
    out_path: pathlib.Path,
    local_include_dirs: list[pathlib.Path] | None = None,
    strip_comments: bool = True,
) -> None:
    '''Embed `cu_path` and its local headers as separate string variables in
    the `_CUDA_SOURCE_BEGIN/END` block of `template_path`. Each header is
    its own dict entry under `LOCAL_HEADERS`; the main TU keeps its
    `#include "..."` directives so the agent's view stays per-file.'''
    cu_src = cu_path.read_text()
    headers = _discover_local_headers(cu_src, cu_path.parent, local_include_dirs)
    if strip_comments:
        cu_src  = _strip_c_comments(cu_src)
        headers = {k: _strip_c_comments(v) for k, v in headers.items()}

    parts = [f'CUDA_SOURCE = {_emit_triple_quoted(cu_src)}', ""]
    if headers:
        parts.append("LOCAL_HEADERS: dict[str, str] = {")
        for name, content in headers.items():
            parts.append(f"    {name!r}: {_emit_triple_quoted(content)},")
        parts.append("}")
    else:
        parts.append("LOCAL_HEADERS: dict[str, str] = {}")

    new_block = f"{_CUDA_SOURCE_BEGIN}\n" + "\n".join(parts) + f"\n{_CUDA_SOURCE_END}"
    result, n_subs = re.subn(
        rf"{re.escape(_CUDA_SOURCE_BEGIN)}\n.*?{re.escape(_CUDA_SOURCE_END)}",
        lambda _: new_block,
        template_path.read_text(),
        flags=re.DOTALL,
    )
    if n_subs == 0:
        raise ValueError(
            f"{template_path} is missing the "
            f"{_CUDA_SOURCE_BEGIN!r} / {_CUDA_SOURCE_END!r} sentinel block."
        )
    out_path.write_text(result)
    total_lines = cu_src.count("\n") + sum(v.count("\n") for v in headers.values())
    logger.info(
        "Embedded kernel (%d lines, %d local header(s)) into %s",
        total_lines, len(headers), out_path.name,
    )


def _emit_triple_quoted(s: str) -> str:
    '''Format `s` as `"""\n...\n"""` with backslashes/triple-quotes escaped
    so the round-trip through Python's string-literal parser is exact.'''
    safe = s.replace("\\", "\\\\").replace('"""', r'\"\"\"')
    return f'"""\n{safe}\n"""'


def _extract_kernel_sources(src: str) -> tuple[str | None, dict[str, str]]:
    '''Recover `(CUDA_SOURCE, LOCAL_HEADERS)` from a candidate's Python source.

    Uses `ast.parse` so the result is robust to raw strings, escape
    variations, and dropped sentinel comments. Falls back to a permissive
    `CUDA_SOURCE` regex (headers only via AST) when the candidate isn't
    yet valid Python.
    '''
    import ast
    cuda_source: str | None = None
    local_headers: dict[str, str] = {}

    def _str_const(node) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"):   v = v[:-1]
            return v
        return None

    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            # Plain `X = ...` and annotated `X: T = ...` look different in
            # the AST but mean the same thing here.
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.value is not None:
                targets = [node.target]
            else:
                continue
            value = node.value
            for tgt in targets:
                if tgt.id == "CUDA_SOURCE":
                    cuda_source = _str_const(value)
                elif tgt.id == "LOCAL_HEADERS" and isinstance(value, ast.Dict):
                    for k_node, v_node in zip(value.keys, value.values):
                        if isinstance(k_node, ast.Constant) \
                                and isinstance(k_node.value, str):
                            v = _str_const(v_node)
                            if v is not None:
                                local_headers[k_node.value] = v
        if cuda_source is not None:
            return cuda_source, local_headers

    # Regex fallback (CUDA_SOURCE only) for mid-edit / invalid-Python candidates.
    m = re.search(
        r'CUDA_SOURCE\s*=\s*(r?)("""|\'\'\')\n?(.*?)\n?\2',
        src, re.DOTALL,
    )
    if m is None:
        return None, {}
    body = m.group(3)
    if not m.group(1):
        try:
            body = body.encode("utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            body = body.replace("\\\\", "\\")
    return body, {}


def _write_back_best_kernel(output_dir: pathlib.Path) -> pathlib.Path | None:
    '''Extract `CUDA_SOURCE` (and any `LOCAL_HEADERS`) from
    `best_candidate_so_far.py` and write them back as a per-file tree
    under `<output_dir>/best_kernel/`.'''
    best_py = output_dir / "best_candidate_so_far.py"
    if not best_py.exists():
        logger.warning("No best_candidate_so_far.py found; skipping kernel write-back.")
        return None

    cuda_body, headers = _extract_kernel_sources(best_py.read_text())
    if cuda_body is None:
        logger.warning("CUDA_SOURCE assignment not found in best candidate; skipping write-back.")
        return None

    if not headers:
        # Single-file kernel — keep the flat artifact.
        out_cu = output_dir / "best_kernel.cu"
        out_cu.write_text(cuda_body)
        logger.info("Best optimized kernel written to %s/%s", output_dir.name, out_cu.name)
        return out_cu

    out_dir = output_dir / "best_kernel"
    out_dir.mkdir(parents=True, exist_ok=True)
    main_cu = out_dir / "kernel.cu"
    main_cu.write_text(cuda_body)
    for name, content in headers.items():
        f = out_dir / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
    logger.info(
        "Best optimized kernel written to %s/best_kernel/ (1 .cu + %d header(s))",
        output_dir.name, len(headers),
    )
    return main_cu


if __name__ == "__main__":
    main()
