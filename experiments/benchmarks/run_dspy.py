"""Paper-exact GEPA+Merge ablation runner.

Uses the DSPy programs, metrics, and feedback functions from gepa-ai/gepa-artifact
(vendored at experiments/vendor/gepa-artifact/) together with gepa_merge's
DSPyAdapter and A1-A4 / B1-B4 merge strategies.

Entry point:

    OPENAI_API_KEY='...' uv run --active python -m experiments.benchmarks.run_dspy \\
        --task ifbench \\
        --model openai/gpt-5-nano \\
        --reflection_lm openai/gpt-4o-mini \\
        --use_merge --merge_selection random --merge_start immediate \\
        --merge_quality \\
        --run_dir runs/smoke_ifbench \\
        --max_metric_calls 500 --seed 0

Supported tasks: ifbench, hotpotqa, hover, musique, twowikimultihopqa.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "experiments" / "vendor" / "gepa-artifact"
if VENDOR.exists() and str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))


def _install_compat_shims():
    """Monkey-patch DSPy symbols that gepa-artifact expects but new DSPy moved/removed.

    gepa-artifact was built against dspy~=3.0; several symbols it imports have
    been relocated in dspy>=3.1:
      - `dspy.dsp.utils.EM` / `F1` → `dspy.evaluate.metrics.EM` / `F1`
    We re-export them from the legacy location so vendored code keeps working.
    """
    import dspy.dsp.utils as _legacy_utils
    from dspy.evaluate import metrics as _new_metrics

    if not hasattr(_legacy_utils, "EM"):
        _legacy_utils.EM = _new_metrics.EM
    if not hasattr(_legacy_utils, "F1"):
        _legacy_utils.F1 = _new_metrics.F1


_install_compat_shims()


# Each entry maps --task value to (import_path, program_index).
# import_path is a module under gepa_artifact.benchmarks.*.
# program_index picks which program from BenchmarkMeta.program (usually 0).
TASKS = {
    "ifbench":  ("gepa_artifact.benchmarks.IFBench",   0),
    "hotpotqa": ("gepa_artifact.benchmarks.hotpotQA",  0),
    "hover":    ("gepa_artifact.benchmarks.hover",     0),
    "musique":  ("gepa_artifact.benchmarks.musique",   0),
    "twowikimultihopqa": ("gepa_artifact.benchmarks.twowikimultihopqa", 0),
    "2wiki":    ("gepa_artifact.benchmarks.twowikimultihopqa", 0),
}


def load_task(task_name: str):
    """Return (program, dataset, metric_with_feedback, feedback_map)."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS)}")

    import_path, prog_idx = TASKS[task_name]
    bench_module = __import__(import_path, fromlist=["benchmark"])
    bm_meta = bench_module.benchmark[0]

    dataset = bm_meta.benchmark()  # triggers init_dataset()
    program = bm_meta.program[prog_idx]

    # Some vendored programs (e.g., IFBenchCoT2StageProgram) override `__call__`
    # but don't define `forward`. DSPy's trace collection (bootstrap_trace_data)
    # bypasses `__call__` and grabs `forward` directly; when `forward` isn't
    # present, DSPy returns a single prediction instead of the (pred, trace)
    # tuple the wrapper expects -> "not enough values to unpack".
    #
    # Fix: rename `__call__` to `forward` so dspy.Module's default `__call__`
    # (which wraps forward with tracing) takes effect.
    cls = type(program)
    if "__call__" in cls.__dict__ and "forward" not in cls.__dict__:
        cls.forward = cls.__dict__["__call__"]
        delattr(cls, "__call__")

    metric_fn = bm_meta.metric_with_feedback
    feedback_maps = bm_meta.feedback_fn_maps or [{}]
    feedback_map = feedback_maps[0]

    # Some benchmarks don't define per-predictor feedback functions.
    # DspyAdapter hard-indexes `feedback_map[predictor_name]` which KeyErrors
    # in that case. Build a generic fallback that routes every predictor's
    # feedback through the benchmark's program-level `metric_with_feedback`.
    predictor_names = [name for name, _ in program.named_predictors()]
    missing = [n for n in predictor_names if n not in feedback_map]
    if missing:
        import dspy as _dspy

        def _make_fallback(name):
            def _fallback(predictor_output, predictor_inputs, module_inputs,
                          module_outputs, captured_trace):
                # module_outputs is a dict-like Prediction payload from the
                # full program run; module_inputs is the dspy.Example.
                if isinstance(module_outputs, dict):
                    pred = _dspy.Prediction(**module_outputs)
                else:
                    pred = module_outputs
                try:
                    result = metric_fn(module_inputs, pred)
                except Exception as e:
                    return {"score": 0.0, "feedback": f"Metric error: {e}"}
                return {
                    "score": getattr(result, "score", None),
                    "feedback": getattr(result, "feedback", "") or "",
                }
            return _fallback

        feedback_map = dict(feedback_map)
        for name in missing:
            feedback_map[name] = _make_fallback(name)

    return program, dataset, metric_fn, feedback_map


def extract_seed_candidate(program) -> dict[str, str]:
    """Extract the current instruction string for each predictor in the DSPy program.

    GEPA treats the seed_candidate as the starting point for optimization; each
    key is a predictor name (e.g., 'generate_response_module.predict') and each
    value is the instruction text (e.g., 'Respond to the query').
    """
    seed = {}
    for name, pred in program.named_predictors():
        try:
            seed[name] = pred.signature.instructions
        except AttributeError:
            seed[name] = ""
    return seed


def _inject_no_think(prompt, messages, tag: str = "/no_think"):
    """Prepend `tag` to the last user message (or the prompt string if no
    messages list). Non-destructive: copies `messages` before mutating.

    Qwen3 models default to thinking mode, which burns ~2000 reasoning tokens
    per call — ~200× the cost of a non-thinking call. `/no_think` is Qwen3's
    chat-template tag that disables thinking for that turn. Paper ran Qwen3-8B
    on local GPUs where reasoning tokens were free; API runs must disable.
    """
    if messages:
        messages = [dict(m) for m in messages]
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and not c.lstrip().startswith(tag):
                    m["content"] = f"{tag}\n{c}"
                break
    elif prompt is not None and isinstance(prompt, str):
        if not prompt.lstrip().startswith(tag):
            prompt = f"{tag}\n{prompt}"
    return prompt, messages


def _make_lm(model: str, **kwargs):
    """dspy.LM constructor that auto-injects /no_think for Qwen3 models.

    Matches by model-family name, not provider — covers all Qwen3 variants
    (4B/8B/14B/32B) on any provider (Together, OpenRouter, DeepInfra, etc.).
    Non-Qwen3 models (gpt-4.1-mini, Llama, etc.) return a vanilla dspy.LM
    unchanged — zero effect on P1/P2.

    Env override: set DISABLE_NO_THINK=1 to skip /no_think injection and use
    paper-exact thinking-on sampling (temperature=0.6, top_p=0.95, top_k=20)
    per Qwen team / GEPA paper E.2.
    """
    import dspy
    import os as _os

    # Prime Intellect inference. When PRIME_API_KEY is set, route through Prime
    # Intellect's OpenAI-compatible endpoint at api.pinference.ai.
    #
    # Routing rules:
    #   - Qwen3 family: default to /no_think (matches the team's existing
    #     Phase A methodology — all reported Qwen3 cells in REPORT.md were
    #     /no_think). Set DISABLE_NO_THINK=1 to opt into paper-exact thinking-on
    #     sampling (temperature=0.6, top_p=0.95, top_k=20) plus a
    #     `reasoning_content`→`content` promotion wrapper for adapters that
    #     can't parse Prime's split format.
    #   - GPT-4.1-mini: pass-through; no special sampling.
    #   - Other models: untouched.
    is_qwen3 = "qwen3" in model.lower()
    is_gpt41_family = "gpt-4.1" in model.lower()
    if (is_qwen3 or is_gpt41_family) and _os.environ.get("PRIME_API_KEY"):
        prime_kwargs = dict(
            api_base="https://api.pinference.ai/api/v1",
            api_key=_os.environ["PRIME_API_KEY"],
        )
        thinking_on = is_qwen3 and _os.environ.get("DISABLE_NO_THINK") == "1"
        if thinking_on:
            # Paper E.2 Qwen3 thinking-on sampling (vLLM honors top_k).
            prime_kwargs.update(temperature=0.6, top_p=0.95, top_k=20)
        prime_kwargs.update(kwargs)
        team_id = _os.environ.get("PRIME_TEAM_ID")
        if team_id:
            prime_kwargs.setdefault("extra_headers", {})["X-Prime-Team-ID"] = team_id
        # litellm strips its dispatch prefix before forwarding, so we need a
        # *double* "openai/" wrap: the outer one tells litellm to use the
        # OpenAI-compatible client (it'll be stripped), the inner one is what
        # Prime actually receives.
        prime_model = model
        for prefix in ("openrouter/", "together_ai/"):
            if prime_model.startswith(prefix):
                prime_model = prime_model[len(prefix):]
                break
        prime_model = f"openai/{prime_model}"

        if is_qwen3 and not thinking_on:
            # Default Prime+Qwen3 path: /no_think. Inject the tag on every
            # request so reasoning never starts and the model produces the
            # structured output directly. Matches REPORT.md's Qwen3 method.
            class _NoThinkPrimeLM(dspy.LM):
                def __call__(self, prompt=None, messages=None, **call_kwargs):
                    prompt, messages = _inject_no_think(prompt, messages)
                    return super().__call__(prompt=prompt, messages=messages, **call_kwargs)

                def forward(self, prompt=None, messages=None, **call_kwargs):
                    prompt, messages = _inject_no_think(prompt, messages)
                    return super().forward(prompt=prompt, messages=messages, **call_kwargs)

            return _NoThinkPrimeLM(prime_model, **prime_kwargs)

        if thinking_on:
            # Prime's vLLM splits Qwen3 thinking into `reasoning_content` and
            # leaves `content` empty when the model runs out of tokens during
            # reasoning — DSPy's ChatAdapter then raises AdapterParseError on
            # text=None. Promote reasoning_content into content when content
            # is empty so the adapter has *something* to parse (even if it's
            # the truncated CoT, that's better than a 0-score parse failure
            # that contaminates GEPA's optimization signal).
            class _ReasoningInlineLM(dspy.LM):
                def forward(self, *args, **call_kwargs):
                    resp = super().forward(*args, **call_kwargs)
                    for ch in getattr(resp, "choices", []) or []:
                        msg = getattr(ch, "message", None)
                        if msg is None:
                            continue
                        rc = getattr(msg, "reasoning_content", None) or ""
                        content = msg.content or ""
                        if not content.strip() and rc.strip():
                            msg.content = rc
                        try:
                            msg.reasoning_content = None
                        except Exception:
                            pass
                    return resp

            return _ReasoningInlineLM(prime_model, **prime_kwargs)

        # gpt-4.1-mini path through Prime: vanilla, no wrapping.
        return dspy.LM(prime_model, **prime_kwargs)

    if "qwen3" in model.lower():
        if _os.environ.get("DISABLE_NO_THINK") == "1":
            # Paper E.2: Qwen3 thinking-on sampling.
            paper_kwargs = dict(temperature=0.6, top_p=0.95, top_k=20)
            paper_kwargs.update(kwargs)

            # OpenRouter / AtlasCloud splits Qwen3's thinking trace into
            # `reasoning_content` (separate field) and only puts the final
            # answer in `content`. When the model "loses" structured output
            # into reasoning_content (content empty), DSPy's ChatAdapter
            # raises AdapterParseError → trace becomes FailedPrediction →
            # GEPA's make_reflective_dataset rejects it → 0 mutations land.
            # Fix: promote reasoning_content into content when content is empty.
            class _ReasoningInlineLM(dspy.LM):
                def forward(self, *args, **call_kwargs):
                    resp = super().forward(*args, **call_kwargs)
                    for ch in getattr(resp, "choices", []) or []:
                        msg = getattr(ch, "message", None)
                        if msg is None:
                            continue
                        rc = getattr(msg, "reasoning_content", None) or ""
                        content = msg.content or ""
                        if not content.strip() and rc.strip():
                            msg.content = rc
                        # Clear reasoning_content so DSPy returns plain
                        # string list (avoids dict-handling edge cases
                        # in some adapters).
                        try:
                            msg.reasoning_content = None
                        except Exception:
                            pass
                    return resp

            return _ReasoningInlineLM(model, **paper_kwargs)

        class _NoThinkLM(dspy.LM):
            def __call__(self, prompt=None, messages=None, **call_kwargs):
                prompt, messages = _inject_no_think(prompt, messages)
                return super().__call__(prompt=prompt, messages=messages, **call_kwargs)

            def forward(self, prompt=None, messages=None, **call_kwargs):
                prompt, messages = _inject_no_think(prompt, messages)
                return super().forward(prompt=prompt, messages=messages, **call_kwargs)

        return _NoThinkLM(model, **kwargs)
    return dspy.LM(model, **kwargs)


def configure_dspy_lm(model: str):
    import dspy
    lm = _make_lm(model)
    dspy.configure(lm=lm)
    return lm


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--model", default="openai/gpt-5-nano",
                        help="task LM (DSPy/litellm format)")
    parser.add_argument("--reflection_lm", default=None,
                        help="reflection LM for GEPA mutation proposals. "
                             "Default: same as --model (paper-exact same-model setup).")
    parser.add_argument("--reflection_max_tokens", type=int, default=None,
                        help="Cap reflection LM output to N tokens (passes max_tokens "
                             "kwarg to dspy.LM). Default None = uncapped (paper-exact). "
                             "Use to test reflective inflation control hypothesis (§6).")

    # Budget / splits
    parser.add_argument("--max_metric_calls", type=int, default=5000)
    parser.add_argument("--reflection_minibatch_size", type=int, default=None,
                        help="Number of examples per reflection step. Default 3 "
                             "(GEPA paper). Raise to reduce 3-shot tie-rejection "
                             "on noisy / discrete-judge tasks.")
    parser.add_argument("--train_size", type=int, default=None,
                        help="Override train split size (default: benchmark default)")
    parser.add_argument("--val_size", type=int, default=None,
                        help="Override val split size (default: benchmark default)")

    # Merge ablation (our extensions)
    parser.add_argument("--use_merge", action="store_true")
    parser.add_argument("--max_merge_invocations", type=int, default=15,
                        help="Cap on merge invocations per run. Paper uses 5; "
                             "we raise to 15 because cap=5 binds asymmetrically "
                             "across start policies (immediate hits it fast, "
                             "score_plateau/budget_proportional may not) and "
                             "contaminates the policy ablation.")
    parser.add_argument("--merge_selection", default="random",
                        choices=["random", "divergence", "score", "complementary", "adaptive_diversity"])
    parser.add_argument("--merge_start", default="immediate",
                        choices=["immediate", "delayed", "periodic", "diversity",
                                 "score_plateau", "budget_proportional"])
    parser.add_argument("--merge_start_param", type=float, default=None)
    # Proposal's 3 merge-algorithm variants
    parser.add_argument("--merge_algorithm", default="original",
                        choices=["original", "combine_all", "summarize_before"],
                        help="How to construct the merged candidate per predictor. "
                             "'original' = paper's Algorithm 4; "
                             "'combine_all' = concatenate disputed prompts; "
                             "'summarize_before' = LLM synthesizes unified prompt.")
    parser.add_argument("--merge_lm", default=None,
                        help="LM for summarize_before variant (litellm format). "
                             "If not set, falls back to --reflection_lm.")

    # Behavioral Adaptive Merge (REPORT.md §16). Off by default.
    parser.add_argument("--adaptive_merge_enabled", action="store_true",
                        help="Enable BehavioralAdaptiveMergePolicy + AdaptiveStartPolicy. "
                             "Forces --merge_selection=adaptive_diversity.")
    parser.add_argument("--adaptive_warmup_frac", type=float, default=0.25)
    # CHANGED 2026-05-01: default 5 → 3 (see adaptive_merge.py:51 trace).
    parser.add_argument("--adaptive_min_frontier_size", type=int, default=3)
    # CHANGED 2026-05-02: was action="store_true" (default off) → now defaults
    # ON with --adaptive_no_plateau_gate to disable. See adaptive_merge.py:63
    # for v2 vs v2.6 empirical evidence (plateau ON wins on qwen).
    parser.add_argument("--adaptive_use_plateau_gate", action="store_true", default=True,
                        help="Enable Layer −1 plateau gate (default True).")
    parser.add_argument("--adaptive_no_plateau_gate", dest="adaptive_use_plateau_gate",
                        action="store_false",
                        help="Disable plateau gate (advanced; v2.6 ablation showed this hurts qwen).")
    # ADDED 2026-05-02 for v2.7 ablation: configurable pair-selection strategy
    # under adaptive_merge_enabled. See api.py:adaptive_pair_selection docstring.
    parser.add_argument("--adaptive_pair_selection", type=str, default="adaptive_diversity",
                        choices=["adaptive_diversity", "random", "divergence", "score", "complementary"],
                        help="Pair-selection strategy when adaptive_merge_enabled (default adaptive_diversity).")
    # ADDED 2026-05-02 (v2.8 ablation): bypass Layer 2 routing — always use `original`.
    parser.add_argument("--adaptive_disable_layer2_routing", action="store_true",
                        help="Bypass A1/A2/A3 Layer 2 gates; always use `original` algorithm. Signals still logged.")
    # ADDED 2026-05-02 (v2.9 ablation): A4 default algorithm override.
    parser.add_argument("--adaptive_a4_default_algorithm", type=str, default="original",
                        choices=["original", "combine_all", "summarize_before"],
                        help="Algorithm for A4 fallback (default 'original'; v2.9 tests 'combine_all').")
    parser.add_argument("--adaptive_plateau_window", type=int, default=3)
    parser.add_argument("--adaptive_plateau_eps", type=float, default=0.0)
    parser.add_argument("--adaptive_parent_strength_quantile", type=float, default=0.50)
    # --adaptive_max_example_agreement removed 2026-05-01 (G2 deletion).
    parser.add_argument("--adaptive_use_maturity_gate", action="store_true", default=True,
                        help="Enable G3 (predictor maturity imbalance). Default True.")
    parser.add_argument("--adaptive_no_maturity_gate", dest="adaptive_use_maturity_gate",
                        action="store_false",
                        help="Disable G3 (the most experimental gate).")
    parser.add_argument("--adaptive_maturity_gini_max", type=float, default=0.50)
    parser.add_argument("--adaptive_use_parent_recent_winrate_gate", action="store_true")
    parser.add_argument("--adaptive_parent_recent_winrate_min", type=float, default=0.50)
    parser.add_argument("--adaptive_L_MAX", type=int, default=4000)
    # REMOVED 2026-05-01: --adaptive_BLOAT_SAFE_RATIO. Growth-ratio gate
    # predicted combine_all output length, but policy no longer routes to
    # combine_all. See adaptive_merge.py for full rationale.
    parser.add_argument("--adaptive_specialization_split_threshold", type=float, default=0.30)
    parser.add_argument("--adaptive_duplicate_jaccard_threshold", type=float, default=0.70)
    parser.add_argument("--adaptive_correctness_threshold", type=float, default=0.5)

    # Diagnostics
    parser.add_argument("--merge_quality", action="store_true",
                        help="Enable Tier A merge-quality sidecars.")
    parser.add_argument("--run_dir", required=True,
                        help="Directory for run artifacts (sidecars).")

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Configure DSPy task LM (shared by all modules in the program)
    configure_dspy_lm(args.model)

    import dspy
    # Paper-exact default: reflection LM = task LM (GEPA paper's teacher_lm=None
    # falls back to dspy.settings.lm). Override with --reflection_lm.
    reflection_lm_name = args.reflection_lm or args.model
    # ADDED 2026-05-03: optional reflection-LM output cap for v3_reflect_capped
    # experiment. When --reflection_max_tokens is None (default), no kwarg
    # is passed → behavior is byte-identical to pre-2026-05-03 behavior.
    _reflect_kwargs = {}
    if args.reflection_max_tokens is not None:
        _reflect_kwargs["max_tokens"] = args.reflection_max_tokens
    reflection_lm = _make_lm(reflection_lm_name, **_reflect_kwargs)

    # Load task: DSPy program + dataset + metric + feedback
    program, dataset, metric_fn, feedback_map = load_task(args.task)

    # Apply split-size overrides if requested (benchmark base class trims to 150/300/300 by default)
    if args.train_size is not None:
        dataset.train_set = dataset.train_set[: args.train_size]
    if args.val_size is not None:
        dataset.val_set = dataset.val_set[: args.val_size]

    same_model = (args.reflection_lm is None or args.reflection_lm == args.model)
    print(f"Task:           {args.task}")
    print(f"Task LM:        {args.model}")
    print(f"Reflection LM:  {reflection_lm_name}{' (same as task LM, paper-exact)' if same_model else ''}")
    print(f"Train / Val:    {len(dataset.train_set)} / {len(dataset.val_set)}")
    print(f"Budget:         {args.max_metric_calls}")
    print(f"Merge:          {args.use_merge}  algorithm={args.merge_algorithm}  start={args.merge_start}")
    print(f"Predictors:     {list(extract_seed_candidate(program).keys())}")
    print("-" * 60)

    from gepa.adapters.dspy_adapter.dspy_adapter import DspyAdapter
    import gepa

    adapter = DspyAdapter(
        student_module=program,
        metric_fn=metric_fn,
        feedback_map=feedback_map,
        reflection_lm=reflection_lm,
    )

    seed_candidate = extract_seed_candidate(program)

    os.makedirs(args.run_dir, exist_ok=True)

    # Build merge_lm callable for summarize_before variant
    merge_lm_callable = None
    if args.merge_algorithm == "summarize_before":
        # Resolve merge LM: explicit --merge_lm wins, else reuse resolved reflection LM
        # (which itself defaults to task model). Never None.
        merge_lm_name = args.merge_lm or reflection_lm_name
        # Note: don't override temperature/max_tokens — reasoning models (gpt-5-*)
        # require temperature=1.0 and max_tokens>=16000; DSPy's default config
        # already respects that.
        # Uses _make_lm so Qwen3-family merge LMs also get /no_think injection.
        _merge_lm_obj = _make_lm(merge_lm_name)
        def merge_lm_callable(prompt: str) -> str:
            # dspy.LM.__call__ returns list[str]; take first
            resp = _merge_lm_obj(prompt)
            if isinstance(resp, list):
                return resp[0] if resp else ""
            return str(resp)

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=dataset.train_set,
        valset=dataset.val_set,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        max_metric_calls=args.max_metric_calls,
        use_merge=args.use_merge,
        max_merge_invocations=args.max_merge_invocations if args.use_merge else 0,
        merge_selection_strategy=args.merge_selection,
        merge_start_policy=args.merge_start,
        merge_start_param=args.merge_start_param,
        merge_algorithm=args.merge_algorithm,
        merge_lm=merge_lm_callable,
        merge_quality=args.merge_quality,
        task_name=args.task,
        run_dir=args.run_dir,
        seed=args.seed,
        # Behavioral adaptive merge (off by default)
        adaptive_merge_enabled=args.adaptive_merge_enabled,
        adaptive_warmup_frac=args.adaptive_warmup_frac,
        adaptive_min_frontier_size=args.adaptive_min_frontier_size,
        adaptive_use_plateau_gate=args.adaptive_use_plateau_gate,
        adaptive_pair_selection=args.adaptive_pair_selection,
        adaptive_disable_layer2_routing=args.adaptive_disable_layer2_routing,
        adaptive_a4_default_algorithm=args.adaptive_a4_default_algorithm,
        adaptive_plateau_window=args.adaptive_plateau_window,
        adaptive_plateau_eps=args.adaptive_plateau_eps,
        adaptive_parent_strength_quantile=args.adaptive_parent_strength_quantile,
        # adaptive_max_example_agreement removed (G2 deletion 2026-05-01)
        adaptive_use_maturity_gate=args.adaptive_use_maturity_gate,
        adaptive_maturity_gini_max=args.adaptive_maturity_gini_max,
        adaptive_use_parent_recent_winrate_gate=args.adaptive_use_parent_recent_winrate_gate,
        adaptive_parent_recent_winrate_min=args.adaptive_parent_recent_winrate_min,
        adaptive_L_MAX=args.adaptive_L_MAX,
        # adaptive_BLOAT_SAFE_RATIO passthrough removed (see argparse note above)
        adaptive_specialization_split_threshold=args.adaptive_specialization_split_threshold,
        adaptive_duplicate_jaccard_threshold=args.adaptive_duplicate_jaccard_threshold,
        adaptive_correctness_threshold=args.adaptive_correctness_threshold,
    )

    best = result.best_candidate
    print("\n=== Optimized Candidate ===")
    for k, v in best.items():
        print(f"--- {k} ---")
        print(v)
        print()

    # Persist best candidate to run_dir
    out = {
        "task": args.task,
        "model": args.model,
        "reflection_lm": args.reflection_lm,
        "use_merge": args.use_merge,
        "merge_selection": args.merge_selection,
        "merge_start": args.merge_start,
        "merge_algorithm": args.merge_algorithm,
        "max_metric_calls": args.max_metric_calls,
        "best_candidate": best,
    }
    with open(os.path.join(args.run_dir, "best_candidate.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
