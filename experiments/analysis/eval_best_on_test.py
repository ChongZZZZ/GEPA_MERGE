"""Post-run held-out test evaluation.

For each completed GEPA run (any subdirectory containing both
``best_candidate.json`` and ``gepa_state.bin``), reconstruct the optimized
program from the saved best_candidate prompts and evaluate it on the
benchmark's held-out ``test_set`` (the third split set up by the
``Benchmark`` base class — never touched during optimization).

This is the missing step to make our numbers directly comparable to the
GEPA paper's Table 1 / Table 2 (which report **test** scores). Our
existing ``best_full_val`` numbers are val-set scores with selection
bias.

Usage::

    cd gepa_merge
    PYTHONPATH=src:experiments/vendor/gepa-artifact .venv/bin/python \
        -m experiments.analysis.eval_best_on_test \
        --runs_root P2_result/phase_a_main P3_result/phase_a_main \
                    P4_result/phase_a_main runs/phase_a_main \
        --out_csv experiments/analysis/output/test_eval_v1.csv

Cost: ~$1-2 per gpt-4.1-mini run (300 test × ~4 LLM calls), ~$0.30 per
qwen3-8b run via OpenRouter. Idempotent — skips runs that already have a
``test_eval.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
VENDOR = REPO_ROOT / "experiments" / "vendor" / "gepa-artifact"
for p in (SRC, VENDOR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the inference-config-aware LM constructor (handles /no_think, Prime
# Intellect, JSON adapter quirks etc) from the run_dspy harness.
from experiments.benchmarks.run_dspy import (  # noqa: E402
    _install_compat_shims,
    _make_lm,
    configure_dspy_lm,
    load_task,
)

_install_compat_shims()


def _inject_prompts(program, prompts: dict[str, str]) -> None:
    """Set each predictor's instruction string to the saved best-candidate value."""
    set_count = 0
    for name, pred in program.named_predictors():
        if name in prompts:
            try:
                pred.signature = pred.signature.with_instructions(prompts[name])
            except AttributeError:
                # Older dspy: mutate in place.
                pred.signature.instructions = prompts[name]
            set_count += 1
    return set_count


def _eval_test(program, test_set, metric_fn, num_threads: int = 8) -> dict[str, Any]:
    """Run ``dspy.Evaluate`` on the held-out test set."""
    import dspy

    evaluator = dspy.Evaluate(
        devset=test_set,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=False,
        # Don't abort on parse failures. Set to 10x test set so dspy's internal
        # parallelizer cancellation threshold (which kicks in even when
        # max_errors is set) stays out of the way for fragile models like
        # Qwen3-8B on long-context inputs.
        max_errors=max(10000, len(test_set) * 10),
    )
    result = evaluator(program)
    # New dspy returns an EvaluationResult with `.score`; older returns float.
    score_val = float(result.score) if hasattr(result, "score") else float(result)
    return {"test_score": score_val, "n_test": len(test_set)}


def _process_run(run_dir: Path, force: bool = False, reroute_openai: bool = False,
                 force_openrouter: bool = False, num_threads: int = 8) -> dict[str, Any] | None:
    bc_path = run_dir / "best_candidate.json"
    if not bc_path.exists():
        return None
    out_path = run_dir / "test_eval.json"
    if out_path.exists() and not force:
        prev = json.load(open(out_path))
        prev["run_dir"] = str(run_dir)
        prev["status"] = "cached"
        return prev

    bc = json.load(open(bc_path))
    task = bc.get("task")
    model = bc.get("model")
    reflection_lm = bc.get("reflection_lm") or model
    prompts: dict[str, str] = bc.get("best_candidate") or {}
    if not (task and model and prompts):
        return {"run_dir": str(run_dir), "status": "skipped_missing_fields"}

    # P1's run harness double-prefixed the provider for some cells →
    # `openai/openai/X`. Strip the duplicate before routing.
    if model.startswith("openai/openai/"):
        model = model[len("openai/"):]

    # `--force_openrouter`: route everything through OpenRouter so a single
    # OR key can serve cells that were originally optimized via direct OpenAI.
    if force_openrouter and model.startswith("openai/"):
        model = "openrouter/" + model

    # P2 ran with `openai/gpt-4.1-mini` (direct OpenAI). To re-evaluate with
    # only an OpenRouter key, rewrite to `openrouter/openai/gpt-4.1-mini`.
    if reroute_openai and model.startswith("openai/"):
        model = "openrouter/" + model

    # Configure DSPy task LM with the same inference settings as the run.
    configure_dspy_lm(model)

    program, dataset, metric_fn, _feedback_map = load_task(task)
    n_set = _inject_prompts(program, prompts)
    if n_set != len(prompts):
        # Predictor names didn't all line up — abort this run cleanly.
        return {
            "run_dir": str(run_dir),
            "status": "predictor_mismatch",
            "expected": list(prompts.keys()),
            "found": [n for n, _ in program.named_predictors()],
        }

    test_set = dataset.test_set
    if not test_set:
        return {"run_dir": str(run_dir), "status": "no_test_set"}

    print(f"  → eval {task} × {model.split('/')[-1]} on {len(test_set)} test examples "
          f"(num_threads={num_threads})...",
          file=sys.stderr, flush=True)
    result = _eval_test(program, test_set, metric_fn, num_threads=num_threads)

    record = {
        "run_dir": str(run_dir),
        "task": task,
        "model": model,
        "reflection_lm": reflection_lm,
        "config": {
            "use_merge": bc.get("use_merge"),
            "merge_algorithm": bc.get("merge_algorithm"),
            "merge_start": bc.get("merge_start"),
        },
        **result,
        "status": "ok",
    }
    with out_path.open("w") as f:
        json.dump(record, f, indent=2)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_root", nargs="+", required=True, type=Path)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--force", action="store_true",
                    help="Re-evaluate runs that already have test_eval.json")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N runs (smoke-test).")
    ap.add_argument("--reroute_openai", action="store_true",
                    help="Rewrite 'openai/X' → 'openrouter/openai/X' so an "
                         "OpenRouter key can serve P2's GPT runs.")
    ap.add_argument("--force_openrouter", action="store_true",
                    help="Same as --reroute_openai but kept distinct: route every "
                         "openai/* model via OR (used when only OR balance is available).")
    ap.add_argument("--num_threads", type=int, default=8,
                    help="dspy.Evaluate parallelism. Lower (e.g. 2) for fragile "
                         "models like Qwen3-8B on long-context tasks where "
                         "concurrent parse failures trigger parallelizer cancellation.")
    args = ap.parse_args()

    run_dirs: list[Path] = []
    for root in args.runs_root:
        if not root.exists():
            print(f"[skip] {root} does not exist", file=sys.stderr)
            continue
        for bc in sorted(root.rglob("best_candidate.json")):
            run_dirs.append(bc.parent)
    if args.limit:
        run_dirs = run_dirs[: args.limit]

    print(f"Found {len(run_dirs)} runs to evaluate.", file=sys.stderr)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for i, rd in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] {rd}", file=sys.stderr)
        try:
            rec = _process_run(rd, force=args.force,
                               reroute_openai=args.reroute_openai,
                               force_openrouter=args.force_openrouter,
                               num_threads=args.num_threads)
        except Exception as e:
            rec = {"run_dir": str(rd), "status": "error", "error": repr(e)}
            print(f"   error: {e!r}", file=sys.stderr)
        if rec is not None:
            rows.append(rec)

    # Flat CSV
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"Wrote {len(rows)} rows → {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
