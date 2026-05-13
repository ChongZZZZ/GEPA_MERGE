"""Post-hoc replay of recorded merge events through the BehavioralAdaptiveMergePolicy.

Read-only. No LM calls. Walks `runs_root/<benchmark>/<config>_s<seed>/{merge_quality.jsonl,gepa_state.bin}`,
loads each cell once, and for every recorded merge event asks the new
adaptive policy what it would have decided at that triplet.

Outputs:
  - <out>/adaptive_replay_decisions.csv  — one row per merge event
  - <out>/adaptive_replay_summary.md     — aggregate report

Replay caveats:
  - We replay against the *final* gepa_state.bin (after the run finished).
    The active-frontier median used by G1 is therefore the post-run frontier,
    not the at-time frontier. This may slightly over- or under-skip early
    merges; the live policy will use the at-time frontier exactly.
  - Layer −1 (AdaptiveStartPolicy) is not replayed: warmup / frontier-size /
    plateau decisions depend on at-time iteration counts that aren't
    reconstructable from the saved state. Those gates are validated live
    in the 5-cell rerun.

Usage::

    cd gepa_merge
    PYTHONPATH=src .venv/bin/python -m experiments.analysis.adaptive_replay \\
        --runs_root P2_result/phase_a_main P3_result/phase_a_main \\
                    P4_result/phase_a_main runs/phase_a_main_qwen \\
        --out experiments/analysis/output/adaptive_replay_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gepa.strategies.adaptive_merge import (  # noqa: E402
    AdaptiveMergeConfig,
    BehavioralAdaptiveMergePolicy,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _load_state_dict(run_dir: Path) -> dict[str, Any] | None:
    """Direct pickle.load of gepa_state.bin (no GEPAState reconstruction).

    Same approach as merge_forensics.py — we only need the data fields,
    not dspy program reconstruction.
    """
    sb = run_dir / "gepa_state.bin"
    if not sb.exists():
        return None
    try:
        with sb.open("rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] failed to load {sb}: {e}", file=sys.stderr)
        return None


def _state_namespace_from_dict(state_dict: dict[str, Any]) -> SimpleNamespace:
    """Wrap the pickled state dict in a SimpleNamespace exposing the attributes
    BehavioralAdaptiveMergePolicy reads. Read-only — we never mutate."""
    progs = state_dict.get("program_candidates", [])
    parents = state_dict.get("parent_program_for_candidate", [])
    val_subscores = state_dict.get("prog_candidate_val_subscores", [])

    # Aggregate scores per candidate (mean over the val set), matching
    # GEPAState.program_full_scores_val_set. Coerce booleans to 0/1.
    scores: list[float] = []
    for sub in val_subscores:
        if not isinstance(sub, dict) or not sub:
            scores.append(0.0)
            continue
        scores.append(sum(float(v) for v in sub.values()) / len(sub))

    pareto_mapping = state_dict.get("program_at_pareto_front_valset", {})
    full_program_trace = state_dict.get("full_program_trace", [])

    return SimpleNamespace(
        program_candidates=progs,
        parent_program_for_candidate=parents,
        prog_candidate_val_subscores=val_subscores,
        program_full_scores_val_set=scores,
        per_program_tracked_scores=scores,
        full_program_trace=full_program_trace,
        get_pareto_front_mapping=lambda: pareto_mapping,
    )


# ---------------------------------------------------------------------------
# Per-cell replay
# ---------------------------------------------------------------------------


def _parse_cell_dirname(name: str) -> tuple[str, str, int]:
    """Parse e.g. 'combine_all_immediate_s0' → ('combine_all', 'immediate', 0).

    Falls back gracefully on non-standard names.
    """
    seed = -1
    if "_s" in name:
        parts = name.rsplit("_s", 1)
        try:
            seed = int(parts[1])
            name = parts[0]
        except ValueError:
            pass
    # Try splitting on known start policy names
    start_policies = (
        "score_plateau", "budget_proportional", "immediate", "delayed",
        "periodic", "diversity",
    )
    for sp in start_policies:
        if name.endswith("_" + sp):
            algo = name[: -(len(sp) + 1)]
            return algo, sp, seed
    if name == "nomerge":
        return "nomerge", "n/a", seed
    return name, "n/a", seed


def _detect_model(runs_root: Path) -> str:
    """Heuristic model name from the runs_root path."""
    parts = runs_root.parts
    name = runs_root.name.lower()
    if "qwen" in name:
        return "qwen3-8b"
    if "gpt" in name:
        return "gpt-4.1-mini"
    # Fall back: search whole path
    full = "/".join(parts).lower()
    if "qwen" in full:
        return "qwen3-8b"
    return "gpt-4.1-mini"


def _replay_event(
    policy: BehavioralAdaptiveMergePolicy,
    state_ns: SimpleNamespace,
    event: dict[str, Any],
    state_dict: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    parent_ids = event.get("parent_ids")
    ancestor = event.get("ancestor_idx")
    if not (isinstance(parent_ids, list) and len(parent_ids) == 2 and isinstance(ancestor, int)):
        return None
    id1, id2 = sorted(parent_ids)
    n_progs = len(state_ns.program_candidates)
    if not (0 <= id1 < n_progs and 0 <= id2 < n_progs and 0 <= ancestor < n_progs):
        return None

    # Layer 1 frontier reference: approximate the at-time frontier by filtering
    # to candidates that existed when this merge fired. We use new_candidate_idx
    # as the cutoff (the merged candidate's slot, i.e., everything below it
    # already existed). Falls back to all candidates if missing.
    new_idx = event.get("new_candidate_idx")
    cutoff = (new_idx if isinstance(new_idx, int) and new_idx > 0
              else max(id1, id2, ancestor) + 1)
    cutoff = max(cutoff, max(id1, id2, ancestor) + 1)

    # At-time scores
    full_scores = list(state_ns.program_full_scores_val_set)
    at_time_scores = full_scores[:cutoff]

    # At-time pareto mapping: keep only candidates < cutoff in each front.
    if state_dict is not None:
        pareto_full = state_dict.get("program_at_pareto_front_valset", {})
        pareto_at_time = {
            k: {i for i in front if i < cutoff}
            for k, front in pareto_full.items()
        }
        pareto_at_time = {k: v for k, v in pareto_at_time.items() if v}
    else:
        pareto_at_time = {}

    # Compute at-time dominators (this matches what the live policy sees).
    if pareto_at_time and at_time_scores:
        from gepa.gepa_utils import find_dominator_programs  # local import
        try:
            frontier = find_dominator_programs(pareto_at_time, at_time_scores)
        except Exception:
            frontier = list(range(cutoff))
    else:
        frontier = list(range(cutoff))
    frontier_scores_for_median = [
        full_scores[i] for i in frontier if 0 <= i < len(full_scores)
    ]

    scores = full_scores  # the policy uses full state scores for the pair itself
    from gepa.strategies.adaptive_merge import _quantile  # local import; pure helper
    ref_score = (
        _quantile(frontier_scores_for_median, policy.config.parent_strength_quantile)
        if frontier_scores_for_median else 0.0
    )
    layer1 = policy._evaluate_layer1(state_ns, id1, id2, scores, ref_score)
    policy._pair_cache[(id1, id2)] = layer1

    # Layer 2 (only meaningful if layer1 passed; we record what it would have
    # been so the report can show "of the events the gate skipped, here's
    # which algo we would have routed them to had they passed")
    algo = policy.algorithm_for_pair(state_ns, ancestor, id1, id2)
    decision = policy.get_cached_decision(ancestor, id1, id2)

    return {
        "parent_a": id1,
        "parent_b": id2,
        "ancestor": ancestor,
        "iteration": event.get("iteration"),
        "actual_event": event.get("event"),
        "actual_full_val_lift": event.get("full_val_lift_over_best_parent"),
        "G1_pass": layer1.signals.get("score_a", 0) >= ref_score and layer1.signals.get("score_b", 0) >= ref_score,
        "G1_score_a": layer1.signals.get("score_a"),
        "G1_score_b": layer1.signals.get("score_b"),
        "G1_ref_score": ref_score,
        "G2_agreement": layer1.signals.get("example_agreement"),
        # G2 was removed from BehavioralAdaptiveMergePolicy 2026-05-01.
        # The example_agreement signal is still computed for observability
        # (it's used for ranking and Layer 2 A2). Always True now since the
        # gate no longer exists. Kept in CSV for backward compat with prior
        # replay outputs that had this column.
        "G2_pass": True,
        "G3_gini": layer1.signals.get("maturity_gini"),
        "G3_unavailable": bool(layer1.signals.get("maturity_gini_unavailable")),
        "G3_pass": (
            (not policy.config.use_maturity_gate)
            or layer1.signals.get("maturity_gini") is None
            or (layer1.signals.get("maturity_gini") or 0.0) <= policy.config.maturity_gini_max
        ),
        "layer1_passed": layer1.passed,
        "layer1_skip_reason": layer1.skip_reason,
        "behavioral_diversity": layer1.behavioral_diversity,
        "selected_algo": decision.algorithm if decision else None,
        "algo_reason": decision.algorithm_reason if decision else None,
        "max_pred_tokens_a": (decision.signals or {}).get("max_predictor_tokens_a") if decision else None,
        "max_pred_tokens_b": (decision.signals or {}).get("max_predictor_tokens_b") if decision else None,
        "predicted_growth_ratio": (decision.signals or {}).get("predicted_growth_ratio") if decision else None,
        "specialization_split": (decision.signals or {}).get("specialization_split") if decision else None,
        "token_jaccard": (decision.signals or {}).get("token_jaccard") if decision else None,
        "final_decision": (
            "skip" if not layer1.passed else (decision.algorithm if decision else "unknown")
        ),
    }


def _build_policies(args) -> dict[str, BehavioralAdaptiveMergePolicy]:
    """Build one policy per gate-toggle setting we want to compare.

    Returns dict: variant_name → policy.
    """
    base = dict(
        warmup_frac=args.warmup_frac,
        min_frontier_size=args.min_frontier_size,
        parent_strength_quantile=args.parent_strength_quantile,
        # max_example_agreement removed (G2 deletion 2026-05-01)
        maturity_gini_max=args.maturity_gini_max,
        L_MAX=args.L_MAX,
        # BLOAT_SAFE_RATIO removed (growth-ratio gate deletion 2026-05-01)
        specialization_split_threshold=args.specialization_split_threshold,
        duplicate_jaccard_threshold=args.duplicate_jaccard_threshold,
        correctness_threshold=args.correctness_threshold,
    )
    return {
        "G3_on": BehavioralAdaptiveMergePolicy(
            AdaptiveMergeConfig(adaptive_merge_enabled=True, use_maturity_gate=True, **base)
        ),
        "G3_off": BehavioralAdaptiveMergePolicy(
            AdaptiveMergeConfig(adaptive_merge_enabled=True, use_maturity_gate=False, **base)
        ),
    }


def _walk_cells(runs_roots: list[Path]) -> Iterable[tuple[Path, Path, str, str, str, int]]:
    """Yield (run_dir, mq_path, model, benchmark, config, seed) for every cell
    that has both gepa_state.bin and merge_quality.jsonl.
    """
    for runs_root in runs_roots:
        if not runs_root.exists():
            print(f"[warn] runs_root does not exist: {runs_root}", file=sys.stderr)
            continue
        model = _detect_model(runs_root)
        for benchmark_dir in sorted(runs_root.iterdir()):
            if not benchmark_dir.is_dir():
                continue
            for cell_dir in sorted(benchmark_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                mq = cell_dir / "merge_quality.jsonl"
                gs = cell_dir / "gepa_state.bin"
                if not mq.exists() or not gs.exists():
                    continue
                algo, start, seed = _parse_cell_dirname(cell_dir.name)
                config = f"{algo}_{start}"
                yield cell_dir, mq, model, benchmark_dir.name, config, seed


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summarize(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        return {"variant": variant, "total_events": 0}

    total = len(rows)
    skips = sum(1 for r in rows if r["final_decision"] == "skip")
    skip_reasons = Counter(r["layer1_skip_reason"] for r in rows if r["final_decision"] == "skip")
    algo_counts = Counter(
        r["final_decision"] for r in rows if r["final_decision"] != "skip"
    )
    algo_reasons = Counter(
        r["algo_reason"] for r in rows if r["final_decision"] != "skip"
    )
    g3_unavailable = sum(1 for r in rows if r.get("G3_unavailable"))

    accepted = [r for r in rows if r["actual_event"] == "accepted"]
    rejected = [r for r in rows if r["actual_event"] == "rejected"]
    accepted_skipped = sum(1 for r in accepted if r["final_decision"] == "skip")
    rejected_skipped = sum(1 for r in rejected if r["final_decision"] == "skip")

    # "Catastrophic" = strongly-negative full_val_lift
    catastrophic = [
        r for r in rows
        if isinstance(r.get("actual_full_val_lift"), (int, float))
        and r["actual_full_val_lift"] <= -0.05  # 5pp drop or worse
    ]
    catastrophic_skipped = sum(1 for r in catastrophic if r["final_decision"] == "skip")
    # "Saved" = either skipped OR re-routed by Layer 2 to a different algorithm
    # than the actual cell used. We don't have the cell's run-time algo on the
    # row, but if Layer 2 chose summarize_before for a known-catastrophic
    # combine_all event, that's a save.
    catastrophic_routed_to_summarize = sum(
        1 for r in catastrophic
        if r["final_decision"] == "summarize_before"
    )
    catastrophic_addressed = catastrophic_skipped + catastrophic_routed_to_summarize

    return {
        "variant": variant,
        "total_events": total,
        "skips": skips,
        "skip_rate": skips / total if total else 0,
        "skip_reasons": dict(skip_reasons),
        "algo_counts": dict(algo_counts),
        "algo_reasons": dict(algo_reasons),
        "g3_unavailable": g3_unavailable,
        "actual_accepted": len(accepted),
        "actual_rejected": len(rejected),
        "accepted_skipped_by_policy": accepted_skipped,
        "rejected_skipped_by_policy": rejected_skipped,
        "catastrophic_events": len(catastrophic),
        "catastrophic_skipped": catastrophic_skipped,
        "catastrophic_routed_to_summarize": catastrophic_routed_to_summarize,
        "catastrophic_addressed": catastrophic_addressed,
    }


def _summary_md(summaries: list[dict[str, Any]], n_cells: int, n_runs_roots: int) -> str:
    lines = []
    lines.append("# Adaptive Merge — Post-Hoc Replay Summary")
    lines.append("")
    lines.append(f"Cells replayed: **{n_cells}** across {n_runs_roots} run root(s).")
    lines.append("")
    lines.append("## Variant comparison (G3 on vs. off)")
    lines.append("")
    lines.append("| metric | " + " | ".join(s["variant"] for s in summaries) + " |")
    lines.append("|---|" + "|".join(["---"] * len(summaries)) + "|")
    metrics_order = [
        "total_events", "skips", "skip_rate", "g3_unavailable",
        "actual_accepted", "actual_rejected",
        "accepted_skipped_by_policy", "rejected_skipped_by_policy",
        "catastrophic_events", "catastrophic_skipped",
        "catastrophic_routed_to_summarize", "catastrophic_addressed",
    ]
    for m in metrics_order:
        row = [m]
        for s in summaries:
            v = s.get(m, "—")
            if isinstance(v, float):
                row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for s in summaries:
        lines.append(f"### {s['variant']}")
        lines.append("")
        lines.append("**Skip reasons (Layer 1):**")
        for k, v in sorted(s.get("skip_reasons", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"  - `{k}`: {v}")
        lines.append("")
        lines.append("**Layer 2 algorithm choices (when not skipped):**")
        for k, v in sorted(s.get("algo_counts", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"  - `{k}`: {v}")
        lines.append("")
        lines.append("**Layer 2 reasons (top):**")
        for k, v in sorted(s.get("algo_reasons", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"  - `{k}`: {v}")
        lines.append("")

    lines.append("## Catastrophe coverage")
    lines.append("")
    lines.append(
        "Definition: a 'catastrophic' event has `full_val_lift_over_best_parent <= -0.05` "
        "(at least a 5 pp regression on the full val set). 'Addressed' = either Layer 1 "
        "skipped the event OR Layer 2 routed it to `summarize_before` (the safe-fallback "
        "algorithm). The headline number is `catastrophic_addressed / catastrophic_events`."
    )
    lines.append("")
    for s in summaries:
        cat = s.get("catastrophic_events", 0)
        skp = s.get("catastrophic_skipped", 0)
        routed = s.get("catastrophic_routed_to_summarize", 0)
        addressed = s.get("catastrophic_addressed", 0)
        rate = (addressed / cat) if cat else 0.0
        lines.append(
            f"- **{s['variant']}**: {addressed}/{cat} addressed ({rate:.0%}) — "
            f"{skp} skipped at Layer 1, {routed} re-routed to summarize_before at Layer 2."
        )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Replay uses the *final* state's Pareto frontier and aggregate scores. "
        "Live policy will use at-time frontier (potentially smaller, lower median).\n"
        "- AdaptiveStartPolicy (Layer −1) is not replayed; warmup/frontier-size/plateau "
        "decisions need at-time iteration counts.\n"
        "- Catastrophic-event detection threshold (-0.05) is heuristic; tweak via "
        "downstream analysis if needed."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--runs_root", nargs="+", required=True, type=Path,
                   help="One or more run-root directories (each containing benchmark/cell subdirs).")
    p.add_argument("--out", required=True, type=Path,
                   help="Output directory for CSV + summary.")
    # Threshold defaults match AdaptiveMergeConfig
    p.add_argument("--warmup_frac", type=float, default=0.25)
    # CHANGED 2026-05-01: 5 → 3 (matches adaptive_merge.py default change).
    p.add_argument("--min_frontier_size", type=int, default=3)
    p.add_argument("--parent_strength_quantile", type=float, default=0.50)
    # --max_example_agreement removed 2026-05-01 (G2 deletion)
    p.add_argument("--maturity_gini_max", type=float, default=0.50)
    p.add_argument("--L_MAX", type=int, default=4000)
    # --BLOAT_SAFE_RATIO removed 2026-05-01 (growth-ratio gate deletion;
    # the predicted-combine_all-length signal is still computed for
    # observability but no longer gates routing).
    p.add_argument("--specialization_split_threshold", type=float, default=0.30)
    p.add_argument("--duplicate_jaccard_threshold", type=float, default=0.70)
    p.add_argument("--correctness_threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "adaptive_replay_decisions.csv"
    md_path = args.out / "adaptive_replay_summary.md"

    policies = _build_policies(args)

    fieldnames = [
        "model", "benchmark", "config", "seed", "cell_id",
        "iteration", "parent_a", "parent_b", "ancestor",
        "actual_event", "actual_full_val_lift",
        "variant",
        "G1_pass", "G1_score_a", "G1_score_b", "G1_ref_score",
        "G2_agreement", "G2_pass",
        "G3_gini", "G3_unavailable", "G3_pass",
        "layer1_passed", "layer1_skip_reason",
        "behavioral_diversity",
        "selected_algo", "algo_reason",
        "max_pred_tokens_a", "max_pred_tokens_b",
        "predicted_growth_ratio", "specialization_split", "token_jaccard",
        "final_decision",
    ]

    all_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    n_cells = 0
    n_events = 0
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cell_dir, mq_path, model, benchmark, config, seed in _walk_cells(args.runs_root):
            state_dict = _load_state_dict(cell_dir)
            if state_dict is None:
                continue
            state_ns = _state_namespace_from_dict(state_dict)
            n_cells += 1
            cell_id = f"{model}/{benchmark}/{config}_s{seed}"
            for event in _read_jsonl(mq_path):
                # Skip "attempted" events that have no parent_ids (rare logging glitches)
                if event.get("event") not in {"accepted", "rejected", "attempted"}:
                    continue
                # Skip events without a paired (rejected) outcome — replay each unique attempt once.
                # We replay BOTH accepted and rejected so the report can compare actual vs. policy.
                for variant, policy in policies.items():
                    # Reset cache per event so replays don't leak across triplets
                    policy._pair_cache.clear()
                    policy._decisions.clear()
                    decision = _replay_event(policy, state_ns, event, state_dict)
                    if decision is None:
                        continue
                    row = {
                        "model": model,
                        "benchmark": benchmark,
                        "config": config,
                        "seed": seed,
                        "cell_id": cell_id,
                        "variant": variant,
                        **decision,
                    }
                    writer.writerow(row)
                    all_rows[variant].append(decision)
                    n_events += 1

    summaries = [_summarize(all_rows[v], v) for v in sorted(all_rows.keys())]
    md_path.write_text(_summary_md(summaries, n_cells, len(args.runs_root)))

    # Also write summaries as JSON for downstream tools
    (args.out / "adaptive_replay_summary.json").write_text(
        json.dumps(summaries, indent=2)
    )

    print(f"replayed {n_cells} cells; wrote {n_events} decision rows to {csv_path}")
    print(f"summary: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
