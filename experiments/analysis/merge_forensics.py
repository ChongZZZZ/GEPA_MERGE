"""Five-step merge forensics — enrich merge_quality.jsonl with derived metrics.

Pipeline (each step gated by a CLI flag; default = Steps 1-3 only, no
external deps, no API cost):

    Step 1 — Performance scope    (just extracts existing full_val_lift_*)
    Step 2 — Lexical scope        (just extracts existing tier_a fields)
    Step 3 — Lineage scope        (parent_generation_depth, is_merge_of_merge,
                                   noop_predictor_rate) — derived from
                                   gepa_state.bin's parent_program_for_candidate
    Step 4 — Semantic scope       (SNR + SLC) — needs sentence-transformers
    Step 5 — Consistency scope    (LLM judge on internal_consistency axis) —
                                   needs OpenRouter (or any LiteLLM-supported)
                                   key

Output: one row per merge attempt across all runs, written as JSONL +
flattened CSV. Columns are stable so downstream pandas / seaborn analysis
is one-line.

Usage::

    cd gepa_merge
    PYTHONPATH=src .venv/bin/python -m experiments.analysis.merge_forensics \\
        --runs_root P2_result/phase_a_main runs/phase_a_main \\
        --out experiments/analysis/output/forensics_v1 \\
        # Optional:
        --semantic                              # Step 4 (~5 min on 200 merges)
        --consistency --judge_lm openrouter/openai/gpt-4.1-mini  # Step 5 (~$0.10)
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Make src/ importable when run as `python -m experiments.analysis.merge_forensics`
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gepa.strategies.merge_quality import split_sentences  # noqa: E402


# ---------------------------------------------------------------------------
# Step 3 — Lineage metrics (no external deps)
# ---------------------------------------------------------------------------


def parent_generation_depth(parents_table: list[list[int | None]], idx: int) -> int:
    """Return the longest lineage depth from candidate ``idx`` back to a seed
    (parents == [None]). For merged candidates we follow both parent edges and
    take the max+1."""
    cache: dict[int, int] = {}

    def _depth(i: int) -> int:
        if i in cache:
            return cache[i]
        ps = parents_table[i] if 0 <= i < len(parents_table) else [None]
        if ps == [None] or not ps or all(p is None for p in ps):
            cache[i] = 0
            return 0
        depths = [_depth(p) for p in ps if p is not None]
        cache[i] = (max(depths) if depths else 0) + 1
        return cache[i]

    return _depth(idx)


def is_merge_of_merge(parents_table: list[list[int | None]], p1_idx: int, p2_idx: int) -> bool:
    """True iff at least one of the two parents was itself produced by a merge
    (i.e. its own parents list has length >= 2). Captures the 'stacked merge'
    pattern that may correlate with negative lift."""
    def _was_merged(i: int) -> bool:
        if not (0 <= i < len(parents_table)):
            return False
        ps = parents_table[i]
        return len([p for p in ps if p is not None]) >= 2

    return _was_merged(p1_idx) or _was_merged(p2_idx)


def noop_predictor_rate(provenance_map_sentence: dict[str, list[str]]) -> float:
    """Fraction of predictors whose merged text is sentence-by-sentence
    'tie' (== both parents identical at every sentence). A high rate means
    'merge had nothing to do' on most of the program."""
    if not provenance_map_sentence:
        return 0.0
    n_noop = sum(
        1 for labels in provenance_map_sentence.values()
        if labels and all(lab == "tie" for lab in labels)
    )
    return n_noop / len(provenance_map_sentence)


# ---------------------------------------------------------------------------
# Step 4 — Semantic scope (lazy-load sentence-transformers)
# ---------------------------------------------------------------------------


_EMBED_MODEL = None


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2"):
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def _flatten_predictor_dict(prompt: dict[str, str]) -> list[str]:
    """Concat all predictor texts -> sentence list (deduped order-preserving)."""
    sents: list[str] = []
    seen: set[str] = set()
    for v in prompt.values():
        for s in split_sentences(v):
            if s not in seen:
                seen.add(s)
                sents.append(s)
    return sents


def semantic_metrics(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    novel_threshold: float = 0.7,
    loss_threshold: float = 0.5,
) -> dict[str, Any]:
    """SNR (semantic novelty rate) + SLC (semantic loss count)."""
    import numpy as np

    m_sents = _flatten_predictor_dict(merged)
    p1_sents = _flatten_predictor_dict(p1)
    p2_sents = _flatten_predictor_dict(p2)
    parent_sents = list({s: None for s in (p1_sents + p2_sents)}.keys())

    if not m_sents or not parent_sents:
        return {
            "snr_semantic_novelty_rate": None,
            "slc_lost_count_p1": None,
            "slc_lost_count_p2": None,
            "slc_lost_count_max": None,
            "n_merged_sentences": len(m_sents),
            "n_p1_sentences": len(p1_sents),
            "n_p2_sentences": len(p2_sents),
        }

    model = _get_embed_model()
    m_emb = model.encode(m_sents, normalize_embeddings=True, show_progress_bar=False)
    p_emb = model.encode(parent_sents, normalize_embeddings=True, show_progress_bar=False)
    sim_m_to_p = m_emb @ p_emb.T  # (n_m, n_parent)
    max_sim_per_m = sim_m_to_p.max(axis=1) if sim_m_to_p.size else np.array([])
    snr = float((max_sim_per_m < novel_threshold).mean()) if len(max_sim_per_m) else 0.0

    def _lost_count(parent_sents_one: list[str]) -> int:
        if not parent_sents_one:
            return 0
        e = model.encode(parent_sents_one, normalize_embeddings=True, show_progress_bar=False)
        sim_p_to_m = e @ m_emb.T  # (n_p, n_m)
        max_sim_per_p = sim_p_to_m.max(axis=1) if sim_p_to_m.size else np.array([])
        return int((max_sim_per_p < loss_threshold).sum()) if len(max_sim_per_p) else 0

    slc_p1 = _lost_count(p1_sents)
    slc_p2 = _lost_count(p2_sents)

    return {
        "snr_semantic_novelty_rate": snr,
        "slc_lost_count_p1": slc_p1,
        "slc_lost_count_p2": slc_p2,
        "slc_lost_count_max": max(slc_p1, slc_p2),
        "n_merged_sentences": len(m_sents),
        "n_p1_sentences": len(p1_sents),
        "n_p2_sentences": len(p2_sents),
    }


# ---------------------------------------------------------------------------
# Step 5 — Consistency scope (LLM judge)
# ---------------------------------------------------------------------------


def _build_judge_lm(model_name: str):
    """Return a LiteLLM-backed LanguageModel callable for `score_rubric`."""
    import litellm

    def _judge(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        resp = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""
    return _judge


def consistency_metrics(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    judge_lm,
) -> dict[str, Any]:
    """Run the existing rubric judge and extract the internal_consistency axis.

    The merged / parent prompts are flattened predictor-by-predictor so the
    judge sees the full optimization-target text in one go (matches what
    `merge_quality_judge` does internally)."""
    from gepa.strategies.merge_quality_judge import score_rubric

    def _flat(d: dict[str, str]) -> str:
        return "\n\n---\n\n".join(f"[{k}]\n{v}" for k, v in d.items())

    out = score_rubric(judge_lm, _flat(merged), _flat(p1), _flat(p2))
    scores = out.get("scores", {}) or {}
    return {
        "judge_clarity": scores.get("clarity"),
        "judge_specificity": scores.get("specificity"),
        "judge_internal_consistency": scores.get("internal_consistency"),
        "judge_coverage_vs_parents": scores.get("coverage_vs_parents"),
        "judge_contradiction_present": (out.get("contradiction") or {}).get("present"),
        "judge_contradiction_span": (out.get("contradiction") or {}).get("span"),
    }


# ---------------------------------------------------------------------------
# Run loader + per-event enrichment
# ---------------------------------------------------------------------------


def _load_state(run_dir: Path) -> dict[str, Any] | None:
    sb = run_dir / "gepa_state.bin"
    if not sb.exists():
        return None
    with sb.open("rb") as f:
        return pickle.load(f)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def enrich_run(
    run_dir: Path,
    do_semantic: bool,
    judge_lm=None,
    judge_only_negative_late_gen: bool = True,
) -> list[dict[str, Any]]:
    """Read one run's merge_quality.jsonl + gepa_state.bin and emit enriched
    records (one per merge attempt). If ``judge_lm`` is provided, optionally
    restrict consistency scoring to negative-lift late-generation merges
    (Step 5 cost control)."""
    mq = run_dir / "merge_quality.jsonl"
    if not mq.exists() or mq.stat().st_size == 0:
        return []
    state = _load_state(run_dir)
    if state is None:
        return []

    parents_table: list[list[int | None]] = state.get("parent_program_for_candidate", [])
    program_candidates: list[dict[str, str]] = state.get("program_candidates", [])

    out_records: list[dict[str, Any]] = []
    for ev in _read_jsonl(mq):
        rec: dict[str, Any] = {
            # Identity
            "run_dir": str(run_dir),
            "run_id": ev.get("run_id"),
            "iteration": ev.get("iteration"),
            "event": ev.get("event"),
            "p1_idx": ev.get("parent_ids", [None, None])[0],
            "p2_idx": ev.get("parent_ids", [None, None])[1],
            "anc_idx": ev.get("ancestor_idx"),
            "new_idx": ev.get("new_candidate_idx"),
            # Step 1: performance
            "lift_full_val": ev.get("full_val_lift_over_best_parent"),
            # Step 2: lexical (extracted from tier_a)
        }
        ta = ev.get("tier_a", {}) or {}
        rec.update({
            "sentence_entropy": ta.get("sentence_provenance_entropy"),
            "predictor_entropy": ta.get("predictor_provenance_entropy"),
            "coverage_p1": (ta.get("content_coverage_fraction") or {}).get("coverage_p1"),
            "coverage_p2": (ta.get("content_coverage_fraction") or {}).get("coverage_p2"),
            "coverage_min": (ta.get("content_coverage_fraction") or {}).get("coverage_min"),
            "novelty_fraction": ta.get("novelty_fraction"),
            "behavioral_delta_rate": ta.get("behavioral_delta_rate"),
        })
        # length_delta total across predictors vs ancestor (compactly)
        ld = ta.get("length_delta") or {}
        rec["length_delta_vs_anc_total"] = sum(
            (per.get("vs_anc", 0) or 0) for per in ld.values()
        ) if isinstance(ld, dict) else 0
        # subsample profile
        srp = ta.get("subsample_rank_profile") or {}
        rec.update({
            "subsample_win": srp.get("win"),
            "subsample_tie": srp.get("tie"),
            "subsample_loss": srp.get("loss"),
        })

        # Step 3: lineage
        p1, p2 = rec["p1_idx"], rec["p2_idx"]
        rec["parent_gen_depth_p1"] = parent_generation_depth(parents_table, p1) if p1 is not None else None
        rec["parent_gen_depth_p2"] = parent_generation_depth(parents_table, p2) if p2 is not None else None
        rec["parent_gen_depth_max"] = max(
            (d for d in [rec["parent_gen_depth_p1"], rec["parent_gen_depth_p2"]] if d is not None),
            default=None,
        )
        rec["is_merge_of_merge"] = (
            is_merge_of_merge(parents_table, p1, p2)
            if p1 is not None and p2 is not None else None
        )
        rec["noop_predictor_rate"] = noop_predictor_rate(ta.get("provenance_map_sentence") or {})

        # Step 4: semantic (lazy)
        if do_semantic and rec["new_idx"] is not None:
            try:
                merged_prompt = program_candidates[rec["new_idx"]]
                p1_prompt = program_candidates[p1] if p1 is not None else {}
                p2_prompt = program_candidates[p2] if p2 is not None else {}
                rec.update(semantic_metrics(merged_prompt, p1_prompt, p2_prompt))
            except (IndexError, KeyError) as e:
                rec["semantic_error"] = repr(e)

        # Step 5: consistency (judge LM)
        if judge_lm is not None and rec["new_idx"] is not None:
            should_judge = True
            if judge_only_negative_late_gen:
                should_judge = (
                    (rec["lift_full_val"] is not None and rec["lift_full_val"] <= 0.0)
                    and (rec["parent_gen_depth_max"] is not None and rec["parent_gen_depth_max"] >= 2)
                )
            if should_judge:
                try:
                    merged_prompt = program_candidates[rec["new_idx"]]
                    p1_prompt = program_candidates[p1] if p1 is not None else {}
                    p2_prompt = program_candidates[p2] if p2 is not None else {}
                    rec.update(consistency_metrics(merged_prompt, p1_prompt, p2_prompt, judge_lm))
                except Exception as e:
                    rec["judge_error"] = repr(e)

        out_records.append(rec)
    return out_records


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _iter_run_dirs(roots: list[Path]) -> Iterable[Path]:
    """Yield every directory under each root that contains gepa_state.bin."""
    for root in roots:
        if not root.exists():
            continue
        for sb in root.rglob("gepa_state.bin"):
            yield sb.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_root", nargs="+", required=True, type=Path,
                    help="One or more directories to walk for runs (each subdir "
                         "with gepa_state.bin is treated as a run).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output directory (will contain merge_enriched.jsonl + "
                         "merge_summary.csv).")
    ap.add_argument("--semantic", action="store_true",
                    help="Enable Step 4 (semantic SNR/SLC). Needs sentence-transformers.")
    ap.add_argument("--consistency", action="store_true",
                    help="Enable Step 5 (LLM judge). Needs --judge_lm.")
    ap.add_argument("--judge_lm", default="openrouter/openai/gpt-4.1-mini",
                    help="LiteLLM model id for judge LM.")
    ap.add_argument("--judge_all", action="store_true",
                    help="By default Step 5 only runs on negative-lift late-gen "
                         "(merge_of_merge) merges to control cost. Pass this to "
                         "judge every accepted merge.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    judge_lm = None
    if args.consistency:
        judge_lm = _build_judge_lm(args.judge_lm)

    all_records: list[dict[str, Any]] = []
    run_dirs = sorted(set(_iter_run_dirs(args.runs_root)))
    print(f"Found {len(run_dirs)} run directories.", file=sys.stderr)
    for i, rd in enumerate(run_dirs, 1):
        print(f"  [{i}/{len(run_dirs)}] {rd}", file=sys.stderr)
        recs = enrich_run(
            rd,
            do_semantic=args.semantic,
            judge_lm=judge_lm,
            judge_only_negative_late_gen=not args.judge_all,
        )
        all_records.extend(recs)

    # Write JSONL
    jsonl_out = args.out / "merge_enriched.jsonl"
    with jsonl_out.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")

    # Write flat CSV (subset of fields safe for pandas)
    csv_out = args.out / "merge_summary.csv"
    if all_records:
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in all_records:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    all_keys.append(k)
        with csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in all_records:
                w.writerow({k: r.get(k) for k in all_keys})

    print(f"Wrote {len(all_records)} merge events to:", file=sys.stderr)
    print(f"  {jsonl_out}", file=sys.stderr)
    print(f"  {csv_out}", file=sys.stderr)


if __name__ == "__main__":
    main()
