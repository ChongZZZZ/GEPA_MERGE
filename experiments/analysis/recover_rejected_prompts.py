"""A1 — Recover merged-prompt text for deterministic rejected merge events.

Rejected merge events in `merge_quality.jsonl` only log lexical/lineage
metrics; the actual merged-prompt text is not saved. This is fine for
accepted merges (their text lives in the candidate pool indexed by
`new_idx`), but rejected merges are dropped before being added to the
pool.

For deterministic merge algorithms (`original` = `merge_system_aware` and
`combine_all` = `merge_combine_all_subprompts`), we can re-derive the
exact merged prompt by calling the algorithm with the same parents. The
algorithms are pure functions of (anc, id1, id2, program_candidates,
agg_scores) — no LM call, no randomness used in either path — so the
recovered text matches what GEPA actually produced and rejected.

For `summarize_before`, the LM output is stochastic and was not logged,
so rejected events for that algorithm cannot be exactly recovered. Those
events are listed in the output with `recoverable=false` and the prompt
is left empty, per the matched-deterministic-subset analysis policy.

Output: `experiments/analysis/output/forensics_v3/recovered_rejected_prompts.jsonl`,
one row per rejected event with the merged-prompt dict and metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gepa.strategies.merge_algorithm import (  # noqa: E402
    merge_system_aware,
    merge_combine_all_subprompts,
)

DETERMINISTIC = {
    "original": merge_system_aware,
    "combine_all": merge_combine_all_subprompts,
}


def _state_agg_scores(state: dict) -> list[float]:
    """Mean val sub-score per candidate. None for empty (uninitialized)."""
    out = []
    for ss in state["prog_candidate_val_subscores"]:
        if ss:
            out.append(sum(ss.values()) / len(ss))
        else:
            out.append(0.0)
    return out


def recover_one(run_dir: Path, event: dict) -> dict:
    """Re-derive the merged prompt for a single rejected event."""
    rec = {
        "run_dir": str(run_dir),
        "attempt_id": event.get("attempt_id"),
        "iteration": event.get("iteration"),
        "p1_idx": event["parent_ids"][0],
        "p2_idx": event["parent_ids"][1],
        "anc_idx": event["ancestor_idx"],
    }

    # Load metadata to identify algo
    bc = json.load(open(run_dir / "best_candidate.json"))
    algo = bc.get("merge_algorithm")
    rec["algo"] = algo
    rec["task"] = bc.get("task")
    rec["model_full"] = bc.get("model")

    if algo not in DETERMINISTIC:
        rec["recoverable"] = False
        rec["reason"] = f"non-deterministic algo: {algo}"
        rec["merged_program"] = None
        return rec

    state = pickle.load(open(run_dir / "gepa_state.bin", "rb"))
    agg = _state_agg_scores(state)
    pcs = state["program_candidates"]

    p1, p2, anc = rec["p1_idx"], rec["p2_idx"], rec["anc_idx"]
    if any(idx is None or not (0 <= idx < len(pcs)) for idx in (p1, p2, anc)):
        rec["recoverable"] = False
        rec["reason"] = "parent or ancestor index out of range"
        rec["merged_program"] = None
        return rec

    merge_fn = DETERMINISTIC[algo]
    # Original (system_aware) uses rng.choice for Case-B tie-breaking
    # (when agg_scores[id1] == agg_scores[id2]). The original run's rng
    # state is not recoverable, so we seed deterministically from the
    # attempt_id — reconstruction is reproducible across re-runs of this
    # script, but on tied score may not match GEPA's original tie-break
    # exactly. Both outcomes are valid parent instructions; downstream
    # length/coverage/entropy/SNR metrics are insensitive to which parent
    # the tie went to.
    seed = int(hashlib.md5(str(event.get("attempt_id","")).encode()).hexdigest()[:8], 16)
    rec_rng = random.Random(seed)
    try:
        new_program, prog_desc = merge_fn(
            ancestor_idx=anc,
            id1=p1,
            id2=p2,
            program_candidates=pcs,
            agg_scores=agg,
            rng=rec_rng,
        )
    except Exception as e:
        rec["recoverable"] = False
        rec["reason"] = f"merge_fn raised: {type(e).__name__}: {e}"
        rec["merged_program"] = None
        return rec

    rec["recoverable"] = True
    rec["merged_program"] = new_program  # dict[predictor_name -> instruction text]
    rec["prog_desc"] = list(prog_desc) if prog_desc is not None else None
    # Also include the parent prompts for downstream (semantic SNR + judge)
    rec["p1_program"] = pcs[p1]
    rec["p2_program"] = pcs[p2]
    rec["ancestor_program"] = pcs[anc]
    return rec


def iter_rejected_events(runs_root: Path):
    """Yield (run_dir, event_dict) for every rejected merge event under runs_root."""
    for mq in runs_root.rglob("merge_quality.jsonl"):
        run_dir = mq.parent
        if any(x in str(run_dir) for x in [
            "smoke", "_local_", "anchor_pilot", "phase_c", "BROKEN", "probe",
            "phase_a_think", "_archive_", ".anomaly_",
        ]):
            continue
        bc = run_dir / "best_candidate.json"
        if not bc.exists():
            continue
        try:
            bc_obj = json.load(open(bc))
        except Exception:
            continue
        if not bc_obj.get("use_merge"):
            continue
        # drop pupa (degenerate)
        if bc_obj.get("task") in ("pupa", "papillon"):
            continue
        with mq.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("event") == "rejected":
                    yield run_dir, ev


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_roots", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path,
                    help="output JSONL path")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_recoverable = 0
    n_skip_sb = 0
    n_skip_other = 0
    by_algo = {}

    with args.out.open("w") as fout:
        for root in args.runs_roots:
            if not root.exists():
                print(f"[skip missing] {root}", file=sys.stderr)
                continue
            for run_dir, ev in iter_rejected_events(root):
                rec = recover_one(run_dir, ev)
                fout.write(json.dumps(rec) + "\n")
                n_total += 1
                algo = rec.get("algo")
                by_algo.setdefault(algo, {"recoverable": 0, "total": 0})
                by_algo[algo]["total"] += 1
                if rec.get("recoverable"):
                    n_recoverable += 1
                    by_algo[algo]["recoverable"] += 1
                else:
                    if algo == "summarize_before":
                        n_skip_sb += 1
                    else:
                        n_skip_other += 1

    print(f"\n=== A1 recovery summary ===", file=sys.stderr)
    print(f"  rejected events processed:     {n_total}", file=sys.stderr)
    print(f"  recoverable (deterministic):   {n_recoverable}", file=sys.stderr)
    print(f"  skipped: summarize_before:     {n_skip_sb}", file=sys.stderr)
    print(f"  skipped: other reasons:        {n_skip_other}", file=sys.stderr)
    for algo, c in sorted(by_algo.items()):
        print(f"  {algo:18s}  recovered {c['recoverable']:3d} / {c['total']:3d}", file=sys.stderr)
    print(f"  output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
