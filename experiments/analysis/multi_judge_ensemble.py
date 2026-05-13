"""3-judge ensemble: re-judge all accepted-deterministic + recovered
rejected merges with claude-sonnet-4.6 and claude-haiku-4.5, on top of
the existing gpt-4.1-mini scores.

Goal: confirm that the surprising finding from forensics_v3
(`judge_clarity` and `judge_specificity` slightly higher on rejected
than accepted, d ≈ −0.46) is not an artifact of single-judge bias.

Inputs:
  - forensics_v3/recovered_rejected_metrics.jsonl  (rejected has gpt-4.1-mini)
  - forensics_v3/accepted_deterministic_metrics.csv  (accepted metadata)
  - per-cell gepa_state.bin  (for accepted: reconstruct merged prompt
                              by re-calling the deterministic merge_fn
                              with the same parents)

Outputs:
  - forensics_v3/multi_judge_metrics.jsonl
    one row per merge event with all 3 judges' scores
    (gpt-4.1-mini from prior data, sonnet-4.6 + haiku-4.5 added now).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.analysis.merge_forensics import (  # noqa: E402
    _build_judge_lm,
    consistency_metrics,
)
from gepa.strategies.merge_algorithm import (  # noqa: E402
    merge_system_aware,
    merge_combine_all_subprompts,
)

DETERMINISTIC = {
    "original": merge_system_aware,
    "combine_all": merge_combine_all_subprompts,
}


def _flat_keys():
    return ["clarity", "specificity", "internal_consistency",
            "coverage_vs_parents", "contradiction_present"]


def _agg_scores(state):
    out = []
    for ss in state["prog_candidate_val_subscores"]:
        out.append(sum(ss.values()) / len(ss) if ss else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rejected_jsonl", required=True, type=Path)
    ap.add_argument("--accepted_csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--judges", nargs="+", default=[
        "openrouter/anthropic/claude-sonnet-4.6",
        "openrouter/anthropic/claude-haiku-4.5",
    ])
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # ---- Load rejected (already has full prompts) ----
    rej_events = []
    with args.rejected_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("recoverable"):
                    rej_events.append(r)
    print(f"Loaded {len(rej_events)} recovered rejected events.", file=sys.stderr)

    # ---- Load accepted metadata ----
    import csv as _csv
    acc_meta = list(_csv.DictReader(open(args.accepted_csv)))
    print(f"Loaded {len(acc_meta)} accepted-deterministic events.", file=sys.stderr)

    # ---- Reconstruct merged prompts for accepted (deterministic algos) ----
    by_run_dir: dict[str, list[dict]] = {}
    for r in acc_meta:
        by_run_dir.setdefault(r["run_dir"], []).append(r)

    acc_events = []
    for run_dir_str, rows in by_run_dir.items():
        run_dir = REPO_ROOT / run_dir_str
        try:
            state = pickle.load(open(run_dir / "gepa_state.bin", "rb"))
            pcs = state["program_candidates"]
            agg = _agg_scores(state)
        except Exception as e:
            print(f"  [skip dir] {run_dir_str}: {e}", file=sys.stderr)
            continue
        for r in rows:
            try:
                p1 = int(r["p1_idx"])
                p2 = int(r["p2_idx"])
                anc = int(r["anc_idx"])
                algo = r["algo"]
                merge_fn = DETERMINISTIC.get(algo)
                if merge_fn is None:
                    continue
                rng = random.Random(int(r.get("iteration", 0) or 0) ^ p1 ^ p2)
                merged_prog, _ = merge_fn(
                    ancestor_idx=anc, id1=p1, id2=p2,
                    program_candidates=pcs, agg_scores=agg, rng=rng,
                )
                acc_events.append({
                    **r,
                    "event": "accepted",
                    "merged_program": merged_prog,
                    "p1_program": pcs[p1],
                    "p2_program": pcs[p2],
                })
            except Exception as e:
                print(f"  [skip ev]: {e}", file=sys.stderr)
                continue

    print(f"Reconstructed {len(acc_events)} accepted merged prompts.", file=sys.stderr)

    # ---- Build new judges ----
    judge_lms = {}
    for judge_id in args.judges:
        short = judge_id.rsplit("/", 1)[-1].replace(".", "_").replace("-", "_")
        judge_lms[short] = _build_judge_lm(judge_id)
        print(f"  judge ready: {short}  ({judge_id})", file=sys.stderr)

    # ---- Combine and judge ----
    all_events = acc_events + rej_events
    print(f"\nTotal events: {len(all_events)} (acc={len(acc_events)} + rej={len(rej_events)})",
          file=sys.stderr)
    print(f"New judges per event: {len(judge_lms)}", file=sys.stderr)

    n_done = 0
    n_err = {short: 0 for short in judge_lms}
    t0 = time.time()

    with args.out.open("w") as fout:
        for r in all_events:
            row = {
                "run_dir": r.get("run_dir"),
                "event": r.get("event"),
                "model": r.get("model"),
                "dataset": r.get("dataset"),
                "algo": r.get("algo"),
                "policy": r.get("policy"),
                "p1_idx": r.get("p1_idx"),
                "p2_idx": r.get("p2_idx"),
                "anc_idx": r.get("anc_idx"),
                "iteration": r.get("iteration"),
            }
            # Pull existing gpt-4.1-mini scores
            for k in _flat_keys():
                row[f"gpt41mini_{k}"] = r.get(f"judge_{k}")

            merged = r.get("merged_program")
            p1 = r.get("p1_program")
            p2 = r.get("p2_program")
            if not (merged and p1 and p2):
                row["error"] = "missing prompts"
                fout.write(json.dumps(row) + "\n")
                n_done += 1
                continue

            for short, lm in judge_lms.items():
                try:
                    cons = consistency_metrics(merged, p1, p2, lm)
                    for k in _flat_keys():
                        row[f"{short}_{k}"] = cons.get(f"judge_{k}")
                    if cons.get("judge_contradiction_span"):
                        row[f"{short}_contradiction_span"] = cons["judge_contradiction_span"]
                except Exception as e:
                    row[f"{short}_error"] = repr(e)[:200]
                    n_err[short] += 1

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            n_done += 1
            if n_done % 20 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(all_events) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(all_events)}] elapsed={elapsed:.0f}s "
                      f"rate={rate:.2f}/s eta={eta:.0f}s err={n_err}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone. {n_done} events, elapsed {elapsed:.0f}s, errors: {n_err}",
          file=sys.stderr)
    print(f"Output: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
