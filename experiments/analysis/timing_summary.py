"""B1 — Per-cell timing summary.

For each Phase A merge cell (8 model-bench × 9 algo-policy = 72 cells +
8 NoMerge cells), compute timing variables that describe when and how
often merge fired during the run, and join with the cell's held-out
test score (from `test_result/all_test_evals_v2.csv`).

NoMerge cells are emitted but with merge-related fields = NaN; they're
included so downstream analyses have the baseline available in one
place.

Output: `experiments/analysis/output/timing_v1/cell_timing.csv`.

Caveat (per plan): n_attempts=0 cells (NoMerge or sparse-merge runs)
get NaN for timing variables. early_merge_ratio defaults to 0 only when
n_attempts > 0; otherwise NaN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from statistics import mean, median


def F(x):
    try:
        return float(x) if x not in (None, "", "None") else None
    except Exception:
        return None


def collect_cells(roots: list[Path]) -> list[dict]:
    """Walk every gepa_state.bin under roots, return rows of cell-level info."""
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for sb in root.rglob("gepa_state.bin"):
            run_dir = sb.parent
            if any(x in str(run_dir) for x in [
                "smoke", "_local_", "anchor_pilot", "phase_c", "BROKEN",
                "probe", "phase_a_think", "_archive_", ".anomaly_",
            ]):
                continue
            bc_path = run_dir / "best_candidate.json"
            if not bc_path.exists():
                continue
            try:
                bc = json.load(open(bc_path))
            except Exception:
                continue
            # Drop pupa cells (degenerate)
            if bc.get("task") in ("pupa", "papillon"):
                continue

            try:
                state = pickle.load(open(sb, "rb"))
            except Exception as e:
                print(f"  [skip state] {run_dir}: {e}", file=sys.stderr)
                continue

            total_iters = int(state.get("i") or 0)

            # Collect merge events
            mq = run_dir / "merge_quality.jsonl"
            attempts: list[int] = []
            n_acc = 0
            if mq.exists():
                with mq.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            ev = json.loads(line)
                        except Exception:
                            continue
                        it = ev.get("iteration")
                        if it is not None:
                            attempts.append(int(it))
                        if ev.get("event") == "accepted":
                            n_acc += 1

            n_att = len(attempts)
            row = {
                "run_dir": str(run_dir),
                "model": "gpt-4.1-mini" if "gpt" in (bc.get("model") or "").lower() else "qwen3-8b",
                "dataset": bc.get("task"),
                "algo": bc.get("merge_algorithm") if bc.get("use_merge") else "nomerge",
                "policy": bc.get("merge_start") if bc.get("use_merge") else "nomerge",
                "use_merge": bool(bc.get("use_merge")),
                "total_iters": total_iters,
                "n_attempts": n_att,
                "n_accepted": n_acc,
            }
            row["cell_id"] = f'{row["model"]}|{row["dataset"]}|{row["algo"]}|{row["policy"]}'

            # Acceptance rate (merge process variable; treat 0/0 as NaN)
            row["acceptance_rate"] = (n_acc / n_att) if n_att > 0 else math.nan

            # Timing summary (per fix #1 + #2: NaN when no events)
            if n_att > 0:
                row["first_merge_iter"] = min(attempts)
                row["last_merge_iter"] = max(attempts)
                row["mean_merge_iter"] = mean(attempts)
                row["median_merge_iter"] = float(median(attempts))
                row["relative_first"] = (
                    row["first_merge_iter"] / total_iters
                    if total_iters > 0 else math.nan
                )
                row["relative_density"] = (
                    n_att / total_iters if total_iters > 0 else math.nan
                )
                # Early-merge: iter ≤ 25% of total_iters
                if total_iters > 0:
                    early_thresh = 0.25 * total_iters
                    row["early_merge_count"] = sum(1 for it in attempts if it <= early_thresh)
                    row["early_merge_ratio"] = row["early_merge_count"] / n_att
                else:
                    row["early_merge_count"] = 0
                    row["early_merge_ratio"] = math.nan
            else:
                row["first_merge_iter"] = math.nan
                row["last_merge_iter"] = math.nan
                row["mean_merge_iter"] = math.nan
                row["median_merge_iter"] = math.nan
                row["relative_first"] = math.nan
                row["relative_density"] = math.nan
                row["early_merge_count"] = 0
                row["early_merge_ratio"] = math.nan

            rows.append(row)
    return rows


def attach_test_scores(rows: list[dict], scores_csv: Path) -> None:
    """Join test_score and NoMerge baseline."""
    raw = list(csv.DictReader(open(scores_csv)))

    # Build lookup: (model, dataset, algo, policy) -> test_score
    scores: dict[tuple, float] = {}
    nomerge_lookup: dict[tuple, float] = {}
    for r in raw:
        m = r["model"]
        d = r["dataset"]
        score = float(r["test_score"])
        if r["use_merge"] == "True":
            algo = r["merge_algorithm"]
            policy = r["merge_start"]
            scores[(m, d, algo, policy)] = score
        else:
            nomerge_lookup[(m, d)] = score
            scores[(m, d, "nomerge", "nomerge")] = score

    n_matched = 0
    for row in rows:
        key = (row["model"], row["dataset"], row["algo"], row["policy"])
        ts = scores.get(key)
        nm = nomerge_lookup.get((row["model"], row["dataset"]))
        row["test_score"] = ts
        row["nomerge_baseline"] = nm
        if ts is not None and nm is not None:
            row["lift_over_nomerge"] = ts - nm
            n_matched += 1
        else:
            row["lift_over_nomerge"] = math.nan

        # Severity (only meaningful for merge cells with lift)
        lift = row.get("lift_over_nomerge")
        if not row.get("use_merge") or lift is None or (isinstance(lift, float) and math.isnan(lift)):
            row["severity_flag"] = ""
        elif lift > 0:
            row["severity_flag"] = "good"
        elif lift > -3:
            row["severity_flag"] = "bad"
        elif lift > -10:
            row["severity_flag"] = "severe"
        else:
            row["severity_flag"] = "catastrophic"

    print(f"  matched test scores: {n_matched}/{len(rows)}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs_roots", nargs="+", required=True, type=Path)
    ap.add_argument("--scores_csv", required=True, type=Path,
                    help="test_result/all_test_evals_v2.csv")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_cells(args.runs_roots)
    print(f"Collected {len(rows)} cells.", file=sys.stderr)

    attach_test_scores(rows, args.scores_csv)

    # Order cols
    cols = [
        "cell_id", "model", "dataset", "algo", "policy", "use_merge",
        "total_iters", "n_attempts", "n_accepted", "acceptance_rate",
        "first_merge_iter", "last_merge_iter",
        "mean_merge_iter", "median_merge_iter",
        "relative_first", "relative_density",
        "early_merge_count", "early_merge_ratio",
        "test_score", "nomerge_baseline", "lift_over_nomerge",
        "severity_flag",
        "run_dir",
    ]

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {args.out}", file=sys.stderr)

    # Summary print
    merge_rows = [r for r in rows if r.get("use_merge")]
    nomerge_rows = [r for r in rows if not r.get("use_merge")]
    print(f"\n=== summary ===", file=sys.stderr)
    print(f"  merge cells: {len(merge_rows)}", file=sys.stderr)
    print(f"  nomerge baseline cells: {len(nomerge_rows)}", file=sys.stderr)
    sev = {}
    for r in merge_rows:
        sev[r["severity_flag"]] = sev.get(r["severity_flag"], 0) + 1
    print(f"  severity distribution among merge cells: {sev}", file=sys.stderr)

    # Catastrophic case
    cata = [r for r in merge_rows if r["severity_flag"] == "catastrophic"]
    if cata:
        print(f"  catastrophic cells (n={len(cata)}):", file=sys.stderr)
        for r in cata:
            print(f"    {r['cell_id']}  lift={r['lift_over_nomerge']:.2f}  "
                  f"first_iter={r['first_merge_iter']}  n_att={r['n_attempts']}",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
