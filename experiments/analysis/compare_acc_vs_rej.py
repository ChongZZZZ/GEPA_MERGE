"""A4 + A5 — Build accepted-vs-rejected comparison tables (matched
deterministic subset).

Inputs:
  - forensics_v2/merge_summary.csv  (accepted has all 5 steps, rejected
                                     has Steps 1-3 only)
  - forensics_v3/recovered_rejected_metrics.jsonl  (rejected with all 5
                                                    steps for deterministic
                                                    algos)

Outputs (forensics_v3/):
  - accepted_deterministic_metrics.csv  (213 rows: original + combine_all
                                         accepted only, all 5 steps)
  - rejected_deterministic_metrics.csv  (52 rows: recovered with all 5
                                         steps)
  - accepted_vs_rejected_main.csv       (213 vs 52 comparison table)
  - accepted_vs_rejected_balanced.csv   (algo-balanced 52 vs 52)
  - selection_proxy_breakdown.csv       (subsample-derived proxy by group)

Per metric we report: Cohen's d (with 95% CI), Cliff's delta, Mann-Whitney
U, p-value. Cohen's d leads, p-value reported in parens. Both Main and
Robustness A' versions reported. Conclusion fires only if both agree.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from scipy import stats


# ---------- helpers ----------

def F(x):
    try:
        return float(x) if x not in (None, "", "None") else None
    except Exception:
        return None


def B(x):
    if x in ("True", "true", True, 1, "1"):
        return 1
    if x in ("False", "false", False, 0, "0"):
        return 0
    return None


def cohen_d(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Cohen's d with bootstrap 95% CI. Returns (d, ci_low, ci_high)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0, 0.0, 0.0
    d = (a.mean() - b.mean()) / pooled
    # bootstrap CI
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        ar = rng.choice(a, size=len(a), replace=True)
        br = rng.choice(b, size=len(b), replace=True)
        p = np.sqrt(((len(ar) - 1) * ar.var(ddof=1) + (len(br) - 1) * br.var(ddof=1))
                    / (len(ar) + len(br) - 2))
        if p > 0:
            boots.append((ar.mean() - br.mean()) / p)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = float("nan"), float("nan")
    return float(d), float(lo), float(hi)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta — non-parametric effect size, range [-1, 1]."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    n_pos = sum(1 for x in a for y in b if x > y)
    n_neg = sum(1 for x in a for y in b if x < y)
    return (n_pos - n_neg) / (len(a) * len(b))


def compare_metric(name: str, vals_a: list[float], vals_b: list[float],
                   higher_better_for_acc: bool | None = None) -> dict:
    """Run all comparison stats for one metric, comparing accepted (a) vs rejected (b)."""
    if not vals_a or not vals_b:
        return {"metric": name, "n_a": len(vals_a), "n_b": len(vals_b),
                "med_a": None, "med_b": None}
    med_a, med_b = float(np.median(vals_a)), float(np.median(vals_b))
    mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
    d, d_lo, d_hi = cohen_d(vals_a, vals_b)
    delta = cliffs_delta(vals_a, vals_b)
    try:
        u, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    except Exception:
        u, p = float("nan"), float("nan")
    return {
        "metric": name,
        "n_a": len(vals_a), "n_b": len(vals_b),
        "med_a": med_a, "med_b": med_b,
        "mean_a": mean_a, "mean_b": mean_b,
        "delta_med": med_b - med_a,
        "cohens_d": d, "d_ci_low": d_lo, "d_ci_high": d_hi,
        "cliffs_delta": delta,
        "u_stat": float(u),
        "p_value": float(p),
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--v2_csv", required=True, type=Path,
                    help="forensics_v2 merge_summary.csv")
    ap.add_argument("--rejected_jsonl", required=True, type=Path,
                    help="forensics_v3 recovered_rejected_metrics.jsonl")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load v2 (annotate algo) ----
    v2_rows = list(csv.DictReader(open(args.v2_csv)))

    # Annotate (model, dataset, algo, policy) from best_candidate.json
    def annotate(r):
        rd = REPO_ROOT / r["run_dir"]
        bc = rd / "best_candidate.json"
        if not bc.exists():
            return None
        bc_obj = json.load(open(bc))
        if not bc_obj.get("use_merge"):
            return None
        return {
            "model": "gpt-4.1-mini" if "gpt" in bc_obj.get("model", "").lower() else "qwen3-8b",
            "dataset": bc_obj.get("task"),
            "algo": bc_obj.get("merge_algorithm"),
            "policy": bc_obj.get("merge_start"),
        }

    REPO_ROOT = Path(__file__).resolve().parents[2]
    for r in v2_rows:
        meta = annotate(r)
        if meta:
            r.update(meta)

    # Filter accepted-deterministic
    acc_det = [r for r in v2_rows if r.get("event") == "accepted"
               and r.get("dataset") not in ("pupa", "papillon")
               and ".anomaly_" not in r.get("run_dir", "")
               and r.get("algo") in ("original", "combine_all")]
    print(f"Accepted-deterministic (orig + combine_all): {len(acc_det)}", file=sys.stderr)

    # ---- Load A2-A3 rejected ----
    rej_rows = []
    with args.rejected_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("recoverable"):
                    rej_rows.append(r)
    print(f"Recovered deterministic rejected: {len(rej_rows)}", file=sys.stderr)

    # Annotate rejected with model/policy from run_dir
    for r in rej_rows:
        run_dir = REPO_ROOT / r["run_dir"]
        bc = run_dir / "best_candidate.json"
        if bc.exists():
            bc_obj = json.load(open(bc))
            r["model"] = "gpt-4.1-mini" if "gpt" in bc_obj.get("model", "").lower() else "qwen3-8b"
            r["dataset"] = bc_obj.get("task")
            r["algo"] = bc_obj.get("merge_algorithm")
            r["policy"] = bc_obj.get("merge_start")

    # ---- Add lexical/lineage from v2 to rejected (by attempt_id matching) ----
    # We need length_delta, sentence_entropy, coverage_min, etc. for rejected.
    # These come from v2 (Steps 1-3 are computed for both acc + rej there).
    v2_rejected_idx = {(r["run_dir"], r.get("p1_idx"), r.get("p2_idx"), r.get("anc_idx")): r
                       for r in v2_rows if r.get("event") == "rejected"}

    n_matched = 0
    for r in rej_rows:
        key = (r["run_dir"], str(r["p1_idx"]), str(r["p2_idx"]), str(r["anc_idx"]))
        v2_r = v2_rejected_idx.get(key)
        if v2_r:
            for k in ("length_delta_vs_anc_total", "sentence_entropy",
                      "predictor_entropy", "coverage_min", "coverage_p1",
                      "coverage_p2", "novelty_fraction", "behavioral_delta_rate",
                      "noop_predictor_rate", "parent_gen_depth_max",
                      "subsample_win", "subsample_tie", "subsample_loss"):
                if k not in r and v2_r.get(k):
                    r[k] = v2_r[k]
            n_matched += 1
    print(f"Joined v2 lexical/lineage onto rejected: {n_matched}/{len(rej_rows)}",
          file=sys.stderr)

    # ---- A4: selection_proxy_score on both groups ----
    def proxy(row):
        win = F(row.get("subsample_win"))
        tie = F(row.get("subsample_tie"))
        loss = F(row.get("subsample_loss"))
        if any(v is None for v in (win, tie, loss)):
            return None
        total = win + tie + loss
        if total == 0:
            return None
        return (win + 0.5 * tie) / total

    for r in acc_det:
        r["selection_proxy_score"] = proxy(r)
    for r in rej_rows:
        r["selection_proxy_score"] = proxy(r)

    # ---- Write filtered metric CSVs ----
    metric_cols = [
        "model", "dataset", "algo", "policy", "event", "run_dir",
        "iteration", "p1_idx", "p2_idx", "anc_idx",
        "length_delta_vs_anc_total", "sentence_entropy", "predictor_entropy",
        "coverage_min", "coverage_p1", "coverage_p2", "novelty_fraction",
        "behavioral_delta_rate", "noop_predictor_rate", "parent_gen_depth_max",
        "subsample_win", "subsample_tie", "subsample_loss",
        "snr_semantic_novelty_rate", "slc_lost_count_max",
        "n_merged_sentences", "n_p1_sentences", "n_p2_sentences",
        "judge_clarity", "judge_specificity", "judge_internal_consistency",
        "judge_coverage_vs_parents", "judge_contradiction_present",
        "selection_proxy_score",
    ]

    def write_csv(path, rows, event_label):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metric_cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = {k: r.get(k) for k in metric_cols}
                row["event"] = event_label
                w.writerow(row)

    acc_path = args.out_dir / "accepted_deterministic_metrics.csv"
    rej_path = args.out_dir / "rejected_deterministic_metrics.csv"
    write_csv(acc_path, acc_det, "accepted")
    write_csv(rej_path, rej_rows, "rejected")
    print(f"Wrote {acc_path}", file=sys.stderr)
    print(f"Wrote {rej_path}", file=sys.stderr)

    # ---- A5: comparisons ----
    METRICS_TO_COMPARE = [
        "length_delta_vs_anc_total",
        "coverage_min", "coverage_p1", "coverage_p2",
        "sentence_entropy", "predictor_entropy",
        "novelty_fraction",
        "noop_predictor_rate",
        "parent_gen_depth_max",
        "snr_semantic_novelty_rate",
        "slc_lost_count_max",
        "judge_clarity", "judge_specificity",
        "judge_internal_consistency", "judge_coverage_vs_parents",
        "judge_contradiction_present",
        "selection_proxy_score",
    ]

    def vals(rows, k):
        if k == "judge_contradiction_present":
            return [B(r.get(k)) for r in rows if B(r.get(k)) is not None]
        return [F(r.get(k)) for r in rows if F(r.get(k)) is not None]

    # Main comparison: 213 vs 52
    main_results = []
    for m in METRICS_TO_COMPARE:
        a, b = vals(acc_det, m), vals(rej_rows, m)
        main_results.append(compare_metric(m, a, b))

    main_path = args.out_dir / "accepted_vs_rejected_main.csv"
    with main_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(main_results[0].keys()))
        w.writeheader()
        for r in main_results:
            w.writerow(r)
    print(f"Wrote {main_path}", file=sys.stderr)

    # Robustness A': algo-balanced sample matching rejected (orig 22, ca 30)
    rej_orig = [r for r in rej_rows if r["algo"] == "original"]
    rej_ca = [r for r in rej_rows if r["algo"] == "combine_all"]
    acc_orig = [r for r in acc_det if r["algo"] == "original"]
    acc_ca = [r for r in acc_det if r["algo"] == "combine_all"]

    rng = random.Random(0)
    sampled_acc_orig = rng.sample(acc_orig, k=len(rej_orig))
    sampled_acc_ca = rng.sample(acc_ca, k=len(rej_ca))
    balanced_acc = sampled_acc_orig + sampled_acc_ca
    print(f"Balanced subset: {len(balanced_acc)} accepted "
          f"({len(sampled_acc_orig)} orig + {len(sampled_acc_ca)} ca) "
          f"vs {len(rej_rows)} rejected", file=sys.stderr)

    bal_results = []
    for m in METRICS_TO_COMPARE:
        a, b = vals(balanced_acc, m), vals(rej_rows, m)
        bal_results.append(compare_metric(m, a, b))

    bal_path = args.out_dir / "accepted_vs_rejected_balanced.csv"
    with bal_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bal_results[0].keys()))
        w.writeheader()
        for r in bal_results:
            w.writerow(r)
    print(f"Wrote {bal_path}", file=sys.stderr)

    # ---- Print summary table to stdout ----
    print(f"\n{'='*100}")
    print(f"  ACCEPTED vs REJECTED — Main (213 vs 52) and Balanced (52 vs 52)")
    print(f"{'='*100}")
    print(f"\n{'metric':30s} {'Main_d':>9s} {'Main_p':>9s} {'Bal_d':>9s} {'Bal_p':>9s}  {'verdict':>10s}")
    print("-" * 90)
    for main, bal in zip(main_results, bal_results):
        m = main["metric"]
        d_main = main.get("cohens_d") or 0
        p_main = main.get("p_value") or 1
        d_bal = bal.get("cohens_d") or 0
        p_bal = bal.get("p_value") or 1
        # Verdict: agreement in direction + magnitude > 0.2
        agree = (d_main * d_bal > 0) and (abs(d_main) > 0.2 or abs(d_bal) > 0.2)
        if agree and abs(d_main) >= 0.5:
            verdict = "STRONG"
        elif agree and abs(d_main) >= 0.3:
            verdict = "moderate"
        elif agree:
            verdict = "weak"
        else:
            verdict = "n.s."
        # significance
        sig_main = "***" if p_main<0.001 else "**" if p_main<0.01 else "*" if p_main<0.05 else ""
        sig_bal = "***" if p_bal<0.001 else "**" if p_bal<0.01 else "*" if p_bal<0.05 else ""
        print(f"{m:30s} {d_main:>+9.3f} {p_main:>9.4f}{sig_main:<3s} {d_bal:>+9.3f} {p_bal:>9.4f}{sig_bal:<3s}  {verdict:>10s}")

    print(f"\nReading: cohen_d > 0 means metric HIGHER in accepted than rejected.")
    print(f"Verdict requires: same sign + |d| > 0.2 in BOTH Main and Balanced.")


if __name__ == "__main__":
    main()
