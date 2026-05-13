"""B2 + B3 — Stratified correlations + severity comparison.

Inputs: `cell_timing.csv` from B1.

For each stratum (by policy / by model / by benchmark / by task type),
compute Pearson r (with bootstrap 95% CI) between timing variables and
lift_over_nomerge.

Hard rule (per plan): never report cross-policy "overall" correlation
as headline, because policy semantics define the timing distribution by
construction.

Per fix #3: acceptance_rate is reported separately as a "merge process
variable" rather than mixed into pure timing variables.

Per fix #4: IFBench task type label is "instruction-following / format-strict".
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, mean

import numpy as np
from scipy import stats


def F(x):
    try:
        if x is None or x == "" or x == "None" or x == "nan":
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def pearson_with_ci(x: list[float], y: list[float], n_boot: int = 2000):
    """Pearson r with bootstrap 95% CI."""
    if len(x) < 3 or len(y) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    xa = np.array(x, dtype=float)
    ya = np.array(y, dtype=float)
    r, p = stats.pearsonr(xa, ya)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xa), len(xa))
        bx = xa[idx]; by = ya[idx]
        if bx.std() > 0 and by.std() > 0:
            br, _ = stats.pearsonr(bx, by)
            boots.append(br)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo, hi = float("nan"), float("nan")
    return float(r), float(lo), float(hi), float(p)


# Variable groups per fix #3
TIMING_VARS = [
    "relative_first",
    "early_merge_ratio",
    "mean_merge_iter",     # raw, less preferred — kept for reference
    "relative_density",
]
PROCESS_VARS = ["acceptance_rate"]  # "merge process variable", separate from timing


def correlate_stratum(rows: list[dict], stratum_name: str, stratum_value: str) -> list[dict]:
    """Compute correlations for all timing+process vars vs lift in a stratum."""
    out = []
    lifts = [F(r["lift_over_nomerge"]) for r in rows]
    for var in TIMING_VARS + PROCESS_VARS:
        var_vals = [F(r[var]) for r in rows]
        # filter pairs where both present
        pairs = [(x, y) for x, y in zip(var_vals, lifts)
                 if x is not None and y is not None]
        if len(pairs) < 3:
            continue
        x = [p[0] for p in pairs]; y = [p[1] for p in pairs]
        r, lo, hi, p = pearson_with_ci(x, y)
        out.append({
            "stratum_kind": stratum_name,
            "stratum_value": stratum_value,
            "n": len(pairs),
            "var_kind": "timing" if var in TIMING_VARS else "merge_process",
            "variable": var,
            "lift_metric": "lift_over_nomerge",
            "pearson_r": r,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": p,
        })
    return out


def task_type(d: str) -> str:
    """Per fix #4."""
    if d in ("hotpotqa", "musique"):
        return "multi_hop"
    if d == "ifbench":
        return "instruction_following_format_strict"
    if d == "hover":
        return "evidence_retrieval"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell_timing_csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw = list(csv.DictReader(open(args.cell_timing_csv)))
    # Use only merge cells (NoMerge has no merge events)
    merge_rows = [r for r in raw if r.get("use_merge") == "True"]
    print(f"Loaded {len(merge_rows)} merge cells.", file=sys.stderr)

    # ===== B2 =====
    all_corr = []

    # By policy
    by_policy: dict[str, list] = defaultdict(list)
    for r in merge_rows:
        by_policy[r["policy"]].append(r)
    for pol, rs in by_policy.items():
        all_corr += correlate_stratum(rs, "policy", pol)

    # By model
    by_model = defaultdict(list)
    for r in merge_rows:
        by_model[r["model"]].append(r)
    for m, rs in by_model.items():
        all_corr += correlate_stratum(rs, "model", m)

    # By benchmark
    by_bench = defaultdict(list)
    for r in merge_rows:
        by_bench[r["dataset"]].append(r)
    for d, rs in by_bench.items():
        all_corr += correlate_stratum(rs, "benchmark", d)

    # By task type
    by_type = defaultdict(list)
    for r in merge_rows:
        by_type[task_type(r["dataset"])].append(r)
    for t, rs in by_type.items():
        all_corr += correlate_stratum(rs, "task_type", t)

    # Write all to one CSV (4 strata in one) for convenience
    out_corr = args.out_dir / "timing_correlations.csv"
    cols = ["stratum_kind", "stratum_value", "n", "var_kind", "variable",
            "lift_metric", "pearson_r", "ci_low", "ci_high", "p_value"]
    with out_corr.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_corr:
            w.writerow(r)
    print(f"Wrote {out_corr}  ({len(all_corr)} rows)", file=sys.stderr)

    # Also split by stratum kind into separate files
    for kind in ("policy", "model", "benchmark", "task_type"):
        sub = [r for r in all_corr if r["stratum_kind"] == kind]
        out_p = args.out_dir / f"timing_correlations_by_{kind}.csv"
        with out_p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sub:
                w.writerow(r)

    # ===== B3 — severity comparison =====
    sev_order = ["good", "bad", "severe", "catastrophic"]
    sev_rows = []
    for sev in sev_order:
        sub = [r for r in merge_rows if r["severity_flag"] == sev]
        if not sub:
            continue
        # collect medians (skip NaN)
        def med(col):
            vals = [F(r[col]) for r in sub]
            vals = [v for v in vals if v is not None]
            return median(vals) if vals else None

        examples = ", ".join(r["cell_id"] for r in sub[:3])
        if len(sub) > 3:
            examples += f", ... ({len(sub)-3} more)"

        sev_rows.append({
            "severity": sev,
            "n_cells": len(sub),
            "median_lift": med("lift_over_nomerge"),
            "median_first_merge_iter": med("first_merge_iter"),
            "median_relative_first": med("relative_first"),
            "median_early_merge_ratio": med("early_merge_ratio"),
            "median_n_attempts": med("n_attempts"),
            "median_relative_density": med("relative_density"),
            "median_acceptance_rate": med("acceptance_rate"),
            "examples": examples,
        })

    out_sev = args.out_dir / "timing_severity_breakdown.csv"
    cols_s = ["severity", "n_cells", "median_lift",
              "median_first_merge_iter", "median_relative_first",
              "median_early_merge_ratio", "median_n_attempts",
              "median_relative_density", "median_acceptance_rate",
              "examples"]
    with out_sev.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols_s)
        w.writeheader()
        for r in sev_rows:
            w.writerow(r)
    print(f"Wrote {out_sev}  ({len(sev_rows)} severity bins)", file=sys.stderr)

    # ===== Print summary =====
    print(f"\n{'='*100}")
    print(f"  B2 — Stratified correlations (Pearson r vs lift_over_nomerge)")
    print(f"{'='*100}")
    for kind in ("policy", "model", "benchmark", "task_type"):
        print(f"\n  Stratum: {kind}")
        sub = [r for r in all_corr if r["stratum_kind"] == kind]
        if not sub:
            continue
        # group by stratum_value
        by_val: dict[str, list[dict]] = defaultdict(list)
        for r in sub:
            by_val[r["stratum_value"]].append(r)
        for val, rs in by_val.items():
            print(f"    [{val}] (n={rs[0]['n']})")
            # Order vars: timing first, then process
            for var_kind in ("timing", "merge_process"):
                vk_rows = [r for r in rs if r["var_kind"] == var_kind]
                for r in vk_rows:
                    sig = ""
                    if not math.isnan(r["p_value"]):
                        sig = "***" if r["p_value"]<0.001 else "**" if r["p_value"]<0.01 else "*" if r["p_value"]<0.05 else ""
                    print(f"      {r['variable']:24s} r={r['pearson_r']:+.3f} "
                          f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] p={r['p_value']:.3f}{sig}  ({r['var_kind']})")

    print(f"\n{'='*100}")
    print(f"  B3 — Severity breakdown")
    print(f"{'='*100}")
    print(f"  {'severity':14s} {'n':>3s} {'med_lift':>10s} {'med_first_it':>13s} "
          f"{'med_rel_first':>14s} {'med_early_r':>13s} {'med_acc_rate':>14s}")
    for r in sev_rows:
        ml = f"{r['median_lift']:+.2f}" if r['median_lift'] is not None else "  -  "
        mf = f"{r['median_first_merge_iter']:.0f}" if r['median_first_merge_iter'] is not None else "-"
        mr = f"{r['median_relative_first']:.3f}" if r['median_relative_first'] is not None else "-"
        me = f"{r['median_early_merge_ratio']:.3f}" if r['median_early_merge_ratio'] is not None else "-"
        ma = f"{r['median_acceptance_rate']:.3f}" if r['median_acceptance_rate'] is not None else "-"
        print(f"  {r['severity']:14s} {r['n_cells']:>3d} {ml:>10s} {mf:>13s} "
              f"{mr:>14s} {me:>13s} {ma:>14s}")


if __name__ == "__main__":
    main()
