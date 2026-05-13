"""Analyze 3-judge ensemble (gpt-4.1-mini + claude-sonnet-4.6 + claude-haiku-4.5).

For each judge axis (clarity, specificity, internal_consistency,
coverage_vs_parents, contradiction_present), report:
  1. Per-judge Cohen's d for accepted vs rejected.
  2. Ensemble (mean of 3 judges) Cohen's d.
  3. Inter-judge Pearson agreement (matrix per axis).
  4. Per-judge mean scores split by event=accepted/rejected.

Goal: verify whether the gpt-4.1-mini single-judge finding
("rejected has higher clarity / specificity than accepted") is
reproduced by claude-sonnet-4.6 and claude-haiku-4.5, or whether it
was a single-judge artifact.

Input: forensics_v3/multi_judge_metrics.jsonl
Output: forensics_v3/multi_judge_compare.csv + console summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats


def F(x):
    try:
        if x in (None, "", "None"):
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


def B(x):
    if x in ("True", "true", True, 1, "1"):
        return 1
    if x in ("False", "false", False, 0, "0"):
        return 0
    return None


def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    print(f"Loaded {len(rows)} merge events.", file=sys.stderr)

    # event split
    acc = [r for r in rows if r.get("event") == "accepted"]
    rej = [r for r in rows if r.get("event") == "rejected"]
    print(f"  accepted: {len(acc)}", file=sys.stderr)
    print(f"  rejected: {len(rej)}", file=sys.stderr)

    JUDGES = ["gpt41mini", "claude_sonnet_4_6", "claude_haiku_4_5"]
    AXES_NUMERIC = ["clarity", "specificity", "internal_consistency", "coverage_vs_parents"]
    AXES_BOOL = ["contradiction_present"]

    # Compute mean ensemble per row for numeric axes
    for r in rows:
        for ax in AXES_NUMERIC:
            vals = [F(r.get(f"{j}_{ax}")) for j in JUDGES]
            vals = [v for v in vals if v is not None]
            r[f"ensemble_{ax}"] = float(np.mean(vals)) if vals else None
        # contradiction: ensemble = OR (any judge flags)
        flags = [B(r.get(f"{j}_contradiction_present")) for j in JUDGES]
        flags = [v for v in flags if v is not None]
        r["ensemble_contradiction_present"] = (1 if any(v == 1 for v in flags) else 0) if flags else None

    # ---- Per-judge + ensemble Cohen's d ----
    print("\n=== Per-axis Cohen's d (accepted vs rejected) per judge + ensemble ===")
    print(f"  {'axis':22s} {'gpt41mini':>10s} {'sonnet-4.6':>11s} {'haiku-4.5':>11s} {'ensemble':>10s}")
    print(f"  {'-'*22} {'-'*10} {'-'*11} {'-'*11} {'-'*10}")

    out_rows = []
    for ax in AXES_NUMERIC:
        d_per = {}
        for j in JUDGES:
            col = f"{j}_{ax}"
            a = [F(r.get(col)) for r in acc]; a = [v for v in a if v is not None]
            b = [F(r.get(col)) for r in rej]; b = [v for v in b if v is not None]
            d_per[j] = cohen_d(a, b)
        # ensemble
        a_ens = [F(r.get(f"ensemble_{ax}")) for r in acc]; a_ens = [v for v in a_ens if v is not None]
        b_ens = [F(r.get(f"ensemble_{ax}")) for r in rej]; b_ens = [v for v in b_ens if v is not None]
        d_ens = cohen_d(a_ens, b_ens)
        print(f"  {ax:22s} {d_per[JUDGES[0]]:>+10.3f} {d_per[JUDGES[1]]:>+11.3f} {d_per[JUDGES[2]]:>+11.3f} {d_ens:>+10.3f}")
        out_rows.append({
            "axis": ax,
            "kind": "numeric",
            "n_acc": len(a_ens), "n_rej": len(b_ens),
            "d_gpt41mini": d_per["gpt41mini"],
            "d_sonnet_4_6": d_per["claude_sonnet_4_6"],
            "d_haiku_4_5": d_per["claude_haiku_4_5"],
            "d_ensemble": d_ens,
        })

    # bool axis
    for ax in AXES_BOOL:
        rates = {}
        for j in JUDGES:
            col = f"{j}_{ax}"
            a = [B(r.get(col)) for r in acc]; a = [v for v in a if v is not None]
            b = [B(r.get(col)) for r in rej]; b = [v for v in b if v is not None]
            rates[j] = (sum(a)/len(a) if a else 0, sum(b)/len(b) if b else 0)

        a_ens = [B(r.get(f"ensemble_{ax}")) for r in acc]; a_ens = [v for v in a_ens if v is not None]
        b_ens = [B(r.get(f"ensemble_{ax}")) for r in rej]; b_ens = [v for v in b_ens if v is not None]
        ens_rate = (sum(a_ens)/len(a_ens) if a_ens else 0, sum(b_ens)/len(b_ens) if b_ens else 0)
        print(f"\n  {ax} (rate accepted / rejected):")
        print(f"  {'gpt41mini':>14s}: acc={rates['gpt41mini'][0]*100:5.2f}%  rej={rates['gpt41mini'][1]*100:5.2f}%")
        print(f"  {'sonnet-4.6':>14s}: acc={rates['claude_sonnet_4_6'][0]*100:5.2f}%  rej={rates['claude_sonnet_4_6'][1]*100:5.2f}%")
        print(f"  {'haiku-4.5':>14s}: acc={rates['claude_haiku_4_5'][0]*100:5.2f}%  rej={rates['claude_haiku_4_5'][1]*100:5.2f}%")
        print(f"  {'ensemble (OR)':>14s}: acc={ens_rate[0]*100:5.2f}%  rej={ens_rate[1]*100:5.2f}%")
        out_rows.append({
            "axis": ax,
            "kind": "bool_rate",
            "n_acc": len(a_ens), "n_rej": len(b_ens),
            "rate_acc_gpt41mini": rates['gpt41mini'][0],
            "rate_rej_gpt41mini": rates['gpt41mini'][1],
            "rate_acc_sonnet": rates['claude_sonnet_4_6'][0],
            "rate_rej_sonnet": rates['claude_sonnet_4_6'][1],
            "rate_acc_haiku": rates['claude_haiku_4_5'][0],
            "rate_rej_haiku": rates['claude_haiku_4_5'][1],
            "rate_acc_ensemble": ens_rate[0],
            "rate_rej_ensemble": ens_rate[1],
        })

    # ---- Inter-judge Pearson agreement per axis ----
    print(f"\n=== Inter-judge Pearson agreement (per axis, all events) ===")
    for ax in AXES_NUMERIC:
        cols = [f"{j}_{ax}" for j in JUDGES]
        # collect rows with all three judges' scores present
        triples = []
        for r in rows:
            v = [F(r.get(c)) for c in cols]
            if all(x is not None for x in v):
                triples.append(v)
        if len(triples) < 5:
            print(f"  {ax}: insufficient data")
            continue
        arr = np.array(triples)
        print(f"\n  {ax} (n={len(triples)}):")
        for i in range(3):
            for j_ in range(i+1, 3):
                if arr[:, i].std() == 0 or arr[:, j_].std() == 0:
                    r_val = float("nan")
                else:
                    r_val, _ = stats.pearsonr(arr[:, i], arr[:, j_])
                print(f"    {JUDGES[i]:14s} ↔ {JUDGES[j_]:14s}  r={r_val:+.3f}")

    # ---- Mean per-judge score split by event ----
    print(f"\n=== Mean per-judge score (out of 5) ===")
    print(f"  {'axis':22s} {'judge':14s}  {'acc_mean':>9s} {'rej_mean':>9s}")
    for ax in AXES_NUMERIC:
        for j in JUDGES:
            col = f"{j}_{ax}"
            a = [F(r.get(col)) for r in acc]; a = [v for v in a if v is not None]
            b = [F(r.get(col)) for r in rej]; b = [v for v in b if v is not None]
            m_a = np.mean(a) if a else float("nan")
            m_b = np.mean(b) if b else float("nan")
            print(f"  {ax:22s} {j:14s}  {m_a:>9.3f} {m_b:>9.3f}")

    # ---- Save CSV ----
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = sorted(set().union(*[set(r.keys()) for r in out_rows]))
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"\nWrote {args.out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
