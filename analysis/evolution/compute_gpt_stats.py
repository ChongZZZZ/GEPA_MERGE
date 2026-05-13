"""Compute (a) reflect-vs-merge per-step Δval per task for phase_a_gpt
and (b) val/test correlation per task by reading per-cell test_eval and
val-best scores.
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "analysis/evolution/candidates.csv"

TASK_DIRS = {
    "hover":     ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hover",
    "hotpotqa":  ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hotpotqa",
    "ifbench":   ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/ifbench",
    "musique":   ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/musique",
    "2wiki":     ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/2wiki",
}


def reflect_vs_merge():
    rows = list(csv.DictReader(open(CSV)))
    rows = [r for r in rows if r["group"] == "phase_a_gpt"]
    by_task = defaultdict(lambda: {"reflect": [], "merge": []})
    by_task_mc = defaultdict(lambda: {"reflect": [], "merge": []})
    for r in rows:
        if r["origin"] not in ("reflect", "merge"):
            continue
        if r["val_delta"] in ("", "None", None):
            continue
        dv = float(r["val_delta"])
        by_task[r["task"]][r["origin"]].append(dv)
        # cost per step ~ mc_at_disc[i] - mc_at_disc[parent]
        # approximation: just take mc_at_disc itself
        if r["mc_at_disc"] not in ("", "None", None):
            by_task_mc[r["task"]][r["origin"]].append(int(r["mc_at_disc"]))

    print("\n=== Reflect vs Merge per-step Δval (GPT phase-A) ===\n")
    print(f"{'task':12s}  {'reflect_n':>8s} {'merge_n':>7s}   "
          f"{'reflect_mean':>12s} {'merge_mean':>10s}   "
          f"{'reflect_sd':>10s} {'merge_sd':>8s}   "
          f"{'merge_zero_pct':>14s}")
    print("-" * 100)
    results = {}
    for task in ("hotpotqa", "ifbench", "hover", "musique", "2wiki"):
        ref = by_task[task]["reflect"]
        mer = by_task[task]["merge"]
        rm = statistics.mean(ref) if ref else float("nan")
        mm = statistics.mean(mer) if mer else float("nan")
        rs = statistics.pstdev(ref) if len(ref) > 1 else 0.0
        ms = statistics.pstdev(mer) if len(mer) > 1 else 0.0
        zero_pct = (sum(1 for d in mer if d == 0.0) / len(mer) * 100.0) if mer else 0.0
        print(f"{task:12s}  {len(ref):>8d} {len(mer):>7d}   "
              f"{rm:>+12.4f} {mm:>+10.4f}   "
              f"{rs:>10.4f} {ms:>8.4f}   "
              f"{zero_pct:>13.1f}%")
        results[task] = {
            "reflect_n": len(ref), "merge_n": len(mer),
            "reflect_mean": rm, "merge_mean": mm,
            "reflect_sd": rs, "merge_sd": ms,
            "merge_zero_pct": zero_pct,
        }
    return results


def val_test_correlation():
    """Per task, compute Pearson r between val-best score and test score
    across the 10 phase-A cells. val-best comes from gepa_state.bin:
    max over all candidates of mean(subscore values)."""
    import pickle

    def best_val_from_state(state_path: Path) -> float | None:
        try:
            d = pickle.load(open(state_path, "rb"))
        except Exception:
            return None
        subs = d.get("prog_candidate_val_subscores", [])
        means = []
        for s in subs:
            if isinstance(s, dict):
                vals = list(s.values())
            else:
                vals = list(s)
            if vals:
                means.append(sum(vals) / len(vals))
        return max(means) if means else None

    print("\n=== Val/Test correlation per task (GPT phase-A, n=10 cells/task) ===\n")
    print(f"{'task':12s}  {'n':>3s}  {'pearson_r':>10s}")
    print("-" * 40)
    results = {}
    for task, parent_dir in TASK_DIRS.items():
        val_scores, test_scores = [], []
        for cell_dir in sorted(parent_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            if ".anomaly_" in cell_dir.name:
                continue
            state = cell_dir / "gepa_state.bin"
            te = cell_dir / "test_eval.json"
            if not state.exists() or not te.exists():
                continue
            v = best_val_from_state(state)
            try:
                t = json.load(open(te)).get("test_score")
            except Exception:
                t = None
            if v is None or t is None:
                continue
            val_scores.append(v)
            test_scores.append(t)
        if len(val_scores) < 3:
            print(f"{task:12s}  {len(val_scores):>3d}  (insufficient)")
            results[task] = {"n": len(val_scores), "r": None}
            continue
        v = np.array(val_scores)
        t = np.array(test_scores)
        r = float(np.corrcoef(v, t)[0, 1])
        print(f"{task:12s}  {len(val_scores):>3d}  {r:>+10.3f}")
        results[task] = {"n": len(val_scores), "r": r}
    return results


def main():
    rvs = reflect_vs_merge()
    vt = val_test_correlation()
    # Print LaTeX table snippets for easy paste
    print("\n\n=== LaTeX table snippet — reflect vs merge (GPT row block) ===")
    for task in ("hotpotqa", "ifbench", "hover", "musique", "2wiki"):
        r = rvs[task]
        print(f"{task:10s} & ${r['reflect_mean']:+.3f}$ & ${r['merge_mean']:+.3f}$ "
              f"& {r['reflect_sd']:.3f} & {r['merge_sd']:.3f} \\\\")
    print("\n=== LaTeX table snippet — val/test r (GPT) ===")
    for task in ("hotpotqa", "musique", "ifbench", "hover", "2wiki"):
        r = vt.get(task, {})
        if r.get("r") is None:
            print(f"{task:10s} & --     & insufficient \\\\")
        else:
            print(f"{task:10s} & ${r['r']:+.2f}$ \\\\")


if __name__ == "__main__":
    main()
