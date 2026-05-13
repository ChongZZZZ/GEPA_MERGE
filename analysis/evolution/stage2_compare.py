"""
Stage 2: Mutation vs Merge quantitative compare.

  2.1 Δval distribution table with bootstrap CI + Mann-Whitney U test
  2.2 Δval split by merge algorithm (original / combine_all / summarize_before)
  2.3 Δlength + instruction-keyword count by origin
  2.4 Efficiency: val_gain per metric_call (rough)
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "analysis/evolution/candidates.csv"


def load():
    rows = list(csv.DictReader(open(CSV_PATH)))
    for r in rows:
        r["idx"] = int(r["idx"])
        r["val"] = float(r["val"]) if r["val"] not in (None, "") else None
        r["val_delta"] = (
            float(r["val_delta"]) if r["val_delta"] not in (None, "", "None") else None
        )
        r["prompt_chars"] = int(r["prompt_chars"])
        r["mc_at_disc"] = (
            int(r["mc_at_disc"]) if r["mc_at_disc"] not in (None, "", "None") else None
        )
    return rows


def bootstrap_ci(data, n_boot=5000, alpha=0.05, rng=None):
    if not data:
        return (None, None)
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.array(data, dtype=float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def algo_of(cell):
    """Parse algorithm from cell name like 'original_score_plateau_s0' → 'original'."""
    if cell.startswith("original_"):
        return "original"
    if cell.startswith("combine_all_"):
        return "combine_all"
    if cell.startswith("summarize_before_"):
        return "summarize_before"
    if cell.startswith("nomerge"):
        return "nomerge"
    return "?"


KEYWORD_REGEXES = {
    "bullets": re.compile(r"\n[\-\*]\s|\n\d+[\.\)]\s"),
    "if_clauses": re.compile(r"\bIf\b"),
    "when_clauses": re.compile(r"\bWhen\b"),
    "must_clauses": re.compile(r"\bmust\b", re.I),
    "ensure_clauses": re.compile(r"\bensure\b", re.I),
    "should_clauses": re.compile(r"\bshould\b", re.I),
}


def count_keywords(text):
    return {k: len(rx.findall(text or "")) for k, rx in KEYWORD_REGEXES.items()}


def main():
    rows = load()
    pa = [r for r in rows if r["group"] == "phase_a_qwen"]

    # ===== 2.1 Δval table with CI + Mann-Whitney =====
    print("=" * 90)
    print("2.1  Δval distribution per task × origin (with bootstrap 95% CI + Mann-Whitney U)")
    print("=" * 90)
    print(f"{'task':<10} {'origin':<8} {'n':>4} {'mean':>9} {'CI 95%':>20} "
          f"{'median':>8} {'%pos':>6} {'%zero':>6} {'%neg':>6}")
    rng = np.random.default_rng(0)
    for task in ["hotpotqa", "ifbench", "hover", "musique", "2wiki"]:
        task_rows = [r for r in pa if r["task"] == task and r["val_delta"] is not None]
        for origin in ["reflect", "merge"]:
            items = [r for r in task_rows if r["origin"] == origin]
            if not items:
                continue
            d = [r["val_delta"] for r in items]
            n = len(d)
            mean = np.mean(d)
            lo, hi = bootstrap_ci(d, rng=rng)
            median = np.median(d)
            pos = sum(1 for x in d if x > 0)
            zero = sum(1 for x in d if x == 0)
            neg = sum(1 for x in d if x < 0)
            print(f"  {task:<8} {origin:<8} {n:>4} {mean:>+9.4f} "
                  f"[{lo:>+.4f}, {hi:>+.4f}]   {median:>+8.4f} "
                  f"{pos/n:>5.0%}  {zero/n:>5.0%}  {neg/n:>5.0%}")
        # Mann-Whitney U test reflect vs merge
        ref_d = [r["val_delta"] for r in task_rows if r["origin"] == "reflect"]
        mrg_d = [r["val_delta"] for r in task_rows if r["origin"] == "merge"]
        if ref_d and mrg_d:
            stat, p = mannwhitneyu(ref_d, mrg_d, alternative="two-sided")
            mark = " ★ " if p < 0.05 else "    "
            print(f"  → Mann-Whitney U (reflect vs merge): p={p:.4f} {mark}")
        print()

    # ===== 2.2 Δval by merge algorithm =====
    print("=" * 90)
    print("2.2  Merge Δval split by algorithm (original / combine_all / summarize_before)")
    print("=" * 90)
    print(f"{'task':<10} {'algo':<18} {'n':>4} {'mean Δval':>12} {'median':>8} "
          f"{'%pos':>6} {'%zero':>6} {'%neg':>6}")
    for task in ["hotpotqa", "ifbench", "hover", "musique", "2wiki"]:
        task_rows = [r for r in pa if r["task"] == task and r["origin"] == "merge"
                     and r["val_delta"] is not None]
        for algo in ["original", "combine_all", "summarize_before"]:
            items = [r for r in task_rows if algo_of(r["cell"]) == algo]
            if not items:
                continue
            d = [r["val_delta"] for r in items]
            n = len(d)
            pos = sum(1 for x in d if x > 0)
            zero = sum(1 for x in d if x == 0)
            neg = sum(1 for x in d if x < 0)
            print(f"  {task:<8} {algo:<18} {n:>4} {np.mean(d):>+12.4f} {np.median(d):>+8.4f} "
                  f"{pos/n:>5.0%}  {zero/n:>5.0%}  {neg/n:>5.0%}")
    print()

    # ===== 2.3 Δlength + instruction count =====
    print("=" * 90)
    print("2.3  Δlength + instruction-keyword growth per step, by origin")
    print("=" * 90)
    # Need parent prompt — re-load gepa_state.bin for full text
    # Instead, parent_chars from candidates.csv (already there as parent's prompt_chars)
    # We can compute Δchars = chars - parent's chars (via lookup)
    char_by_idx = {(r["task"], r["cell"], r["idx"]): r["prompt_chars"] for r in pa}

    # Get full prompt text from gepa_state for keyword counting
    cell_dirs = {}
    for r in pa:
        if (r["task"], r["cell"]) in cell_dirs:
            continue
        for root in [
            ROOT / f"results/sec4_configuration_sweep_seed0/qwen3-8b/{r['task']}/{r['cell']}",
            ROOT / f"results/sec4_configuration_sweep_seed0/qwen3-8b/{r['task']}/{r['cell']}",
            ROOT / f"results/sec4_configuration_sweep_seed0/results/sec4_configuration_sweep_seed0/qwen3-8b/musique/{r['cell']}",
            ROOT / f"runs/phase_a_main_qwen/{r['task']}/{r['cell']}",
        ]:
            if (root / "gepa_state.bin").exists():
                cell_dirs[(r["task"], r["cell"])] = root
                break

    # Load all programs (text per candidate) lazily into a dict keyed by (task, cell, idx)
    import pickle
    text_by_idx = {}
    for (task, cell), root in cell_dirs.items():
        try:
            s = pickle.load(open(root / "gepa_state.bin", "rb"))
            for i, c in enumerate(s.get("program_candidates", [])):
                if isinstance(c, dict):
                    text_by_idx[(task, cell, i)] = "\n".join(c.values())
        except Exception:
            continue

    # Compute per-step Δkeyword for reflect vs merge
    import ast
    delta_by_origin = {"reflect": defaultdict(list), "merge": defaultdict(list)}
    delta_chars_by_origin = {"reflect": [], "merge": []}
    for r in pa:
        if r["origin"] not in ("reflect", "merge") or r["val_delta"] is None:
            continue
        parents = ast.literal_eval(r["parents"]) if r["parents"] else []
        if not parents:
            continue
        # parent's max chars
        parent_chars = []
        parent_kw = {k: [] for k in KEYWORD_REGEXES}
        for p in parents:
            if p is None:
                continue
            ck = (r["task"], r["cell"], p)
            if ck in char_by_idx:
                parent_chars.append(char_by_idx[ck])
            if ck in text_by_idx:
                pkw = count_keywords(text_by_idx[ck])
                for k, v in pkw.items():
                    parent_kw[k].append(v)
        if not parent_chars:
            continue
        d_chars = r["prompt_chars"] - max(parent_chars)
        delta_chars_by_origin[r["origin"]].append(d_chars)
        # keyword delta: child keywords - max(parent keywords) per keyword
        my_text = text_by_idx.get((r["task"], r["cell"], r["idx"]))
        if my_text is not None:
            mkw = count_keywords(my_text)
            for k, v in mkw.items():
                base = max(parent_kw[k]) if parent_kw[k] else 0
                delta_by_origin[r["origin"]][k].append(v - base)

    print(f"\n  {'origin':<8} {'metric':<16} {'n':>5} {'mean':>9} {'median':>8}")
    for origin in ["reflect", "merge"]:
        d = delta_chars_by_origin[origin]
        if d:
            print(f"  {origin:<8} {'Δchars':<16} {len(d):>5} {np.mean(d):>+9.0f} {np.median(d):>+8.0f}")
        for k in KEYWORD_REGEXES:
            arr = delta_by_origin[origin][k]
            if not arr:
                continue
            print(f"  {origin:<8} {('Δ'+k):<16} {len(arr):>5} {np.mean(arr):>+9.2f} {np.median(arr):>+8.0f}")
        print()

    # ===== 2.4 Efficiency =====
    print("=" * 90)
    print("2.4  Efficiency: val_gain per metric_call (mc_at_disc difference)")
    print("=" * 90)
    # mc_at_disc[i] - mc_at_disc[i-1] = cost spent producing candidate i
    # we can use this only if candidates are sorted by mc; assume they are
    rng = np.random.default_rng(1)
    print(f"  {'origin':<8} {'n':>5} {'mean cost':>10} {'mean Δval':>10} {'val/cost x1000':>16}")
    for origin in ["reflect", "merge"]:
        items = [r for r in pa if r["origin"] == origin and r["val_delta"] is not None
                 and r["mc_at_disc"] is not None]
        if not items:
            continue
        # Need parent's mc_at_disc to compute step cost
        cost_by_idx = {(r["task"], r["cell"], r["idx"]): r["mc_at_disc"] for r in pa}
        costs = []
        gains = []
        for r in items:
            import ast
            parents = ast.literal_eval(r["parents"]) if r["parents"] else []
            parent_mcs = [cost_by_idx.get((r["task"], r["cell"], p)) for p in parents
                          if p is not None]
            parent_mcs = [m for m in parent_mcs if m is not None]
            if not parent_mcs:
                continue
            step_cost = r["mc_at_disc"] - max(parent_mcs)
            if step_cost <= 0:
                continue
            costs.append(step_cost)
            gains.append(r["val_delta"])
        if costs:
            mean_cost = np.mean(costs)
            mean_gain = np.mean(gains)
            ratio = (mean_gain / mean_cost) * 1000 if mean_cost > 0 else 0
            print(f"  {origin:<8} {len(costs):>5} {mean_cost:>10.1f} {mean_gain:>+10.4f} {ratio:>+16.4f}")


if __name__ == "__main__":
    main()
