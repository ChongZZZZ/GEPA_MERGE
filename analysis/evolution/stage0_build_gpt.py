"""Stage 0 (GPT extension): extract phase_a_gpt candidates from existing run dirs.

Walks the GPT phase-A run directories, loads each gepa_state.bin, derives:
  group, task, cell, idx, origin, parents, val, parent_val, val_delta,
  mc_at_disc, prompt_chars, n_predictors

Appends to analysis/evolution/candidates.csv with group="phase_a_gpt".
Matches the schema produced for phase_a_qwen.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "analysis/evolution/candidates.csv"

# Each entry: (task, parent_dir). 10 cells expected per task.
TASKS = [
    ("hover",     ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hover"),
    ("hotpotqa",  ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hotpotqa"),
    ("ifbench",   ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/ifbench"),
    ("musique",   ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/musique"),
    ("2wiki",     ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/2wiki"),
]


def normalize_cell_name(dir_name: str) -> str:
    """Map raw dir name to the same canonical form used by phase_a_qwen rows.

    Existing CSV uses forms like 'combine_all_budget_proportional_s0'. The
    P*_result subdirs use the same canonical form, so this is identity for
    those. The results/sec4_configuration_sweep_seed0/gpt-4.1-mini/musique side also uses the same form.
    """
    # Drop any trailing ".anomaly_*" suffix on summarize_before_bp s0.
    if ".anomaly_" in dir_name:
        return dir_name.split(".anomaly_")[0]
    return dir_name


def extract_cell(task: str, cell_dir: Path) -> list[dict]:
    state_path = cell_dir / "gepa_state.bin"
    if not state_path.exists():
        return []
    try:
        d = pickle.load(open(state_path, "rb"))
    except Exception as e:
        print(f"  [skip] {cell_dir.name}: load error {e}")
        return []

    progs = d["program_candidates"]
    parents = d["parent_program_for_candidate"]
    val_subs = d["prog_candidate_val_subscores"]
    mc_disc = d["num_metric_calls_by_discovery"]
    n_predictors = len(progs[0]) if progs else 0

    def cand_val(i: int) -> float | None:
        if i is None or i < 0 or i >= len(val_subs):
            return None
        sub = val_subs[i]
        if not sub:
            return None
        # subscores are dict[example_idx -> bool|float] OR list[bool|float]
        if isinstance(sub, dict):
            vals = list(sub.values())
        else:
            vals = list(sub)
        if not vals:
            return None
        return float(sum(vals)) / len(vals)

    def cand_chars(i: int) -> int:
        if i < 0 or i >= len(progs):
            return 0
        return sum(len(v) for v in progs[i].values())

    rows = []
    cell = normalize_cell_name(cell_dir.name)
    for i in range(len(progs)):
        parent_list = parents[i] if i < len(parents) else [None]
        # Determine origin.
        non_none_parents = [p for p in parent_list if p is not None]
        if not non_none_parents:
            origin = "seed"
            parent_val = None
            val_delta = None
        elif len(non_none_parents) == 1:
            origin = "reflect"
            parent_val = cand_val(non_none_parents[0])
        else:
            origin = "merge"
            # Convention: take max of parent vals as the reference parent_val
            # (matches what Qwen pipeline did: gain over the better parent).
            pvs = [cand_val(p) for p in non_none_parents]
            pvs = [p for p in pvs if p is not None]
            parent_val = max(pvs) if pvs else None

        val = cand_val(i)
        if origin == "seed":
            val_delta = None
        else:
            val_delta = (
                val - parent_val if (val is not None and parent_val is not None) else None
            )

        rows.append({
            "group": "phase_a_gpt",
            "task": task,
            "cell": cell,
            "idx": i,
            "origin": origin,
            "parents": str(parent_list),
            "val": "" if val is None else val,
            "parent_val": "" if parent_val is None else parent_val,
            "val_delta": "" if val_delta is None else val_delta,
            "mc_at_disc": mc_disc[i] if i < len(mc_disc) else "",
            "prompt_chars": cand_chars(i),
            "n_predictors": n_predictors,
        })
    return rows


def main():
    all_new_rows: list[dict] = []
    cell_count = 0
    for task, parent_dir in TASKS:
        if not parent_dir.exists():
            print(f"[skip] {parent_dir} (missing)")
            continue
        print(f"=== {task} from {parent_dir} ===")
        for sub in sorted(parent_dir.iterdir()):
            if not sub.is_dir():
                continue
            # Skip if name ends in '.log' or known anomaly tags
            if sub.name.endswith(".log"):
                continue
            if ".anomaly_" in sub.name:
                print(f"  [skip anomaly] {sub.name}")
                continue
            rows = extract_cell(task, sub)
            if rows:
                cell_count += 1
                all_new_rows.extend(rows)
                print(f"  {sub.name}: {len(rows)} candidates")
    print()
    print(f"Total cells: {cell_count}")
    print(f"Total new rows: {len(all_new_rows)}")

    # Append to existing CSV
    existing = list(csv.DictReader(open(CSV)))
    fieldnames = list(existing[0].keys()) if existing else list(all_new_rows[0].keys())
    out_rows = existing + all_new_rows
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            # Ensure all keys present for every row.
            for k in fieldnames:
                r.setdefault(k, "")
            w.writerow(r)
    print(f"Wrote {CSV} ({len(out_rows)} total rows)")


if __name__ == "__main__":
    main()
