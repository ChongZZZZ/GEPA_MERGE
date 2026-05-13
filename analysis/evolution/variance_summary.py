"""Variance summary across multi-seed cells we currently have on disk.

For each (model, task, config) collected over seeds 0, 1, 2:
  - raw test_score per seed
  - within-seed Δ vs NoMerge (Δ_s = score(config, s) - score(nomerge, s))
  - mean Δ and sample std across the seeds present

Sources:
  GPT-4.1-mini:
    s=0 archive:
      hotpotqa  : results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hotpotqa/<cfg>_s0
      ifbench   : results/sec4_configuration_sweep_seed0/gpt-4.1-mini/ifbench/<cfg>_s0
      hover     : results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hover/<cfg>_s0
      musique   : results/sec4_configuration_sweep_seed0/gpt-4.1-mini/musique/<cfg>_s0
      2wiki     : results/sec4_configuration_sweep_seed0/gpt-4.1-mini/2wiki/<cfg>_s0
    s=1, s=2 multiseed:
      hotpotqa, ifbench  : paper_final/p2/logs/test_eval_gpt_<task>_<cfg>_s{1,2}.csv
      hover, musique, 2wiki(adaptive): paper_final/p1/logs/test_eval_gpt_<task>_<cfg>_s{1,2}.csv
  Qwen3-8B:
    s=0 archive:
      hotpotqa  : results/sec4_configuration_sweep_seed0/qwen3-8b/hotpotqa
      hover     : results/sec4_configuration_sweep_seed0/qwen3-8b/hover
      ifbench   : results/sec4_configuration_sweep_seed0/qwen3-8b/ifbench
      musique   : results/sec4_configuration_sweep_seed0/qwen3-8b/musique
      2wiki     : runs/phase_a_main_qwen/2wiki
    s=1, s=2 multiseed:
      adaptive_merge/runs_phase_a_seed{1,2}/qwen_<task>_<cfg>_s{1,2}/test_eval.json
      paper_final/p3/logs/test_eval_qwen_<task>_<cfg>_s{1,2}.csv (current session)
      paper_final/p4/logs/test_eval_qwen_<task>_<cfg>_s{1,2}.csv (current session)
"""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def score_from_csv(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get("test_score")
                if v not in (None, "", "None"):
                    return float(v)
    except Exception:
        return None
    return None


def score_from_json(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        v = d.get("test_score")
        return float(v) if v is not None else None
    except Exception:
        return None


# ───────────────────────── GPT ─────────────────────────
# Maps (task, cfg) -> {seed: score}
GPT_S0_ROOT = {
    "hotpotqa": ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hotpotqa",
    "ifbench":  ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/ifbench",
    "hover":    ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/hover",
    "musique":  ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/musique",
    "2wiki":    ROOT / "results/sec4_configuration_sweep_seed0/gpt-4.1-mini/2wiki",
}
GPT_S0_CFG_NAME = {
    "nomerge":  "nomerge",
    "orig_imm": "original_immediate",
    "adaptive": None,  # no s=0 adaptive run on most GPT tasks
}
# adaptive_s0 paths (some have, some don't)
GPT_ADAPTIVE_S0 = {
    "hotpotqa": ROOT / "results/sec6_adaptive_seed0_gpt/gpt_hotpotqa_adaptive_s0",
    "ifbench":  ROOT / "results/sec6_adaptive_seed0_gpt/gpt_ifbench_adaptive_s0",
    "hover":    ROOT / "results/sec6_adaptive_seed0_gpt/gpt_hover_adaptive_s0",
    "musique":  ROOT / "results/sec6_adaptive_seed0_gpt/gpt_musique_adaptive_s0",
    "2wiki":    ROOT / "results/sec6_adaptive_seed0_gpt/gpt_2wiki_adaptive_s0",
}
# s=1/s=2 sources
# Non-adaptive cells (nomerge, orig_imm) live under sec4_variance_audit_seed{1,2}.
# Adaptive cells live under sec6_behavioral_probe/gpt-4.1-mini/<task>/adaptive_s<seed>/,
# with a flat CSV mirror in sec6_behavioral_probe/test_eval_csvs/ for callers that
# prefer reading by filename.
GPT_VARIANCE_ROOTS = [
    ROOT / "results/sec4_variance_audit_seed1/gpt-4.1-mini",
    ROOT / "results/sec4_variance_audit_seed2/gpt-4.1-mini",
]
GPT_PROBE_RUNS_ROOT = ROOT / "results/sec6_behavioral_probe/gpt-4.1-mini"
GPT_PROBE_CSV_ROOT  = ROOT / "results/sec6_behavioral_probe/test_eval_csvs"

GPT_TASKS  = ["hotpotqa", "ifbench", "hover", "musique", "2wiki"]
GPT_CFGS   = ["nomerge", "orig_imm", "adaptive"]


def gpt_score(task: str, cfg: str, seed: int) -> float | None:
    if seed == 0:
        if cfg == "adaptive":
            p = GPT_ADAPTIVE_S0[task] / "test_eval.json"
            return score_from_json(p)
        cfg_full = GPT_S0_CFG_NAME[cfg]
        p = GPT_S0_ROOT[task] / f"{cfg_full}_s0" / "test_eval.json"
        return score_from_json(p)
    # s=1, s=2
    # 1) Non-adaptive (nomerge, orig_imm): variance-audit subtree
    if cfg in GPT_S0_CFG_NAME and GPT_S0_CFG_NAME.get(cfg):
        cfg_full = GPT_S0_CFG_NAME[cfg]
        for root in GPT_VARIANCE_ROOTS:
            p = root / task / f"{cfg_full}_s{seed}" / "test_eval.json"
            if p.exists():
                v = score_from_json(p)
                if v is not None:
                    return v
    if cfg == "nomerge":
        for root in GPT_VARIANCE_ROOTS:
            p = root / task / f"nomerge_s{seed}" / "test_eval.json"
            if p.exists():
                v = score_from_json(p)
                if v is not None:
                    return v
    # 2) Adaptive (cfg == "adaptive"): prefer the CSV mirror in test_eval_csvs/
    # (those are the snapshots whose values are quoted in Figure 4 / Table 7),
    # then fall back to the run-dir JSON for any cell not in the CSV mirror.
    p_csv = GPT_PROBE_CSV_ROOT / f"test_eval_gpt_{task}_{cfg}_s{seed}.csv"
    if p_csv.exists():
        v = score_from_csv(p_csv)
        if v is not None:
            return v
    if cfg == "adaptive":
        p_json = GPT_PROBE_RUNS_ROOT / task / f"adaptive_s{seed}" / "test_eval.json"
        if p_json.exists():
            v = score_from_json(p_json)
            if v is not None:
                return v
    return None


# ───────────────────────── Qwen ─────────────────────────
QWEN_S0_ROOT = {
    "hotpotqa": ROOT / "results/sec4_configuration_sweep_seed0/qwen3-8b/hotpotqa",
    "hover":    ROOT / "results/sec4_configuration_sweep_seed0/qwen3-8b/hover",
    "ifbench":  ROOT / "results/sec4_configuration_sweep_seed0/qwen3-8b/ifbench",
    "musique":  ROOT / "results/sec4_configuration_sweep_seed0/qwen3-8b/musique",
    "2wiki":    ROOT / "results/sec4_configuration_sweep_seed0/qwen3-8b/2wiki",
}
QWEN_S0_CFG_NAME = {
    "nomerge":   "nomerge",
    "orig_imm":  "original_immediate",
    "orig_bp":   "original_budget_proportional",
    "combine_imm": "combine_all_immediate",
    "adaptive":  None,  # archived elsewhere
}
QWEN_ADAPTIVE_S0 = {
    "hotpotqa": ROOT / "results/sec6_adaptive_seed0_qwen/qwen_hotpotqa_adaptive_s0",
    "ifbench":  ROOT / "results/sec6_adaptive_seed0_qwen/qwen_ifbench_adaptive_s0",
    "hover":    ROOT / "results/sec6_adaptive_seed0_qwen/qwen_hover_adaptive_s0",
    "musique":  ROOT / "results/sec6_adaptive_seed0_qwen/qwen_musique_adaptive_s0",
    "2wiki":    ROOT / "results/sec6_adaptive_seed0_qwen/qwen_2wiki_adaptive_s0",
}

# s=1, s=2 multiseed
# Variance-audit cells now live at
#   sec4_variance_audit_seed{1,2}/qwen3-8b/<task>/<algo>_<timing>_s<seed>/test_eval.json
# Behavioral-probe seeds 1, 2 stay under the paper_final-style p3/p4 trees.
QWEN_VARIANCE_ROOTS = [
    ROOT / "results/sec4_variance_audit_seed1/qwen3-8b",
    ROOT / "results/sec4_variance_audit_seed2/qwen3-8b",
]
QWEN_PROBE_RUNS_ROOT = ROOT / "results/sec6_behavioral_probe/qwen3-8b"
QWEN_PROBE_CSV_ROOT  = ROOT / "results/sec6_behavioral_probe/test_eval_csvs"

# Legacy abbreviation → full cfg name (used to translate the
# `qwen_<task>_<cfg>_s<seed>` CSV filenames in test_eval_csvs/).
QWEN_CFG_ABBREV = {
    "nomerge":      "nomerge",
    "adaptive":     "adaptive",
    "orig_imm":     "original_immediate",
    "orig_plat":    "original_score_plateau",
    "orig_bp":      "original_budget_proportional",
    "combine_imm":  "combine_all_immediate",
    "combine_plat": "combine_all_score_plateau",
    "combine_bp":   "combine_all_budget_proportional",
    "sum_imm":      "summarize_before_immediate",
    "sum_plat":     "summarize_before_score_plateau",
    "sum_bp":       "summarize_before_budget_proportional",
}


def qwen_score(task: str, cfg: str, seed: int) -> float | None:
    if seed == 0:
        if cfg == "adaptive":
            p = QWEN_ADAPTIVE_S0[task] / "test_eval.json"
            return score_from_json(p)
        # Prefer the sec4 configuration-sweep nomerge_s0 (the matched
        # baseline used by Tables 3, 7, 8 in the paper). Fall back to the
        # behavioral-probe nomerge_s0 only when the sweep cell is missing.
        cfg_full = QWEN_S0_CFG_NAME.get(cfg)
        if cfg_full:
            p = QWEN_S0_ROOT[task] / f"{cfg_full}_s0" / "test_eval.json"
            v = score_from_json(p)
            if v is not None:
                return v
        if cfg == "nomerge":
            p_json = QWEN_PROBE_RUNS_ROOT / task / f"nomerge_s0" / "test_eval.json"
            if p_json.exists():
                v = score_from_json(p_json)
                if v is not None:
                    return v
            p_csv = QWEN_PROBE_CSV_ROOT / f"test_eval_qwen_{task}_{cfg}_s0.csv"
            if p_csv.exists():
                v = score_from_csv(p_csv)
                if v is not None:
                    return v
        return None

    # seed >= 1
    # 1) Variance-audit cells (orig_imm / orig_bp / combine_imm / sum_imm / etc.)
    cfg_full = QWEN_S0_CFG_NAME.get(cfg)
    if cfg_full:
        for root in QWEN_VARIANCE_ROOTS:
            p = root / task / f"{cfg_full}_s{seed}" / "test_eval.json"
            if p.exists():
                v = score_from_json(p)
                if v is not None:
                    return v

    # 2) Behavioral-probe cells (cfg in {"adaptive", "nomerge"} typically)
    cfg_full_probe = QWEN_CFG_ABBREV.get(cfg, cfg)
    p_json = QWEN_PROBE_RUNS_ROOT / task / f"{cfg_full_probe}_s{seed}" / "test_eval.json"
    if p_json.exists():
        v = score_from_json(p_json)
        if v is not None:
            return v
    p_csv = QWEN_PROBE_CSV_ROOT / f"test_eval_qwen_{task}_{cfg}_s{seed}.csv"
    if p_csv.exists():
        v = score_from_csv(p_csv)
        if v is not None:
            return v
    return None


# ───────────────────────── Report ─────────────────────────
def fmt(v: float | None, w: int = 6) -> str:
    return f"{v:>{w}.2f}" if isinstance(v, float) else f"{'--':>{w}}"


def report(model: str, tasks: list[str], cfgs: list[str], score_fn):
    print(f"\n{'='*100}")
    print(f"  {model} — per-seed test scores and within-seed Δ vs NoMerge")
    print(f"{'='*100}")
    print(f"{'task':10s} {'config':12s}   {'s=0':>6} {'s=1':>6} {'s=2':>6}    "
          f"{'Δ_s0':>6} {'Δ_s1':>6} {'Δ_s2':>6}    {'mean':>6} {'std':>5}")
    print("-" * 100)
    for task in tasks:
        nm = {s: score_fn(task, "nomerge", s) for s in (0, 1, 2)}
        for cfg in cfgs:
            sc = {s: score_fn(task, cfg, s) for s in (0, 1, 2)}
            deltas = []
            for s in (0, 1, 2):
                if sc[s] is not None and nm[s] is not None:
                    deltas.append(sc[s] - nm[s])
            if cfg == "nomerge":
                # NoMerge is baseline -> all Δ = 0 by definition; only show seeds
                print(f"{task:10s} {cfg:12s}   {fmt(sc[0])} {fmt(sc[1])} {fmt(sc[2])}    "
                      f"{'--':>6} {'--':>6} {'--':>6}    {'--':>6} {'--':>5}")
                continue
            mean = statistics.mean(deltas) if deltas else None
            sd = statistics.stdev(deltas) if len(deltas) >= 2 else None
            d_str = []
            for s in (0, 1, 2):
                if sc[s] is not None and nm[s] is not None:
                    d_str.append(f"{(sc[s]-nm[s]):>+6.2f}")
                else:
                    d_str.append(f"{'--':>6}")
            mean_s = f"{mean:>+6.2f}" if mean is not None else f"{'--':>6}"
            sd_s = f"{sd:>5.2f}" if sd is not None else f"{'--':>5}"
            print(f"{task:10s} {cfg:12s}   {fmt(sc[0])} {fmt(sc[1])} {fmt(sc[2])}    "
                  f"{d_str[0]} {d_str[1]} {d_str[2]}    {mean_s} {sd_s}")


def main():
    report("GPT-4.1-mini", GPT_TASKS, GPT_CFGS, gpt_score)
    qwen_tasks = ["hotpotqa", "hover", "ifbench", "musique", "2wiki"]
    qwen_cfgs  = ["nomerge", "orig_imm", "orig_bp", "combine_imm", "adaptive"]
    report("Qwen3-8B", qwen_tasks, qwen_cfgs, qwen_score)


if __name__ == "__main__":
    main()
