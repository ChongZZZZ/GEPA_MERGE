"""B4 — Plots for Phase B timing analysis.

Five plots:
1. catastrophic_case_study.png   ← money shot (2 panels: ifbench + hotpotqa)
2. scatter_panel_by_policy.png   (relative_first × lift, faceted by policy)
3. first_merge_iter_vs_lift_by_policy.png   (raw first_iter; supplementary per fix #2)
4. early_merge_ratio_vs_lift_by_model.png
5. merge_density_vs_lift.png

Per fix #5: catastrophic case study caption is conservative — "consistent
with a failure mode where ..." rather than causal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def F(x):
    try:
        if x is None or x == "" or x == "None" or x == "nan":
            return None
        v = float(x)
        return None if math.isnan(v) else v
    except Exception:
        return None


# ---------- catastrophic case study ----------

def load_score_trajectory(run_dir: Path):
    """Return (discovery_iters, running_max_val_score, merge_events).

    discovery_iters: when each candidate was first added (we use index 0..N-1
    as the discovery order proxy — the more accurate timing uses
    full_program_trace if needed).
    running_max_val_score: cumulative max mean-val-score across candidates so far.
    merge_events: list of (iter, event, accepted) tuples from merge_quality.jsonl.
    """
    state = pickle.load(open(run_dir / "gepa_state.bin", "rb"))
    val_subs = state["prog_candidate_val_subscores"]
    n = len(val_subs)
    means = []
    for ss in val_subs:
        if ss:
            means.append(sum(ss.values()) / len(ss))
        else:
            means.append(None)
    # Running max ignoring None
    cur = -float("inf")
    rmax = []
    cand_indices = []
    for i, m in enumerate(means):
        if m is None:
            continue
        if m > cur:
            cur = m
        rmax.append(cur)
        cand_indices.append(i)

    # Use full_program_trace if available to get the iteration each candidate
    # was discovered. Fall back to index order.
    trace = state.get("full_program_trace") or []
    candidate_iter = {}
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        new_idx = entry.get("new_program_idx") or entry.get("new_candidate_idx")
        if new_idx is not None and new_idx not in candidate_iter:
            candidate_iter[new_idx] = entry.get("i", entry.get("iteration"))

    iters = [candidate_iter.get(i, i) for i in cand_indices]

    merge_events = []
    mq = run_dir / "merge_quality.jsonl"
    if mq.exists():
        for line in mq.open():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            it = ev.get("iteration")
            if it is None:
                continue
            merge_events.append((int(it), ev.get("event")))
    return iters, rmax, merge_events


def find_run_dir(roots: list[Path], cell_id: str) -> Path | None:
    """cell_id = model|dataset|algo|policy"""
    parts = cell_id.split("|")
    if len(parts) != 4:
        return None
    model, ds, algo, policy = parts
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
            bc = run_dir / "best_candidate.json"
            if not bc.exists():
                continue
            try:
                bc_obj = json.load(open(bc))
            except Exception:
                continue
            row_model = "gpt-4.1-mini" if "gpt" in (bc_obj.get("model") or "").lower() else "qwen3-8b"
            if row_model != model:
                continue
            if bc_obj.get("task") != ds:
                continue
            if algo == "nomerge":
                if not bc_obj.get("use_merge"):
                    return run_dir
            else:
                if bc_obj.get("merge_algorithm") == algo and bc_obj.get("merge_start") == policy:
                    return run_dir
    return None


def plot_catastrophic_case_study(roots: list[Path], out_path: Path):
    """2-panel figure: each panel shows the catastrophic cell trajectory
    alongside a same-task "winner" cell trajectory."""
    cases = [
        # (catastrophic_cell, winner_cell, title)
        ("qwen3-8b|ifbench|combine_all|immediate",
         "qwen3-8b|ifbench|original|budget_proportional",
         "Qwen3-8B × IFBench"),
        ("qwen3-8b|hotpotqa|original|immediate",
         "qwen3-8b|hotpotqa|original|score_plateau",
         "Qwen3-8B × HotpotQA"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax, (cata, winner, title) in zip(axes, cases):
        for cell_id, color, label_prefix in [(cata, "tab:red", "catastrophic"),
                                              (winner, "tab:green", "winner")]:
            run_dir = find_run_dir(roots, cell_id)
            if run_dir is None:
                print(f"  [skip] not found: {cell_id}", file=sys.stderr)
                continue
            iters, rmax, merge_events = load_score_trajectory(run_dir)
            if not rmax:
                continue
            # final score value
            final_label = f"{label_prefix}: {cell_id.split('|')[-2]} × {cell_id.split('|')[-1]} (final={rmax[-1]*100:.1f}%)"
            ax.plot(iters, [r * 100 for r in rmax], color=color, label=final_label,
                    linewidth=2)
            # merge events: triangle-up = accepted, triangle-down = rejected
            for it, event in merge_events:
                marker = "^" if event == "accepted" else "v"
                # find score at that iter
                ax.scatter([it], [_score_at(iters, rmax, it) * 100],
                           marker=marker, color=color, s=70,
                           edgecolors="black", linewidths=0.5, zorder=5)

        ax.set_xlabel("iteration")
        ax.set_ylabel("running max val score (%)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Catastrophic case study — early merge trajectories on Qwen3-8B\n"
                 "(△ = accepted merge; ▽ = rejected merge)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}", file=sys.stderr)
    plt.close(fig)


def _score_at(iters, scores, target_iter):
    """Find the running-max score value at or just before target_iter."""
    last = 0.0
    for it, s in zip(iters, scores):
        if it <= target_iter:
            last = s
        else:
            break
    return last


# ---------- scatter plots ----------

def load_cells(csv_path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(csv_path)))
    out = []
    for r in rows:
        if r.get("use_merge") != "True":
            continue
        rec = {
            "model": r["model"], "dataset": r["dataset"],
            "algo": r["algo"], "policy": r["policy"],
            "lift": F(r["lift_over_nomerge"]),
            "relative_first": F(r["relative_first"]),
            "first_merge_iter": F(r["first_merge_iter"]),
            "early_merge_ratio": F(r["early_merge_ratio"]),
            "relative_density": F(r["relative_density"]),
        }
        if rec["lift"] is not None:
            out.append(rec)
    return out


ALGO_MARKER = {"original": "o", "combine_all": "s", "summarize_before": "^"}
MODEL_COLOR = {"gpt-4.1-mini": "tab:blue", "qwen3-8b": "tab:orange"}
POLICY_COLOR = {"immediate": "tab:red", "score_plateau": "tab:green",
                "budget_proportional": "tab:purple"}


def plot_scatter_panel_by_policy(cells: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, pol in zip(axes, ["immediate", "score_plateau", "budget_proportional"]):
        sub = [c for c in cells if c["policy"] == pol and c["relative_first"] is not None]
        for c in sub:
            ax.scatter(c["relative_first"], c["lift"],
                       c=MODEL_COLOR[c["model"]],
                       marker=ALGO_MARKER[c["algo"]],
                       s=80, edgecolors="black", linewidths=0.5,
                       alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("relative_first_merge (= first_iter / total_iters)")
        ax.set_title(f"policy = {pol}  (n={len(sub)})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("lift over NoMerge (pp)")

    # legend
    handles = []
    for m, c in MODEL_COLOR.items():
        handles.append(plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                                   markeredgecolor="black", label=m, markersize=8))
    for a, mk in ALGO_MARKER.items():
        handles.append(plt.Line2D([], [], marker=mk, color="black",
                                   markerfacecolor="white", label=a, markersize=8))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=2, fontsize=8)
    fig.suptitle("Phase B exploratory: relative_first × lift, by policy", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}", file=sys.stderr)
    plt.close(fig)


def plot_first_iter_raw_supplement(cells: list[dict], out_path: Path):
    """Per fix #2: raw first_merge_iter as supplementary."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, pol in zip(axes, ["immediate", "score_plateau", "budget_proportional"]):
        sub = [c for c in cells if c["policy"] == pol and c["first_merge_iter"] is not None]
        for c in sub:
            ax.scatter(c["first_merge_iter"], c["lift"],
                       c=MODEL_COLOR[c["model"]],
                       marker=ALGO_MARKER[c["algo"]],
                       s=80, edgecolors="black", linewidths=0.5, alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("first_merge_iter (raw)")
        ax.set_title(f"policy = {pol}  (n={len(sub)})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("lift over NoMerge (pp)")
    fig.suptitle("Supplementary: raw first_merge_iter × lift, by policy\n"
                 "(main analysis uses relative_first; raw differs across runs with different total_iters)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}", file=sys.stderr)
    plt.close(fig)


def plot_early_merge_ratio_by_model(cells: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, m in zip(axes, ["gpt-4.1-mini", "qwen3-8b"]):
        sub = [c for c in cells if c["model"] == m and c["early_merge_ratio"] is not None]
        for c in sub:
            ax.scatter(c["early_merge_ratio"], c["lift"],
                       c=POLICY_COLOR[c["policy"]],
                       marker=ALGO_MARKER[c["algo"]],
                       s=80, edgecolors="black", linewidths=0.5, alpha=0.85)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        ax.set_xlabel("early_merge_ratio (merges fired in first 25% of iters)")
        ax.set_title(f"model = {m}  (n={len(sub)})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("lift over NoMerge (pp)")
    handles = []
    for p, c in POLICY_COLOR.items():
        handles.append(plt.Line2D([], [], marker="o", color="w", markerfacecolor=c,
                                   markeredgecolor="black", label=p, markersize=8))
    for a, mk in ALGO_MARKER.items():
        handles.append(plt.Line2D([], [], marker=mk, color="black",
                                   markerfacecolor="white", label=a, markersize=8))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=2, fontsize=8)
    fig.suptitle("Phase B exploratory: early_merge_ratio × lift, by model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}", file=sys.stderr)
    plt.close(fig)


def plot_merge_density(cells: list[dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sub = [c for c in cells if c["relative_density"] is not None]
    for c in sub:
        ax.scatter(c["relative_density"], c["lift"],
                   c=MODEL_COLOR[c["model"]],
                   marker=ALGO_MARKER[c["algo"]],
                   s=70, edgecolors="black", linewidths=0.5, alpha=0.85)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("relative_density (n_attempts / total_iters)")
    ax.set_ylabel("lift over NoMerge (pp)")
    ax.set_title(f"Phase B exploratory: merge density × lift  (n={len(sub)})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"  saved {out_path}", file=sys.stderr)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell_timing_csv", required=True, type=Path)
    ap.add_argument("--runs_roots", nargs="+", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=== plotting catastrophic case study ===", file=sys.stderr)
    plot_catastrophic_case_study(args.runs_roots,
                                  args.out_dir / "catastrophic_case_study.png")

    cells = load_cells(args.cell_timing_csv)
    print(f"\n=== plotting {len(cells)} cells ===", file=sys.stderr)
    plot_scatter_panel_by_policy(cells, args.out_dir / "scatter_panel_by_policy.png")
    plot_first_iter_raw_supplement(cells, args.out_dir / "first_merge_iter_vs_lift_by_policy.png")
    plot_early_merge_ratio_by_model(cells, args.out_dir / "early_merge_ratio_vs_lift_by_model.png")
    plot_merge_density(cells, args.out_dir / "merge_density_vs_lift.png")


if __name__ == "__main__":
    main()
