"""Side-by-side comparison: adaptive live re-run vs. Phase A baselines.

Reads `test_eval.json` from:
  - `--adaptive_root` cells named `<model>_<task>_adaptive_s<seed>`
  - `--baseline_roots` cells named `<task>/<algo>_<start>_s<seed>` (Phase A layout)

Emits a markdown report grouped by (task, model) with three columns:
  - NoMerge baseline
  - Best fixed merge (max test_score over the 9 algo×start cells)
  - Adaptive merge

Read-only.

Usage::

    PYTHONPATH=src .venv/bin/python -m experiments.analysis.compare_adaptive_to_phase_a \\
        --adaptive_root runs/adaptive_live_v1 \\
        --baseline_roots P2_result/phase_a_main P3_result/phase_a_main \\
                         P4_result/phase_a_main runs/phase_a_main_qwen \\
        --out experiments/analysis/output/adaptive_live_v1_compare.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _read_test_eval(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _detect_model(runs_root: Path) -> str:
    name = "/".join(runs_root.parts).lower()
    if "qwen" in name:
        return "qwen3-8b"
    return "gpt-4.1-mini"


def _walk_baseline(roots: list[Path]) -> dict:
    """Return: scores[(task, model, seed)][config] = test_score.

    config is one of: "nomerge", "<algo>_<start>" for the merge cells.
    """
    scores: dict = defaultdict(dict)
    for root in roots:
        if not root.exists():
            continue
        model = _detect_model(root)
        for task_dir in sorted(root.iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            for cell_dir in sorted(task_dir.iterdir()):
                if not cell_dir.is_dir():
                    continue
                te = _read_test_eval(cell_dir / "test_eval.json")
                if te is None or te.get("status") != "ok":
                    continue
                # Parse cell name e.g. "combine_all_immediate_s0" or "nomerge_s0"
                name = cell_dir.name
                seed = -1
                if "_s" in name:
                    parts = name.rsplit("_s", 1)
                    try:
                        seed = int(parts[1])
                        name = parts[0]
                    except ValueError:
                        pass
                config = name  # "nomerge" or "<algo>_<start>"
                key = (task, model, seed)
                # Latest write wins; for duplicates across run-roots we'd
                # ideally pick the most-recent, but test_eval.json doesn't
                # carry a timestamp.
                scores[key][config] = float(te.get("test_score", 0))
    return scores


def _walk_adaptive(root: Path) -> dict:
    """Return: scores[(task, model, seed)] = test_score for adaptive cells."""
    out: dict = {}
    if not root.exists():
        return out
    for cell_dir in sorted(root.iterdir()):
        if not cell_dir.is_dir():
            continue
        # Expected: <model>_<task>_adaptive_s<seed>
        name = cell_dir.name
        if "_adaptive_s" not in name:
            continue
        prefix, _, seed_str = name.rpartition("_s")
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        prefix = prefix[: -len("_adaptive")] if prefix.endswith("_adaptive") else prefix
        # prefix is now "<model_label>_<task>"
        if prefix.startswith("gpt_"):
            model = "gpt-4.1-mini"
            task = prefix[4:]
        elif prefix.startswith("qwen_"):
            model = "qwen3-8b"
            task = prefix[5:]
        else:
            continue
        te = _read_test_eval(cell_dir / "test_eval.json")
        if te is None or te.get("status") != "ok":
            continue
        out[(task, model, seed)] = float(te.get("test_score", 0))
    return out


def _build_report(baseline: dict, adaptive: dict) -> str:
    keys = sorted(set(baseline.keys()) | set(adaptive.keys()))
    if not keys:
        return "(no cells found)\n"
    lines = ["# Adaptive Live Re-Run — comparison vs. Phase A baselines", ""]
    lines.append("All scores are test-set evaluations from `test_eval.json` (`test_score`).")
    lines.append(
        "**Best fixed merge** = max test_score across the available "
        "{original, combine_all, summarize_before} × {immediate, score_plateau, "
        "budget_proportional} cells in Phase A archive."
    )
    lines.append("")
    lines.append("| task | model | seed | NoMerge | Best fixed merge | Adaptive | Δ vs NoMerge | Δ vs Best fixed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for task, model, seed in keys:
        bs = baseline.get((task, model, seed), {})
        nomerge = bs.get("nomerge")
        merge_cells = {k: v for k, v in bs.items() if k != "nomerge"}
        best_merge = max(merge_cells.values()) if merge_cells else None
        adapt = adaptive.get((task, model, seed))
        def _fmt(x):
            return f"{x:.2f}" if isinstance(x, (int, float)) else "—"
        def _delta(a, b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return f"{a - b:+.2f}"
            return "—"
        lines.append(
            f"| {task} | {model} | {seed} | {_fmt(nomerge)} | {_fmt(best_merge)} | "
            f"{_fmt(adapt)} | {_delta(adapt, nomerge)} | {_delta(adapt, best_merge)} |"
        )
    lines.append("")
    lines.append(
        "**Reading the deltas:** the §16.5 lower-bound argument predicts "
        "Δ vs NoMerge ≥ −2 pp (seed noise). Δ vs Best fixed should be ≥ 0 "
        "on cells where the best fixed merge was a healthy lift, and clearly "
        "positive on the §15 catastrophic cells (where the best fixed merge "
        "was a regression that adaptive avoids)."
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--adaptive_root", required=True, type=Path)
    p.add_argument("--baseline_roots", nargs="+", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    baseline = _walk_baseline(args.baseline_roots)
    adaptive = _walk_adaptive(args.adaptive_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_build_report(baseline, adaptive))
    print(
        f"baseline cells: {sum(len(v) for v in baseline.values())} "
        f"across {len(baseline)} (task,model,seed) groups"
    )
    print(f"adaptive cells: {len(adaptive)}")
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
