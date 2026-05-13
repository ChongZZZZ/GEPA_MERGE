"""Offline Tier B judge driver.

Walks a directory of completed GEPA runs, locates each run's
``candidates.jsonl`` + ``merge_quality.jsonl`` sidecars, filters events to the
pre-registered cells, and writes an augmented JSONL with ``tier_b`` blocks.

This script **never** touches the optimizer budget. It re-reads sidecars
produced by :class:`gepa.logging.merge_quality_callback.MergeQualityCallback`
and calls the judge via LiteLLM (or a user-provided callable in tests).

Typical usage::

    uv run python experiments/analysis/run_judge.py \\
        --runs_root runs/ \\
        --cells random,immediate complementary,diversity score,immediate \\
        --judge_lm openai/gpt-4o-mini \\
        --out judged/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make the gepa package importable even when the script is run directly.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from gepa.strategies.merge_quality_judge import (  # noqa: E402
    JudgeRunStats,
    cell_filter_from_set,
    judge_events,
    load_candidates,
    load_events,
    write_events,
)


def _build_litellm_judge(model_name: str):
    """Return a ``LanguageModel`` callable backed by LiteLLM."""
    import litellm  # local import so the stub tests don't need the dep

    def _judge(prompt):
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        completion = litellm.completion(model=model_name, messages=messages)
        return completion.choices[0].message.content  # type: ignore[attr-defined]

    return _judge


def _discover_run_dirs(runs_root: Path) -> list[Path]:
    """Return every directory under ``runs_root`` containing both sidecars."""
    out: list[Path] = []
    if not runs_root.exists():
        return out
    for path in sorted(runs_root.rglob("merge_quality.jsonl")):
        run_dir = path.parent
        if (run_dir / "candidates.jsonl").exists():
            out.append(run_dir)
    return out


def _parse_cells(raw: list[str]) -> set[tuple[str, str]]:
    """Parse ``A,B`` strings into ``(selection, start)`` tuples."""
    cells: set[tuple[str, str]] = set()
    for item in raw:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"Bad --cells entry {item!r}; expected 'selection,start'")
        cells.add((parts[0], parts[1]))
    return cells


def run(
    runs_root: str,
    cells: set[tuple[str, str]],
    judge_lm_name: str,
    out_root: str,
    only_accepted: bool,
    dry_run: bool,
    judge_callable=None,
) -> JudgeRunStats:
    """Main driver; returns aggregated stats across all runs."""
    runs_root_p = Path(runs_root)
    out_root_p = Path(out_root)
    out_root_p.mkdir(parents=True, exist_ok=True)
    filt = cell_filter_from_set(cells)

    run_dirs = _discover_run_dirs(runs_root_p)
    if not run_dirs:
        print(f"[run_judge] No runs found under {runs_root_p}")
        return JudgeRunStats()

    if judge_callable is None and not dry_run:
        judge_callable = _build_litellm_judge(judge_lm_name)

    total = JudgeRunStats()
    for run_dir in run_dirs:
        cand_path = run_dir / "candidates.jsonl"
        evt_path = run_dir / "merge_quality.jsonl"
        prompts = load_candidates(str(cand_path))
        events = load_events(str(evt_path))
        matching = [e for e in events if filt(e)]
        print(
            f"[run_judge] {run_dir}: {len(events)} events, {len(matching)} in pre-registered cells"
        )
        if dry_run or not matching:
            continue

        augmented, stats = judge_events(
            events=events,
            prompts_by_idx=prompts,
            judge_lm=judge_callable,
            cell_filter=filt,
            judge_lm_name=judge_lm_name,
            only_accepted=only_accepted,
        )

        # Mirror the run's relative path under out_root so collisions across runs
        # are impossible.
        rel = run_dir.relative_to(runs_root_p) if run_dir.is_relative_to(runs_root_p) else Path(run_dir.name)
        out_dir = out_root_p / rel
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "merge_quality_judged.jsonl"
        write_events(str(out_path), augmented)

        meta = {
            "run_dir": str(run_dir),
            "judge_lm": judge_lm_name,
            "cells": sorted([list(c) for c in cells]),
            "total_judge_calls": stats.total_judge_calls,
            "events_scored": stats.events_scored,
            "events_skipped": stats.events_skipped,
            "parse_errors": stats.parse_errors,
            "skipped_reasons": stats.skipped_reasons,
            "only_accepted": only_accepted,
        }
        with open(out_dir / "judge_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        total.total_judge_calls += stats.total_judge_calls
        total.events_scored += stats.events_scored
        total.events_skipped += stats.events_skipped
        total.parse_errors += stats.parse_errors
        for k, v in stats.skipped_reasons.items():
            total.skipped_reasons[k] = total.skipped_reasons.get(k, 0) + v

    print(
        f"[run_judge] Done. total_judge_calls={total.total_judge_calls}, "
        f"events_scored={total.events_scored}, parse_errors={total.parse_errors}"
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Tier B merge-quality judge.")
    parser.add_argument("--runs_root", required=True, help="Directory containing GEPA run subdirs.")
    parser.add_argument(
        "--cells",
        nargs="+",
        required=True,
        help='Pre-registered cells as "selection,start" tokens, e.g. "random,immediate".',
    )
    parser.add_argument("--judge_lm", default="openai/gpt-4o-mini")
    parser.add_argument("--out", default="judged/", help="Output directory.")
    parser.add_argument(
        "--only_accepted",
        action="store_true",
        help="Score only accepted merges (cheaper; loses rejected-vs-accepted contrast).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List what would be judged without calling the LLM.",
    )
    args = parser.parse_args()

    cells = _parse_cells(args.cells)
    run(
        runs_root=args.runs_root,
        cells=cells,
        judge_lm_name=args.judge_lm,
        out_root=args.out,
        only_accepted=args.only_accepted,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
