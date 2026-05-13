"""A2 + A3 — Compute Step 4 (semantic) and Step 5 (judge) metrics on
recovered deterministic rejected merge events.

Input: `recovered_rejected_prompts.jsonl` from `recover_rejected_prompts.py`
       (each line has merged_program + p1_program + p2_program + ancestor_program)

Output: `recovered_rejected_metrics.jsonl` — same rows with semantic SNR/SLC
        and judge rubric scores added.

The semantic and judge functions are imported from `merge_forensics.py`
to ensure consistency with how they're applied to accepted merges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the exact same metric implementations
from experiments.analysis.merge_forensics import (  # noqa: E402
    semantic_metrics,
    consistency_metrics,
    _build_judge_lm,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, type=Path,
                    help="recovered_rejected_prompts.jsonl from A1")
    ap.add_argument("--output", required=True, type=Path,
                    help="output JSONL with semantic + judge metrics added")
    ap.add_argument("--judge_lm", default="openrouter/openai/gpt-4.1-mini",
                    help="LiteLLM model id for judge LM")
    ap.add_argument("--skip_semantic", action="store_true",
                    help="Skip Step 4 (debug)")
    ap.add_argument("--skip_judge", action="store_true",
                    help="Skip Step 5 (debug)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    judge_lm = None
    if not args.skip_judge:
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("ERROR: OPENROUTER_API_KEY not set. "
                  "Source .env.local first.", file=sys.stderr)
            sys.exit(1)
        judge_lm = _build_judge_lm(args.judge_lm)

    rows = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_recoverable = sum(1 for r in rows if r.get("recoverable"))
    print(f"Loaded {len(rows)} rejected events, {n_recoverable} recoverable.",
          file=sys.stderr)

    n_processed = 0
    n_judge_err = 0
    n_sem_err = 0

    with args.output.open("w") as fout:
        for i, r in enumerate(rows, 1):
            out = dict(r)  # shallow copy preserves all fields

            if not r.get("recoverable"):
                # Pass through unchanged with empty metrics
                fout.write(json.dumps(out) + "\n")
                continue

            merged = r["merged_program"]
            p1 = r["p1_program"]
            p2 = r["p2_program"]

            # Step 4 — semantic
            if not args.skip_semantic:
                try:
                    sem = semantic_metrics(merged, p1, p2)
                    out.update(sem)
                except Exception as e:
                    out["semantic_error"] = repr(e)
                    n_sem_err += 1

            # Step 5 — judge
            if judge_lm is not None:
                try:
                    cons = consistency_metrics(merged, p1, p2, judge_lm)
                    out.update(cons)
                except Exception as e:
                    out["judge_error"] = repr(e)
                    n_judge_err += 1

            fout.write(json.dumps(out) + "\n")
            n_processed += 1
            if i % 10 == 0:
                print(f"  [{i}/{len(rows)}] processed (judge errors: {n_judge_err})",
                      file=sys.stderr)

    print(f"\nWrote {len(rows)} rows → {args.output}", file=sys.stderr)
    print(f"  Processed: {n_processed}", file=sys.stderr)
    print(f"  Semantic errors: {n_sem_err}", file=sys.stderr)
    print(f"  Judge errors: {n_judge_err}", file=sys.stderr)


if __name__ == "__main__":
    main()
