#!/usr/bin/env bash
# Held-out test evaluation for completed GEPA runs.
#
# Walks RUNS_ROOT and, for every subdirectory containing best_candidate.json,
# reconstructs the optimized program and evaluates it on the benchmark's
# held-out test_set (300 examples by default). Writes test_eval.json next to
# best_candidate.json and aggregates to an optional CSV.
#
# REQUIRED ENV
#   OPENAI_API_KEY (or OPENROUTER_API_KEY / PRIME_API_KEY — see run_dspy.py)
#
# KNOBS
#   RUNS_ROOT : directory of completed runs (default: runs/sweep)
#   OUT_CSV   : aggregate CSV path (default: <RUNS_ROOT>/test_eval_summary.csv)
#   PYBIN     : python binary (default: .venv/bin/python)
#
# Example
#   OPENAI_API_KEY=... RUNS_ROOT=runs/sweep bash experiments/run_test_eval.sh
set -e

: "${RUNS_ROOT:=runs/sweep}"
: "${OUT_CSV:=${RUNS_ROOT}/test_eval_summary.csv}"
: "${PYBIN:=.venv/bin/python}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHONPATH="src:experiments/vendor/gepa-artifact" "$PYBIN" \
    -m experiments.analysis.eval_best_on_test \
    --runs_root "$RUNS_ROOT" \
    --out_csv "$OUT_CSV"
