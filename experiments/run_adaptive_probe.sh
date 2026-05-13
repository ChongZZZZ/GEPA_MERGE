#!/usr/bin/env bash
# Behavioral probe (§6) — runs the locked four-layer adaptive merge policy.
#
# This invokes the same run_dspy entry point as the configuration sweep, but
# with --use_merge --adaptive_merge_enabled and the default thresholds from
# AdaptiveMergeConfig (warmup_frac=0.25, plateau_window=3, maturity_gini_max=0.50,
# specialization_split_threshold=0.30, duplicate_jaccard_threshold=0.70).
#
# REQUIRED ENV
#   OPENAI_API_KEY (or compatible)
#
# KNOBS
#   TASK_LIST  : default "ifbench hotpotqa hover"
#   MODEL      : default openai/gpt-4.1-mini
#   SEED_ONLY  : default "0 1 2"
#   RUNS_ROOT  : default runs/adaptive_probe
#   INCLUDE_NOMERGE : "yes" | "no" (default yes — matched baselines for each seed)
#
# Example
#   OPENAI_API_KEY=... MODEL=openai/gpt-4.1-mini bash experiments/run_adaptive_probe.sh
set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${TASK_LIST:=ifbench hotpotqa hover}"
: "${MODEL:=openai/gpt-4.1-mini}"
: "${REFLECTION_LM:=$MODEL}"
: "${RUNS_ROOT:=runs/adaptive_probe}"
: "${SEED_ONLY:=0 1 2}"
: "${INCLUDE_NOMERGE:=yes}"

: "${BUDGET_HOTPOTQA:=6871}"
: "${BUDGET_IFBENCH:=3593}"
: "${BUDGET_HOVER:=7051}"
: "${BUDGET_MUSIQUE:=6871}"
: "${BUDGET_TWOWIKIMULTIHOPQA:=6871}"

budget_for() {
    case "$1" in
        hotpotqa) echo "$BUDGET_HOTPOTQA" ;;
        ifbench)  echo "$BUDGET_IFBENCH" ;;
        hover)    echo "$BUDGET_HOVER" ;;
        musique)  echo "$BUDGET_MUSIQUE" ;;
        twowikimultihopqa|2wiki) echo "$BUDGET_TWOWIKIMULTIHOPQA" ;;
        *) echo 5000 ;;
    esac
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASKS=($TASK_LIST)
SEEDS=($SEED_ONLY)

run_one() {
    local run_dir="$1"; local task="$2"; local budget="$3"; shift 3
    mkdir -p "$(dirname "$run_dir")"
    if [ -f "$run_dir/best_candidate.json" ]; then
        echo "[skip] $run_dir"; return 0
    fi
    echo "[run] $run_dir (budget=$budget)"
    PYTHONPATH=src .venv/bin/python -m experiments.benchmarks.run_dspy \
        --task "$task" --model "$MODEL" --reflection_lm "$REFLECTION_LM" \
        --max_metric_calls "$budget" --run_dir "$run_dir" "$@" \
        2>&1 | tee "$run_dir.log"
}

for task in "${TASKS[@]}"; do
    budget=$(budget_for "$task")
    for seed in "${SEEDS[@]}"; do
        run_one "$RUNS_ROOT/${task}/adaptive_s${seed}" "$task" "$budget" \
            --use_merge --adaptive_merge_enabled --seed "$seed"
        if [ "$INCLUDE_NOMERGE" = "yes" ]; then
            run_one "$RUNS_ROOT/${task}/nomerge_s${seed}" "$task" "$budget" \
                --seed "$seed"
        fi
    done
done

echo ""
echo "Probe runs complete under $RUNS_ROOT"
echo "Run experiments/run_test_eval.sh with RUNS_ROOT=$RUNS_ROOT to get held-out test scores."
