#!/usr/bin/env bash
# Configuration sweep — merge_algorithm × start_policy ablation (§4 of the paper).
#
# Grid per benchmark:
#   3 merge_algorithms × 3 start_policies × N seeds + N NoMerge baselines
#
# Methodology follows the GEPA paper (Agrawal et al., ICLR 2026):
#   - Reflection LM == Task LM by default (teacher_lm=None falls back to dspy.settings.lm).
#   - Per-benchmark rollout budget matches MIPROv2's usage (paper §E.4).
#   - minibatch=3 (paper default).
#   - max_merge_invocations=15 (paper uses 5; raised to avoid asymmetric
#     binding across start policies — cap=5 hits immediate early while
#     score_plateau / budget_proportional may not hit it at all).
#
# REQUIRED ENV
#   OPENAI_API_KEY (or compatible — Prime Intellect / OpenRouter both work; see
#                   experiments/benchmarks/run_dspy.py for provider routing)
#
# KNOBS (env vars)
#   TASK_LIST          : benchmarks to run. Default "ifbench hotpotqa hover".
#                        Valid: hotpotqa hover ifbench musique twowikimultihopqa
#   MODEL              : task LM. e.g. openai/gpt-4.1-mini or qwen/qwen3-8b.
#   REFLECTION_LM      : defaults to $MODEL.
#   RUNS_ROOT          : output directory (default: runs/sweep).
#   MERGE_ALGO_ONLY    : subset of {original combine_all summarize_before}.
#   START_ONLY         : subset of {immediate score_plateau budget_proportional}.
#   SEED_ONLY          : space-separated seeds. Default "0 1 2".
#   INCLUDE_NOMERGE    : "yes" | "no". Default yes.
#   BUDGET_<TASK>      : override per-benchmark metric-call budget.
#
# Examples
#   # Full sweep on default 3 tasks, 3 seeds, 3 algos × 3 timings:
#   OPENAI_API_KEY=... MODEL=openai/gpt-4.1-mini bash experiments/run_phase_a.sh
#
#   # Single configuration check:
#   OPENAI_API_KEY=... MODEL=openai/gpt-4.1-mini TASK_LIST=hotpotqa \
#       MERGE_ALGO_ONLY=original START_ONLY=immediate SEED_ONLY=0 \
#       INCLUDE_NOMERGE=no bash experiments/run_phase_a.sh
set -e

: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${TASK_LIST:=ifbench hotpotqa hover}"
: "${MODEL:=openai/gpt-4.1-mini}"
: "${REFLECTION_LM:=$MODEL}"   # paper-exact: same model for task + reflection
: "${RUNS_ROOT:=runs/sweep}"

# Paper-exact per-benchmark budget (reverse-engineered from MIPROv2 usage).
# Override via env, e.g. BUDGET_HOTPOTQA=5000.
: "${BUDGET_HOTPOTQA:=6871}"
: "${BUDGET_IFBENCH:=3593}"
: "${BUDGET_HOVER:=7051}"
# MuSiQue: per-example paragraphs are passed in-context (no external retriever),
# program structure mirrors HotpotMultiHop (4 predictors) → match HotpotQA budget.
: "${BUDGET_MUSIQUE:=6871}"
# 2WikiMultiHopQA: same in-context-paragraphs / 4-predictor shape as MuSiQue
# → match HotpotQA/MuSiQue budget.
: "${BUDGET_TWOWIKIMULTIHOPQA:=6871}"

budget_for() {
    case "$1" in
        hotpotqa) echo "$BUDGET_HOTPOTQA" ;;
        ifbench)  echo "$BUDGET_IFBENCH"  ;;
        hover)    echo "$BUDGET_HOVER"    ;;
        musique)  echo "$BUDGET_MUSIQUE"  ;;
        twowikimultihopqa|2wiki) echo "$BUDGET_TWOWIKIMULTIHOPQA" ;;
        *) echo "5000" ;;  # sensible fallback for unknown tasks
    esac
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Apply split filters
if [ -n "${MERGE_ALGO_ONLY:-}" ]; then
    MERGE_ALGOS=($MERGE_ALGO_ONLY)
else
    MERGE_ALGOS=(original combine_all summarize_before)
fi

if [ -n "${START_ONLY:-}" ]; then
    START_POLICIES=($START_ONLY)
else
    # Default: 3 policies. `periodic` dropped — `score_plateau` strictly
    # subsumes it (same intent, score-driven instead of clock-driven).
    # `periodic` is still registered and can be opted in via START_ONLY.
    START_POLICIES=(immediate score_plateau budget_proportional)
fi

if [ -n "${SEED_ONLY:-}" ]; then
    SEEDS=($SEED_ONLY)
else
    SEEDS=(0 1 2)
fi

TASKS=($TASK_LIST)

# Default: include NoMerge only when NOT filtering merge_algo.
if [ -z "${INCLUDE_NOMERGE:-}" ]; then
    if [ -n "${MERGE_ALGO_ONLY:-}" ]; then
        INCLUDE_NOMERGE=no
    else
        INCLUDE_NOMERGE=yes
    fi
fi

# Count expected runs for progress display
expected_merge=$(( ${#TASKS[@]} * ${#MERGE_ALGOS[@]} * ${#START_POLICIES[@]} * ${#SEEDS[@]} ))
expected_nomerge=0
if [ "$INCLUDE_NOMERGE" = "yes" ]; then
    expected_nomerge=$(( ${#TASKS[@]} * ${#SEEDS[@]} ))
fi
expected_total=$(( expected_merge + expected_nomerge ))

count=0
skipped=0

run_one() {
    local run_dir="$1"
    local task="$2"
    local budget="$3"
    shift 3
    mkdir -p "$(dirname "$run_dir")"
    if [ -f "$run_dir/best_candidate.json" ]; then
        echo "[skip] $run_dir (already complete)"
        skipped=$((skipped + 1))
        return 0
    fi
    count=$((count + 1))
    echo "[$count/$expected_total] $run_dir (budget=$budget)"
    PYTHONPATH=src .venv/bin/python -m experiments.benchmarks.run_dspy \
        --task "$task" \
        --model "$MODEL" \
        --reflection_lm "$REFLECTION_LM" \
        --max_metric_calls "$budget" \
        --merge_quality \
        --run_dir "$run_dir" \
        "$@" \
        2>&1 | tee "$run_dir.log"
}

# --- Merge cells: task × merge_algo × start_policy × seed ---
for task in "${TASKS[@]}"; do
    budget=$(budget_for "$task")
    for algo in "${MERGE_ALGOS[@]}"; do
        for start in "${START_POLICIES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_dir="$RUNS_ROOT/${task}/${algo}_${start}_s${seed}"
                run_one "$run_dir" "$task" "$budget" \
                    --use_merge \
                    --merge_algorithm "$algo" \
                    --merge_start "$start" \
                    --seed "$seed"
            done
        done
    done
done

# --- NoMerge baselines: task × seed (only if this slice owns them) ---
if [ "$INCLUDE_NOMERGE" = "yes" ]; then
    for task in "${TASKS[@]}"; do
        budget=$(budget_for "$task")
        for seed in "${SEEDS[@]}"; do
            run_dir="$RUNS_ROOT/${task}/nomerge_s${seed}"
            run_one "$run_dir" "$task" "$budget" \
                --seed "$seed"
        done
    done
fi

echo ""
echo "Sweep slice complete: $count new runs, $skipped skipped."
echo "Tasks:         ${TASKS[*]}"
echo "Merge algos:   ${MERGE_ALGOS[*]}"
echo "Start policy:  ${START_POLICIES[*]}"
echo "Seeds:         ${SEEDS[*]}"
echo "NoMerge:       $INCLUDE_NOMERGE"
echo "Model:         $MODEL   (reflection LM: $REFLECTION_LM)"
echo ""
echo "Aggregate / analyse results with the scripts under analysis/."
