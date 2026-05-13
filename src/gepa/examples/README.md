# Evaluation & Experiment Guide

This directory contains evaluation datasets and the unified experiment runner for the **Adaptive Merge Point Detection** project (MIT NLP, Spring 2026).

---

## Datasets

| File | Task | Metric | HuggingFace Dataset |
|------|------|--------|---------------------|
| `hotpotqa.py` | Multi-hop QA | F1 + Exact Match | `hotpot_qa` (distractor) |
| `hover.py` | Fact verification | Accuracy (SUPPORTED / NOT_SUPPORTED) | `hover-nlp/hover` |
| `ifbench.py` | Instruction following | Constraint satisfaction rate | `allenai/IFBench` |

Each file exposes:
- `init_dataset(train_size, val_size) → (trainset, valset, testset)`
- A task-specific `evaluator(data, response) → EvaluationResult`

---

## Running Experiments

### Single run

```bash
uv run python src/gepa/examples/run_eval.py \
    --task hotpotqa \
    --model openai/gpt-4.1-mini \
    --reflection_lm openai/gpt-4o \
    --max_metric_calls 150
```

### With Merge (original GEPA behavior)

```bash
uv run python src/gepa/examples/run_eval.py \
    --task hotpotqa \
    --model openai/gpt-4.1-mini \
    --use_merge \
    --output_file results/hotpotqa_gpt4mini_merge.json
```

### With Adaptive Merge (proposal contribution)

The new behavioral adaptive merge is enabled through:

- `adaptive_merge_enabled=True` on `gepa.optimize`
- `BehavioralAdaptiveMergePolicy` (Layer 1 behavioral skip gate + per-event Layer 2 algorithm selection)
- `AdaptiveStartPolicy` (Layer −1 warmup + frontier-size + optional plateau gate)
- `merge_selection_strategy="adaptive_diversity"` (outer pair selection)

See `gepa_merge/experiments/adaptive_merge_implementation_notes.md` for the full design, threshold defaults, and how to run the post-hoc replay.

The legacy prompt-divergence `AdaptiveMergePolicy` (and its `merge_min_score_threshold` / `merge_divergence_threshold` kwargs) has been removed; a repo-wide audit confirmed no live caller. See the implementation notes for the deletion record.

---

## Model Scale Experiment (Qwen3)

Run Ollama locally, then:

```bash
# No API cost — open-weight models
for MODEL in ollama/qwen3:4b ollama/qwen3:8b ollama/qwen3:14b; do
    uv run python src/gepa/examples/run_eval.py \
        --task hotpotqa \
        --model $MODEL \
        --reflection_lm ollama/qwen3:14b \
        --use_merge \
        --output_file results/hotpotqa_${MODEL//\//_}.json
done
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/gepa/strategies/merge_policy.py` | **New** — `MergePolicy` Protocol + `AlwaysMergePolicy` |
| `src/gepa/proposer/merge.py` | `MergeProposer.__init__` accepts `merge_policy`; `schedule_if_needed` consults policy |
| `src/gepa/core/engine.py` | Calls `schedule_if_needed(state)` instead of inline `merges_due += 1` |
| `src/gepa/api.py` | Adds `use_merge`, `merge_start_policy`, `merge_selection_strategy`, `merge_algorithm` (+ adaptive-merge wiring; see implementation notes) |
| `src/gepa/examples/hotpotqa.py` | **New** — HotpotQA dataset + F1 evaluator |
| `src/gepa/examples/hover.py` | **New** — HoVer dataset + classification evaluator |
| `src/gepa/examples/ifbench.py` | **New** — IFBench dataset + constraint evaluator |
| `src/gepa/examples/run_eval.py` | **New** — Unified CLI runner |
