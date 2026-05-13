# Adaptive Merge — Post-Hoc Replay Summary

Cells replayed: **73** across 5 run root(s).

## Variant comparison (G3 on vs. off)

| metric | G3_off | G3_on |
|---|---|---|
| total_events | 296 | 296 |
| skips | 245 | 250 |
| skip_rate | 0.828 | 0.845 |
| g3_unavailable | 0 | 0 |
| actual_accepted | 234 | 234 |
| actual_rejected | 62 | 62 |
| accepted_skipped_by_policy | 197 | 200 |
| rejected_skipped_by_policy | 48 | 50 |
| catastrophic_events | 10 | 10 |
| catastrophic_skipped | 9 | 9 |
| catastrophic_routed_to_summarize | 1 | 1 |
| catastrophic_addressed | 10 | 10 |

### G3_off

**Skip reasons (Layer 1):**
  - `G1_parent_strength`: 181
  - `G2_behavioral_complementarity`: 64

**Layer 2 algorithm choices (when not skipped):**
  - `summarize_before`: 43
  - `combine_all`: 6
  - `original`: 2

**Layer 2 reasons (top):**
  - `bloat_risk_growth_over_safe_ratio`: 38
  - `safe_complementary_combine_all`: 6
  - `bloat_risk_predictor_over_LMAX`: 3
  - `near_duplicate`: 2
  - `specialization_split`: 2

### G3_on

**Skip reasons (Layer 1):**
  - `G1_parent_strength`: 181
  - `G2_behavioral_complementarity`: 64
  - `G3_maturity_imbalance`: 5

**Layer 2 algorithm choices (when not skipped):**
  - `summarize_before`: 38
  - `combine_all`: 6
  - `original`: 2

**Layer 2 reasons (top):**
  - `bloat_risk_growth_over_safe_ratio`: 34
  - `safe_complementary_combine_all`: 6
  - `bloat_risk_predictor_over_LMAX`: 3
  - `near_duplicate`: 2
  - `specialization_split`: 1

## Catastrophe coverage

Definition: a 'catastrophic' event has `full_val_lift_over_best_parent <= -0.05` (at least a 5 pp regression on the full val set). 'Addressed' = either Layer 1 skipped the event OR Layer 2 routed it to `summarize_before` (the safe-fallback algorithm). The headline number is `catastrophic_addressed / catastrophic_events`.

- **G3_off**: 10/10 addressed (100%) — 9 skipped at Layer 1, 1 re-routed to summarize_before at Layer 2.
- **G3_on**: 10/10 addressed (100%) — 9 skipped at Layer 1, 1 re-routed to summarize_before at Layer 2.

## Caveats

- Replay uses the *final* state's Pareto frontier and aggregate scores. Live policy will use at-time frontier (potentially smaller, lower median).
- AdaptiveStartPolicy (Layer −1) is not replayed; warmup/frontier-size/plateau decisions need at-time iteration counts.
- Catastrophic-event detection threshold (-0.05) is heuristic; tweak via downstream analysis if needed.
