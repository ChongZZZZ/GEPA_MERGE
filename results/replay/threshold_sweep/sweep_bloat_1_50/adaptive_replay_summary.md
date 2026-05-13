# Adaptive Merge — Post-Hoc Replay Summary

Cells replayed: **73** across 5 run root(s).

## Variant comparison (G3 on vs. off)

| metric | G3_off | G3_on |
|---|---|---|
| total_events | 296 | 296 |
| skips | 181 | 193 |
| skip_rate | 0.611 | 0.652 |
| g3_unavailable | 0 | 0 |
| actual_accepted | 234 | 234 |
| actual_rejected | 62 | 62 |
| accepted_skipped_by_policy | 154 | 162 |
| rejected_skipped_by_policy | 27 | 31 |
| catastrophic_events | 10 | 10 |
| catastrophic_skipped | 8 | 8 |
| catastrophic_routed_to_summarize | 1 | 1 |
| catastrophic_addressed | 9 | 9 |

### G3_off

**Skip reasons (Layer 1):**
  - `G1_parent_strength`: 181

**Layer 2 algorithm choices (when not skipped):**
  - `original`: 75
  - `summarize_before`: 40

**Layer 2 reasons (top):**
  - `safe_complementary_original`: 73
  - `bloat_risk_growth_over_safe_ratio`: 33
  - `specialization_split`: 4
  - `bloat_risk_predictor_over_LMAX`: 3
  - `near_duplicate`: 2

### G3_on

**Skip reasons (Layer 1):**
  - `G1_parent_strength`: 181
  - `G3_maturity_imbalance`: 12

**Layer 2 algorithm choices (when not skipped):**
  - `original`: 71
  - `summarize_before`: 32

**Layer 2 reasons (top):**
  - `safe_complementary_original`: 69
  - `bloat_risk_growth_over_safe_ratio`: 27
  - `bloat_risk_predictor_over_LMAX`: 3
  - `near_duplicate`: 2
  - `specialization_split`: 2

## Catastrophe coverage

Definition: a 'catastrophic' event has `full_val_lift_over_best_parent <= -0.05` (at least a 5 pp regression on the full val set). 'Addressed' = either Layer 1 skipped the event OR Layer 2 routed it to `summarize_before` (the safe-fallback algorithm). The headline number is `catastrophic_addressed / catastrophic_events`.

- **G3_off**: 9/10 addressed (90%) — 8 skipped at Layer 1, 1 re-routed to summarize_before at Layer 2.
- **G3_on**: 9/10 addressed (90%) — 8 skipped at Layer 1, 1 re-routed to summarize_before at Layer 2.

## Caveats

- Replay uses the *final* state's Pareto frontier and aggregate scores. Live policy will use at-time frontier (potentially smaller, lower median).
- AdaptiveStartPolicy (Layer −1) is not replayed; warmup/frontier-size/plateau decisions need at-time iteration counts.
- Catastrophic-event detection threshold (-0.05) is heuristic; tweak via downstream analysis if needed.
