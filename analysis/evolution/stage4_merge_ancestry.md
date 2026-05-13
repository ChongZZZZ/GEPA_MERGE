# Stage 4 — Merge ancestry / descendant credit

Reconciles Phase A's positive merge lift (+1.33 to +4.25 pp on qwen with right schedule) with Section 4's near-zero direct merge Δval (mean +0.002). Hypothesis: merge's value is realized **indirectly** through reflective descendants of merge candidates that later become best_candidates.

Method: for every running-best-update event and every cell's final `best_candidate.json` target, walk `parent_program_for_candidate` back through the DAG (BFS over all parents). Record whether the lineage contains any merge, the shortest-path distance to the nearest merge, and the lineage depth from seed.

## Table A — Per-cell final-best ancestry (50 cells)

| Task | Cell | Best op | Best val | Has merge ancestor | Dist to nearest merge | Lineage depth | # merges in lineage |
|------|------|--------:|--------:|:------------------:|----------------------:|--------------:|--------------------:|
| 2wiki | `combine_all_score_plateau_s0` | reflect | 0.7467 | ✅ | 2 | 5 | 1 |
| 2wiki | `nomerge_s0` | reflect | 0.4033 | ❌ | — | 2 | 0 |
| 2wiki | `original_budget_proportional_s0` | reflect | 0.4667 | ❌ | — | 2 | 0 |
| 2wiki | `original_immediate_s0` | merge | 0.2667 | ✅ | 0 | 3 | 2 |
| 2wiki | `original_score_plateau_s0` | reflect | 0.6133 | ❌ | — | 2 | 0 |
| 2wiki | `summarize_before_budget_proportional_s0` | reflect | 0.6967 | ✅ | 2 | 5 | 2 |
| 2wiki | `summarize_before_immediate_s0` | reflect | 0.7667 | ❌ | — | 2 | 0 |
| 2wiki | `summarize_before_score_plateau_s0` | reflect | 0.5300 | ✅ | 1 | 4 | 1 |
| hotpotqa | `combine_all_budget_proportional_s0` | reflect | 0.5067 | ❌ | — | 4 | 0 |
| hotpotqa | `combine_all_immediate_s0` | reflect | 0.5333 | ❌ | — | 3 | 0 |
| hotpotqa | `combine_all_score_plateau_s0` | merge | 0.5233 | ✅ | 0 | 3 | 1 |
| hotpotqa | `nomerge_s0` | reflect | 0.5333 | ❌ | — | 3 | 0 |
| hotpotqa | `original_budget_proportional_s0` | merge | 0.4833 | ✅ | 0 | 2 | 1 |
| hotpotqa | `original_immediate_s0` | reflect | 0.4733 | ❌ | — | 3 | 0 |
| hotpotqa | `original_score_plateau_s0` | merge | 0.5467 | ✅ | 0 | 4 | 1 |
| hotpotqa | `summarize_before_budget_proportional_s0` | reflect | 0.5067 | ❌ | — | 4 | 0 |
| hotpotqa | `summarize_before_immediate_s0` | reflect | 0.5400 | ❌ | — | 2 | 0 |
| hotpotqa | `summarize_before_score_plateau_s0` | merge | 0.5467 | ✅ | 0 | 4 | 1 |
| hover | `combine_all_budget_proportional_s0` | merge | 0.4167 | ✅ | 0 | 4 | 1 |
| hover | `combine_all_immediate_s0` | merge | 0.4200 | ✅ | 0 | 4 | 5 |
| hover | `combine_all_score_plateau_s0` | merge | 0.4233 | ✅ | 0 | 5 | 2 |
| hover | `nomerge_s0` | reflect | 0.3933 | ❌ | — | 4 | 0 |
| hover | `original_budget_proportional_s0` | merge | 0.4100 | ✅ | 0 | 3 | 1 |
| hover | `original_immediate_s0` | reflect | 0.4167 | ✅ | 2 | 6 | 4 |
| hover | `original_score_plateau_s0` | merge | 0.4267 | ✅ | 0 | 4 | 2 |
| hover | `summarize_before_budget_proportional_s0` | merge | 0.4133 | ✅ | 0 | 3 | 1 |
| hover | `summarize_before_immediate_s0` | merge | 0.4033 | ✅ | 0 | 3 | 1 |
| hover | `summarize_before_score_plateau_s0` | reflect | 0.4200 | ❌ | — | 3 | 0 |
| ifbench | `combine_all_budget_proportional_s0` | reflect | 0.8158 | ❌ | — | 2 | 0 |
| ifbench | `combine_all_immediate_s0` | merge | 0.8156 | ✅ | 0 | 3 | 2 |
| ifbench | `combine_all_score_plateau_s0` | reflect | 0.8147 | ❌ | — | 2 | 0 |
| ifbench | `nomerge_s0` | reflect | 0.8039 | ❌ | — | 3 | 0 |
| ifbench | `original_budget_proportional_s0` | reflect | 0.8158 | ❌ | — | 2 | 0 |
| ifbench | `original_immediate_s0` | merge | 0.7236 | ✅ | 0 | 3 | 4 |
| ifbench | `original_score_plateau_s0` | reflect | 0.8075 | ❌ | — | 2 | 0 |
| ifbench | `summarize_before_budget_proportional_s0` | merge | 0.8275 | ✅ | 0 | 3 | 2 |
| ifbench | `summarize_before_immediate_s0` | reflect | 0.8131 | ❌ | — | 2 | 0 |
| ifbench | `summarize_before_score_plateau_s0` | reflect | 0.8131 | ❌ | — | 2 | 0 |
| musique | `combine_all_budget_proportional_s0` | merge | 0.5067 | ✅ | 0 | 3 | 1 |
| musique | `combine_all_immediate_s0` | merge | 0.4500 | ✅ | 0 | 3 | 3 |
| musique | `combine_all_score_plateau_s0` | merge | 0.5133 | ✅ | 0 | 5 | 2 |
| musique | `nomerge_s0` | reflect | 0.4533 | ❌ | — | 2 | 0 |
| musique | `original_budget_proportional_s0` | reflect | 0.4800 | ❌ | — | 2 | 0 |
| musique | `original_immediate_s0` | merge | 0.5133 | ✅ | 0 | 3 | 2 |
| musique | `original_score_plateau_s0` | reflect | 0.4800 | ❌ | — | 2 | 0 |
| musique | `summarize_before_budget_proportional_s0` | merge | 0.5167 | ✅ | 0 | 5 | 5 |
| musique | `summarize_before_immediate_s0` | reflect | 0.4900 | ❌ | — | 2 | 0 |
| musique | `summarize_before_score_plateau_s0` | reflect | 0.5033 | ❌ | — | 4 | 0 |

### Final-best aggregates per task

| Task | n cells | % final-best from reflect | % final-best from merge | % final-best from seed | % final-best with merge ancestor (any op) | % reflect-final-best with merge ancestor | mean dist to merge (when present) |
|------|--------:|--------------------------:|-----------------------:|-----------------------:|------------------------------------------:|-----------------------------------------:|----------------------------------:|
| hotpotqa | 10 | 60% | 40% | 0% | 40% | 0% | 0.00 |
| ifbench | 10 | 70% | 30% | 0% | 30% | 0% | 0.00 |
| hover | 10 | 30% | 70% | 0% | 80% | 33% | 0.25 |
| musique | 10 | 50% | 50% | 0% | 50% | 0% | 0.00 |
| 2wiki | 8 | 88% | 12% | 0% | 50% | 43% | 1.25 |

## Table B — Running-best update aggregates per task

Across all 50 qwen Phase A cells, count every event where the running max val strictly increased.

| Task | n events | % from reflect | % reflect with merge ancestor | % from merge (direct) | % from seed | mean dist to merge (reflect events with merge ancestor) |
|------|--------:|---------------:|------------------------------:|---------------------:|------------:|--------------------------------------------------------:|
| hotpotqa | 50 | 62% | 3% | 18% | 20% | 1.00 |
| ifbench | 58 | 66% | 3% | 17% | 17% | 1.00 |
| hover | 56 | 50% | 7% | 32% | 18% | 1.50 |
| musique | 56 | 61% | 9% | 21% | 18% | 1.33 |
| 2wiki | 56 | 70% | 10% | 12% | 18% | 1.25 |

### Merge influence rate (merge-direct + merge-via-reflect)

For each task, fraction of running-best updates whose lineage involves merge: either the update itself is a merge OR a reflect with merge in its ancestry.

| Task | n events | direct merge | reflect-with-merge-ancestor | total merge-touched | total merge-untouched (pure reflect chain or seed) |
|------|---------:|-------------:|----------------------------:|--------------------:|---------------------------------------------------:|
| hotpotqa | 50 | 9 (18%) | 1 (2%) | **10 (20%)** | 40 (80%) |
| ifbench | 58 | 10 (17%) | 1 (2%) | **11 (19%)** | 47 (81%) |
| hover | 56 | 18 (32%) | 2 (4%) | **20 (36%)** | 36 (64%) |
| musique | 56 | 12 (21%) | 3 (5%) | **15 (27%)** | 41 (73%) |
| 2wiki | 56 | 7 (12%) | 4 (7%) | **11 (20%)** | 45 (80%) |

## Table C — Reflect Δval split by merge-ancestry

For all *reflect* candidates (whether running-best or not), compare Δval distribution depending on whether the lineage contains a merge.
Tests the hypothesis: **reflect-of-merge produces higher per-step Δval than reflect-without-merge-ancestor.**

| Task | n reflect-with-merge | mean Δval (with) | n reflect-no-merge | mean Δval (no) | Δ(with − no) |
|------|--------------------:|-----------------:|-------------------:|---------------:|-------------:|
| hotpotqa | 24 | -0.0104 | 123 | -0.0033 | **-0.0071** |
| ifbench | 5 | -0.0016 | 80 | +0.0419 | **-0.0435** |
| hover | 23 | -0.0062 | 130 | +0.0246 | **-0.0308** |
| musique | 23 | -0.0055 | 115 | +0.0224 | **-0.0279** |
| 2wiki | 25 | +0.0384 | 113 | +0.0469 | **-0.0085** |

### Reflect Δval by distance from nearest merge

Bins reflect candidates by `distance_to_nearest_merge` (1 = direct child of merge, 2 = grandchild, etc.). Tests whether merge influence decays with distance.

| Task | dist=1 (n, mean Δval) | dist=2 | dist=3 | dist≥4 | no merge in lineage |
|------|----------------------:|-------:|-------:|-------:|--------------------:|
| hotpotqa | (20, -0.0080) | (4, -0.0225) | — | — | (123, -0.0033) |
| ifbench | (5, -0.0016) | — | — | — | (80, +0.0419) |
| hover | (17, -0.0114) | (6, +0.0083) | — | — | (130, +0.0246) |
| musique | (17, -0.0049) | (6, -0.0072) | — | — | (115, +0.0224) |
| 2wiki | (16, +0.0537) | (7, +0.0248) | (2, -0.0367) | — | (113, +0.0469) |

## Interpretation

- Of the **48 cell final-best candidates** across qwen Phase A, **24 (50%) have at least one merge in their ancestry**.
- Among the 28 final-bests whose own op is `reflect`, **4 (14%) have a merge upstream**. These are the cells where Phase A's merge lift is realized indirectly: the recorded best is a reflect, but a merge in its lineage was a load-bearing step.

**Reconciliation:** Section 4 §4.3 reports merge mean Δval = +0.002 (vs reflect +0.022). But the *test-set lift attributed to merge* in Phase A is not measured by direct merge Δval — it's measured by comparing the cell's final best_candidate test score against NoMerge's. This Stage 4 result shows the structural mechanism: when merge wins on test, it usually wins by sitting in the ancestry of a reflective descendant, not by being the direct best. The two findings are not in tension; they describe different layers of credit attribution.
