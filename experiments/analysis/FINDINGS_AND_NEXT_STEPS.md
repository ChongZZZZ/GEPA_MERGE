# Phase A — Findings and Next Steps

**Last updated:** 2026-04-28, after Phase A test-eval grid + 5-step forensic
pipeline completion.

This doc captures (1) what we have learned so far, (2) the open questions,
and (3) the concrete plan for the next two analyses (deeper accepted-vs-
rejected forensics + timing analysis) before any adaptive-policy work.

---

## 1. What we have

### 1.1 Test-score grid — 80 cells, 4 benchmarks × 2 models

`test_result/all_test_evals_v2.csv`. Held-out test eval (300 examples per
cell), not val. Coverage: 4 benchmarks (hotpotqa, ifbench, hover, musique)
× 2 models (gpt-4.1-mini, qwen3-8b) × 9 (algorithm × start-policy) cells +
1 NoMerge baseline = 80 cells. n=1 per cell.

### 1.2 Forensic pipeline — 409 merge events with all 5 steps

`experiments/analysis/output/forensics_v2/merge_summary.csv`. One row per
merge attempt across all 84 use_merge runs. Steps 1–5 metrics for accepted
events (323), Steps 1–3 only for rejected events (86). Step 4–5 missing
on rejected because the merged-prompt text is not logged for rejected
attempts.

### 1.3 Implementation artifacts

- 5-step pipeline: `experiments/analysis/merge_forensics.py`
- Test-eval driver: `experiments/analysis/eval_best_on_test.py`
- 3-lane test-eval launcher: `experiments/run_test_eval_3lane.sh`
- Pre-registered hypotheses: `hypotheses.md`
- Pipeline design doc: `experiments/analysis/MERGE_PROMPT_EVAL.md`

---

## 2. Findings so far

### 2.1 Hypothesis verdicts

| | claim | result |
| --- | --- | --- |
| **H1** | ≥+1 pp avg merge lift across 4 benches per model | ✅ passes — gpt-4.1-mini +3.39, qwen3-8b +3.15 |
| **H2** | Winner repairs paper's `(original, immediate)` on Qwen3 by ≥+1 pp on ≥3/4 benches | ✅ partial (3/4) — hotpot +12.00, ifbench +4.93, musique +1.33; hover paper-default already wins |
| **H3** | Cross-LM winner consistency (same cell, or each model's winner in other's top-2) | ❌ fails strictly — best (algo, policy) is benchmark-specific |
| **H4** | Cross-scale Qwen3 4B/14B confirmation | ⏰ deferred |
| **T1** | Three algorithms separable on (entropy × coverage) plane | ✅ partial — combine_all clearly distinct; original vs summarize_before only separable on semantic SNR (Step 4) |
| **T2** | Compositional quadrant Pareto-dominates shuffle and lossy | ✅ partial — shuffle quadrant clearly worst (median lift −0.67 pp); compositional does not dominate degenerate |
| **T3** | combine_all > summarize_before > original on prompt-bloat | ✅ strong — combine_all +60% length vs original; summarize_before +7% |

### 2.2 Top empirical findings (with caveats)

1. **Paper Observation 5 is decisively repaired.** The paper claims merge
   degrades Qwen3-8B by −2.45 pp. With proper start-policy ablation,
   Qwen3-8B benefits on 4/4 benchmarks (+1.33 to +4.25 pp). The repair
   is a hyperparameter (`score_plateau` or `budget_proportional` instead
   of `immediate`), not a new algorithm.

2. **Musique is the cleanest case for merge.** gpt-4.1-mini × musique:
   all 9 merge cells beat NoMerge (max +6.34 pp). qwen3-8b × musique:
   7/9 beat NoMerge.

3. **Catastrophic failure mode confirmed.** qwen3-8b × ifbench × `combine_all
   × immediate` → 9.69% test (−22.28 pp vs NoMerge). Mechanism documented:
   small model + 60% prompt bloat (T3) + format-strict evaluator =
   instruction-following collapse.

4. **Algorithm signatures (T1 + T3 + Step 4):**
   - `original` = **shuffler** — SNR = 0, length ≈ baseline. Picks one
     parent's predictor text per disputed predictor; never introduces
     novel content.
   - `combine_all` = **accumulator** — SNR = 0, length +60%. Concatenates
     both parents on disputed predictors; never loses content but never
     creates it.
   - `summarize_before` = **rewriter** — SNR = 0.016, length +7%, novelty
     non-zero on 36.6% of merges. LM synthesizes a unified instruction
     from both parents on Case-B predictors.

5. **Step 5 (LLM judge) has poor resolution on accepted merges.** 313/314
   merges scored 5/5 on `internal_consistency`. Reason: GEPA's subsample
   gate already filters out the obviously-bad merges. The judge needs to
   be re-targeted at *rejected* merges to demonstrate value (Phase A
   deep-dive item).

6. **GEPA's accept/reject is not predicted by prompt-level features.**
   Logistic regression on lexical features alone cannot beat the
   majority-class baseline (acc = 0.801 = baseline). This is a *negative*
   finding; it has implications for adaptive-policy design (cannot rely
   on prompt-static signals alone, must include behavioral signals).

### 2.3 Caveats baked in

- **n=1 per cell.** Differences ≤2 pp may be seed noise. Pattern-level
  conclusions (consistent direction across benchmarks) are stronger than
  any single cell.
- **Multiple-comparison risk.** "Best of 9" winners inflate apparent
  effect sizes; we report the magnitude across cells, not headlines from
  any single best cell.
- **Paper-equivalent budget cap interaction.** We set
  `max_merge_invocations=15` vs paper's 5; verified non-binding (max
  observed = 12) so cap is not a confounder.
- **Pupa dropped.** Qwen3-8B catastrophically fails to produce parsable
  output on pupa. Replaced with musique. Documented in
  `experiments/PHASE_A_PLAN.md`.

---

## 3. Open questions before adaptive policy

The next two analyses are gated *before* any Stage 2 adaptive work,
because Stage 2 design depends on which signals actually carry information
about merge quality.

### 3.1 Accepted vs Rejected forensics deep-dive (Phase A')

The pipeline currently runs Step 4–5 only on accepted merges, where the
judge saturates at 5/5. The honest research question is: **can prompt-level
features explain GEPA's accept/reject decisions, with subsample outcome
held out?**

Caveat learned from preliminary work: subsample profile predicts
accept/reject at acc = 0.985 — but this is a **sanity check** (subsample
*is* the gate GEPA uses), not a finding. The real question is what
prompt-level features (length, coverage, entropy, semantic SNR, judge
axes) add *over the majority baseline*, with subsample held out.

### 3.2 Timing × quality association (Phase B exploratory)

Phase A shows policy matters: `(original, immediate)` on Qwen3 × hotpotqa
loses 10.67 pp; same algo with `score_plateau` gains 1.33 pp. The
proposal-level hypothesis "merge fires too early on immature lineages"
deserves direct investigation, but with n=72 cells it is **exploratory
not confirmatory**.

---

## 4. Plan: Phase A' (accepted vs rejected) — final version

### 4.1 Strict scope

Reconstruct merged prompts only for **deterministic algorithms** (`original`
+ `combine_all`). Do not re-call LM for `summarize_before` rejected events
— a re-called LM produces a *different* prompt, not the one GEPA
rejected.

- Recoverable rejected: 22 (original) + 30 (combine_all) = **52 events**
- Matched accepted-deterministic: 111 + 102 = **213 events**

`summarize_before` accepted (101 events) goes to appendix only, with
explicit note that the comparison is biased.

### 4.2 Pipeline steps

| step | task | cost | time |
| --- | --- | --- | --- |
| **A1** | Reconstruct deterministic rejected prompts via re-call of `merge_system_aware` / `merge_combine_all_subprompts` with same parents. Deterministic, no LM call. | $0 | 30 min |
| **A2** | Step 4 (semantic SNR/SLC) on 52 recovered rejected. Local sentence-transformers. | $0 | 5 min |
| **A3** | Step 5 (judge) on all 213 accepted-deterministic + 52 rejected. | $0.13 | 10 min |
| **A4** | Compute `selection_proxy_score = (subsample_win + 0.5 * subsample_tie) / total_compared` on both groups. Explicit label, never called "lift". | $0 | 5 min |
| **A5** | Matched comparison: Main (213 vs 52) + Robustness A' (algo-balanced 52 vs 52). | $0 | 20 min |
| **A6** | Five nested logistic models with **GroupKFold by cell_id** (`model+benchmark+algo+policy`) — not random split. | $0 | 20 min |
| **A7** | Findings doc + Option B decision (full_val rerun gated). | $0 | 30 min |

**Total Phase A': ~2 hours, ~$0.13 OR.**

### 4.3 Reporting standards (hard rules)

1. **Effect size leads, p-value follows.** Each metric reports Cohen's d
   (or Cliff's δ for ordinal) with 95% CI; p-value in parentheses.
   Reason: 213 vs 52 imbalance + multiple comparisons inflate p-values.

2. **Both Main and Robustness reported.** Main = full 213 vs 52;
   Robustness A' = balanced 52 vs 52. Conclusion only fires if both
   versions agree in direction.

3. **GroupKFold ≠ random split.** Logistic uses 5-fold GroupKFold on
   `cell_id` to prevent within-cell event correlation from inflating AUC.
   Reported metrics: accuracy, **balanced accuracy**, ROC-AUC, PR-AUC,
   confusion matrix — not accuracy alone.

4. **Selection proxy ≠ lift.** All plots and tables involving rejected
   events use the label `selection_proxy_score`. Caption explicitly
   notes: "Rejected merges do not have full validation scores. We use
   subsample outcome as a local selection proxy. This is not equivalent
   to held-out performance."

5. **Sanity check labeled as such.** Logistic Model E (with subsample
   profile) is labeled "Sanity check (Model E)" everywhere; it confirms
   our pipeline correctly reflects GEPA's gate but is not a research
   finding.

### 4.4 Five logistic models

| model | features added |
| --- | --- |
| A | lexical only (length, entropy, coverage, novelty, behavioral) |
| B | + lineage (gen_depth, noop_predictor_rate) |
| C | + semantic (SNR, SLC) |
| D | + judge axes (clarity, specificity, internal_consistency, coverage_vs_parents, contradiction) |
| **E** | **+ subsample profile (sanity check, not finding)** |

Per model: train+test under GroupKFold(5) by cell_id. Report mean ±
std across folds for accuracy / balanced accuracy / ROC-AUC / PR-AUC.

### 4.5 Three pre-registered outcomes

| outcome | implication for adaptive design |
| --- | --- |
| **A**: rejected significantly more bloated / lower coverage | adaptive trigger should include length / coverage thresholds |
| **B**: judge axes or SNR significantly distinguish accepted vs rejected | adaptive trigger should include judge or SNR gate |
| **C**: no significant prompt-level difference | adaptive policy uses **conservative multi-signal trigger** to *reduce risky merges*, not to *identify globally optimal merges*. Honest framing: "single prompt-level metrics have weak predictive power; we use multi-signal trigger as a defensive heuristic." |

All three outcomes are publishable. Outcome C is the most likely given
preliminary lexical-only logistic finding (Model A acc = baseline) and
should not be framed as a failure.

### 4.6 Option B decision rule

After A6 finishes, fire Option B (re-run full validation on the 52
recovered rejected prompts) only if any of:

- rejected significantly more bloated (Cohen's d ≥ 0.4 on length)
- rejected SNR/SLC significantly different (Cohen's d ≥ 0.3)
- judge clarity or internal_consistency differs (Cohen's d ≥ 0.3)
- Model C or D shows ROC-AUC ≥ Model A/B + 0.05

Otherwise: write into limitations as future work, do not fire. Reason:
3 hours of API time and ~$10 should not be spent on a comparison whose
preliminary signal is weak.

### 4.7 Output structure

```
experiments/analysis/output/forensics_v3/
  recovered_rejected_metrics.csv
  accepted_deterministic_metrics.csv
  accepted_vs_rejected_main.csv         # 213 vs 52
  accepted_vs_rejected_balanced.csv     # 52 vs 52 robustness
  logistic_models_grouped.csv
  appendix_all_acc_vs_recoverable_rej.csv
  findings.md
  plots/
    length_delta_acc_vs_rej.png
    coverage_min_acc_vs_rej.png
    snr_acc_vs_rej.png
    judge_acc_vs_rej.png
    selection_proxy_acc_vs_rej.png
    logistic_roc_curves_groupkfold.png
```

---

## 5. Plan: Phase B — timing × quality (exploratory)

### 5.1 Framing

Phase B is **exploratory**, not confirmatory. n=72 cells (8 model-bench
× 9 algo-policy) is too small for strong causal claims. The output is
descriptive: "early merge is *associated with* negative lift in stratum
X", not "early merge *causes* failure."

The visual money shot is the **catastrophic case study**: qwen3 × ifbench
× `combine_all × immediate` (lift −22.28 pp) plotted as a timeline
alongside what `score_plateau` would have done. This is a single-cell
story, but it is the cleanest evidence that timing matters.

### 5.2 Per-cell timing variables

For each of 72 cells, compute:

| var | definition |
| --- | --- |
| `first_merge_iter` | iter of first merge fired |
| `mean_merge_iter`, `median_merge_iter` | iter distribution stats |
| `n_merges`, `n_accepted_merges` | total / accepted attempt counts |
| `accepted_merge_rate` | accepted / total |
| `total_iters` | run final iteration count |
| `relative_first` | first_merge_iter / total_iters ∈ [0, 1] |
| `relative_density` | n_merges / total_iters |
| `early_merge_count` | merges with iter ≤ 25% of total_iters |
| `early_merge_ratio` | early_count / n_merges |
| `lift_over_nomerge` | cell_test_score − NoMerge_test_score (same model+bench) |
| `severity_flag` | "good" (lift > 0), "bad" (−3 < lift ≤ 0), "severe" (−10 < lift ≤ −3), "catastrophic" (lift ≤ −10) |

### 5.3 Stratified correlations (with confounding controls)

Hard rule: no cross-policy correlation as headline (policy *defines* the
timing distribution by design — cross-policy r is a function of policy
labels, not timing).

Compute Pearson r + bootstrap 95% CI for `(timing variable, lift)`
within:

1. each policy (immediate / score_plateau / budget_proportional)
2. each model (gpt-4.1-mini / qwen3-8b)
3. each benchmark
4. multi-hop tasks (hotpotqa + musique) vs format-strict (ifbench) vs
   evidence-retrieval (hover)

### 5.4 Severity comparison

For each stratum, compare timing variables across severity groups (good
/ bad / severe / catastrophic). Catastrophic group is n=1 in our data
— present as case study, not statistic.

### 5.5 Plots

| plot | purpose |
| --- | --- |
| `catastrophic_case_study.png` | Money shot: timeline of qwen3 × ifbench × `combine_all × immediate` with merge events overlaid on score trajectory |
| `scatter_panel_by_policy.png` | x=relative_first, y=lift, faceted by policy, colored by model |
| `first_merge_iter_vs_lift_by_policy.png` | within-policy scatter |
| `early_merge_ratio_vs_lift_by_model.png` | within-model scatter |
| `merge_density_vs_lift.png` | density × lift |

### 5.6 Wording

- "is associated with" not "causes"
- "consistent with the hypothesis that" not "demonstrates that"
- "the catastrophic outlier" gets case-study treatment, not p-value claim
- Sample sizes always reported (e.g., "within Qwen3, n=36")

**Phase B total: ~1 hour, $0.**

---

## 6. After Phase A' + B: Stage 2 adaptive policy

This section is intentionally brief. The exact design depends on Phase A'
outcome (A/B/C above).

### 6.1 Design space (skeleton)

GEPA's `merge_start_policy.py` already implements 6 policies, including
`DiversityTriggeredStartPolicy` (B4) using `prompt_divergence`.
Composition is the natural path:

```python
class AdaptiveMergePointPolicy:
    """Adaptive: AND-gate over divergence + plateau + lineage guards.

    Designed informed by Phase A' outcome:
    - Divergence guard (T1): require Pareto-front pair with prompt_divergence ≥ τ_div
    - Plateau guard (H2): require best score stalled for K iterations
    - Lineage guard (Step 3 finding): require diverse pair with gen_depth ≤ τ_depth
    """
```

### 6.2 Scopes for evaluation

- **Min viable** (poster-quality): 1 model × 1 dataset × 3 conditions × 1
  seed = 3 runs. Pick gpt-4.1-mini × musique (cleanest merge benefit
  ground). adaptive vs phase-A winner vs NoMerge. ~6h, ~$20.
- **Recommended** (paper-quality): 2 models × 2 datasets (musique +
  ifbench) × 3 conditions × 1 seed = 12 runs. ~30h, ~$60. Adds the
  catastrophic-failure-avoidance demonstration on ifbench.
- **Stretch**: 2 models × 4 datasets × 3 conditions × 2 seeds = 48 runs.
  ~5 days, ~$200.

### 6.3 Three pre-registered outcomes for Stage 2

| outcome | paper framing |
| --- | --- |
| adaptive ≥ best-fixed-cell | "Forensic-informed triggers can match or exceed any single fixed schedule." |
| adaptive ≈ best-fixed-cell (within 1 pp) | "Adaptive achieves best-fixed performance without per-task hyperparameter tuning — robust default." |
| adaptive < best-fixed-cell | "Naive composition of forensic signals underperforms expert-tuned schedules; suggests adaptive trigger learning is non-trivial. Negative result motivates future work on learned triggers." |

### 6.4 Risk-management framing (always available)

Even if adaptive < best-fixed on raw score, if it *avoids* the
catastrophic failure (qwen3 × ifbench × `combine_all × immediate` =
−22 pp), that is itself a publishable result. Defensive merging is a
publishable contribution distinct from optimal merging.

---

## 7. Decision log

This section tracks the methodological decisions made during analysis.

| date | decision | rationale |
| --- | --- | --- |
| 2026-04-23 | Drop pupa, replace with musique | Qwen3-8B catastrophic parse failure on pupa; documented in `experiments/PHASE_A_PLAN.md` |
| 2026-04-25 | `max_merge_invocations=15` (vs paper 5) | Avoid asymmetric cap binding across policies; verified non-binding (max observed = 12) |
| 2026-04-26 | Held-out test eval (vs val) | Eliminate val-set selection bias; comparable to paper Table 1/2 |
| 2026-04-27 | Patch `eval_best_on_test.py`: `max_errors=10000`, `--num_threads 2` | Fix dspy parallelizer cancellation on Qwen3 musique |
| 2026-04-28 | Move duplicate qwen3 × hotpotqa runs to `runs/_archive_dup/` | Deduplicate forensic counts (was double-counted at 18 cells, now 9) |
| 2026-04-28 | Phase A' scope: deterministic-only rejected reconstruction | Re-calling LM for summarize_before rejected produces synthetic, not-original prompts |
| 2026-04-28 | Reporting: effect size leads, GroupKFold logistic | Avoid p-value inflation under class imbalance + within-cell correlation |
| 2026-04-28 | Selection proxy ≠ lift | No full validation on rejected; do not conflate subsample outcome with held-out performance |

---

## 8. Hard rules to enforce (summary)

These came up across multiple revisions and are worth listing in one
place:

1. **Subsample-predicts-accept is sanity check, not finding.** Logistic
   Model E is labeled accordingly.
2. **Rejected reconstruction only for deterministic algorithms.**
   `summarize_before` rejected events are excluded; do not re-call LM.
3. **No "lift" without full validation.** Use `selection_proxy_score`
   for subsample-based comparisons.
4. **Effect size > p-value.** Report Cohen's d / Cliff's δ first, p in
   parens.
5. **GroupKFold by cell_id, not random split.** Prevents within-cell
   event correlation from inflating AUC.
6. **Both Main + Robustness for the matched comparison.** Conclusion
   fires only if both agree.
7. **Timing analysis is exploratory.** "Associated with", not "causes";
   case study leads, p-value follows.
8. **n=1 per cell caveat is explicit everywhere.** Differences ≤2 pp
   on lift may be seed noise.

---

## 9. Pointers

- Hypotheses (pre-registration): [hypotheses.md](../../hypotheses.md)
- Pipeline design: [MERGE_PROMPT_EVAL.md](MERGE_PROMPT_EVAL.md)
- Phase A experimental plan: [PHASE_A_PLAN.md](../PHASE_A_PLAN.md)
- Phase A' raw output: `experiments/analysis/output/forensics_v3/`
- Phase B raw output: `experiments/analysis/output/timing_v1/`
- Test-score CSV: `test_result/all_test_evals_v2.csv`
