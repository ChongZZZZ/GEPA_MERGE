# Phase A' — Accepted vs Rejected Forensics: Findings

**Generated:** 2026-04-28, after A1–A6 completion.
**Inputs:** `forensics_v2/merge_summary.csv` + `forensics_v3/recovered_rejected_metrics.jsonl`
**Outputs in this directory:** comparison CSVs (Main + Balanced), logistic results, plots.

---

## TL;DR

**Prompt-level forensic metrics cannot predict GEPA's accept/reject
decisions.** Across 17 metrics covering lexical, lineage, semantic, and
LLM-judge axes, none meaningfully separate accepted from rejected merges.
Five nested logistic models confirm: ROC-AUC ≤ 0.58 for any combination
of prompt-level features (random = 0.5). Only when the subsample profile
(GEPA's documented gate) is added does AUC jump to 0.987 — a sanity
check, not a finding.

This is **Outcome C** from our pre-registered three branches. It has
direct implications for adaptive-policy design: Stage 2 cannot rely on
prompt-static signals alone.

A surprising secondary finding: **LM-judge clarity and specificity are
slightly *higher* on rejected merges than accepted** (Cohen's d = −0.46
in the balanced subset, p < 0.05). The LM judge is mis-aligned with
GEPA's behavioral gate.

---

## 1. Setup

### 1.1 Sample composition

- **Accepted-deterministic:** 213 merges (`original` 111 + `combine_all` 102)
- **Recovered rejected-deterministic:** 52 merges (`original` 22 + `combine_all` 30)
- **Excluded:** 26 `summarize_before` rejected (LM output not logged, cannot
  be deterministically recovered)
- **Pupa excluded:** Qwen3-8B catastrophically fails to produce parsable
  output on pupa; merge events from pupa are degenerate.

### 1.2 Metric inventory (17 metrics, 5 scopes)

| scope | metrics |
|-------|---------|
| lexical (Step 2) | length_delta_vs_anc_total, sentence_entropy, predictor_entropy, coverage_min, coverage_p1, coverage_p2, novelty_fraction |
| lineage (Step 3) | noop_predictor_rate, parent_gen_depth_max |
| semantic (Step 4) | snr_semantic_novelty_rate, slc_lost_count_max |
| judge (Step 5) | judge_clarity, judge_specificity, judge_internal_consistency, judge_coverage_vs_parents, judge_contradiction_present |
| selection proxy | selection_proxy_score = (subsample_win + 0.5 × subsample_tie) / total |

### 1.3 Reporting standards

- Effect size (Cohen's d, Cliff's δ) leads; p-value secondary.
- Both Main (213 vs 52) and Balanced (algo-balanced 52 vs 52) reported.
- Logistic regression uses **GroupKFold(5) by cell_id**
  (= `model+benchmark+algo+policy`) to avoid within-cell event correlation
  inflating AUC.

### 1.4 Algorithm signatures (from broader Phase A forensics)

Across all 314 accepted merges (excluding pupa/anomaly), the three merge
algorithms produce distinguishable prompt-level fingerprints. These are
**descriptive of what each algorithm produces**, computed before the
accepted-vs-rejected comparison.

| algo | n | sentence_entropy | coverage_min | semantic SNR | length_delta vs ancestor |
|------|---:|------------------:|-------------:|--------------:|-------------------------:|
| `original` | 111 | 0.731 | 0.608 | **0.000** | baseline (~985 chars) |
| `combine_all` | 102 | **0.820 ↑** | **0.759 ↑** | 0.000 | **+60% (~1585 chars)** |
| `summarize_before` | 101 | 0.739 | 0.616 | **0.016 ↑** (36.6% nonzero) | +7% (~1053 chars) |

**Three-algorithm behavioral signatures:**

- **`original` = shuffler.** Selects one parent's predictor text on
  Case-B disputed predictors; never introduces sentences absent from
  parents (SNR = 0). Length stays close to baseline.
- **`combine_all` = accumulator.** Concatenates both parents' predictor
  text on disputed predictors; never loses content but never creates
  it (SNR = 0). Prompts grow ~60% longer than `original` — this is the
  documented mechanism behind `combine_all × immediate` catastrophic
  failure on small-model + format-strict tasks.
- **`summarize_before` = rewriter.** LM synthesizes a unified
  instruction from both parents on disputed predictors. SNR is non-zero
  for 36.6% of merges (the LM produced sentences absent from either
  parent), but length stays modest because the LM does not concatenate.

**Cross-reference:** the parallel **start-policy signatures** (timing
+ acceptance behavior) are documented in
[`../timing_v1/findings.md`](../timing_v1/findings.md) Section 4.
Together, the algorithm signature (prompt-level) and policy signature
(timing-level) span the orthogonal axes that define each (algo,
policy) cell's expected behavior.

---

## 2. A5 — Per-metric comparison (Main + Balanced)

| metric | Main d | Bal d | verdict | direction |
|--------|--------|-------|---------|-----------|
| **selection_proxy_score** | **+2.482** | **+2.669** | **STRONG** | accepted >> rejected (sanity ✓) |
| slc_lost_count_max | +0.320 | +0.239 | moderate (marginal) | acc loses slightly more parent content |
| coverage_p1 | +0.252 | +0.456 * | weak | acc slightly higher p1 coverage |
| sentence_entropy | −0.202 | −0.254 | weak | acc slightly less mixing |
| predictor_entropy | −0.206 | −0.354 | weak | acc less predictor-level entropy |
| **judge_clarity** | **−0.220** | **−0.457 *** | weak | **rejected higher clarity** (surprising) |
| **judge_specificity** | **−0.205** | **−0.457 *** | weak | **rejected higher specificity** (surprising) |
| length_delta_vs_anc_total | +0.122 | +0.209 | weak | acc slightly longer |
| coverage_min | −0.141 | −0.011 | n.s. | indistinguishable |
| coverage_p2 | −0.176 | −0.176 | n.s. | indistinguishable |
| novelty_fraction | −0.037 | +0.146 | n.s. | indistinguishable |
| noop_predictor_rate | −0.014 | +0.048 | n.s. | indistinguishable |
| parent_gen_depth_max | +0.048 | −0.169 | n.s. | indistinguishable |
| **snr_semantic_novelty_rate** | 0.000 | 0.000 | n.s. | both groups all-zero (deterministic algos don't introduce novel sentences) |
| **judge_internal_consistency** | −0.076 | 0.000 | n.s. | judge cannot separate |
| judge_coverage_vs_parents | 0.000 | 0.000 | n.s. | both saturated at 5/5 |
| judge_contradiction_present | +0.076 | 0.000 | n.s. | indistinguishable (both ~0%) |

`*` = p < 0.05; `***` = p < 0.001 (Mann-Whitney U).

**Reading guide:** Cohen's d > 0 = metric higher in accepted than rejected.
Verdict requires (a) same sign in Main and Balanced, (b) |d| > 0.2 in at
least one. STRONG = both ≥ 0.5.

---

## 3. A6 — Five nested logistic models

GroupKFold(5) by cell_id, `class_weight="balanced"`. Mean ± std across
folds.

| model | features added | ROC-AUC | PR-AUC | bal_acc | accuracy |
|-------|----------------|---------|--------|---------|----------|
| **A** | lexical only | **0.569** ± 0.06 | 0.872 | 0.570 | 0.517 |
| **B** | + lineage | 0.550 ± 0.06 | 0.865 | 0.567 | 0.521 |
| **C** | + semantic (SNR, SLC) | 0.558 ± 0.06 | 0.868 | 0.529 | 0.521 |
| **D** | + judge (4 axes + contradiction) | **0.575** ± 0.05 | 0.873 | 0.556 | 0.536 |
| **E** | + subsample profile (sanity) | **0.987** ± 0.02 | 0.996 | 0.970 | 0.985 |

**Top 5 standardized coefficients (Model D):**
1. coverage_p1: +1.02
2. coverage_min: −0.91
3. slc_lost_count_max: +0.56
4. judge_clarity: −0.55  ← rejected has higher clarity
5. length_delta_vs_anc_total: +0.35

**Top 5 in Model E (sanity):**
1. subsample_loss: −2.93  ← dominant gate signal
2. subsample_tie: +2.70
3. length_delta_vs_anc_total: +1.08
4. noop_predictor_rate: +0.99
5. subsample_win: +0.93

**Interpretation:**
- ROC-AUC for prompt-level models (A–D): 0.55–0.58. Random performance is
  0.5. So prompt-level features extract essentially no information about
  acceptance.
- PR-AUC ≈ 0.87 looks high, but the prior is 80% — even random scoring
  achieves PR-AUC ≈ 0.80 under this prior. Reporting both confirms the
  models are not picking up real signal.
- Model E confirms the implementation correctly mirrors GEPA's documented
  subsample-based selection. **This is a sanity check, not a research
  finding.**
- The judge_clarity coefficient in Model D (−0.55) reproduces the A5
  comparison: clearer prompts (per LLM judge) are *more likely to be
  rejected*. Direction is opposite to "judge can identify good merges".

---

## 4. Side-finding: single-judge anti-correlation does NOT replicate across judges

### 4.1 What gpt-4.1-mini alone said

The single-judge analysis (gpt-4.1-mini only) showed a small but consistent
**negative** relationship between LM-judge axes and acceptance:

| axis | gpt-4.1-mini d (Main) | gpt-4.1-mini d (Balanced) | logistic D coef |
|------|-------------------------|------------------------------|------------------|
| judge_clarity | −0.220 | −0.457 * | −0.55 |
| judge_specificity | −0.205 | −0.457 * | (subsumed) |

Reading: "rejected merges score *higher* on judge clarity / specificity
than accepted merges, by gpt-4.1-mini's rubric."

### 4.2 Multi-judge ensemble (gpt-4.1-mini + sonnet-4.6 + haiku-4.5)

To check whether this anti-correlation was a single-judge artifact, we
re-judged all 265 merges with claude-sonnet-4.6 and claude-haiku-4.5
(530 additional API calls, ~$5 OR).

**Per-judge Cohen's d (accepted vs rejected, Main 213 vs 52):**

| axis | gpt-4.1-mini | claude-sonnet-4.6 | claude-haiku-4.5 | **ensemble (mean)** |
|------|-------------:|---------------------:|---------------------:|---------------------:|
| clarity | **−0.220** | −0.102 | **+0.046** | **−0.104** |
| specificity | −0.205 | +0.027 | −0.058 | −0.077 |
| internal_consistency | −0.076 | −0.001 | −0.031 | −0.034 |
| coverage_vs_parents | 0.000 | −0.068 | −0.130 | −0.146 |

**The single-judge gpt-4.1-mini effect (d ≈ −0.22) shrinks to ensemble
d ≈ −0.10 — essentially null.** Sonnet and haiku do not reproduce the
direction (haiku is even slightly *positive* on clarity).

### 4.3 Why this matters

Three methodology observations from running 3 judges in parallel:

1. **Wildly different scoring scales.** Same merges, same rubric:

   | judge | mean clarity (accepted) |
   |-------|--------------------------|
   | gpt-4.1-mini | 4.96 (saturated near max) |
   | claude-sonnet-4.6 | 3.57 (mid-scale, discriminating) |
   | claude-haiku-4.5 | 2.24 (low end) |

2. **Inter-judge agreement is moderate at best.** Pearson r on clarity
   axis: gpt↔sonnet = 0.30, gpt↔haiku = 0.15, sonnet↔haiku = 0.40. The
   two Claude models agree with each other more than either does with
   gpt-4.1-mini.

3. **Contradiction-flag rate diverges enormously.** gpt-4.1-mini flags
   <1% of merges as containing contradictions; sonnet flags ~60%; haiku
   flags ~95%. The rubric's notion of "contradiction" is interpreted
   very differently by each model.

### 4.4 Conclusion (updated)

The right framing for the paper:

> "When evaluated by a single LLM judge (gpt-4.1-mini), accepted merges
> score slightly lower on clarity and specificity than rejected
> (Cohen's d ≈ −0.22). This finding does **not** replicate when scoring
> with claude-sonnet-4.6 (d ≈ −0.10) or claude-haiku-4.5 (d ≈ +0.05),
> and the 3-judge ensemble effect is d ≈ −0.10 — essentially null. We
> conclude the original anti-correlation was primarily a single-judge
> bias artifact, and that LLM-judge rubric scoring of merged prompts
> is not well-calibrated across judge model families."

### 4.5 Implication for adaptive policy design

The earlier conclusion stands and is *strengthened*:

- Single-judge clarity scoring is unreliable as a trigger signal —
  three judges disagree about which prompts are clearer.
- Ensemble judge effect is null — judge does not separate accepted from
  rejected merges in any direction.
- Adaptive policy must rely on **behavioral signals** (subsample win
  rate, score plateau, parent strength), not prompt-static LM
  judgment.

---

## 5. Outcome and Option B decision

The outcomes pre-registered in `FINDINGS_AND_NEXT_STEPS.md` Section 4.5:

- **Outcome A** (rejected significantly more bloated / lower coverage):
  ❌ **NOT supported**. length d = +0.12 to +0.21, coverage indistinguishable.
- **Outcome B** (judge / semantic distinguishes acc vs rej):
  ❌ **NOT supported**. SNR identical (both 0). Judge axes indistinguishable
  on internal_consistency / coverage_vs_parents / contradiction. Clarity
  shows weak signal in *wrong direction*.
- **Outcome C** (no significant prompt-level difference):
  ✅ **SUPPORTED**.

### Option B decision (re-run full_val on 52 recovered rejected)

Pre-registered firing thresholds:
- length Cohen's d ≥ 0.4: **NOT MET** (0.21)
- SNR/SLC d ≥ 0.3: **borderline** (SLC 0.32, but logistic Model C ≤ Model A — no information added)
- judge clarity/IC d ≥ 0.3: **NOT MET in useful direction** (clarity = −0.46, but predicts *rejection*, not quality)
- Model C/D AUC ≥ A/B + 0.05: **NOT MET** (Model D − Model A = +0.006)

**Decision: skip Option B.** Documented as future work. Reason: signals
are too weak to expect that running 52 full-validation recomputations
(~3 hours, ~$10) would change the conclusion.

---

## 6. What this means for the paper

### 6.1 Honest negative finding (publishable)

> "Prompt-level forensic metrics describe what merges look like (T1
> algorithm fingerprints, T3 prompt-bloat) but do not predict GEPA's
> accept/reject decision. Across 17 metrics in 5 scopes — lexical,
> lineage, semantic, LLM-judge, and a subsample-derived selection proxy
> — only the proxy (i.e., the behavioral signal GEPA itself uses)
> achieves above-chance prediction (ROC-AUC = 0.987, sanity check).
> Lexical and semantic features alone yield ROC-AUC ≤ 0.58 (random).
> Prompt-level static analysis is therefore complementary to, not a
> substitute for, behavioral evaluation."

### 6.2 Surprising side-finding (publishable, brief)

> "An LLM judge (gpt-4.1-mini) scoring merged prompts on clarity and
> specificity shows weak negative correlation with GEPA's behavioral
> acceptance (Cohen's d = −0.46, p < 0.05 in the balanced subset).
> Judge-clearer prompts are slightly more likely to fail the subsample
> behavioral test. We attribute this to selection effect (the judge
> evaluates surface form; GEPA evaluates output behavior) and recommend
> against using LLM-judge-clarity-style triggers in adaptive merge
> policies."

### 6.3 Implication for Stage 2

The adaptive policy cannot use lexical, semantic, or LLM-judge signals
as primary triggers — they have no predictive power for merge
acceptance. The remaining options:

1. **Behavioral signals**: parent subsample win rate, score plateau,
   Pareto-front rank. These are correlated with acceptance by
   construction (Model E sanity check).
2. **Conservative multi-signal triggers**: combine multiple weak signals
   (lineage divergence + plateau + parent strength) as an AND-gate that
   reduces *risky* merges, not as a predictor of *good* merges. Frame
   honestly as defensive heuristic.

This matches the third pre-registered Stage 2 framing in
`FINDINGS_AND_NEXT_STEPS.md` Section 6.3:

> "Naive composition of forensic signals underperforms expert-tuned
> schedules; suggests adaptive trigger learning is non-trivial. Negative
> result motivates future work on learned triggers."

---

## 7. Caveats and limitations

1. **Recovered rejected = deterministic algorithms only** (52 of 78
   rejected events). `summarize_before` rejected merges (26 events) are
   excluded; their LM outputs were not logged and re-calling the LM
   would produce a different prompt than the one GEPA actually rejected.
2. **No held-out full validation on rejected merges.** Comparisons use
   `selection_proxy_score` derived from subsample outcome, not full-val
   `lift`. Future work: re-run full validation to enable direct
   comparison.
3. **Class imbalance** (80% accepted / 20% rejected) is handled with
   `class_weight="balanced"` in logistic regression and Mann-Whitney U
   (rank-based, robust to unbalanced sizes), but PR-AUC reporting helps
   contextualize accuracy.
4. **GroupKFold by cell_id** prevents within-cell contamination but
   events from the same run remain partially correlated through shared
   parents. Reported AUC is still a slight overestimate of
   generalization to truly novel cells.
5. **Single LM judge** (gpt-4.1-mini) at 1–5 rubric. The negative judge
   finding could partially reflect judge-model bias rather than a
   universal property of LLM-judge-style filtering.
6. **n_rejected = 52 in deterministic subset** is moderate. Statistical
   power for detecting d < 0.3 is limited.

---

## 8. Pointers

- Comparison tables: `accepted_vs_rejected_main.csv`,
  `accepted_vs_rejected_balanced.csv`
- Logistic results: `logistic_models_grouped.csv`
- Per-event metrics: `accepted_deterministic_metrics.csv`,
  `rejected_deterministic_metrics.csv`
- Recovered prompts: `recovered_rejected_metrics.jsonl`
- Plan: [`../FINDINGS_AND_NEXT_STEPS.md`](../FINDINGS_AND_NEXT_STEPS.md)
- Pipeline design: [`../MERGE_PROMPT_EVAL.md`](../MERGE_PROMPT_EVAL.md)
