# Phase A — Pre-registered Hypotheses (v2)

**Registered**: 2026-04-24, before Phase A launch on 2026-04-25.
**Supersedes**: v1 registered 2026-04-16. Scope changed on 2026-04-23:
we pivoted from ablating selection strategies (A1-A4) to ablating **merge
algorithms** (3 variants: `original`, `combine_all`, `summarize_before`)
because A1-A4 showed poor differentiation in sparse valid-pair regimes
(we observed most merge attempts had ≤1 valid pair, so ranking strategies
had no effect). v1's H1-H4 assumed the old design and are moot; v2 keeps
v1's Tier A offline analyses (now T1-T3) since those test merge-behavior
metrics independent of experimental design.

**Commit at registration**: `1587a34` — defining the 3 merge algorithms,
3 start policies, and 80-run ablation harness. Any post-hoc code change
will be tagged and cross-referenced to affected numbers.

**Experimental setup**: see
[experiments/PHASE_A_PLAN.md](experiments/PHASE_A_PLAN.md). Briefly:
3 merge algorithms × 3 start policies × 1 seed × 4 benchmarks × 2
task LMs + NoMerge baselines = 80 runs, paper-exact per-benchmark
rollout budgets, reflection LM = task LM (paper `teacher_lm=None`).

---

## Motivation

GEPA paper (ICLR 2026) Observation 5, highlighted in the paper:

> "GEPA+Merge works especially well for GPT-4.1 Mini, [but] lead to
> performance degradation when used with Qwen3 8B. [...] We attribute
> these discrepancies to the way the rollout budget is allocated between
> reflective mutation and crossover, and the timing of invocation of the
> crossover strategy."

Paper reports +1.14% on GPT-4.1 Mini, **−2.45%** on Qwen3-8B with fixed
hyperparameters. This is the concrete failure mode we aim to repair.

**Our hypothesis**: paper's `original` merge (Algorithm 4 System-Aware
Merge) picks one parent's prompt on Case-B disputed predictors and
discards the other. For smaller task LMs like Qwen3-8B that rely more on
prompt content, this information loss dominates the benefit of merging.
Algorithms that preserve more information (`combine_all` via concat) or
synthesize across parents (`summarize_before` via LM) should behave
differently; a signal-driven start policy (`score_plateau`) should avoid
firing merge when mutation is still productive.

---

## Winner-discovery hypotheses (80-run ablation)

### H1 — Primary: at least one config beats NoMerge on each model

**Claim**: On each task LM M ∈ {gpt-4.1-mini, Qwen3-8B}, there exists at
least one (merge_algorithm, start_policy) configuration whose full-val
score averaged across the 4 benchmarks **beats NoMerge by ≥+1 absolute
percentage point**.

**Test**:
1. `lift[M, algo, policy, bench] = score[M, algo, policy, bench] − score[M, NoMerge, bench]`
2. `avg_lift[M, algo, policy] = mean over 4 benches of lift[M, algo, policy, ·]`
3. For each M: `winner[M] = argmax over (algo, policy) of avg_lift`
4. H1 holds iff `avg_lift[M, winner[M]] ≥ +1.0 pp` for both models.

**If refuted**: merge is inconclusive in our setup. Paper pivots to a
"negative result" framing — still publishable, weaker claim.

### H2 — Qwen3-8B repair: our winner beats paper's default on Qwen3-8B

**Claim**: On Qwen3-8B, the winner config beats paper's default
`(original, immediate)` by **≥+1 pp on ≥3 of 4 benchmarks**.

**Test**: Per-benchmark comparison of winner vs (original, immediate)
on Qwen3-8B. H2 holds iff winner wins (lift ≥ +1 pp) on ≥3 benchmarks.

**Why this matters**: H2 is the direct "we fixed paper's Observation 5"
claim. If H2 fails, the repair is elsewhere (timing, scale-specific).

### H3 — Winner consistency across task LMs

**Claim**: Winner on gpt-4.1-mini is identical to winner on Qwen3-8B,
**or** each model's winner appears in the other's top-2.

**Test**: Rank 9 (algo, policy) cells by avg_lift within each model.
H3 holds iff:
- Same winner in both models, **OR**
- Each model's winner appears in the other's rank-1 or rank-2.

**Why this matters**: H3 supports a "task-LM-invariant strategy" claim.
If H3 fails but H1 passes → per-model strategy matters; weaker but still
publishable finding.

### H4 — Cross-scale: winner transfers to Qwen3-4B and Qwen3-14B

**Claim**: On HotpotQA, the Qwen3-8B winner config beats NoMerge by
≥+1 pp on both Qwen3-4B and Qwen3-14B.

**Test**: Phase B.2 runs winner + NoMerge × {4B, 14B} × HotpotQA × 1
seed. H4 holds iff both sizes show lift ≥ +1 pp.

**Why this matters**: Extends claim from "2 models" to "Qwen3 4B → 14B
capability scale (5× params)".

---

## Pre-registered predictions (ranked by prior confidence)

1. **HIGH**: `(original, immediate)` — paper default — does NOT win on
   Qwen3-8B. (If this prediction fails, paper's Obs 5 was a fluke.)

2. **MEDIUM-HIGH**: Winner's merge_algorithm is `summarize_before` or
   `combine_all`, not `original` — information preservation > discard.

3. **MEDIUM**: Winner's start_policy is `score_plateau`, not `immediate`
   — signal-driven firing avoids wasting merge budget.

4. **LOW-MEDIUM**: Most likely single winner cell:
   `(summarize_before, score_plateau)`. **Alternative**: on Qwen3-8B,
   `(combine_all, immediate)` may win if Qwen handles long prompts
   adequately and `summarize_before`'s LM-in-the-loop introduces noise
   the 8B model can't recover from.

5. **LOW (near-coinflip)**: H3 holds. Our priors ≈ 50/50.

6. **HIGH**: H4 holds given H1 holds on Qwen3-8B. Scaling behavior is
   rarely non-monotonic.

---

## Tier A offline analyses (run post-Phase-A, from sidecars)

These use `merge_quality.jsonl` and `candidates.jsonl` sidecars — no
additional LM calls. Inherited from v1 with metric definitions unchanged.

### T1 — Algorithm behavior on the 2D diagnostic plane

**Claim**: The 3 merge algorithms occupy distinguishable regions of the
(sentence_provenance_entropy × content_coverage_fraction.coverage_min)
plane, each with a characteristic signature:
- `original`: low entropy, high coverage (shuffle zone)
- `combine_all`: high entropy, high coverage (compositional zone)
- `summarize_before`: high entropy, low-to-mid coverage (novel zone)

**Test**: 2D KDE per algorithm from all accepted merges pooled across
models and benches. Report mean (entropy, coverage) per algorithm with
10K-bootstrap 95% ellipses. Claim T1 supported if algorithm centroids
are pairwise non-overlapping at 95% level.

**Null baseline**: label-permutation of sentence parent tags per merge
event (1K shuffles). Real entropy distribution must differ from null
(Kolmogorov-Smirnov p < 0.01) to trust the metric.

### T2 — Joint signal predicts lift (compositional quadrant hypothesis)

**Claim**: Among accepted merges, partitioning by medians of
(sentence_provenance_entropy, content_coverage_fraction.coverage_min)
yields 4 quadrants. The compositional quadrant (hi entropy × hi coverage)
has higher median `full_val_lift_over_best_parent` than the shuffle
(lo entropy × hi coverage) and lossy (hi entropy × lo coverage) quadrants.

**Test**: Kruskal-Wallis across 4 quadrants on lift; if p < 0.0125
(Bonferroni across T1-T3), Dunn post-hoc with Bonferroni on 3 pairwise
contrasts (compositional vs each of shuffle, lossy, degenerate),
one-sided since direction is pre-registered.

**Claim T2 supported iff**: Kruskal-Wallis p < 0.0125 AND
compositional median > both shuffle and lossy medians (Dunn p < 0.05).

### T3 — Algorithm × prompt-bloat tradeoff

**Claim**: `combine_all` produces merges with longer prompts than
`original` or `summarize_before`, and the prompt-length cost scales with
merge count within a run.

**Test**: Linear regression of `length_delta` on `num_merges_so_far`
within each (run, algorithm). Report slope per algorithm, 95% CI from
per-run bootstrap. Expected slopes: combine_all >> summarize_before >
original.

**Supporting analysis**: bytes-per-pp-gain efficiency — total prompt
length growth per winning-lift percentage point. Use to argue "you get
+X% for Y extra tokens of prompt."

---

## Secondary plots (no hypothesis test, shown for intuition)

- **Per-benchmark winner heatmap** — 4 × 3 × 3 grid (bench × algo × policy) of
  lift over NoMerge, one heatmap per model. Visual check whether winner
  is uniform or bench-specific.
- **Start-policy timing histogram** — distribution of iteration-when-
  first-merge-fires, by policy × model. Expected: `immediate` at iter 1,
  `score_plateau` concentrated mid-late, `budget_proportional` at fixed
  ≈25%-budget mark.
- **Cross-scale scaling curve** — lift vs Qwen3 size (4B, 8B, 14B),
  HotpotQA. Expected: monotone or slight inverted-U.

---

## Multiple-comparison plan

- Primary family: {H1, H2, H3, H4} — Bonferroni α/4 = 0.0125 per claim
  on tests that use p-values.
- H1, H3 use point-estimate decision rules (no p-value) given n=1.
- Tier A family: {T1, T2, T3} — separate Bonferroni α/3 = 0.0167 per.
- Two-sided tests except where direction pre-registered (T2 quadrant
  ordering is one-sided).

---

## What WOULD falsify our overall narrative

Paper story is robust to any **one** of these; multiple compound
failures would force a rewrite:
- **H1 refuted on both models** → merge genuinely not beneficial in our
  setup (unlikely given paper finds +1.14% on GPT-4.1 Mini).
- **H2 refuted** → we did not repair the Qwen3-8B regression, only added
  variants. Paper claim weakens from "we fix Obs 5" to "we extend the
  ablation".
- **H3 refuted AND H1 passes** → per-model strategy. Cross-LM claim
  weakens to "context-dependent merge strategy."
- **T1 refuted (algorithms' 2D centroids overlap)** → our algorithms
  don't actually differ in behavior. Unlikely given code inspection.

We commit to reporting outcomes **as observed**. No post-hoc reframing
to hide failed predictions.

---

## Hard-cut contingencies

- Phase A wall-clock slips past 4/28 → cut Phase B.2 (H4 deferred to
  follow-up); keep H1-H3 + T1-T3 on the original 80-run grid.
- A teammate's slice fails completely (e.g., API ban) → re-run on lead's
  account; if still impossible, drop that teammate's (model, benches) and
  report partial grid. Missing cells are marked as NA, not imputed.
- Sidecar corruption on some runs → drop those runs from T1-T3; report n
  used per test.

---

## Caveats (pre-registered, not excuses)

- **n=1 per cell** → no significance tests on winner identification,
  only point-estimate comparisons. The +1 pp threshold is larger than
  typical single-seed inter-config noise on these benchmarks (paper
  Table 2 inter-method gaps are 3–10 pp).
- **"Winner"** is a point estimate; paper framing will be "best-performing
  configuration", not "significantly better".
- If schedule permits additional seeds on the winner cell only (not the
  full grid), we will report seed count for every claim.

---

## Out of scope (explicit)

- NLI-based contradiction detection — deferred, needs external model.
- Sentence-transformer semantic coverage — Jaccard-only as lexical lower
  bound.
- LLM-judge rubric evaluation (old v1 H4) — dropped to preserve timeline
  for poster 5/7 and paper 5/12.
- Adaptive start policy learned from Phase A data (Phase C) — considered
  but cut 4/23 to prioritize poster polish. Filed as follow-up.
