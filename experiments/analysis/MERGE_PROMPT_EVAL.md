# Merge Prompt Evaluation — Design

**Owner:** Phase A analysis pipeline (`merge_forensics.py`).
**Audience:** poster / paper readers; future maintainers of this fork.

---

## 1. Motivation — why task accuracy alone is insufficient

The GEPA paper measures merge by a single number: full-val score after
optimization. That number is necessary but not sufficient. It tells us
*whether* merge helped, not *why* — and the "why" is exactly what our
project promised to deliver.

A merge attempt produces a child prompt by combining two parents'
instructions. Three very different things can happen at the prompt
level, and they can all yield the same downstream score:

1. **Shuffle** — child = parent A on some predictors, parent B on
   others. No new content. Picks one parent's text per disputed
   predictor.
2. **Compositional** — child contains material from *both* parents on
   at least one predictor. Real cross-pollination, no information loss.
3. **Lossy** — child drops information that was present in either
   parent (e.g. a strict format constraint). Looks compositional in
   length but is functionally degraded.

Two cells that both score 50% on the val set can hide drastically
different generative behaviour. To make a defensible claim about
*which merge algorithm wins and why*, we must look inside the merged
prompt itself.

The paper-reported failure mode (Observation 5: merge degrades on
Qwen3-8B) is most plausibly a *lossy*-class failure. Without
prompt-level instrumentation we cannot test that hypothesis at all —
we can only observe the score and shrug.

---

## 2. Design — five complementary scopes

Each merge attempt that GEPA's `MergeQualitySidecar` records gets
augmented with metrics from five orthogonal scopes. Each scope
answers a different question; combining them recovers the causal
chain *prompt structure → behaviour → score*.

| Step | Scope | Question it answers | Cost | Status |
| ---- | ----- | -------------------- | ---- | ------ |
| 1 | Performance | Did this merge actually help? | free | implemented |
| 2 | Lexical | What does the merged text *look like* relative to its parents? | free | implemented |
| 3 | Lineage | What was the merge *born from* — single mutation, prior merge, deep ancestry? | free | implemented |
| 4 | Semantic | What did the merge *say* relative to its parents (beyond surface form)? | ~$0 (local embeddings) | implemented, gated by `--semantic` |
| 5 | Consistency | Is the merged prompt internally coherent and free of contradictions? | ~$0.10/run (LLM judge) | implemented, gated by `--consistency` |

Steps 1-3 are zero-cost and run by default. Steps 4-5 are opt-in
because they require a model (sentence-transformers / OpenRouter).
The tiered design lets us iterate quickly on cheap signals and
escalate only when the cheap ones leave a question open.

---

## 3. Each step — what, why, how, what we learn

### Step 1 — Performance scope

**What:** lift over best parent on the full val set; lift over both
parents; whether the merge was *promoted* into the candidate pool.

**Why:** the ground-truth answer to "did this attempt help". Every
other scope is correlated against this.

**How:** read directly from
`runs/<task>/<cell>/merge_quality.jsonl` —
`full_val_lift_over_best_parent` and friends are written by GEPA's
sidecar at evaluation time. Step 1 just normalises the field names and
joins them with the run's `(model, task, algorithm, start_policy)`
metadata.

**What it provides:** the y-axis for every plot. It also lets us
discard merges that GEPA itself rejected (negative lift or below the
selection threshold), so downstream scopes only describe the
*surviving* population.

### Step 2 — Lexical scope

**What:**
`sentence_provenance_entropy` (how much the child mixes parent text
sentence-by-sentence — 0 = pure copy of one parent, 1 = perfectly
balanced),
`content_coverage_fraction.coverage_min` (fraction of the rarer
parent's content carried into the child — 0 = lossy, 1 = lossless),
`length_delta` (token-count change relative to the longer parent).

**Why:** these three metrics form the diagnostic plane the paper
needs. Entropy distinguishes shuffle from compositional; coverage
distinguishes compositional from lossy; length captures the
prompt-bloat side-effect of compositional merges.

**How:** same source as Step 1 — the sidecar already computes these
during the merge attempt. Step 2 just lifts them into the per-attempt
row.

**What it provides:**

- Per-algorithm 2D fingerprint on the (entropy × coverage) plane (T1
  in `hypotheses.md`). The three algorithms should occupy
  pairwise-non-overlapping regions; if they don't, our algorithms are
  not behaviourally distinct and the rest of the story collapses.
- The four-quadrant lift test (T2): does the **compositional**
  quadrant (high entropy × high coverage) Pareto-dominate the
  **shuffle** and **lossy** quadrants? This is the strongest causal
  claim the lexical scope can support without semantics.

### Step 3 — Lineage scope

**What:** for each merged candidate,
`parent_generation_depth` (longest chain from candidate back to a
seed),
`is_merge_of_merge` (is at least one parent itself a merge product?),
`noop_predictor_rate` (fraction of the program's predictors where
both parents had identical text — those predictors carry no merge
signal).

**Why:** lift correlates with where in the search tree a merge
happens. Late-tree merges (deep generation depth, parents are
themselves merges) are stacking more interventions and are
plausibly noisier / more brittle. A high `noop_predictor_rate`
means the merge was effectively a no-op on most of the program — its
score signal is therefore *not* attributable to merge logic at all.

**How:** reconstructed from `gepa_state.bin`'s
`parent_program_for_candidate` field — a complete parent table for
every candidate in the run. We walk it recursively.

**What it provides:**

- Lets us split T2 by lineage: does the compositional-quadrant
  advantage hold *only* on first-generation merges, or also on
  merges-of-merges? This determines whether we should recommend
  capping merge depth as a hyperparameter.
- Filters out `noop_predictor_rate ≈ 1` rows from the headline
  averages — these are degenerate "merges" that carry no signal.

### Step 4 — Semantic scope

**What:**
`SNR` (semantic novelty rate) = fraction of merged-prompt sentences
whose nearest-neighbour cosine similarity to *any* parent sentence is
below a threshold (0.7) — i.e. genuinely new content,
`SLC` (semantic loss count) = number of parent sentences whose
nearest-neighbour similarity to *any* merged sentence is below 0.5 —
i.e. content that disappeared.

**Why:** the lexical scope counts *string* novelty, which is the wrong
unit. "Always answer concisely" and "Be brief" are lexically
different but semantically identical; lexical entropy/coverage credit
the merge with information the model can't actually use. Semantic
embeddings collapse paraphrase into a single point, giving us a
content-level rather than surface-level reading.

**How:** sentence-transformers `all-MiniLM-L6-v2` (lazy-loaded; CPU is
fine). For each merge attempt we encode all sentences from
{merged, p1, p2}, compute pairwise cosine, and apply the two
thresholds. Cost is ~5 minutes for a corpus of ~200 merge events.

**What it provides:**

- A semantic version of the (entropy × coverage) plane. If the
  algorithms still separate after de-paraphrasing, T1 is robust. If
  they collapse, the lexical separation we see is mostly stylistic
  shuffling, not real content recombination.
- SLC lets us audit `combine_all` — paper claims it preserves
  information; we can now check whether *semantic* coverage matches
  *lexical* coverage on Qwen-class summarisation steps.

### Step 5 — Consistency scope

**What:** an LLM judge scores the merged prompt on a four-axis
rubric: clarity, specificity, internal consistency, coverage relative
to parents. Each axis is 1-5 with a brief rationale.

**Why:** the most common failure mode for `combine_all` is
*self-contradiction*: it concatenates parents, and if the parents
disagree (e.g. p1 says "always cite sources", p2 says "answer in one
word") the merged prompt becomes incoherent. None of Steps 1-4
detect this — entropy and coverage both score the contradictory
merge favourably. A judge call is the only cheap way to catch it.

**How:** a single API call per attempt, structured-output rubric.
~$0.0005 per attempt at gpt-4.1-mini; ~$0.10 for a full run's
~200 merges. Single-LM judge bias is acknowledged; the paper's claim
is comparative across algorithms, not absolute, so within-judge
ranking is what matters.

**What it provides:**

- The "lossy" quadrant of T2 has a known lexical signature. The
  consistency axis lets us further split it into
  *information-loss-only* vs *contradiction-and-loss*. The latter
  is the failure mode we conjecture explains paper Obs 5 on Qwen3-8B
  — small models are less robust to contradictory instructions.
- Complements SLC: SLC says "information disappeared"; the judge
  says "and the result is incoherent". Both being high is the
  textbook bad merge.

---

## 4. How the steps combine

The five scopes are independent measurements; their analytic value
is in joining them.

For every merge attempt we end up with one row containing:

- the four `(model, task, algorithm, start_policy)` keys,
- Step 1's lift fields,
- Step 2's three lexical metrics,
- Step 3's three lineage metrics,
- (optionally) Step 4's SNR + SLC,
- (optionally) Step 5's four rubric scores.

This is dumped to `forensics.csv` (flattened) and `forensics.jsonl`
(canonical). Every plot in the paper / poster comes out of a one-line
pandas groupby on this table.

The three top-level analyses we plan to run on it (T1-T3 in
`hypotheses.md`):

- **T1 — algorithm fingerprint.** 2D KDE per algorithm on
  (entropy × coverage), with a label-permutation null. Distinguishes
  whether the three algorithms are *empirically* different.
- **T2 — quadrant lift.** Median lift in each of the four quadrants
  defined by entropy/coverage medians. Kruskal-Wallis + Dunn
  post-hoc. The compositional quadrant should out-perform shuffle
  and lossy.
- **T3 — prompt-bloat tradeoff.** Slope of `length_delta` against
  `num_merges_so_far`, per algorithm. Tells us how much prompt
  inflation a `combine_all` user is signing up for, in tokens per
  winning percentage point.

Together, T1 establishes that the algorithms are doing different
things; T2 establishes that one of those things (compositional with
preservation) is what helps; T3 establishes the cost of paying for
that benefit.

---

## 5. Why this matters for the paper

We have five (model, benchmark) cells of test scores from Phase A.
That gives us a *score* table, but every cell is a single
point estimate at n=1, so we cannot defend
"`combine_all` is best" with statistical significance from accuracy
alone. The merge-forensics signal is what makes the paper credible:

- It's recorded **per merge attempt**, of which there are ~50-200 per
  run, vs n=1 per cell on the score side. So we have ~10³ rows of
  forensics signal vs ~10² rows of accuracy signal — enough power for
  the algorithm-level comparisons that motivate the work.
- It's mechanistic. Score curves invite "maybe it's noise" objections
  reviewers love. Showing that `combine_all` produces measurably
  longer, semantically broader, more contradiction-prone prompts than
  `original` makes the algorithm-level claim qualitative-then-
  quantitative, which is much harder to dismiss.
- It explains observed failures. Paper Obs 5 (merge hurts Qwen3-8B)
  becomes a testable mechanism rather than an empirical surprise: we
  predict that on Qwen-failed cells, lossy + contradictory quadrants
  account for the negative lift. If we see that, we have a
  *mechanism* for the failure that generalises beyond Qwen3-8B.

---

## 6. What we expect to find (pre-registered, see `hypotheses.md`)

- T1 supported: three algorithm centroids non-overlapping at 95% on
  the (entropy × coverage) plane. `original` near low-entropy /
  high-coverage; `combine_all` near high-entropy / high-coverage;
  `summarize_before` near high-entropy / mid-coverage.
- T2 supported: compositional quadrant median lift > both shuffle and
  lossy median lift, Kruskal-Wallis p < 0.0125 (Bonferroni for
  T1-T3).
- T3 supported: slope of `length_delta` vs `num_merges_so_far` is
  `combine_all` ≫ `summarize_before` > `original`. Bytes-per-pp
  efficiency lets us put a price on the lift each algorithm buys.

If any of these fail, we report the failure as observed. The
algorithm × scope decomposition is informative either way:
- T1 fails → algorithms are *not* meaningfully different; project
  pivots to a "negative result on algorithm choice" framing.
- T2 fails → compositional is no better than shuffle; the
  recommendation flips to `original`.
- T3 fails → no bloat penalty; `combine_all` is free lunch and we
  recommend it unconditionally.

Each of those failure modes is publishable on its own; the design is
robust to outcome.

---

## 7. Pointers

- Implementation: `experiments/analysis/merge_forensics.py`
- Pre-registration: `hypotheses.md` (T1-T3)
- Sample run command: see module docstring header.
- Output schema: `forensics.csv` columns are stable across all five
  steps; downstream notebooks can rely on column names not changing.
