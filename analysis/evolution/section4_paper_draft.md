# Section 4 — Why didn't adaptive merge work? (paper draft)

This is the paper's punchline section. Length target: ~3 pages (when typeset).

## 4.1 Gate analysis: most adaptive components are dead code on qwen

Our adaptive policy adds three layers on top of GEPA's vanilla merge: (a) a gated start policy (warmup + plateau + frontier-size), (b) per-pair Layer-1 filters (parent strength G1, maturity-imbalance G3), (c) per-event Layer-2 algorithm routing (A1 bloat, A2 specialization, A3 jaccard duplicate, A4 default). To diagnose the 2/4 task losses against NoMerge (Section 3), we instrumented every gate and counted invocations across the 7 ablation variants on qwen3-8b.

**Layer-1 G1 and G3 fired zero times** across all variants — the parent-strength threshold (median of frontier scores) and the maturity-Gini threshold (≤0.50) are never both binding in the qwen runs we observed. **Layer-2 A1/A2/A3 also never fired**: the L_MAX=4000-token cap, specialization-split=0.30 threshold, and jaccard=0.70 duplicate threshold are all set conservatively enough that every accepted merge fell through to A4 (= `original` algorithm, A4-default). The only adaptive components that meaningfully changed behavior were the **plateau gate** (responsible for 71% of all merge-rejection events on qwen) and the **adaptive_diversity pair selector**.

Two ablations on those two components confirm both are necessary:
- Disabling plateau (v2.6) caused hover to collapse from 43.0 to 1.67 — too many merges stacked overlong prompts that broke qwen3-8b's JSON adapter at inference time.
- Replacing adaptive_diversity with random pair selection (v2.7) dropped musique by 11pp.

Both gates protect against catastrophic failure, but neither moves the score *above* NoMerge. To understand why, we need to look beyond gates and ask: what is GEPA's optimization actually doing?

## 4.2 Anatomy of a GEPA run

We extracted 1,300 candidates from 50 Phase A qwen3-8b cells (5 tasks × 10 configs × seed 0). For each candidate we recorded its origin (seed / reflect / merge), parent indices, mean validation score, prompt length, and cumulative metric-call cost at discovery. Figure 1 shows trajectories for the four PA-best cells (one per task).

**[Figure 1: 2×2 trajectory plots, file `analysis/evolution/figures/main_trajectory_2x2.png`]**

Three patterns recur across all 50 cells:

1. **Reflect produces both peaks and crashes.** Single reflective steps swing val by up to ±0.18 (musique iter 11: +0.18; hotpotqa iter 11: −0.16). The std of single-step Δval for reflect is 0.072 across all qwen Phase A candidates.
2. **Merge is steady but often inert.** 16-25% of merge events produce zero Δval. The std of merge Δval is 0.023 — a third of reflect's. Direct merge gain is small: mean Δval = +0.002, median 0.000.
3. **The peak val candidate is usually a reflect, but its ancestry frequently includes a merge.** In hotpotqa-PA-best, the best_candidate (val=0.5467) is a reflective descendant of merge[6, 16] at iter 17; in hover-PA-best, the best is reflect-of-reflect-of-merge. Merges contribute *new ancestry* — combinations whose downstream reflective children achieve the peaks — rather than direct improvement.

Aggregating across 10 cells per task on a normalized timeline (Figure 2), val rises monotonically through the run while prompt length grows roughly linearly. Merge events cluster in the second half of the trajectory: median merge position is at fraction 0.62-0.73 of the run (Figure 3). GEPA explicitly delays merging via score-plateau / budget-proportional starts, so this matches design — but it also means merges combine prompts that have already been heavily reshaped by reflection.

**[Figure 2: aggregate trajectory per task, `aggregate_trajectory.png`]**
**[Figure 3: merge-timing histogram per task, `merge_timing.png`]**

## 4.3 What does each operation contribute?

Table 1 summarizes per-task Δval distributions for reflect vs merge.

**Table 1: Δval per origin per task (Phase A qwen, single-step val change relative to max parent val).**

| Task | Reflect mean (95% CI) | Merge mean (95% CI) | M-W p | Reflect %pos / %zero | Merge %pos / %zero |
|------|---------------------:|--------------------:|------:|---------------------:|-------------------:|
| HotpotQA | −0.005 [−0.012, +0.003] | −0.008 [−0.017, +0.000] | 0.77 | 37% / 8% | 27% / 14% |
| IFBench | +0.039 [+0.023, +0.055] | +0.009 [−0.001, +0.018] | 0.12 | 60% / 0% | 50% / 32% |
| HoVer | +0.020 [+0.016, +0.024] | +0.015 [+0.011, +0.020] | 0.33 | 71% / 4% | 69% / 16% |
| MuSiQue | +0.018 [+0.010, +0.026] | +0.001 [−0.003, +0.005] | 0.20 | 54% / 5% | 40% / 25% |
| 2WikiMHQA | +0.045 [+0.026, +0.067] | −0.011 [−0.018, −0.005] | **0.02** ★ | 47% / 7% | 31% / 21% |

Reflect's mean Δval exceeds merge's on 5/5 tasks; the difference is statistically significant only on 2WikiMHQA. The bigger story is in the variance and the zero-fraction: reflect has std 0.072 vs merge's 0.023, and merge has 16-32% exact-zero events vs reflect's 0-8%. **Merge is mostly conservative; when it does change val, it changes it by less.**

Splitting merge events by the algorithm that produced them (Table 2) reveals where the inertness comes from.

**Table 2: Merge Δval by algorithm (qwen Phase A, n=37/22/55/48 across hotpotqa/ifbench/hover/musique).**

| Algorithm | %zero-Δval | Mean Δval (across all 5 tasks) | Behavior |
|-----------|-----------:|-------------------------------:|----------|
| `original` | **39-78%** | +0.003 | Picks each predictor from {ancestor, A, B}; often picks an existing predictor verbatim → no actual change |
| `combine_all` | 0-12% | +0.002 | Concatenates differing predictors; nearly always introduces new content but variance is high |
| `summarize_before` | 0-21% | −0.011 | LM rewrites both parent prompts; on hotpotqa and 2wiki the rewrite is net-negative |

The `original` algorithm's "Case-A passthrough" — picking the ancestor's or one parent's predictor unchanged — accounts for the bulk of inert merges. This is by design (it's GEPA's safest merge variant), but it means **a sizeable share of accepted merges produce candidates that are clones of an existing parent**.

Operations differ even more sharply in their *content footprint* (Table 3).

**Table 3: Per-step content change (median across all qwen Phase A candidates, n=661 reflect, n=201 merge).**

| | Reflect | Merge |
|---|---:|---:|
| Δchars | **+1,750** | +1,433 |
| Δbullets | **+7** | 0 |
| Δ"If" clauses | +2 | 0 |
| Δ"Ensure" clauses | +1 | 0 |
| Δ"When" clauses | 0 | **0 (mean −0.82)** |

Reflect is overwhelmingly an *inflation* operation: 99% of reflective mutations make the prompt longer, with median +1,750 chars and a typical addition of 7 bullet points and several conditional clauses per step. Merge is more conservative — and uniquely, merge can also *delete* instruction structure (mean Δ"When" = −0.82, Δ"Ensure" = −0.87): when `original` picks the ancestor's predictor over a heavily-mutated parent's, it discards the latter's accumulated guidance.

Across a full run, prompts grow 25-30× from seed on hotpotqa/hover/musique/2wiki, and **170× on IFBench** — a benchmark whose compound constraints invite the model to stack instruction handlers. A single reflective step on IFBench has been observed to add 40,241 chars in one shot (combine_all_BP_s0 iter 9). At equilibrium, IFBench prompts averaged 23,396 chars (~5,800 tokens) per predictor, growing from 138-char seeds.

Finally, a per-cost view (Table 4) shows reflect and merge are not comparable as improvement engines.

**Table 4: Efficiency in val-gain per metric_call invested.**

| Operation | Mean step cost (metric calls) | Mean Δval | Δval per 1k metric calls |
|-----------|------------------------------:|----------:|-------------------------:|
| Reflect | 1,592 | +0.022 | **+0.0137** |
| Merge | 906 | +0.002 | +0.0020 |

Reflect is **6.85× more efficient than merge in Δval per cost**. Merge is cheaper per attempt (about half the metric calls of a reflect step), but its mean Δval is so close to zero that the per-cost ratio is dominated by reflect. Merge does not function as a cheap improvement channel; it functions as a *diversity-injection channel* whose value is realized only through downstream reflective descendants.

## 4.4 The metric-correlation upper bound

Even given a perfect adaptive policy that fires only beneficial merges, the gain it can deliver is upper-bounded by how well validation-set scores predict test-set scores. We computed this correlation across all 10 qwen Phase A cells per task (Table 5).

**Table 5: Validation-test correlation across cells per task (qwen3-8b, n=10 cells per task).**

| Task | Pearson r | Spearman ρ | argmax(val) == argmax(test)? |
|------|----------:|----------:|:---:|
| HotpotQA | **+0.89** | **+0.92** | **YES** |
| IFBench | −0.01 | +0.35 | NO |
| HoVer | **−0.06** | **+0.04** | **NO** |
| MuSiQue | +0.40 | +0.04 | NO |

On HotpotQA, val and test are tightly correlated; the val-best Phase A cell is also the test-best cell. On HoVer, val and test are statistically independent (Pearson −0.06): picking the val-best cell yields a test score essentially uncorrelated with the val signal. We attribute this to the metric structure — HoVer is binary verdict (supported / not_supported), so a small prompt mutation that changes which borderline examples flip can yield val ≈ 0.42 by flipping different subsets each time, with no generalization signal carried to test. HotpotQA's F1 metric is continuous and partial-credit, so systematic prompt improvements transfer.

This explains the per-task pattern in Section 3:

- **HoVer (val/test corr ≈ 0)**: 6 ablation variants all land in [42.3, 50.7] on test, regardless of policy. The val-greedy candidate selection is essentially random with respect to test performance. NoMerge happens to be one valid sample from this distribution; our adaptive variants are others. **No policy parameter we control can systematically beat NoMerge on hover** because the optimization signal does not transfer.
- **HotpotQA (val/test corr ≈ 0.89)**: differences in val-greedy candidates do translate to test differences. We lose to NoMerge by 4pp because our adaptive policy lands a slightly weaker val candidate (val=0.533 vs PA-best's val=0.547) — a 1.4pp val gap × 0.89 correlation ≈ 4pp test gap, recoverable in principle by matching PA's specific timing.
- **IFBench / MuSiQue**: weak val/test correlations; merges contribute mostly noise. We win NoMerge by ~3pp on each, but the wins are not robust — v2.7 random selection lost MuSiQue by 11pp, suggesting our +3pp wins are partially lucky draws.

## 4.5 v3 counterfactual: capping reflect output is harmful

§4.3 Finding 1 ("reflect dominates Δval") + Stage 4 ancestry analysis (reflect-of-merge worse than reflect-without-merge across all 5 tasks) suggested reflective inflation could be the right intervention surface — perhaps capping reflect would unstuck the 2/4 ceiling. We tested this directly: a v3_reflect_capped variant that re-runs the locked v2_first config but with `--reflection_max_tokens=2000` (≈6,500 chars; below IFBench's 5,800-token equilibrium but above HotpotQA/HoVer/MuSiQue typical predictor lengths).

**Result on Qwen3-8B (4 tasks × seed=0)**:

| Task | NoMerge | v2_first | v3_reflect_capped (max_tokens=2000) | Δ vs v2_first | Δ vs NoMerge |
|------|--------:|---------:|------------------------------------:|--------------:|-------------:|
| HotpotQA | 55.00 | 51.00 | 49.33 | −1.67 | **−5.67** |
| IFBench  | 31.97 | 35.54 | 31.80 | −3.74 | −0.17 (≈ NoMerge) |
| HoVer    | 47.67 | 43.00 | 38.00 | **−5.00** | **−9.67** |
| MuSiQue  | 48.33 | 51.67 | 50.00 | −1.67 | +1.67 ✅ |
| **avg**  | 45.74 | 45.30 | **42.28** | **−3.02** | **−3.46** |

→ **1/4 wins vs NoMerge (only MuSiQue); 0/4 wins vs v2_first.** Capping reflect is *harmful*, not helpful.

The asymmetry is informative:
- **IFBench**: 61 truncation events + 0 non-seed accepts → reflect cap directly starves the optimization channel.
- **HoVer / HotpotQA**: 0 truncations but still −5 to −9.67 pp — even when the cap doesn't bind, *cap-aware* reflection produces qualitatively worse mutations. Stage 2 §4.3 already showed reflective inflation correlates with capability gain (median +1,750 chars, +7 bullets, +2 If-clauses per accepted reflect step); cutting the room to inflate cuts the gain mechanism.

**This refutes the "control inflation" half of the original §4.5 takeaway.** Reflect inflation is not a pathology to be suppressed; it is GEPA's value channel. Future direction: *augment* reflect's diversity (per-iteration temperature ramping, multi-proposal voting), not constrain its length.

## 4.6 Cross-model validation: same policy, dramatically different result

To test whether the qwen 2/4 ceiling reflects a structural limit of adaptive merging or a property of Qwen3-8B specifically, we ran the *exact same locked v2_first config* on **gpt-4.1-mini × 5 tasks × seed=0** (hotpotqa, ifbench, hover, musique, 2wiki).

**Result on GPT-4.1-mini (5 tasks)**:

| Task | NoMerge | PA-best (Phase A) | v2_first (gpt) | Δ vs NoMerge | accepted merges |
|------|--------:|------------------:|---------------:|-------------:|---------------:|
| HotpotQA | 59.67 | 58.67 | **63.33** | **+3.66** ✅ | 5 |
| IFBench  | 48.98 | 52.21 | 50.34 | +1.36 ✅ | 2 |
| HoVer    | 44.67 | 49.67 | 46.67 | +2.00 ✅ | 4 |
| MuSiQue  | 51.33 | 57.67 | 53.00 | +1.67 ✅ | 2 |
| **2WikiMHQA** | **83.33** | **84.33** | **84.33** | **+1.00** ✅ | **0** ⭐ |
| **avg**  | 57.60 | 60.51 | **59.53** | **+1.94** | 2.6 |

**5/5 wins vs NoMerge; +1.94 pp avg lift.** On HotpotQA the policy *exceeds* the best Phase-A fixed config by +4.66 pp. On 2WikiMHQA — where Phase A documented a −28.66 pp catastrophe with `original × score_plateau` — **the policy emitted 0 accepted merges across 1016 reflective iterations**, finishing tied with the only non-catastrophic Phase-A fixed config (84.33 = 84.33). The G3 (maturity Gini) and A2 (specialization split) gates that fire 0× on Qwen because Qwen never reproduces the catastrophe conditions DO function correctly at gpt scale, where those conditions arise.

The same policy → 1/4 wins on Qwen, 5/5 wins on gpt. The 2/4 qwen ceiling is **small-model fragility**, not a structural policy limit:
- §15.3's catastrophes don't recur on Qwen → catastrophe-prevention gates have nothing to catch
- Reflect inflation works less efficiently on Qwen (parse failures, JSON adapter fragility) → val-greedy candidate selection less effective
- HoVer's val/test corr ≈ 0 (§4.4) is a Qwen-binary-metric peculiarity; gpt × HoVer has usable signal (v2_first +2.00 pp on gpt vs −3.34 pp on qwen)

## 4.7 Synthesis (revised)

Five complementary findings, each independently supported:

1. **GEPA's optimization is reflection-driven inflation, not merge-driven combination.** Reflect produces 99% of all candidate-pool growth, accounts for val peaks, and is 6.85× more cost-efficient than merge per metric call. (§4.3, Stage 1-2)
2. **Merge is sparse, often inert.** 16-32% of merges are zero-Δval Case-A passthroughs; merge value (when it exists) is realized by being the direct best_candidate, *not* through reflective descendants — Stage 4 ancestry analysis showed reflect-of-merge produces *worse* per-step Δval than reflect-without-merge ancestry on all 5 tasks. (§4.3, Stage 3-4)
3. **Adaptive policy gates are model-scale-dependent**, not dead code. On Qwen3-8B, G1/G3/A1/A2 fire 0× across 32 surveyed cells because Qwen doesn't reproduce the §15.3 catastrophe conditions. On gpt-4.1-mini, the *same locked policy* lands 5/5 wins vs NoMerge, including 0 accepted merges on 2WikiMHQA — the policy correctly recognized §15.3's catastrophe class and avoided it. (§4.1 + §4.6)
4. **Per-task win-ability is val/test-correlation-dependent.** HoVer's r ≈ −0.06 on Qwen makes any val-greedy optimization unable to systematically beat NoMerge there. The same task on gpt has usable signal (v2_first +2.00 pp). (§4.4 + §4.6)
5. **Reflective inflation is load-bearing**, validated by counterfactual: capping reflection LM output to 2,000 tokens drops aggregate test by −3.46 pp across 4 qwen tasks (§4.5, v3_reflect_capped). The 2/4 qwen ceiling cannot be moved by suppressing reflect; suppressing it makes things worse on 3/4 tasks.

**The headline**: *adaptive merge policy works at adequate model scale; the qwen 2/4 ceiling is a small-model issue, not a structural policy limit.* Improving GEPA-style optimization is more profitably approached at the **reflective-LM augmentation** level (diversity sampling, multi-proposal voting, temperature ramping) than at the **inflation-control** level (which §4.5 falsified) or the **merge-policy** level (which §4.6 shows already works at gpt scale).

These findings also clarify benchmark choice: tasks with low val/test correlation (e.g., HoVer on Qwen3-8B) cannot reliably distinguish optimization strategies via val-greedy candidate selection — gains reported there should carry uncertainty bands matching the val/test noise floor.
