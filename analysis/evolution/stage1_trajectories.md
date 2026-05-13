# Stage 1: Anatomy of a GEPA run — trajectories per cell

Methodology: extracted (idx, origin, val, prompt_chars, parents, mc_at_disc) for every candidate in 50 qwen Phase A cells (5 tasks × 10 configs). Below are 4 representative cells (each task's PA-best merge config) plus their NoMerge counterparts.

## Key findings

1. **Reflect 是主要的 val-改进 channel, 但 noisy:** swings of ±0.10 in single steps; merge typically Δval∈[-0.02, +0.05].
2. **Case-A merges are inert:** 16-25% of merge events produce zero Δval AND zero Δchars (the `original` algorithm picked an identical predictor from one parent). These are accounting events, not real evolutions.
3. **The peak val candidate is often a reflect-of-merge, not the merge itself:** in hotpotqa-PA-best, merge[3,12] at iter 13 produced val=0.5233 (peak so far); iter 17 merge[6,16] became val=0.5467 (overall best). Both merges contribute *new ancestry*, the reflect chain off them captures the peak.
4. **Single reflect can grow prompt by 40k chars:** ifbench iter 9 reflect of [2] added 40,241 chars to the prompt. Reflective_full_program isn't a small edit — when it goes wide, it goes very wide.

## Cell-level trajectories

### hotpotqa / original_score_plateau_s0 (PA best, test=56.33, val=0.5467)

```
idx orig    val    Δval   chars   Δchar  parents          milestone
  0 seed   .4133      ?    303    +0    [None]            seed
  1 ref    .4267 +.0133   3478 +3175    [0]               + first reflect inflation
  2 ref    .3867 -.0400   5744 +2266    [1]
  3 ref    .4267 +.0000   5056 -688     [1]
  4 ref    .3233 -.0900   1822 -3234    [0]
  5 ref    .4067 -.0200   5167 +3345    [1]
  6 ref    .4133 +.0067   6982 +1815    [5]
  7 ref    .3800 -.0333   2869 -4113    [0]
  8 ref    .3800 -.0333   7323 +4454    [6]
  9 ref    .3933 +.0067   6301 -1022    [2]
 10 ref    .4100 -.0167   7521 +1220    [3]
 11 ref    .2533 -.1600   9177 +1656    [6]               worst, val crashed
 12 ref    .4833 +.1033   4794 -4383    [7]               recovery
 13 mrg    .5233 +.0400   9547 +4753    [3, 12]           ★ first merge → new peak
 14 ref    .4267 +.0000   7414 -2133    [3]
 15 mrg    .4267 +.0000   7414 +0       [6, 14]           inert merge (case-A)
 16 ref    .5300 +.1033   6572 -842     [3]               reflect catches up
 17 mrg    .5467 +.0167  10076 +3504    [6, 16]           ★★ best_candidate
 18 ref    .5167 -.0133   8341 -1735    [16]
 19 ref    .5367 +.0067   7534 -807     [16]
```

→ Merges at 13 and 17 produce the val peaks. Inert merge at 15 (zero Δval, zero Δchars) is a `original`-Case-A passthrough.

### hotpotqa / nomerge_s0 (NoMerge, test=55.0, val=0.5333)

```
idx orig    val    Δval   chars   Δchar  parents
  0 seed   .4133      ?    303    +0    [None]
  1 ref    .4267 +.0133   3478 +3175    [0]
  2-11    chaotic reflect: val ∈ [0.25, 0.49], chars ∈ [1822, 9177]
 12 ref    .4867 +.0033 (peak so far)
 15 ref    .5333 +.1067   6758 -451     [3]               ★ best, by reflect alone
 16-17  recovery
```

→ Same seed, same early reflects (iter 1-4 identical to original_score_plateau because cache!), but no merge channel. Best val 0.5333 vs PA-best's 0.5467 — 1.3 pp gap that the merge contributes.

### hover / original_immediate_s0 (PA best, test=50.67, val=0.4167)

```
idx orig    val   Δval   chars  Δchar  parents     comment
  0 seed   .27       ?    290   +0    [None]
  1 ref    .35   +.077   2342  +2052  [0]
  2 ref    .32   +.047   1723   -619  [0]
  3 mrg    .37   +.023   3775  +2052  [1, 2]      ★ early merge!
  4 ref    .37   +.023   4100   +325  [1]
  5 mrg    .37   +.000   4100     +0  [2, 4]      inert
  6 ref    .34   +.023   4066    -34  [2]
  7 mrg    .38   +.010   6443  +2377  [5, 6]      productive merge
  8 ref    .40   +.023   7846  +1403  [7]         reflect off merge → climbs
  9 mrg    .37   +.000   4100  -3746  [3, 5]      inert
 10 ref    .38   +.013   6042  +1942  [9]
 11 mrg    .38   +.000   6042    +0   [6, 10]     inert
 12 ref    .38   -.003   7716  +1674  [7]
 13 mrg    .38   +.000   7716    +0   [9, 12]     inert
 14 ref    .33   -.013   4759  -2957  [1]
 15 mrg    .35   +.020   6192  +1433  [2, 14]
 16 ref    .37   -.017   6942   +750  [11]
 17 ref    .42   +.050   9198  +2256  [16]        ★★ best_candidate (reflect)
 18 mrg    .40   +.000   7846  -1352  [8, 9]      inert
 19 ref    .32   +.007   4268  -3578  [2]
 20 mrg    .36   -.007   6320  +2052  [3, 19]
 21 ref    .32   -.007   5937   -383  [19]
```

→ 9 merges total, 5 inert (zero Δval/Δchars), 4 productive (small Δval). Best is reflect-of-reflect-of-merge — the merge ancestry matters but isn't the direct peak.

### ifbench / combine_all_budget_proportional_s0 (PA best, test=36.22, val=0.8158)

```
idx orig    val   Δval     chars     Δchar  parents     comment
  0 seed   .60       ?       138       +0   [None]
  1 ref    .69   +.089       679     +541   [0]
  2 ref    .69   +.092       875     +196   [0]
  3 ref    .69   -.008      2134    +1259   [2]
  4 ref    .59   -.008      1312     -822   [0]
  5 ref    .74   +.140      1007     -305   [0]
  6 mrg    .79   +.050      3003    +1996   [3, 5]      ★ merge → new peak
  7 ref    .82   +.125      2466     -537   [1]         ★ reflect → higher peak
  8 mrg    .82   -.000      3251     +785   [2, 7]      no-op merge (val unchanged)
  9 ref    .81   +.119  >>>>43492 +40241    [2]         ★★★ 40k char inflation in 1 step
 10 mrg    .81   +.001     44507    +1015   [5, 9]
 11 ref    .80   -.014     46255    +1748   [10]        best_candidate (val 0.8158)
```

→ Iter 9 reflect adds **40k chars in a single mutation**. The model's reflective LM produced a massive instruction-handler dump. After this, every candidate carries 44k+ chars. This is how IFBench prompts grow 170x.

### musique / summarize_before_immediate_s0 (PA best, test=52.33, val=0.49)

```
idx orig    val   Δval   chars  parents
  0 seed   .31       ?    307   [None]
  1 ref    .31   +.000  4228   [0]
  2 ref    .34   +.027  2594   [0]
  3 ref    .39   +.057  4254   [2]
  4 mrg    .39   -.003  5749   [1, 3]      neutral merge
  5 ref    .42   +.027  7463   [4]
  6 ref    .30   -.010  5631   [1]         val regression
  7 ref    .33   +.020  2985   [0]
  8 mrg    .39   +.003  5677   [4, 7]      neutral
  9 ref    .30   -.010  2400   [0]         val regression
 10 mrg    .39   -.003  5689   [3, 7]
 11 ref    .49   +.180  6253   [1]         ★★ best_candidate (reflect)
 12-19  oscillation, no improvement on val
```

→ Best is a reflect (iter 11), Δval=+0.18 in one step. 5 merges, all near-zero Δval. Merge contributes nothing direct; reflect off accumulated parents drives the peak.

## Distilled observations across the 4 PA-best cells

| Observation | Evidence |
|---|---|
| **Reflect produces both peaks and crashes**: max single-step Δval +0.18 (musique iter 11), min −0.16 (hotpotqa iter 11). std ~0.07. | hotpotqa: −0.09 to +0.10; ifbench: −0.01 to +0.14; hover: −0.07 to +0.08; musique: −0.04 to +0.18. |
| **Merge produces small steady gains or no-ops**: max Δval +0.05, mean 0.002, with 16-25% exact-zero Δval. | hover original_immediate: 9 merges, 5 inert. ifbench: 1 of 3 merges inert. hotpotqa: 2 of 3 merges inert. |
| **The peak candidate is usually a reflect**, but its ancestry includes 1+ merges. | hotpotqa best at iter 17 (merge[6,16]). Its parent #16 is itself reflect-of-#3, but #3 was the lower-val pre-merge state. The crucial step is iter 13's merge that introduced [3,12] → produced #13 with val 0.52, which became the parent lineage for the best at iter 17. |
| **Prompt length grows mostly via reflect, but sometimes catastrophically**. ifbench iter 9 +40k chars from a single reflect. | Median per-step Δchars: reflect +1750, merge +1433. Tails are very different: reflect 99th percentile is ~+10k+; merge 99th percentile is ~+5k. |
| **NoMerge can match merge cells closely** when val/test correlates well; the gap to best-merge is 1-3 pp. | hotpotqa nomerge val=0.5333 vs PA-best val=0.5467, gap 0.013 → 1.33 pp test gap (55.0 vs 56.33). |

## What this section will say in the paper

> "GEPA's optimization is asymmetric: reflective mutations drive both improvements and regressions in val score, while merges contribute primarily through *ancestry* — they introduce new combinations whose downstream reflective descendants achieve the val peaks. Direct merge Δval is small (mean +0.002, median 0) and a sizeable fraction (16-25%) of merges have zero effect at all (the `original` algorithm Case-A passthrough). On the operations level, GEPA optimizes by amassing instruction text: reflect adds ~1.7k chars per step (median); after 20 steps prompts grow 25-30× from seed. On instruction-rich benchmarks (IFBench), single reflects can add 40k chars at once, growing prompts 170× across a run."
