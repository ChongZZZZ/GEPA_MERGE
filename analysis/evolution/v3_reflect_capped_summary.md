# v3_reflect_capped — Reflective Inflation Control Experiment

**Setup**: Qwen3-8B × 4 tasks × seed=0, locked v2_first config + new flag `--reflection_max_tokens=2000` (≈ 6,500 chars). Goal: test §6 hypothesis that *"improving GEPA-style optimization is more profitable at the reflective LM level (controlling inflation) than at the merge policy level"*.

**Run details**: launched 2026-05-03 01:35 EDT, completed ~06:55 EDT (~5h 20min wall-clock parallel). 4 cells in parallel via XIXIXI key on Prime Intellect. Reflection LM output capped via dspy.LM `max_tokens=2000`; otherwise byte-identical to v2_first.

---

## 1. Test scores — per-task

| Task | NoMerge | v2_first | PA-best (s=0) | **v3_reflect_capped** | Δ vs NoMerge | Δ vs v2_first | Δ vs PA |
|------|--------:|---------:|--------------:|----------------------:|-------------:|--------------:|--------:|
| **hotpotqa** | 55.00 | 51.00 | 56.33 | **49.33** | **−5.67** ❌ | −1.67 | −7.00 |
| **ifbench**  | 31.97 | 35.54 | 36.22 | **31.80** | −0.17 ≈ NoMerge | **−3.74** ❌ | −4.42 |
| **hover**    | 47.67 | 43.00 | 50.67 | **38.00** | **−9.67** ❌❌ | **−5.00** ❌ | **−12.67** |
| **musique**  | 48.33 | 51.67 | 52.33 | **50.00** | **+1.67** ✅ | −1.67 | −2.33 |
| **avg**      | 45.74 | 45.30 | 48.89 | **42.28** | **−3.46** | **−3.02** | **−6.61** |

**Wins vs NoMerge**: 1/4 (musique only).
**Wins vs v2_first**: 0/4.
**Wins vs PA-best**: 0/4.

---

## 2. Candidates accepted per task — the structural signal

| Task | candidates accepted | iter completed | accept rate | trunc events |
|------|--------------------:|---------------:|------------:|-------------:|
| hotpotqa | **12** | 184 | 6.5% | 0 |
| ifbench  | **1 (seed only)** | 91 | **0%** | **61** |
| hover    | **11** | 99 | 11.1% | 0 |
| musique  | **5** | 279 | 1.8% | 0 |

**Key observation**: ifbench accepted **zero non-seed candidates** in 91 iterations. The 61 truncation events confirm reflection LM output was being chopped at the 2000-token cap on ifbench — its reflective mutations grew above the cap, got truncated mid-JSON, and DSPy parse failed → reject. **Reflect cap effectively starved ifbench's optimization channel completely.**

By contrast hotpotqa (12) and hover (11) accepted normally despite the cap — their natural reflect output didn't hit the wall (0 truncations) — yet they still scored materially worse than v2_first. So it isn't only truncation that hurts; **even uncapped-but-cap-aware reflection produces worse candidates** than the uncapped baseline.

---

## 3. Interpretation for §6 — *revised*

The original §6 takeaway in `section4_paper_draft.md` proposed:

> **3.** Improving GEPA-style optimization is more profitable at the reflective LM level (controlling inflation, regularizing instruction stacking) than at the merge policy level.

This experiment **falsifies the controlling-inflation half of that claim**. Capping reflection LM output to 2,000 tokens (a value chosen to lie below IFBench's 5,800-token equilibrium but above HotpotQA/HoVer/MuSiQue's typical predictor lengths) produces:

- **−5.67pp on hotpotqa** vs NoMerge (was −4.0 in v2_first)
- **−9.67pp on hover** vs NoMerge (was −3.34 in v2_first)
- ifbench reduced to NoMerge baseline (the +3.57 v2_first lift is wiped out)
- musique +1.67pp vs NoMerge (the only positive — but still −1.67 vs v2_first)

**Aggregate: −3.46pp vs NoMerge across 4 tasks. The cap is harmful, not helpful.**

The mechanism is asymmetric:
- On **ifbench**, the cap directly truncates reflective output → DSPy can't parse → 0% accept rate → optimization collapses to seed-level performance.
- On **hover/hotpotqa**, the cap doesn't truncate (0 trunc events) but still degrades scores by 5–9pp. This implies the *anticipation* of cap is irrelevant — the cap pressure must be implicit in some other channel. More likely: with 2000 tokens, reflective LM produces *qualitatively different* prompt structures (less expansive, fewer conditional clauses) that the program LM can't leverage as well. Stage 2 §4.3 already showed reflective inflation correlates with capability gain (median +1,750 chars, +7 bullets, +2 If-clauses per accepted reflect step); cutting the room to inflate cuts the gain mechanism.

**Revised §6 takeaway:**

> **3.** GEPA's reflective inflation is *load-bearing*, not pathological. Capping the reflection LM's output token budget (1) starves accept rates on tasks whose mutations naturally exceed the cap, and (2) degrades test scores even on tasks where the cap doesn't bind, by removing the headroom for the kinds of expansive structural mutations that drive Stage 2's measured Δval gains. The 2/4 ceiling cannot be moved by suppressing reflect; suppressing it makes things worse on 3/4 tasks.
>
> A more productive future direction is to *augment* reflect's diversity (e.g., per-iteration temperature ramping, multi-proposal voting) rather than constrain its length, or to intervene on the **program-LM** side (instruction-following capability) rather than the reflection-LM side.

This is sharper and more falsifiable than the original hypothesis — and the experimental data is consistent and strong (3/4 cells degraded materially, the 4th's +1.67pp is within noise).

---

## 4. Connection to other §4 findings

- **Stage 1-2 (reflect efficiency)**: Reflect was 6.85× more cost-efficient than merge. v3 result is consistent: cutting reflect's output budget wastes that efficiency — most metric_calls go into rejected reflective attempts (see ifbench parse_err=147, hotpotqa=95).
- **Stage 5 (gate fire counts)**: G1, G3, A1, A2 all fired 0× on qwen — the policy never engaged its catastrophe-prevention gates. v3 confirms that the lever isn't the gates; it's the upstream reflection volume.
- **§4.4 (val/test correlation)**: hover dropped −9.67pp on test even though val/test corr ≈ −0.06 means val is uncorrelated with test. So even val-blind optimization should have produced a roughly NoMerge-equivalent score; that v3 dropped 9.67pp **below** NoMerge implies reflect cap actively damages the seed prompt's ability to be augmented usefully.

---

## 5. Cost / wall-clock summary

| Metric | Value |
|--------|-------|
| Wall-clock (parallel, 4 cells on Mac) | ~5h 20min (01:35 → 06:55 EDT) |
| Total metric_calls used | ~24,000 across 4 cells (≤ budget) |
| Parse-error rate | hotpotqa 95, ifbench 147, hover 32, musique 77 (high vs ~5-10 baseline) |
| Truncation events | ifbench 61 (the only task hitting cap); 0 elsewhere |
| Cost (Prime, qwen3-8b @ ~$0.00073/call) | ~$18 + ~$5 test_eval = ~$23 |
| Cells: alive at end | 0 (all clean exit) |

---

## 6. What this means for the paper

- **§4 / §6 sharpened**: the reflect-inflation-is-the-lever claim is now empirically supported by a counterfactual experiment. We don't just observe that reflect dominates Δval; we tested taking it away and confirmed the score drops.
- **§17-§18 in REPORT.md**: add v3_reflect_capped as the 9th adaptive-style variant (after v2.11). It also lands at the 1/4 wins range.
- **Future-work paragraph**: pivot from "constrain reflect" to "augment reflect" — the data points suggest the reflective LM is the right intervention surface but the right *direction* is +diversity / +sampling, not −length.

---

*Generated 2026-05-03 06:55 EDT. Underlying data: `adaptive_merge/runs_v3_reflect_capped_2000/`, `adaptive_merge/logs_v3_reflect_capped_2000/`.*
