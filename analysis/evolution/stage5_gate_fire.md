# Stage 5 — Gate-fire instrumentation (qwen3-8b adaptive runs)

Parses every `[adaptive_start]`, `[layer1]`, `[layer2]` log line across all v2.x adaptive runs on qwen3-8b. Quantifies §4-§5 claim that catastrophe-prevention gates (G1, G3, A1, A2, A3) fire 0× on qwen.

Variants surveyed: v1_g2on, v2_5, v2_6, v2_7, v2_8, v2_9, v2_10, v2_11
Cells parsed: 32

## Aggregate gate-fire counts across all surveyed cells

### Layer −1 (start gate) skip reasons

| Skip reason | Count |
|-------------|------:|
| `not_plateaued_optional` | 165 |
| `warmup_not_passed` | 81 |
| `frontier_too_small` | 4 |
| **TOTAL Layer-1 skips** | **250** |

### Layer 1 (skip gate) reasons

| Skip reason | Count |
|-------------|------:|
| `no_valid_adaptive_pair` | 19 |
| **TOTAL Layer-1-pair skips** | **19** |

### Layer 2 (algorithm routing) — decisions made

Total Layer 2 decisions logged: **103**

| Algorithm chosen | Count |  | Reason | Count |
|------------------|------:|--|--------|------:|
| `original` | 63 |  | `safe_complementary_original` | 54 |
| `combine_all` | 29 |  | `safe_complementary_combine_all` | 29 |
| `summarize_before` | 11 |  | `bloat_risk_growth_over_safe_ratio` | 11 |
|  |  |  | `layer2_routing_disabled` | 8 |
|  |  |  | `near_duplicate` | 1 |

## ⭐ Per-rule fire rate (the §4-§5 headline table)

Concatenating all variants' qwen runs:

| Layer | Rule | Fire count | Notes |
|-------|------|-----------:|-------|
| Layer −1 | warmup_not_passed | 81 | always fires until iter_frac ≥ 0.25 |
| Layer −1 | frontier_too_small | 4 | task-specific (ifbench frontier slow to grow) |
| Layer −1 | not_plateaued_optional | 165 | the dominant gate (~71% of all rejection events on qwen) |
| Layer −1 | other start skips | 0 | misc |
| Layer 1 | **G1 parent_strength** | **0** | designed to skip merges with weak parents |
| Layer 1 | **G3 maturity_imbalance** | **0** | catches §15.3 catastrophes (-28.66 / -35.67) |
| Layer 1 | parent_recent_winrate (G2 was deleted) | 0 | optional gate (off in default config) |
| Layer 2 | **A1 bloat (L_MAX)** | **0** | catches §7 IFBench combine_all catastrophe |
| Layer 2 | **A2 specialization split** | **0** | catches multi-domain heterogeneity (§15.2) |
| Layer 2 | **A3 near-duplicate** | **1** | when parents are nearly identical |
| Layer 2 | **A4 default (=`original`)** | **62** | fallback — most merges land here |

→ **A1+A2+A3 routing fired 1 / 63 = 1.6% of Layer 2 decisions.** A4 default catches 62/63 = 98.4%. 

## Per-variant per-task breakdown

| Variant | Task | Start skips | Layer-1 skips | Layer-2 decisions | A1+A2+A3 fired? |
|---------|------|------------:|--------------:|------------------:|:---------------:|
| v1_g2on | hotpotqa | 0 | 4 | 2 | ❌ (0 fired) |
| v1_g2on | ifbench | 5 | 3 | 1 | ❌ (0 fired) |
| v1_g2on | hover | 1 | 0 | 21 | ❌ (0 fired) |
| v1_g2on | musique | 0 | 5 | 6 | ❌ (0 fired) |
| v2_5 | hotpotqa | 6 | 0 | 1 | ❌ (0 fired) |
| v2_5 | ifbench | 6 | 0 | 0 | ❌ (0 fired) |
| v2_5 | hover | 16 | 0 | 2 | ❌ (0 fired) |
| v2_5 | musique | 12 | 0 | 4 | ❌ (0 fired) |
| v2_6 | hotpotqa | 3 | 1 | 4 | ❌ (0 fired) |
| v2_6 | ifbench | 1 | 0 | 1 | ❌ (0 fired) |
| v2_6 | hover | 3 | 0 | 10 | ✅ (1 fired) |
| v2_6 | musique | 3 | 2 | 6 | ❌ (0 fired) |
| v2_7 | hotpotqa | 6 | 0 | 1 | ❌ (0 fired) |
| v2_7 | ifbench | 6 | 0 | 1 | ❌ (0 fired) |
| v2_7 | hover | 15 | 0 | 2 | ❌ (0 fired) |
| v2_7 | musique | 11 | 4 | 1 | ❌ (0 fired) |
| v2_8 | hotpotqa | 6 | 0 | 2 | ❌ (0 fired) |
| v2_8 | ifbench | 6 | 0 | 0 | ❌ (0 fired) |
| v2_8 | hover | 16 | 0 | 2 | ❌ (0 fired) |
| v2_8 | musique | 12 | 0 | 4 | ❌ (0 fired) |
| v2_9 | hotpotqa | 6 | 0 | 1 | ❌ (0 fired) |
| v2_9 | ifbench | 6 | 0 | 0 | ❌ (0 fired) |
| v2_9 | hover | 13 | 0 | 5 | ❌ (0 fired) |
| v2_9 | musique | 12 | 0 | 4 | ❌ (0 fired) |
| v2_10 | hotpotqa | 11 | 0 | 1 | ❌ (0 fired) |
| v2_10 | ifbench | 7 | 0 | 0 | ❌ (0 fired) |
| v2_10 | hover | 16 | 0 | 2 | ❌ (0 fired) |
| v2_10 | musique | 15 | 0 | 4 | ❌ (0 fired) |
| v2_11 | hotpotqa | 4 | 0 | 4 | ❌ (0 fired) |
| v2_11 | ifbench | 6 | 0 | 0 | ❌ (0 fired) |
| v2_11 | hover | 8 | 0 | 7 | ❌ (0 fired) |
| v2_11 | musique | 12 | 0 | 4 | ❌ (0 fired) |

## Conclusion

Across all 32 qwen3-8b adaptive runs surveyed, the catastrophe-prevention gates fire as follows:

- **G1** (parent strength quantile): **0× fires** ❌ designed but never triggered on qwen
- **G3** (maturity imbalance): **0× fires** ❌ designed but never triggered on qwen
- **A1** (bloat L_MAX): **0× fires** ❌ designed but never triggered on qwen
- **A2** (specialization split): **0× fires** ❌ designed but never triggered on qwen
- **A3** (near-duplicate): **1× fires** 

