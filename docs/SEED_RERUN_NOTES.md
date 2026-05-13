# Phase A Seed Rerun — s=1 + s=2 Test Scores

Generated 2026-05-05 10:39 EDT. All 24 cells complete (12 per seed).

## Per-cell scores

| cell | s1 | s2 | mean | Δ (s2−s1) |
|---|---:|---:|---:|---:|
| qwen_hotpotqa_nomerge | 53.33 | 47.33 | 50.33 | −6.00 |
| qwen_hotpotqa_orig_imm | 45.33 | 45.67 | 45.50 | +0.34 |
| qwen_hotpotqa_orig_plat | 54.00 | 45.67 | 49.84 | **−8.33** |
| qwen_hover_combine_plat | 46.33 | 44.00 | 45.16 | −2.33 |
| qwen_hover_nomerge | 41.67 | 44.00 | 42.84 | +2.33 |
| qwen_hover_orig_imm | 43.33 | 40.33 | 41.83 | −3.00 |
| qwen_ifbench_combine_bp | 35.37 | 30.61 | 32.99 | −4.76 |
| qwen_ifbench_nomerge | 36.05 | 35.20 | 35.62 | −0.85 |
| qwen_ifbench_orig_imm | 36.39 | 32.31 | 34.35 | −4.08 |
| qwen_musique_nomerge | 51.67 | 41.67 | 46.67 | **−10.00** |
| qwen_musique_orig_imm | 33.33 | 49.33 | 41.33 | **+16.00** |
| qwen_musique_sum_imm | 52.00 | 44.67 | 48.34 | −7.33 |

## By task

| task | n | mean Δ | range |
|---|---:|---:|---|
| hotpotqa | 3 | −4.66 | [−8.33, +0.34] |
| hover | 3 | −1.00 | [−3.00, +2.33] |
| ifbench | 3 | −3.23 | [−4.76, −0.85] |
| musique | 3 | −0.44 | [−10.00, +16.00] |

## Observations

1. **Seed noise is non-trivial**: 4 cells move ≥6pp between seeds (hotpotqa_orig_plat −8.33, musique_nomerge −10, musique_orig_imm **+16**, musique_sum_imm −7.33).
2. **musique is the noisiest task**: range spans 26pp across configs. Suggests tiny dev set / high reflection variance.
3. **hover most stable**: all 3 configs within ±3pp.
4. **s=1 generally higher than s=2** (10/12 cells) — could reflect different reflective-rationale paths, not a systematic effect.

## Next step

Aggregate with s=0 archive scores (REPORT.md §5 / §17 already documents s=0). With three seeds we can report mean ± std per cell and do paired tests across configs.

## Files

- s=1: `adaptive_merge/runs_phase_a_seed1/qwen_*_s1/test_eval.json`
- s=2: `adaptive_merge/runs_phase_a_seed2/qwen_*_s2/test_eval.json`
