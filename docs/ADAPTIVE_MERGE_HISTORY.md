# Adaptive Merge — Result Directory

All artifacts produced by the Behavioral Adaptive Merge work (REPORT.md §16)
live here. The implementation source code is unchanged in
`gepa_merge/src/gepa/strategies/adaptive_merge.py` and
`gepa_merge/src/gepa/proposer/adaptive_pair_selection.py`; this folder is for
**outputs and analysis** only.

## Layout

```
adaptive_merge/
├── runs/      Per-cell GEPA run directories (best_candidate.json,
│              gepa_state.bin, merge_quality.jsonl, test_eval.json).
│              Cell name: <model>_<task>_adaptive_s<seed>/
├── replay/    Post-hoc replay of the 80-cell Phase A archive through the
│              new policy (read-only; no LM calls).
│                - adaptive_replay_decisions.csv
│                - adaptive_replay_summary.md
│                - adaptive_replay_summary.json
├── compare/   Comparison reports vs. Phase A baselines.
│                - adaptive_live_v1_compare.md (after live re-run completes)
└── logs/      Per-cell launcher logs from run_adaptive_live.sh.
```

## How outputs got here

- **Replay** (already complete; 73 cells / 296 events; 100% catastrophe coverage):
  ```bash
  cd gepa_merge
  PYTHONPATH=src .venv/bin/python -m experiments.analysis.adaptive_replay \
      --runs_root P2_result/phase_a_main P3_result/phase_a_main \
                  P4_result/phase_a_main runs/phase_a_main \
                  runs/phase_a_main_qwen \
      --out adaptive_merge/replay
  ```

- **Live re-run** (gpt-4.1-mini + qwen3-8b × 4 tasks, seed=0, ~$55, ~36h sequential):
  ```bash
  PRIME_API_KEY=... PRIME_TEAM_ID=... \
      bash experiments/run_adaptive_live.sh
  ```
  Writes per-cell artifacts to `adaptive_merge/runs/` and logs to `adaptive_merge/logs/`.

- **Comparison** (after live re-run):
  ```bash
  PYTHONPATH=src .venv/bin/python -m experiments.analysis.compare_adaptive_to_phase_a \
      --adaptive_root adaptive_merge/runs \
      --baseline_roots P2_result/phase_a_main P3_result/phase_a_main \
                       P4_result/phase_a_main runs/phase_a_main_qwen \
      --out adaptive_merge/compare/adaptive_live_v1_compare.md
  ```

See `gepa_merge/experiments/adaptive_merge_implementation_notes.md` for the full
design + caveats, and the project-level
`/Users/zhaochong/Desktop/NLP_Project/AdaptiveMergePolicy.md` for the canonical
plan.
