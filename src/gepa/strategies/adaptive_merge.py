# Copyright (c) 2026 the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# Behavioral adaptive merge policy (REPORT.md §16).
#
# Implements:
#   - Layer −1 start gate  (AdaptiveStartPolicy)
#   - Layer 1 skip gates    (BehavioralAdaptiveMergePolicy.should_schedule)
#   - Outer pair selection  (BehavioralAdaptiveMergePolicy.rank_pairs)
#   - Layer 2 algo choice   (BehavioralAdaptiveMergePolicy.algorithm_for_pair)
#
# AdaptiveMergePolicy does not replace GEPA structural legality. It only
# strengthens GEPA-valid triplets with behavioral safety checks.

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gepa.gepa_utils import find_dominator_programs

if TYPE_CHECKING:
    from gepa.core.state import GEPAState
    from gepa.logging.logger import LoggerProtocol


_log = logging.getLogger("gepa.adaptive_merge")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveMergeConfig:
    """All thresholds for the behavioral adaptive merge policy.

    Defaults are chosen a priori from REPORT.md §16; do not tune per-(model,
    benchmark). The post-hoc replay (--sweep) is the proper mechanism for
    threshold sensitivity analysis.
    """

    # Master switch (also exposed at gepa.optimize level)
    adaptive_merge_enabled: bool = False

    # Layer −1 (start gate)
    warmup_frac: float = 0.25
    # CHANGED 2026-05-01: was 5 → now 3.
    # Reason: live v2 qwen × ifbench run logged "frontier_too_small
    # |frontier|=4 < 5" repeatedly; ifbench frontier *never* grew to 5
    # within its 3593-metric-call budget, so 0 merges fired. ifbench's
    # multi-constraint metric is harder to dominate per-example, so the
    # frontier grows slowly. 3 is the structural minimum (you need ≥3
    # candidates to form a non-degenerate ancestor + 2 descendants
    # triplet), so this restores the "merge can fire if structurally
    # possible" baseline without making it more aggressive on
    # easier-to-dominate benchmarks like hotpotqa/musique (which already
    # had |frontier| ≥ 5 well before warmup completed).
    min_frontier_size: int = 3
    # CHANGED 2026-05-02: was False → now True.
    # Empirical evidence (v2 vs v2.6 ablation on qwen3-8b, seed=0, 4 tasks):
    #   v2 (plateau ON):  hotpotqa=51.0, ifbench=35.54, hover=43.0,  musique=51.67
    #   v2.6 (plateau OFF): hotpotqa=51.0, ifbench=32.65, hover=1.67, musique=49.0
    # Plateau OFF was 0/4 wins, 3/4 cells dropped 2.67–41.33 pp; hover
    # collapsed catastrophically because plateau OFF allowed 7 accepted
    # merges (vs 1 in plateau ON) which compounded into a corrupted prompt
    # that broke test-time inference. Phase A's `score_plateau` start policy
    # finding was empirically vindicated. Lock plateau ON as default.
    use_plateau_gate: bool = True
    plateau_window: int = 3
    plateau_eps: float = 0.0
    min_iter_before_merge: int = 5  # used only if max_metric_calls is None

    # Layer 1 (skip gate)
    parent_strength_quantile: float = 0.50  # G1: median (relative threshold, not magic number)
    # G2 (behavioral_complementarity) was removed. Reasons:
    #   1. Any specific agreement threshold (0.85, 0.84, 0.80, ...) was a magic
    #      number with no principled derivation; in 20 observed merges on qwen
    #      it produced 0/20 skips, i.e. it was a no-op on qwen.
    #   2. Its "is this pair worth merging" intent is already covered by the
    #      outer `rank_pairs` selection (which picks max-diversity) and by
    #      GEPA's native subsample gate (which rejects merges whose child
    #      doesn't beat max(parent_subsample_sums) — near-duplicate parents
    #      produce near-duplicate children that fail naturally).
    use_maturity_gate: bool = True  # G3 master toggle
    maturity_gini_max: float = 0.50  # G3
    use_parent_recent_winrate_gate: bool = False  # optional, off by default
    parent_recent_winrate_min: float = 0.50

    # Layer 2 (algorithm selection)
    L_MAX: int = 4000  # A1: absolute predictor-length cap (defensive; rarely fires in adaptive mode)
    # REMOVED 2026-05-01: BLOAT_SAFE_RATIO was the threshold for the
    # `predicted_combine_all_length / max(parent_lengths)` growth check.
    # That signal was specifically simulating "if combine_all were applied,
    # how much would the program grow?". After the A4-default flip
    # (combine_all → original), the policy never routes to combine_all, so
    # the growth-ratio gate was predicting an algorithm we never invoke.
    # The signal is still computed and logged in `sigs` for observability
    # (so replay tooling can compare adaptive vs Phase A combine_all runs),
    # but it no longer gates the routing decision. L_MAX (above) is retained
    # as a defensive absolute-length cap that triggers `summarize_before` if
    # any predictor — from any source, including reflective_full_program — has
    # grown past the danger zone.
    specialization_split_threshold: float = 0.30  # A2
    duplicate_jaccard_threshold: float = 0.70  # A3
    # ADDED 2026-05-02 (v2.8 ablation): bypass Layer 2 routing entirely.
    # When True, _evaluate_layer2 returns ("original", "layer2_routing_disabled")
    # immediately — A1 (L_MAX), A2 (specialization_split), A3 (near_duplicate)
    # are skipped, all merges use `original` algorithm. Signals are still
    # computed and logged for observability. Hypothesis being tested: v2's
    # hover (-4.67 vs NoMerge) and hotpotqa (-4 vs NoMerge) regressions come
    # from A1/A2 routing some merges to `summarize_before` whose LM-rewritten
    # output is worse than `original`'s per-predictor pick. Phase A's
    # `original_immediate` (= no Layer 2 routing, always `original`) gets
    # 50.67 on hover and 56.33 on hotpotqa, far above our v2 (43.0/51.0).
    disable_layer2_routing: bool = False
    # ADDED 2026-05-02 (v2.9 ablation): A4 default algorithm override.
    # Locked default = "original" (per A4 flip 2026-05-01). v2.9 tests if
    # "combine_all" recovers hover (Phase A combine_all_immediate=50.0,
    # score_plateau=42.33; current v2 hover=43.0). Risks musique (PA
    # combine_all_immediate=45.0 vs 48.33 NoMerge) but with our plateau-on
    # stack we'd land closer to combine_all_score_plateau=51.33. Allowed
    # values: "original" / "combine_all" / "summarize_before".
    a4_default_algorithm: str = "original"

    # Misc
    correctness_threshold: float = 0.5  # binarize per-example continuous scores

    # Injected at construction time (not user-visible)
    max_metric_calls: int | None = None


# ---------------------------------------------------------------------------
# Decision record (canonical cache entry)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveMergeDecision:
    """Cache entry per (ancestor, id1, id2). Owns Layer 1 + Layer 2 result."""

    ancestor: int
    id1: int
    id2: int
    # Layer 1
    layer1_passed: bool
    layer1_skip_reason: str | None
    # Layer 2 (only meaningful when layer1_passed=True)
    algorithm: str | None  # "original" / "combine_all" / "summarize_before"
    algorithm_reason: str | None
    # Observability
    behavioral_diversity: float
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Layer1Result:
    """Per-pair Layer 1 result (no ancestor yet)."""

    id1: int
    id2: int
    passed: bool
    skip_reason: str | None
    behavioral_diversity: float
    signals: dict[str, Any]


# ---------------------------------------------------------------------------
# Pure-function signal computers (reusable from live + replay)
# ---------------------------------------------------------------------------


def binarize(scores: dict, threshold: float) -> dict:
    """Map continuous per-example scores to 0/1 correctness."""
    return {k: (1 if v >= threshold else 0) for k, v in scores.items()}


def example_level_agreement(corr_a: dict, corr_b: dict) -> float:
    """Fraction of common examples where corr_a[k] == corr_b[k].

    Returns 1.0 if no common examples (degenerate; treat as "too similar").
    """
    common = set(corr_a.keys()) & set(corr_b.keys())
    if not common:
        return 1.0
    matches = sum(1 for k in common if corr_a[k] == corr_b[k])
    return matches / len(common)


def behavioral_diversity(corr_a: dict, corr_b: dict) -> float:
    """A_only_correct + B_only_correct over common examples.

    This is the "complementary correctness" form (§16.2 Outer-loop) and
    is preferred over `1 - agreement` because it weights complementary
    failure modes equally with complementary successes.
    """
    common = set(corr_a.keys()) & set(corr_b.keys())
    if not common:
        return 0.0
    a_only = sum(1 for k in common if corr_a[k] == 1 and corr_b[k] == 0)
    b_only = sum(1 for k in common if corr_b[k] == 1 and corr_a[k] == 0)
    return (a_only + b_only) / len(common)


def specialization_split(corr_a: dict, corr_b: dict) -> float:
    """A2 signal: same numerator as behavioral_diversity but interpreted as
    'how much do the parents specialize to disjoint subsets'."""
    return behavioral_diversity(corr_a, corr_b)


def token_jaccard(prog_a: dict, prog_b: dict) -> float:
    """Word-level Jaccard across all shared predictor prompts."""
    keys = set(prog_a.keys()) & set(prog_b.keys())
    if not keys:
        return 0.0
    a_words: set[str] = set()
    b_words: set[str] = set()
    for k in keys:
        a_words |= set(prog_a[k].split())
        b_words |= set(prog_b[k].split())
    union = a_words | b_words
    if not union:
        return 1.0
    return len(a_words & b_words) / len(union)


def prompt_token_count(prog: dict) -> int:
    """Approximate total token count via len(text) // 4 across all predictors."""
    return sum(max(1, len(v) // 4) for v in prog.values())


def max_predictor_token_count(prog: dict) -> int:
    """Max single-predictor token count (A1 bloat trigger)."""
    if not prog:
        return 0
    return max(max(1, len(v) // 4) for v in prog.values())


def predicted_combine_all_length(
    prog_anc: dict,
    prog_a: dict,
    prog_b: dict,
) -> int:
    """Estimate the token count if combine_all were applied to (anc, a, b).

    Mirrors merge_combine_all_subprompts case logic without invoking it:
        - Case C (a == b): take a
        - Case A (one unchanged): take the changed
        - Case B (both diverged): pred_a + template_overhead + pred_b
    """
    template_overhead = 6  # ~"---\\n\\nAdditional guidance:\\n\\n" tokens
    keys = set(prog_anc.keys()) & set(prog_a.keys()) & set(prog_b.keys())
    if not keys:
        return prompt_token_count(prog_a)
    total = 0
    for k in keys:
        anc, a, b = prog_anc[k], prog_a[k], prog_b[k]
        a_t = max(1, len(a) // 4)
        b_t = max(1, len(b) // 4)
        if a == b:
            total += a_t
        elif anc == a:
            total += b_t
        elif anc == b:
            total += a_t
        else:
            total += a_t + b_t + template_overhead
    return total


def per_predictor_mutation_count(
    candidate_idx: int,
    parent_program_for_candidate: Sequence,
    program_candidates: Sequence[dict],
) -> dict[str, int] | None:
    """Walk lineage from candidate to seed, counting predictor changes per step.

    Returns None if lineage is missing or candidate is the seed. The walk
    follows the first parent at each step (deterministic). Two-parent merges
    contribute via the first parent only — this is sufficient for measuring
    *how mutated* this lineage is, since the second parent's lineage is
    captured separately via its own walk.
    """
    if candidate_idx < 0 or candidate_idx >= len(program_candidates):
        return None
    pred_names = list(program_candidates[candidate_idx].keys())
    counts = {pred: 0 for pred in pred_names}
    visited: set[int] = set()
    current = candidate_idx
    walked = False
    while True:
        if current >= len(parent_program_for_candidate):
            break
        parents = parent_program_for_candidate[current]
        if not parents:
            break
        parent = parents[0]
        if parent is None:
            break
        if parent in visited:
            break
        if parent < 0 or parent >= len(program_candidates):
            break
        visited.add(parent)
        for pred in pred_names:
            cur_val = program_candidates[current].get(pred)
            par_val = program_candidates[parent].get(pred)
            if cur_val != par_val:
                counts[pred] += 1
        current = parent
        walked = True
    if not walked:
        return None
    return counts


def gini_coefficient(values: Sequence[float]) -> float:
    """Standard Gini over a non-negative count vector. Returns 0 if all zero."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sum(values)
    if s <= 0:
        return 0.0
    sorted_vals = sorted(values)
    cum = 0.0
    for i, v in enumerate(sorted_vals, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def predictor_maturity_gini(
    id1: int,
    id2: int,
    parent_program_for_candidate: Sequence,
    program_candidates: Sequence[dict],
) -> float | None:
    """G3 signal: imbalance across per-predictor mutation counts of (A, B).

    Returns None if neither lineage is walkable (caller should log
    'maturity_gini_unavailable' and either skip the gate or treat as pass
    based on `use_maturity_gate`).
    """
    counts_a = per_predictor_mutation_count(id1, parent_program_for_candidate, program_candidates)
    counts_b = per_predictor_mutation_count(id2, parent_program_for_candidate, program_candidates)
    if counts_a is None and counts_b is None:
        return None
    pred_names = (
        set(counts_a.keys()) if counts_a else set()
    ) | (set(counts_b.keys()) if counts_b else set())
    combined = []
    for pred in pred_names:
        a_c = counts_a.get(pred, 0) if counts_a else 0
        b_c = counts_b.get(pred, 0) if counts_b else 0
        combined.append(a_c + b_c)
    return gini_coefficient(combined)


# ---------------------------------------------------------------------------
# Logging guard (Caution 2)
# ---------------------------------------------------------------------------


def safe_log_adaptive(
    state: "GEPAState | None",
    logger: "LoggerProtocol | None",
    key: str,
    value: Any,
) -> None:
    """Write `adaptive_<key> = value` to state.full_program_trace[-1] iff that
    entry is the current merge attempt. On any mismatch, fall back to the
    proposer logger and the module logger; never pollute an unrelated trace
    entry.

    The check confirms (a) trace exists, (b) last entry is a dict, (c) entry
    has invoked_merge=True (set by MergeProposer.propose at line 319 before
    any adaptive code runs).
    """
    full_key = f"adaptive_{key}"
    wrote_to_trace = False
    if state is not None:
        trace = getattr(state, "full_program_trace", None)
        if isinstance(trace, list) and trace:
            last = trace[-1]
            if isinstance(last, dict) and last.get("invoked_merge") is True:
                last[full_key] = value
                wrote_to_trace = True
    if not wrote_to_trace:
        msg = f"[adaptive] {key}={value} (trace skip; no current merge entry)"
        if logger is not None:
            try:
                logger.log(msg)
            except Exception:
                _log.info(msg)
        else:
            _log.info(msg)


# ---------------------------------------------------------------------------
# Layer −1 — AdaptiveStartPolicy
# ---------------------------------------------------------------------------


class AdaptiveStartPolicy:
    """Layer −1 of the adaptive pipeline.

    Hard checks:
      1. Warmup fraction (state.total_num_evals / max_metric_calls >= warmup_frac).
         If max_metric_calls is None, fall back to iteration count vs config.min_iter_before_merge.
      2. Active frontier size (>= config.min_frontier_size) computed via
         find_dominator_programs on state.get_pareto_front_mapping().

    Optional:
      3. Plateau gate (config.use_plateau_gate). Tracks best score across
         allow_merge calls; allows merge only when best score has not improved
         by more than plateau_eps over the last plateau_window calls.
    """

    def __init__(
        self,
        config: AdaptiveMergeConfig,
        logger: "LoggerProtocol | None" = None,
    ):
        self.config = config
        self.logger = logger
        # Plateau-tracking state (only used if config.use_plateau_gate=True)
        self._best_history: list[float] = []

    # MergeStartPolicy Protocol
    def allow_merge(self, state: "GEPAState") -> bool:
        cfg = self.config

        # 1) Warmup
        if cfg.max_metric_calls is not None and cfg.max_metric_calls > 0:
            consumed = float(getattr(state, "total_num_evals", 0))
            frac = consumed / cfg.max_metric_calls
            if frac < cfg.warmup_frac:
                self._log_skip("warmup_not_passed", f"frac={frac:.3f} < {cfg.warmup_frac}")
                return False
        else:
            current_iter = int(getattr(state, "i", -1))
            if current_iter < cfg.min_iter_before_merge:
                self._log_skip(
                    "warmup_not_passed",
                    f"iter={current_iter} < min_iter_before_merge={cfg.min_iter_before_merge}",
                )
                return False

        # 2) Frontier size
        pareto_mapping = state.get_pareto_front_mapping()
        tracked_scores = list(getattr(state, "per_program_tracked_scores", state.program_full_scores_val_set))
        active_frontier = find_dominator_programs(pareto_mapping, tracked_scores)
        if len(active_frontier) < cfg.min_frontier_size:
            self._log_skip(
                "frontier_too_small",
                f"|frontier|={len(active_frontier)} < {cfg.min_frontier_size}",
            )
            return False

        # 3) Plateau (optional)
        if cfg.use_plateau_gate:
            if not tracked_scores:
                return False
            current_best = max(tracked_scores)
            self._best_history.append(current_best)
            if len(self._best_history) < cfg.plateau_window + 1:
                self._log_skip(
                    "not_plateaued_optional",
                    f"history_len={len(self._best_history)} < window+1={cfg.plateau_window + 1}",
                )
                return False
            window = self._best_history[-(cfg.plateau_window + 1):]
            improvement = window[-1] - window[0]
            if improvement > cfg.plateau_eps:
                self._log_skip(
                    "not_plateaued_optional",
                    f"improvement={improvement:.4f} > eps={cfg.plateau_eps}",
                )
                return False

        return True

    def _log_skip(self, reason: str, detail: str) -> None:
        msg = f"[adaptive_start] skip reason={reason} {detail}"
        if self.logger is not None:
            try:
                self.logger.log(msg)
                return
            except Exception:
                pass
        _log.info(msg)


# ---------------------------------------------------------------------------
# Layer 1 + 2 — BehavioralAdaptiveMergePolicy
# ---------------------------------------------------------------------------


class BehavioralAdaptiveMergePolicy:
    """REPORT.md §16 implementation.

    Owns the canonical decision cache (per Caution 1):
      - `_pair_cache: dict[(id1,id2), _Layer1Result]`        — Layer 1 results
      - `_decisions:  dict[(ancestor,id1,id2), AdaptiveMergeDecision]` — full triplet records

    Cache lifetime: from `should_schedule(state, ...)` to the next call.
    Both `rank_pairs` and `algorithm_for_pair` read/write the same caches so
    they cannot disagree about a triplet.
    """

    def __init__(
        self,
        config: AdaptiveMergeConfig,
        logger: "LoggerProtocol | None" = None,
    ):
        self.config = config
        self.logger = logger
        self._pair_cache: dict[tuple[int, int], _Layer1Result] = {}
        self._decisions: dict[tuple[int, int, int], AdaptiveMergeDecision] = {}

    # -----------------------------------------------------------------
    # Public API — MergePolicy Protocol
    # -----------------------------------------------------------------

    def should_schedule(
        self,
        state: "GEPAState",
        merge_candidates: Sequence[int],
    ) -> bool:
        """Layer 1 entry point. Resets caches, evaluates Layer 1 gates per
        pair, returns True iff at least one pair survives."""
        self._pair_cache.clear()
        self._decisions.clear()

        if len(merge_candidates) < 2:
            self._log("[layer1] no_valid_adaptive_pair: <2 candidates")
            return False

        scores = list(getattr(state, "per_program_tracked_scores", state.program_full_scores_val_set))
        # G1 reference: median of active-frontier scores
        frontier_scores = [scores[i] for i in merge_candidates if 0 <= i < len(scores)]
        if not frontier_scores:
            self._log("[layer1] no_valid_adaptive_pair: empty frontier scores")
            return False
        ref_score = _quantile(frontier_scores, self.config.parent_strength_quantile)

        any_passed = False
        cands = list(merge_candidates)
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                if key in self._pair_cache:
                    continue
                result = self._evaluate_layer1(state, key[0], key[1], scores, ref_score)
                self._pair_cache[key] = result
                if result.passed:
                    any_passed = True

        if not any_passed:
            self._log("[layer1] no_valid_adaptive_pair: all skipped")
        return any_passed

    def rank_pairs(
        self,
        state: "GEPAState",
        merge_candidates: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Outer pair selection over the *current* merge_candidates.

        Uses the Layer 1 cache populated by `should_schedule` as a hint;
        lazily evaluates any pair not in the cache (without clearing
        existing entries — only `should_schedule` clears). This keeps
        rank_pairs and should_schedule strictly consistent for any pair
        they both touch.
        """
        scores = list(
            getattr(state, "per_program_tracked_scores", state.program_full_scores_val_set)
        )
        cands = sorted({c for c in merge_candidates if 0 <= c < len(state.program_candidates)})
        if len(cands) < 2:
            return []
        # G1 reference: median of active-frontier scores
        frontier_scores = [scores[i] for i in cands if 0 <= i < len(scores)]
        ref_score = (
            _quantile(frontier_scores, self.config.parent_strength_quantile)
            if frontier_scores else 0.0
        )

        passing: list[_Layer1Result] = []
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                key = (a, b)
                result = self._pair_cache.get(key)
                if result is None:
                    result = self._evaluate_layer1(state, a, b, scores, ref_score)
                    self._pair_cache[key] = result
                if result.passed:
                    passing.append(result)
        if not passing:
            return []

        def sort_key(r: _Layer1Result) -> tuple:
            min_score = (
                min(scores[r.id1], scores[r.id2])
                if 0 <= r.id1 < len(scores) and 0 <= r.id2 < len(scores)
                else 0.0
            )
            bloat_penalty = max(
                max_predictor_token_count(state.program_candidates[r.id1])
                if r.id1 < len(state.program_candidates) else 0,
                max_predictor_token_count(state.program_candidates[r.id2])
                if r.id2 < len(state.program_candidates) else 0,
            )
            return (
                -r.behavioral_diversity,  # higher diversity first
                -min_score,                # break ties by stronger weakest parent
                bloat_penalty,             # lower bloat first
                r.id1, r.id2,              # deterministic tie-break
            )

        ranked = sorted(passing, key=sort_key)
        return [(r.id1, r.id2) for r in ranked]

    def algorithm_for_pair(
        self,
        state: "GEPAState",
        ancestor: int,
        id1: int,
        id2: int,
    ) -> str:
        """Layer 2 entry point. Idempotent via _decisions cache.

        The proposer's wrapper in merge.py calls this after Layer 0 has
        produced a structurally-valid triplet (ancestor, id1, id2). Returns
        one of {"original", "combine_all", "summarize_before"}.
        """
        # Normalize pair order so cache hits regardless of caller convention
        a, b = (id1, id2) if id1 <= id2 else (id2, id1)
        key = (ancestor, a, b)
        cached = self._decisions.get(key)
        if cached is not None and cached.algorithm is not None:
            return cached.algorithm

        # Reuse Layer 1 result if available (same pair, regardless of ancestor)
        pair_key = (a, b)
        layer1 = self._pair_cache.get(pair_key)
        if layer1 is None:
            # Fallback: compute inline (rare; defensive).
            scores = list(
                getattr(state, "per_program_tracked_scores", state.program_full_scores_val_set)
            )
            frontier = [scores[i] for i in (a, b) if 0 <= i < len(scores)]
            ref_score = min(frontier) if frontier else 0.0
            layer1 = self._evaluate_layer1(state, a, b, scores, ref_score)
            self._pair_cache[pair_key] = layer1

        algo, reason, signals = self._select_algorithm(state, ancestor, a, b, layer1)
        decision = AdaptiveMergeDecision(
            ancestor=ancestor,
            id1=a,
            id2=b,
            layer1_passed=layer1.passed,
            layer1_skip_reason=layer1.skip_reason,
            algorithm=algo,
            algorithm_reason=reason,
            behavioral_diversity=layer1.behavioral_diversity,
            signals={**layer1.signals, **signals},
        )
        self._decisions[key] = decision
        self._log(
            f"[layer2] decision pair=({a},{b}) ancestor={ancestor} algo={algo} reason={reason}"
        )
        return algo

    # Inspection helper (used by replay + tests; not part of MergePolicy)
    def get_cached_decision(
        self, ancestor: int, id1: int, id2: int
    ) -> AdaptiveMergeDecision | None:
        a, b = (id1, id2) if id1 <= id2 else (id2, id1)
        return self._decisions.get((ancestor, a, b))

    # -----------------------------------------------------------------
    # Internals — Layer 1
    # -----------------------------------------------------------------

    def _evaluate_layer1(
        self,
        state: "GEPAState",
        id1: int,
        id2: int,
        scores: list[float],
        ref_score: float,
    ) -> _Layer1Result:
        cfg = self.config
        signals: dict[str, Any] = {}

        # G1: parent strength
        s1 = scores[id1] if 0 <= id1 < len(scores) else 0.0
        s2 = scores[id2] if 0 <= id2 < len(scores) else 0.0
        signals["score_a"] = s1
        signals["score_b"] = s2
        signals["ref_score"] = ref_score
        if s1 < ref_score or s2 < ref_score:
            return _Layer1Result(
                id1=id1, id2=id2, passed=False,
                skip_reason="G1_parent_strength",
                behavioral_diversity=0.0,
                signals=signals,
            )

        # Prepare per-example correctness vectors (used by Layer 2 A2 and for
        # behavioral_diversity ranking in `rank_pairs`).
        sub_a = state.prog_candidate_val_subscores[id1] if id1 < len(state.prog_candidate_val_subscores) else {}
        sub_b = state.prog_candidate_val_subscores[id2] if id2 < len(state.prog_candidate_val_subscores) else {}
        corr_a = binarize(sub_a, cfg.correctness_threshold)
        corr_b = binarize(sub_b, cfg.correctness_threshold)
        # Compute observability signals (no longer used as a gate — G2 was
        # removed; see docstring on AdaptiveMergeConfig.parent_strength_quantile).
        agreement = example_level_agreement(corr_a, corr_b)
        diversity = behavioral_diversity(corr_a, corr_b)
        signals["example_agreement"] = agreement
        signals["behavioral_diversity"] = diversity

        # G3: predictor maturity imbalance (toggleable)
        if cfg.use_maturity_gate:
            gini = predictor_maturity_gini(
                id1, id2,
                state.parent_program_for_candidate,
                state.program_candidates,
            )
            signals["maturity_gini"] = gini
            if gini is None:
                signals["maturity_gini_unavailable"] = True
                # Per spec: do not crash, do not penalize. Skip this gate.
                self._log(
                    f"[layer1] maturity_gini_unavailable pair=({id1},{id2}); skipping G3"
                )
            elif gini > cfg.maturity_gini_max:
                return _Layer1Result(
                    id1=id1, id2=id2, passed=False,
                    skip_reason="G3_maturity_imbalance",
                    behavioral_diversity=diversity,
                    signals=signals,
                )

        # Optional: parent recent subsample winrate (off by default)
        if cfg.use_parent_recent_winrate_gate:
            winrate_a = _recent_subsample_winrate(state, id1)
            winrate_b = _recent_subsample_winrate(state, id2)
            signals["recent_winrate_a"] = winrate_a
            signals["recent_winrate_b"] = winrate_b
            if winrate_a is not None and winrate_a < cfg.parent_recent_winrate_min:
                return _Layer1Result(
                    id1=id1, id2=id2, passed=False,
                    skip_reason="parent_recent_winrate_low_a",
                    behavioral_diversity=diversity,
                    signals=signals,
                )
            if winrate_b is not None and winrate_b < cfg.parent_recent_winrate_min:
                return _Layer1Result(
                    id1=id1, id2=id2, passed=False,
                    skip_reason="parent_recent_winrate_low_b",
                    behavioral_diversity=diversity,
                    signals=signals,
                )

        return _Layer1Result(
            id1=id1, id2=id2, passed=True,
            skip_reason=None,
            behavioral_diversity=diversity,
            signals=signals,
        )

    # -----------------------------------------------------------------
    # Internals — Layer 2
    # -----------------------------------------------------------------

    def _select_algorithm(
        self,
        state: "GEPAState",
        ancestor: int,
        id1: int,
        id2: int,
        layer1: _Layer1Result,
    ) -> tuple[str, str, dict[str, Any]]:
        cfg = self.config
        sigs: dict[str, Any] = {}

        prog_a = state.program_candidates[id1] if id1 < len(state.program_candidates) else {}
        prog_b = state.program_candidates[id2] if id2 < len(state.program_candidates) else {}
        prog_anc = state.program_candidates[ancestor] if ancestor < len(state.program_candidates) else {}

        # Compute signals up front (always, for observability — even when
        # disable_layer2_routing bypasses the gates).
        max_pred_a = max_predictor_token_count(prog_a)
        max_pred_b = max_predictor_token_count(prog_b)
        sigs["max_predictor_tokens_a"] = max_pred_a
        sigs["max_predictor_tokens_b"] = max_pred_b

        # ADDED 2026-05-02 (v2.8 ablation): if disable_layer2_routing, skip
        # A1/A2/A3 and always return `original`. Signals computed below are
        # still logged. Hypothesis: A1/A2 routing some merges to summarize_before
        # is what hurts hover/hotpotqa vs NoMerge.
        if cfg.disable_layer2_routing:
            # Still compute the rest of the signals for observability
            total_a = prompt_token_count(prog_a)
            total_b = prompt_token_count(prog_b)
            sigs["predicted_combine_all_length"] = predicted_combine_all_length(prog_anc, prog_a, prog_b)
            sigs["predicted_growth_ratio"] = sigs["predicted_combine_all_length"] / max(total_a, total_b, 1)
            sub_a = state.prog_candidate_val_subscores[id1] if id1 < len(state.prog_candidate_val_subscores) else {}
            sub_b = state.prog_candidate_val_subscores[id2] if id2 < len(state.prog_candidate_val_subscores) else {}
            if sub_a and sub_b:
                corr_a = binarize(sub_a, cfg.correctness_threshold)
                corr_b = binarize(sub_b, cfg.correctness_threshold)
                sigs["specialization_split"] = specialization_split(corr_a, corr_b)
            sigs["token_jaccard"] = token_jaccard(prog_a, prog_b)
            return "original", "layer2_routing_disabled", sigs

        # A1 — bloat risk
        if max(max_pred_a, max_pred_b) > cfg.L_MAX:
            return "summarize_before", "bloat_risk_predictor_over_LMAX", sigs

        # Observability-only: predicted combine_all length and growth ratio.
        # REMOVED 2026-05-01 as a gate — the growth ratio specifically simulates
        # `combine_all` output length, but the policy never routes to
        # combine_all (A4 was flipped from combine_all → original on the same
        # date). Using a fictional algorithm's predicted length to gate routing
        # was logically incoherent. We still compute + log the values so
        # post-hoc replay can compare adaptive vs Phase A combine_all runs and
        # so future analyses can study the bloat we *would* have caused had we
        # taken the combine_all path.
        total_a = prompt_token_count(prog_a)
        total_b = prompt_token_count(prog_b)
        predicted = predicted_combine_all_length(prog_anc, prog_a, prog_b)
        denom = max(total_a, total_b, 1)
        growth = predicted / denom
        sigs["predicted_combine_all_length"] = predicted
        sigs["predicted_growth_ratio"] = growth

        # A2 — specialization split (reuse correctness vectors via Layer 1 signals)
        # Recompute here to be safe in case _pair_cache was populated outside should_schedule.
        sub_a = state.prog_candidate_val_subscores[id1] if id1 < len(state.prog_candidate_val_subscores) else {}
        sub_b = state.prog_candidate_val_subscores[id2] if id2 < len(state.prog_candidate_val_subscores) else {}
        if sub_a and sub_b:
            corr_a = binarize(sub_a, cfg.correctness_threshold)
            corr_b = binarize(sub_b, cfg.correctness_threshold)
            spec = specialization_split(corr_a, corr_b)
            sigs["specialization_split"] = spec
            if spec > cfg.specialization_split_threshold:
                return "summarize_before", "specialization_split", sigs
        else:
            sigs["specialization_split_unavailable"] = True

        # A3 — near duplicate
        jacc = token_jaccard(prog_a, prog_b)
        sigs["token_jaccard"] = jacc
        if jacc > cfg.duplicate_jaccard_threshold:
            return "original", "near_duplicate", sigs

        # A4 — safe complementary default
        # CHANGED 2026-05-01: was "combine_all" → now "original".
        # CHANGED 2026-05-02 (v2.9): default value comes from
        # cfg.a4_default_algorithm to allow per-run override. Default is
        # still "original"; v2.9 tests "combine_all" for hover recovery.
        return cfg.a4_default_algorithm, f"safe_complementary_{cfg.a4_default_algorithm}", sigs

    # -----------------------------------------------------------------
    # Logging helper
    # -----------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            try:
                self.logger.log(msg)
                return
            except Exception:
                pass
        _log.info(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantile(values: Sequence[float], q: float) -> float:
    """Simple quantile (linear interpolation between order statistics)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _recent_subsample_winrate(state: "GEPAState", program_idx: int) -> float | None:
    """Best-effort scrape of recent subsample win events for `program_idx` from
    state.full_program_trace. Returns None if no events found.

    GEPA writes id1_subsample_scores / id2_subsample_scores / new_program_subsample_scores
    on each merge attempt (merge.py:428-430). We use these as the only available
    'recent subsample history'. Since this is optional and off by default, the
    implementation is intentionally minimal.
    """
    trace = getattr(state, "full_program_trace", None)
    if not isinstance(trace, list):
        return None
    wins = 0
    total = 0
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        merged = entry.get("merged_entities")
        if not (isinstance(merged, tuple) and len(merged) == 3):
            continue
        id1, id2, _ = merged
        if program_idx == id1:
            p_scores = entry.get("id1_subsample_scores")
            n_scores = entry.get("new_program_subsample_scores")
        elif program_idx == id2:
            p_scores = entry.get("id2_subsample_scores")
            n_scores = entry.get("new_program_subsample_scores")
        else:
            continue
        if isinstance(p_scores, list) and isinstance(n_scores, list) and len(p_scores) == len(n_scores) and p_scores:
            for ps, ns in zip(p_scores, n_scores):
                total += 1
                if ps >= ns:
                    wins += 1
    if total == 0:
        return None
    return wins / total
