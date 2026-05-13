"""Adaptive-diversity pair selection for GEPA merge.

Strategy registered as ``"adaptive_diversity"`` in
``gepa.proposer.merge_selection.SELECTION_STRATEGIES``.

This strategy is a *thin delegate* to the active
``BehavioralAdaptiveMergePolicy`` — the policy owns the canonical Layer 1
decision cache (per Caution 1 in the implementation plan). Doing the ranking
inside the policy guarantees ``should_schedule`` and the pair selection
cannot disagree about which pairs survived the gate.

The existing ``SELECTION_STRATEGIES`` callsite (proposer/merge.py) now passes
``state`` and ``policy`` via ``**kwargs``; existing strategies ignore them.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence


def rank_pairs_by_adaptive_diversity(
    merge_candidates: Sequence[int],
    rng: random.Random,
    state=None,
    policy=None,
    **kwargs,
) -> list[tuple[int, int]]:
    """Rank pairs using BehavioralAdaptiveMergePolicy's cached Layer 1 results.

    Falls back to a randomized order if the policy is missing or doesn't
    expose ``rank_pairs`` (so a misconfiguration can't silently crash the
    proposer; the global enable flag in `api.py` is what actually wires this
    strategy in, so the fallback path should never run in practice).
    """
    if policy is not None and hasattr(policy, "rank_pairs") and state is not None:
        return policy.rank_pairs(state, list(merge_candidates))

    pairs = list(itertools.combinations(merge_candidates, 2))
    rng.shuffle(pairs)
    return pairs
