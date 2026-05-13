"""Start policies for GEPA merge (Dimension B of the ablation study).

Each policy decides whether a merge should be scheduled at the current
iteration. Inserted as an additional gate inside
`MergeProposer.schedule_if_needed`, before the existing `MergePolicy`
check. Designed so `ImmediateStartPolicy` is a perfect no-op preserving
original GEPA behavior.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol

from gepa.proposer.merge_selection import prompt_divergence

if TYPE_CHECKING:
    from gepa.core.state import GEPAState


class MergeStartPolicy(Protocol):
    """Decides whether merge is allowed to be scheduled right now."""

    def allow_merge(self, state: "GEPAState") -> bool: ...


class ImmediateStartPolicy:
    """B1: allow merge from the first successful mutation (original GEPA behavior)."""

    def allow_merge(self, state: "GEPAState") -> bool:
        return True


class DelayedStartPolicy:
    """B2: only allow merge once the candidate pool has grown to `min_candidates`."""

    def __init__(self, min_candidates: int = 4):
        if min_candidates < 2:
            raise ValueError("min_candidates must be >= 2 for merge to be meaningful")
        self.min_candidates = min_candidates

    def allow_merge(self, state: "GEPAState") -> bool:
        return len(state.program_candidates) >= self.min_candidates


class PeriodicStartPolicy:
    """B3: allow merge every N-th successful mutation.

    The engine calls `schedule_if_needed` only on iterations that found a new
    program, so each `allow_merge` call corresponds to one successful mutation.
    """

    def __init__(self, merge_every_n: int = 3):
        if merge_every_n < 1:
            raise ValueError("merge_every_n must be >= 1")
        self.merge_every_n = merge_every_n
        self._mutation_count = 0

    def allow_merge(self, state: "GEPAState") -> bool:
        self._mutation_count += 1
        if self._mutation_count >= self.merge_every_n:
            self._mutation_count = 0
            return True
        return False


class DiversityTriggeredStartPolicy:
    """B4: allow merge only when the Pareto front's average pairwise prompt
    divergence meets `diversity_threshold`."""

    def __init__(self, diversity_threshold: float = 0.3):
        if not 0.0 <= diversity_threshold <= 1.0:
            raise ValueError("diversity_threshold must be in [0, 1]")
        self.diversity_threshold = diversity_threshold

    def allow_merge(self, state: "GEPAState") -> bool:
        front_indices: set[int] = set()
        for front in state.program_at_pareto_front_valset.values():
            front_indices |= front
        candidates = list(front_indices)
        if len(candidates) < 2:
            return False

        divs = [
            prompt_divergence(state.program_candidates[i], state.program_candidates[j])
            for i, j in itertools.combinations(candidates, 2)
        ]
        avg_div = sum(divs) / len(divs) if divs else 0.0
        return avg_div >= self.diversity_threshold


class ScorePlateauStartPolicy:
    """B5: allow merge only once the best Pareto score has stalled for
    `plateau_patience` consecutive calls.

    The 'best score' is the max of `state.per_program_tracked_scores`, which
    is the mean val-subset score of each candidate. Each call to `allow_merge`
    corresponds to one successful mutation (see engine contract). When the
    best score doesn't strictly improve `plateau_patience` times in a row,
    mutation is assumed to have stalled and merge fires. The counter resets
    once a merge is permitted, so the next merge waits for another plateau.
    """

    def __init__(self, plateau_patience: int = 3):
        if plateau_patience < 1:
            raise ValueError("plateau_patience must be >= 1")
        self.plateau_patience = plateau_patience
        self._best_so_far: float = float("-inf")
        self._stalled_count: int = 0

    def allow_merge(self, state: "GEPAState") -> bool:
        scores = state.per_program_tracked_scores
        if not scores:
            return False
        current_best = max(scores)
        if current_best > self._best_so_far:
            self._best_so_far = current_best
            self._stalled_count = 0
            return False
        self._stalled_count += 1
        if self._stalled_count >= self.plateau_patience:
            self._stalled_count = 0
            return True
        return False


class BudgetProportionalStartPolicy:
    """B6: allow merge only once a fixed fraction of the optimization budget
    has been consumed.

    Normalizes the delayed-start idea across benchmarks with different
    `max_metric_calls`: a candidate-count threshold (B2) means very different
    things at budget=500 vs budget=5000, while "33% of budget consumed"
    transfers cleanly. `max_metric_calls` is passed in from api.optimize.

    Default is 0.33 (was 0.25). Rationale: on small-budget tasks (PUPA=2426,
    IFBench=3593, D_pareto=111–300), a 0.25 trigger lands around iter ~15
    out of ~60 total iterations, which is close enough to `immediate`'s
    iter ~3 trigger that the two policies produce near-indistinguishable
    search trees. 0.33 shifts the trigger to ~iter 20–25, providing clearer
    separation between `immediate` / `budget_proportional` / `score_plateau`
    in the Phase A ablation. See experiments/PHASE_A_PLAN.md.
    """

    def __init__(self, budget_fraction: float = 0.33, max_metric_calls: int | None = None):
        if not 0.0 <= budget_fraction <= 1.0:
            raise ValueError("budget_fraction must be in [0, 1]")
        if max_metric_calls is None or max_metric_calls <= 0:
            raise ValueError("max_metric_calls must be provided and > 0")
        self.budget_fraction = budget_fraction
        self.max_metric_calls = max_metric_calls

    def allow_merge(self, state: "GEPAState") -> bool:
        consumed = getattr(state, "total_num_evals", 0)
        return consumed >= self.budget_fraction * self.max_metric_calls


START_POLICIES: dict[str, type[MergeStartPolicy]] = {
    "immediate": ImmediateStartPolicy,
    "delayed": DelayedStartPolicy,
    "periodic": PeriodicStartPolicy,
    "diversity": DiversityTriggeredStartPolicy,
    "score_plateau": ScorePlateauStartPolicy,
    "budget_proportional": BudgetProportionalStartPolicy,
}
