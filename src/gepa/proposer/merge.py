# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import math
import random
from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy

from gepa.core.adapter import Candidate, DataInst, RolloutOutput
from gepa.core.callbacks import (
    EvaluationEndEvent,
    EvaluationStartEvent,
    GEPACallback,
    notify_callbacks,
)
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState, ObjectiveScores, ProgramIdx
from gepa.gepa_utils import find_dominator_programs
from gepa.logging.logger import LoggerProtocol
from gepa.proposer.base import CandidateProposal, ProposeNewCandidate
from gepa.proposer.merge_selection import SELECTION_STRATEGIES
from gepa.strategies.merge_policy import AlwaysMergePolicy, MergePolicy
from gepa.strategies.merge_start_policy import ImmediateStartPolicy, MergeStartPolicy

AncestorLog = tuple[int, int, int]
MergeDescription = tuple[int, int, tuple[int, ...]]
MergeAttempt = tuple[Candidate, ProgramIdx, ProgramIdx, ProgramIdx] | None


def does_triplet_have_desirable_predictors(
    program_candidates: Sequence[Candidate],
    ancestor: ProgramIdx,
    id1: ProgramIdx,
    id2: ProgramIdx,
) -> bool:
    found_predictors: list[tuple[int, int]] = []
    pred_names = list(program_candidates[ancestor].keys())
    for pred_idx, pred_name in enumerate(pred_names):
        pred_anc = program_candidates[ancestor][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]
        if (pred_anc == pred_id1 or pred_anc == pred_id2) and pred_id1 != pred_id2:
            same_as_ancestor_id = 1 if pred_anc == pred_id1 else 2
            found_predictors.append((pred_idx, same_as_ancestor_id))

    return len(found_predictors) > 0


def filter_ancestors(
    i: ProgramIdx,
    j: ProgramIdx,
    common_ancestors: Iterable[ProgramIdx],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    agg_scores: Sequence[float],
    program_candidates: Sequence[Candidate],
) -> list[ProgramIdx]:
    filtered_ancestors: list[ProgramIdx] = []
    for ancestor in common_ancestors:
        if (i, j, ancestor) in merges_performed[0]:
            continue

        if agg_scores[ancestor] > agg_scores[i] or agg_scores[ancestor] > agg_scores[j]:
            continue

        if not does_triplet_have_desirable_predictors(program_candidates, ancestor, i, j):
            continue

        filtered_ancestors.append(ancestor)
    return filtered_ancestors


def find_common_ancestor_pair(
    rng: random.Random,
    parent_list: Sequence[Sequence[int | None]],
    program_indexes: Sequence[int],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    agg_scores: Sequence[float],
    program_candidates: Sequence[Candidate],
    max_attempts: int = 10,
    pair_iter: Iterator[tuple[int, int]] | None = None,
) -> tuple[int, int, int] | None:
    def get_ancestors(node: int, ancestors_found: set[int]) -> list[int]:
        parents = parent_list[node]
        for parent in parents:
            if parent is not None and parent not in ancestors_found:
                ancestors_found.add(parent)
                get_ancestors(parent, ancestors_found)

        return list(ancestors_found)

    for _ in range(max_attempts):
        if pair_iter is not None:
            try:
                i, j = next(pair_iter)
            except StopIteration:
                return None
        else:
            if len(program_indexes) < 2:
                return None
            i, j = rng.sample(list(program_indexes), 2)
        if i == j:
            continue

        if j < i:
            i, j = j, i

        ancestors_i = get_ancestors(i, set())
        ancestors_j = get_ancestors(j, set())

        if j in ancestors_i or i in ancestors_j:
            # If one is an ancestor of the other, we cannot merge them
            continue

        common_ancestors = set(ancestors_i) & set(ancestors_j)
        common_ancestors = filter_ancestors(i, j, common_ancestors, merges_performed, agg_scores, program_candidates)
        if common_ancestors:
            # Select a random common ancestor
            common_ancestor = rng.choices(
                list(common_ancestors),
                k=1,
                weights=[agg_scores[ancestor] for ancestor in common_ancestors],
            )[0]
            return (i, j, common_ancestor)

    return None


def sample_and_attempt_merge_programs_by_common_predictors(
    agg_scores: Sequence[float],
    rng: random.Random,
    merge_candidates: Sequence[int],
    merges_performed: tuple[list[AncestorLog], list[MergeDescription]],
    program_candidates: Sequence[Candidate],
    parent_program_for_candidate: Sequence[Sequence[int | None]],
    has_val_support_overlap: Callable[[ProgramIdx, ProgramIdx], bool] | None = None,
    max_attempts: int = 10,
    ranked_pairs: list[tuple[int, int]] | None = None,
    merge_algorithm_fn: Callable[..., tuple[dict[str, str], tuple]] | None = None,
    merge_lm: Callable[[str], str] | None = None,
) -> MergeAttempt:
    if len(merge_candidates) < 2:
        return None
    if len(parent_program_for_candidate) < 3:
        return None

    # Build a single iterator over ranked pairs so position persists across outer retries.
    pair_iter: Iterator[tuple[int, int]] | None = iter(ranked_pairs) if ranked_pairs is not None else None

    for _ in range(max_attempts):
        ids_to_merge = find_common_ancestor_pair(
            rng,
            parent_program_for_candidate,
            list(merge_candidates),
            merges_performed=merges_performed,
            agg_scores=agg_scores,
            program_candidates=program_candidates,
            max_attempts=max_attempts,
            pair_iter=pair_iter,
        )
        if ids_to_merge is None:
            if pair_iter is not None:
                # Ranked list exhausted; no point in further retries.
                return None
            continue
        id1, id2, ancestor = ids_to_merge

        if (id1, id2, ancestor) in merges_performed[0]:
            continue
        assert agg_scores[ancestor] <= agg_scores[id1], "Ancestor should not be better than its descendants"
        assert agg_scores[ancestor] <= agg_scores[id2], "Ancestor should not be better than its descendants"
        assert id1 != id2, "Cannot merge the same program"

        # Now we have a common ancestor, which is outperformed by both its descendants
        # Construct the merged candidate via the selected merge algorithm
        # (default = paper Algorithm 4 "System-Aware Merge" if not specified).
        pred_names = set(program_candidates[ancestor].keys())
        assert pred_names == set(program_candidates[id1].keys()) == set(program_candidates[id2].keys()), (
            "Predictors should be the same across all programs"
        )

        if merge_algorithm_fn is None:
            # Lazy default — avoids a circular import at module top
            from gepa.strategies.merge_algorithm import merge_system_aware as _default
            merge_algorithm_fn = _default

        new_program, new_prog_desc = merge_algorithm_fn(
            ancestor, id1, id2,
            list(program_candidates),
            agg_scores, rng,
            merge_lm=merge_lm,
        )

        if (id1, id2, new_prog_desc) in merges_performed[1]:
            # This triplet has already been merged, so we skip it
            continue

        if has_val_support_overlap and not has_val_support_overlap(id1, id2):
            # Not enough overlapping validation support for candidates
            continue

        merges_performed[1].append((id1, id2, new_prog_desc))

        return new_program, id1, id2, ancestor

    return None


class MergeProposer(ProposeNewCandidate[DataId]):
    """
    Implements merge flow that combines compatible descendants of a common ancestor.

    - Find merge candidates among Pareto front dominators
    - Attempt a merge via sample_and_attempt_merge_programs_by_common_predictors
    - Subsample eval on valset-driven selected indices
    - Return proposal if merge's subsample score >= max(parents)
    The engine handles full eval + adding to state.
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        valset: DataLoader[DataId, DataInst],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
        use_merge: bool,
        max_merge_invocations: int,
        val_overlap_floor: int = 5,
        rng: random.Random | None = None,
        callbacks: list[GEPACallback] | None = None,
        merge_policy: MergePolicy | None = None,
        start_policy: MergeStartPolicy | None = None,
        selection_strategy: str = "random",
        merge_algorithm: str = "original",
        merge_lm: Callable[[str], str] | None = None,
    ):
        if selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"Unknown selection_strategy: {selection_strategy}. "
                f"Supported: {sorted(SELECTION_STRATEGIES)}"
            )
        self.selection_strategy = selection_strategy
        # Merge-algorithm choice (paper proposal: 3 fixed variants)
        from gepa.strategies.merge_algorithm import get_merge_algorithm
        self.merge_algorithm_name = merge_algorithm
        self.merge_algorithm_fn = get_merge_algorithm(merge_algorithm)
        self.merge_lm = merge_lm
        self.logger = logger
        self.valset = valset
        self.evaluator = evaluator
        self.use_merge = use_merge
        self.max_merge_invocations = max_merge_invocations
        self.rng = rng if rng is not None else random.Random(0)
        self.callbacks = callbacks
        self.merge_policy: MergePolicy = merge_policy if merge_policy is not None else AlwaysMergePolicy()
        self.start_policy: MergeStartPolicy = start_policy if start_policy is not None else ImmediateStartPolicy()

        if val_overlap_floor <= 0:
            raise ValueError("val_overlap_floor should be a positive integer")
        self.val_overlap_floor = val_overlap_floor
        # Internal counters matching original behavior
        self.merges_due = 0
        self.total_merges_tested = 0
        self.merges_performed: tuple[list[AncestorLog], list[MergeDescription]] = ([], [])

        # Toggle controlled by engine: set True when last iter found new program
        self.last_iter_found_new_program = False

    def schedule_if_needed(self, state: GEPAState | None = None) -> None:
        if not (self.use_merge and self.total_merges_tested < self.max_merge_invocations):
            return
        if state is not None and not self.start_policy.allow_merge(state):
            self.logger.log("Merge skipped by start policy.")
            return
        if state is not None:
            merge_candidates = find_dominator_programs(
                state.get_pareto_front_mapping(), list(state.program_full_scores_val_set)
            )
            if not self.merge_policy.should_schedule(state, merge_candidates):
                self.logger.log("Merge skipped by policy: candidates not sufficiently strong or divergent.")
                return
        self.merges_due += 1

    def select_eval_subsample_for_merged_program(
        self,
        scores1: dict[DataId, float],
        scores2: dict[DataId, float],
        num_subsample_ids: int = 5,
    ) -> list[DataId]:
        common_ids = list(set(scores1.keys()) & set(scores2.keys()))

        p1 = [idx for idx in common_ids if scores1[idx] > scores2[idx]]
        p2 = [idx for idx in common_ids if scores2[idx] > scores1[idx]]
        p3 = [idx for idx in common_ids if idx not in p1 and idx not in p2]

        n_each = max(1, math.ceil(num_subsample_ids / 3))
        selected: list[DataId] = []
        for bucket in (p1, p2, p3):
            if len(selected) >= num_subsample_ids:
                break
            available = [idx for idx in bucket if idx not in selected]
            take = min(len(available), n_each, num_subsample_ids - len(selected))
            if take > 0:
                selected += self.rng.sample(available, k=take)

        remaining = num_subsample_ids - len(selected)
        if remaining > 0:
            unused = [idx for idx in common_ids if idx not in selected]
            if len(unused) >= remaining:
                selected += self.rng.sample(unused, k=remaining)
            elif common_ids:
                selected += self.rng.choices(common_ids, k=remaining)

        return selected[:num_subsample_ids]

    def propose(self, state: GEPAState[RolloutOutput, DataId]) -> CandidateProposal[DataId] | None:
        i = state.i + 1
        state.full_program_trace[-1]["invoked_merge"] = True

        # Only attempt when scheduled by engine and after a new program in last iteration
        if not (self.use_merge and self.last_iter_found_new_program and self.merges_due > 0):
            self.logger.log(f"Iteration {i}: No merge candidates scheduled")
            return None

        pareto_front_programs = state.get_pareto_front_mapping()

        tracked_scores: Sequence[float] = getattr(
            state, "per_program_tracked_scores", state.program_full_scores_val_set
        )
        merge_candidates = find_dominator_programs(pareto_front_programs, list(tracked_scores))

        def has_val_support_overlap(id1: ProgramIdx, id2: ProgramIdx) -> bool:
            common_ids = set(state.prog_candidate_val_subscores[id1].keys()) & set(
                state.prog_candidate_val_subscores[id2].keys()
            )
            return len(common_ids) >= self.val_overlap_floor

        ranked_pairs: list[tuple[int, int]] | None = None
        if self.selection_strategy != "random" and len(merge_candidates) >= 2:
            strategy_fn = SELECTION_STRATEGIES[self.selection_strategy]
            ranked_pairs = strategy_fn(
                merge_candidates=merge_candidates,
                rng=self.rng,
                program_candidates=state.program_candidates,
                scores=tracked_scores,
                val_subscores=state.prog_candidate_val_subscores,
                state=state,                # extra: adaptive strategies use this
                policy=self.merge_policy,   # extra: adaptive strategies use this
            )

        # Per-event Layer 2 algorithm selection (Caution 1).
        # If the merge_policy exposes algorithm_for_pair, wrap merge_algorithm_fn
        # so the algorithm is chosen per (ancestor, id1, id2). The wrapper has
        # a tiny defensive cache but always delegates to the policy so the
        # policy's canonical _decisions cache is the single source of truth.
        algo_fn = self.merge_algorithm_fn
        if hasattr(self.merge_policy, "algorithm_for_pair"):
            from gepa.strategies.merge_algorithm import get_merge_algorithm
            policy = self.merge_policy
            default_algo_fn = self.merge_algorithm_fn
            _local_cache: dict[tuple[int, int, int], "Callable[..., tuple[dict[str, str], tuple]]"] = {}

            def algo_fn(ancestor, id1, id2, *a, **kw):  # type: ignore[no-redef]
                key = (ancestor, id1, id2)
                fn = _local_cache.get(key)
                if fn is None:
                    chosen_name = policy.algorithm_for_pair(state, ancestor, id1, id2)
                    fn = get_merge_algorithm(chosen_name) if chosen_name else default_algo_fn
                    _local_cache[key] = fn
                return fn(ancestor, id1, id2, *a, **kw)

        merge_output = sample_and_attempt_merge_programs_by_common_predictors(
            agg_scores=list(tracked_scores),
            rng=self.rng,
            merge_candidates=merge_candidates,
            merges_performed=self.merges_performed,
            program_candidates=state.program_candidates,
            parent_program_for_candidate=state.parent_program_for_candidate,
            has_val_support_overlap=has_val_support_overlap,
            ranked_pairs=ranked_pairs,
            merge_algorithm_fn=algo_fn,
            merge_lm=self.merge_lm,
        )

        if merge_output is None:
            self.logger.log(f"Iteration {i}: No merge candidates found")
            return None

        new_program, id1, id2, ancestor = merge_output
        state.full_program_trace[-1]["merged"] = True
        state.full_program_trace[-1]["merged_entities"] = (id1, id2, ancestor)
        self.merges_performed[0].append((id1, id2, ancestor))
        self.logger.log(f"Iteration {i}: Merged programs {id1} and {id2} via ancestor {ancestor}")

        # Adaptive metadata (Caution 2: only writes if trace[-1] is the current
        # merge attempt; falls back to the logger otherwise).
        if hasattr(self.merge_policy, "get_cached_decision"):
            from gepa.strategies.adaptive_merge import safe_log_adaptive
            decision = self.merge_policy.get_cached_decision(ancestor, id1, id2)
            if decision is not None:
                safe_log_adaptive(state, self.logger, "algorithm", decision.algorithm)
                safe_log_adaptive(state, self.logger, "algorithm_reason", decision.algorithm_reason)
                safe_log_adaptive(state, self.logger, "behavioral_diversity", decision.behavioral_diversity)
                safe_log_adaptive(state, self.logger, "layer1_signals", decision.signals)

        subsample_ids = self.select_eval_subsample_for_merged_program(
            state.prog_candidate_val_subscores[id1],
            state.prog_candidate_val_subscores[id2],
        )
        if not subsample_ids:
            self.logger.log(
                f"Iteration {i}: Skipping merge of {id1} and {id2} due to insufficient overlapping val coverage"
            )
            return None

        assert set(subsample_ids).issubset(state.prog_candidate_val_subscores[id1].keys())
        assert set(subsample_ids).issubset(state.prog_candidate_val_subscores[id2].keys())
        id1_sub_scores = [state.prog_candidate_val_subscores[id1][k] for k in subsample_ids]
        id2_sub_scores = [state.prog_candidate_val_subscores[id2][k] for k in subsample_ids]
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids

        mini_devset = self.valset.fetch(subsample_ids)

        # Notify evaluation start for merged candidate
        notify_callbacks(
            self.callbacks,
            "on_evaluation_start",
            EvaluationStartEvent(
                iteration=i,
                candidate_idx=None,
                batch_size=len(mini_devset),
                capture_traces=False,
                parent_ids=[id1, id2],
                inputs=mini_devset,
                is_seed_candidate=False,
            ),
        )

        outputs_by_id, scores_by_id, objective_by_id, actual_evals_count = state.cached_evaluate_full(
            new_program, subsample_ids, self.valset.fetch, self.evaluator
        )
        new_sub_scores = [scores_by_id[eid] for eid in subsample_ids]
        outputs = [outputs_by_id[eid] for eid in subsample_ids]

        notify_callbacks(
            self.callbacks,
            "on_evaluation_end",
            EvaluationEndEvent(
                iteration=i,
                candidate_idx=None,
                scores=new_sub_scores,
                has_trajectories=False,
                parent_ids=[id1, id2],
                outputs=outputs,
                trajectories=None,
                objective_scores=[objective_by_id[eid] for eid in subsample_ids] if objective_by_id else None,
                is_seed_candidate=False,
            ),
        )

        state.full_program_trace[-1]["id1_subsample_scores"] = id1_sub_scores
        state.full_program_trace[-1]["id2_subsample_scores"] = id2_sub_scores
        state.full_program_trace[-1]["new_program_subsample_scores"] = new_sub_scores

        # Count evals via hook mechanism
        state.increment_evals(actual_evals_count)

        # Acceptance will be evaluated by engine (>= max(parents))
        return CandidateProposal(
            candidate=new_program,
            parent_program_ids=[id1, id2],
            subsample_indices=subsample_ids,
            subsample_scores_before=[sum(id1_sub_scores), sum(id2_sub_scores)],
            subsample_scores_after=new_sub_scores,
            tag="merge",
            metadata={"ancestor": ancestor},
        )
