"""Candidate pair selection strategies for GEPA merge.

Each ranking function returns a list of (i, j) pairs sorted by priority
(best first). The merge logic iterates over them until one passes the
existing validation checkpoints (ancestry, common ancestor, divergence, etc.).
"""

import itertools
import random
from collections.abc import Sequence


def prompt_divergence(prompt_a: dict[str, str], prompt_b: dict[str, str]) -> float:
    """Word-set symmetric difference / union across all prompt components."""
    total, diff = 0, 0
    for key in set(prompt_a) & set(prompt_b):
        a_words = set(prompt_a[key].split())
        b_words = set(prompt_b[key].split())
        diff += len(a_words.symmetric_difference(b_words))
        total += len(a_words | b_words) or 1
    return diff / total if total > 0 else 0.0


def rank_pairs_random(
    merge_candidates: Sequence[int],
    rng: random.Random,
    **kwargs,
) -> list[tuple[int, int]]:
    """Shuffle all pairs randomly. This is the original GEPA behavior."""
    pairs = list(itertools.combinations(merge_candidates, 2))
    rng.shuffle(pairs)
    return pairs


def rank_pairs_by_divergence(
    merge_candidates: Sequence[int],
    program_candidates: Sequence[dict[str, str]],
    **kwargs,
) -> list[tuple[int, int]]:
    """Rank pairs by prompt divergence, highest first."""
    scored = []
    for i, j in itertools.combinations(merge_candidates, 2):
        div = prompt_divergence(program_candidates[i], program_candidates[j])
        scored.append((i, j, div))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(i, j) for i, j, _ in scored]


def rank_pairs_by_score(
    merge_candidates: Sequence[int],
    scores: Sequence[float],
    **kwargs,
) -> list[tuple[int, int]]:
    """Rank pairs by sum of val scores, highest first."""
    scored = []
    for i, j in itertools.combinations(merge_candidates, 2):
        scored.append((i, j, scores[i] + scores[j]))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(i, j) for i, j, _ in scored]


def rank_pairs_by_complementarity(
    merge_candidates: Sequence[int],
    val_subscores: Sequence[dict],
    **kwargs,
) -> list[tuple[int, int]]:
    """
    Rank pairs by how complementary they are on the validation set.
    Complementarity = number of val examples where exactly one candidate
    scores above the per-example median.
    """
    all_val_ids: set = set()
    for idx in merge_candidates:
        all_val_ids |= set(val_subscores[idx].keys())

    # Per-example median across merge candidates
    medians: dict = {}
    for vid in all_val_ids:
        vals = sorted(
            val_subscores[idx].get(vid, 0.0) for idx in merge_candidates
        )
        medians[vid] = vals[len(vals) // 2]

    # For each candidate, the set of examples it is "strong" on
    strong_sets: dict[int, set] = {}
    for idx in merge_candidates:
        strong_sets[idx] = {
            vid for vid, s in val_subscores[idx].items()
            if s > medians.get(vid, 0.0)
        }

    scored = []
    for i, j in itertools.combinations(merge_candidates, 2):
        complementarity = len(strong_sets[i].symmetric_difference(strong_sets[j]))
        scored.append((i, j, complementarity))
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(i, j) for i, j, _ in scored]


from gepa.proposer.adaptive_pair_selection import rank_pairs_by_adaptive_diversity

SELECTION_STRATEGIES = {
    "random": rank_pairs_random,
    "divergence": rank_pairs_by_divergence,
    "score": rank_pairs_by_score,
    "complementary": rank_pairs_by_complementarity,
    "adaptive_diversity": rank_pairs_by_adaptive_diversity,
}
