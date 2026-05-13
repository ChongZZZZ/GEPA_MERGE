"""Merge algorithms — how to construct a new candidate from two descendants
that share a common ancestor (the DESIRABLE triplet has already been chosen
by checkpoints 1-4 in proposer/merge.py).

Three variants, mapped to the project proposal:

  1. ``original``          — paper Algorithm 4 "System-Aware Merge":
                             per-predictor pick from {ancestor, id1, id2}
                             according to DESIRABLE / score rules.

  2. ``combine_all``       — "Combine-all-subprompts variant":
                             concatenate both candidates' instruction text
                             per disputed predictor (Case B in system-aware
                             terminology). Case A / Case C are identical
                             to system_aware.

  3. ``summarize_before``  — "Summarize-before-merge variant":
                             for disputed predictors, call an LLM to
                             synthesize a unified instruction that
                             preserves both candidates' contributions.

All three keep the upstream 4-checkpoint gate (ancestry, common-ancestor,
score-violation, DESIRABLE) and only change how ``new_program`` is built.

Contract:
    merge_fn(
        ancestor_idx, id1, id2,
        program_candidates, agg_scores, rng,
        merge_lm=None,         # required only by summarize_before
    ) -> (new_program: dict[str, str], new_prog_desc: tuple)

``new_prog_desc`` is a per-predictor trace used by merge.py's dedup logic
to ensure the same triplet/result combination isn't tried twice. For
algorithms that produce deterministic outputs this is the tuple of picked
``candidate_idx`` values; for ``combine_all`` / ``summarize_before`` we
inject sentinel strings (``"__concat__"`` / ``"__synth__"``) when we merge
rather than pick.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable


CONCAT_SENTINEL = "__concat__"
SYNTH_SENTINEL = "__synth__"

# Named logger so Phase A analysis can grep `.log` files for fallback counts:
#   grep "SYNTH_FALLBACK" runs/phase_a/*/summarize_before_*/*.log | wc -l
# A non-trivial count in any summarize_before cell means that cell is polluted
# by concat-shaped merges and its comparison vs combine_all is weakened.
_log = logging.getLogger("gepa.merge.summarize_before")


def _pick_unchanged_side(pred_anc, pred_id1, pred_id2, id1, id2):
    """Case A helper: one descendant unchanged, the other diverged.
    Returns (chosen_value, chosen_idx) where chosen_idx is the diverged one.
    """
    if pred_anc == pred_id1:
        return pred_id2, id2
    return pred_id1, id1


def _pick_higher_score(id1, id2, agg_scores, program_candidates, pred_name, rng):
    """Case B helper: both descendants diverged; take higher agg score, random on tie."""
    if agg_scores[id1] > agg_scores[id2]:
        winner = id1
    elif agg_scores[id2] > agg_scores[id1]:
        winner = id2
    else:
        winner = rng.choice([id1, id2])
    return program_candidates[winner][pred_name], winner


# ---------------------------------------------------------------------------
# Variant 1 — Original GEPA Merge (Algorithm 4)
# ---------------------------------------------------------------------------

def merge_system_aware(
    ancestor_idx: int,
    id1: int,
    id2: int,
    program_candidates: list[dict[str, str]],
    agg_scores,
    rng,
    merge_lm: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], tuple]:
    """Paper Algorithm 4. Per-predictor pick:
      - Case A (exactly one descendant changed): take the changed one.
      - Case B (both changed differently): take higher agg_score.
      - Case C (equal): take id1 (arbitrary).
    """
    new_program = deepcopy(program_candidates[ancestor_idx])
    new_prog_desc: tuple = ()

    pred_names = set(program_candidates[ancestor_idx].keys())
    for pred_name in pred_names:
        pred_anc = program_candidates[ancestor_idx][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]

        # Order matters: check C before A/B so that the
        # pred_id1 == pred_id2 != pred_anc case (descendants converged on the
        # same new prompt) is not misrouted to Case B.
        if pred_id1 == pred_id2:
            # Case C — descendants agree (covers pred_anc == pred_id1 == pred_id2 too)
            new_program[pred_name] = program_candidates[id1][pred_name]
            new_prog_desc = (*new_prog_desc, id1)
        elif pred_anc == pred_id1 or pred_anc == pred_id2:
            # Case A — exactly one descendant diverged
            value, chosen_idx = _pick_unchanged_side(pred_anc, pred_id1, pred_id2, id1, id2)
            new_program[pred_name] = value
            new_prog_desc = (*new_prog_desc, chosen_idx)
        else:
            # Case B — both diverged in different directions
            value, chosen_idx = _pick_higher_score(id1, id2, agg_scores, program_candidates, pred_name, rng)
            new_program[pred_name] = value
            new_prog_desc = (*new_prog_desc, chosen_idx)

    return new_program, new_prog_desc


# ---------------------------------------------------------------------------
# Variant 2 — Combine-All-Subprompts
# ---------------------------------------------------------------------------

_CONCAT_TEMPLATE = (
    "{a}\n\n"
    "---\n\n"
    "Additional guidance:\n"
    "{b}"
)


def merge_combine_all_subprompts(
    ancestor_idx: int,
    id1: int,
    id2: int,
    program_candidates: list[dict[str, str]],
    agg_scores,
    rng,
    merge_lm: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], tuple]:
    """Concatenate both candidates' instruction text per disputed predictor.

    - Case A (one unchanged): take the changed version (same as system_aware).
    - Case B (both changed differently): CONCATENATE the two texts.
    - Case C (equal): identical to system_aware.

    Rationale: never lose a mutation's information; let the downstream LM
    handle redundancy. Trade-off is prompt length grows with merge count.
    """
    new_program = deepcopy(program_candidates[ancestor_idx])
    new_prog_desc: tuple = ()

    pred_names = set(program_candidates[ancestor_idx].keys())
    for pred_name in pred_names:
        pred_anc = program_candidates[ancestor_idx][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]

        # Order matters: Case C first so that convergent descendants are not
        # routed to Case B and needlessly concatenated with themselves.
        if pred_id1 == pred_id2:
            # Case C — descendants agree
            new_program[pred_name] = program_candidates[id1][pred_name]
            new_prog_desc = (*new_prog_desc, id1)
        elif pred_anc == pred_id1 or pred_anc == pred_id2:
            # Case A — exactly one descendant diverged
            value, chosen_idx = _pick_unchanged_side(pred_anc, pred_id1, pred_id2, id1, id2)
            new_program[pred_name] = value
            new_prog_desc = (*new_prog_desc, chosen_idx)
        else:
            # Case B — concat both versions in score order (higher first)
            if agg_scores[id2] > agg_scores[id1]:
                first, second = pred_id2, pred_id1
            else:
                first, second = pred_id1, pred_id2
            new_program[pred_name] = _CONCAT_TEMPLATE.format(a=first, b=second)
            new_prog_desc = (*new_prog_desc, CONCAT_SENTINEL)

    return new_program, new_prog_desc


# ---------------------------------------------------------------------------
# Variant 3 — Summarize-Before-Merge
# ---------------------------------------------------------------------------

_SYNTH_PROMPT_TEMPLATE = """\
You are helping merge two versions of a task instruction. Two candidate
instructions for the same module are shown below. Write a single unified
instruction that preserves the substantive guidance from both. The result
will be used directly by a downstream model, so:

- Do NOT refer to "version A" / "version B" or mention the merge process.
- Do NOT add meta-commentary, prefaces, or lists of what you changed.
- Output ONLY the unified instruction, nothing else.
- Preserve task-specific constraints, formats, and examples from both versions.

VERSION A:
{a}

VERSION B:
{b}

UNIFIED INSTRUCTION:"""


def merge_summarize_before(
    ancestor_idx: int,
    id1: int,
    id2: int,
    program_candidates: list[dict[str, str]],
    agg_scores,
    rng,
    merge_lm: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], tuple]:
    """For disputed predictors (Case B), call an LLM to synthesize a unified
    instruction from both candidates. Case A / Case C identical to system_aware.

    ``merge_lm`` must be a callable ``str -> str``; typically wraps a litellm
    completion with small max_tokens. If None, falls back to concat (so the
    caller never crashes if misconfigured).
    """
    new_program = deepcopy(program_candidates[ancestor_idx])
    new_prog_desc: tuple = ()

    pred_names = set(program_candidates[ancestor_idx].keys())
    for pred_name in pred_names:
        pred_anc = program_candidates[ancestor_idx][pred_name]
        pred_id1 = program_candidates[id1][pred_name]
        pred_id2 = program_candidates[id2][pred_name]

        # Order matters: Case C first so that convergent descendants skip the
        # synthesis LLM call entirely.
        if pred_id1 == pred_id2:
            # Case C — descendants agree
            new_program[pred_name] = program_candidates[id1][pred_name]
            new_prog_desc = (*new_prog_desc, id1)
        elif pred_anc == pred_id1 or pred_anc == pred_id2:
            # Case A — exactly one descendant diverged
            value, chosen_idx = _pick_unchanged_side(pred_anc, pred_id1, pred_id2, id1, id2)
            new_program[pred_name] = value
            new_prog_desc = (*new_prog_desc, chosen_idx)
        else:
            # Case B — LLM synthesis
            prompt = _SYNTH_PROMPT_TEMPLATE.format(a=pred_id1, b=pred_id2)
            if merge_lm is None:
                # Fallback: concat (so calls don't crash if LM not wired).
                # Should never fire in Phase A — run_dspy.py always wires a
                # merge_lm for summarize_before. If it does, every merge in
                # the run is silently concat-shaped.
                _log.warning(
                    "SYNTH_FALLBACK pred=%s reason=no_merge_lm "
                    "(summarize_before received merge_lm=None — this run's "
                    "entire Case-B stream degrades to concat)",
                    pred_name,
                )
                synthesized = _CONCAT_TEMPLATE.format(a=pred_id1, b=pred_id2)
            else:
                try:
                    synthesized = merge_lm(prompt).strip()
                    if not synthesized:
                        _log.warning(
                            "SYNTH_FALLBACK pred=%s reason=empty_response "
                            "(merge_lm returned empty string; using concat)",
                            pred_name,
                        )
                        synthesized = _CONCAT_TEMPLATE.format(a=pred_id1, b=pred_id2)
                except Exception as e:
                    _log.warning(
                        "SYNTH_FALLBACK pred=%s reason=exception "
                        "exc_type=%s exc_msg=%s (using concat)",
                        pred_name, type(e).__name__, str(e)[:200],
                    )
                    synthesized = _CONCAT_TEMPLATE.format(a=pred_id1, b=pred_id2)
            new_program[pred_name] = synthesized
            new_prog_desc = (*new_prog_desc, SYNTH_SENTINEL)

    return new_program, new_prog_desc


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MERGE_ALGORITHMS: dict[str, Callable[..., tuple[dict[str, str], tuple]]] = {
    "original":            merge_system_aware,
    "combine_all":         merge_combine_all_subprompts,
    "summarize_before":    merge_summarize_before,
}


def get_merge_algorithm(name: str) -> Callable[..., tuple[dict[str, str], tuple]]:
    """Lookup by name; raise ValueError with helpful message on unknown."""
    if name not in MERGE_ALGORITHMS:
        raise ValueError(
            f"Unknown merge_algorithm={name!r}. "
            f"Options: {sorted(MERGE_ALGORITHMS.keys())}"
        )
    return MERGE_ALGORITHMS[name]
