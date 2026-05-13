"""Tier B LLM-as-judge for merge quality (offline, post-hoc).

This module is decoupled from the optimizer: it operates on already-serialized
prompts (via ``candidates.jsonl``) and merge events (via ``merge_quality.jsonl``)
produced by :class:`gepa.logging.merge_quality_callback.MergeQualityCallback`.

Why offline? Tier B is gated on three pre-registered cells
(see ``hypotheses.md``), so running it inline would (a) change the optimization
trajectory vs. the 13 un-judged cells and (b) burn the run budget on scoring
rather than optimization. Instead the judge reads sidecar files after Phase A
completes and writes augmented records.

Public entry points:

- :func:`score_merge_event` — judge a single merge attempt. Runs the rubric
  prompt + a position-swapped pairwise-vs-best-parent prompt and returns the
  :class:`TierBResult` dict that's appended to the event record.
- :func:`judge_events` — iterate over events + candidates, scoring those that
  match the pre-registered cells.

All LLM calls go through the existing :class:`LanguageModel` protocol from
``gepa.proposer.reflective_mutation.base`` so callers can swap in LiteLLM,
OpenAI, a stub for tests, etc. Calls **never** touch ``state.increment_evals``;
this module tracks its own count on the returned object.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from gepa.proposer.reflective_mutation.base import LanguageModel

RUBRIC_AXES: tuple[str, ...] = (
    "clarity",
    "specificity",
    "internal_consistency",
    "coverage_vs_parents",
)

PAIRWISE_CHOICES: tuple[str, ...] = ("merged", "parent", "tie")


# ----------------------------------------------------------------------
# Anchor exemplars (frozen 2026-04-19 via anchor-pilot run; edit only with
# pre-registration amendment).
#
# Each axis gets one 2/5 exemplar and one 5/5 exemplar. The judge sees these
# inline in the prompt so scoring is anchored, not free-floating.
# ----------------------------------------------------------------------

ANCHOR_EXEMPLARS: dict[str, dict[int, str]] = {
    "clarity": {
        2: (
            "Figure out the answer based on what you know and respond however "
            "makes sense. Try to be helpful."
        ),
        5: (
            "Answer the question in one sentence. If the question has multiple "
            "parts, answer each part in order, separated by semicolons."
        ),
    },
    "specificity": {
        2: "Answer questions using your knowledge.",
        5: (
            "Answer multi-hop factual questions by first identifying the bridge "
            "entity, then using it to locate the final answer. Output only the "
            "final answer string (no explanation, no 'Answer:' prefix)."
        ),
    },
    "internal_consistency": {
        2: (
            "Answer concisely in exactly one sentence. Provide a full paragraph "
            "with reasoning steps before the answer."
        ),
        5: (
            "Answer concisely in one sentence. Do not include reasoning in the "
            "output."
        ),
    },
    "coverage_vs_parents": {
        2: (
            "Parents required: (a) one-sentence answers; (b) avoiding 'I don't "
            "know'. Merged: 'Reply briefly.' — loses (b) entirely."
        ),
        5: (
            "Parents required: (a) one-sentence answers; (b) avoiding 'I don't "
            "know'. Merged: 'Answer in one sentence; if unsure, give your best "
            "guess rather than declining.' — both retained and unified."
        ),
    },
}


# ----------------------------------------------------------------------
# Prompt templates
# ----------------------------------------------------------------------

_RUBRIC_PROMPT_TEMPLATE = """\
You are a strict evaluator of prompt quality. Score the merged prompt on four \
1-5 axes. 1 = poor, 3 = acceptable, 5 = excellent. Use the anchor exemplars to \
calibrate — do not be lenient.

==== ANCHORS (study before scoring) ====
{anchors}

==== PARENT 1 ====
{p1}

==== PARENT 2 ====
{p2}

==== MERGED PROMPT (to score) ====
{merged}

Instructions:
1. Score each axis as an integer 1-5.
2. For internal_consistency, also output {{"contradiction_present": true|false, \
"contradiction_span": "..." or null}}.
3. Return a single JSON object with keys: clarity, specificity, \
internal_consistency, coverage_vs_parents, contradiction_present, \
contradiction_span. No prose outside the JSON.

JSON:"""


_PAIRWISE_PROMPT_TEMPLATE = """\
Compare two candidate prompts for the same task. For each of the four axes, \
pick which candidate is better, or "tie".

==== PARENT 1 ====
{p1}

==== PARENT 2 ====
{p2}

==== CANDIDATE A ====
{cand_a}

==== CANDIDATE B ====
{cand_b}

Which candidate is better on each axis? Options per axis: "A", "B", "tie". \
Return a single JSON object with keys clarity, specificity, \
internal_consistency, coverage_vs_parents. No prose outside the JSON.

JSON:"""


def _format_anchors() -> str:
    lines: list[str] = []
    for axis in RUBRIC_AXES:
        lines.append(f"-- {axis} --")
        for score in (2, 5):
            lines.append(f"  [score={score}] {ANCHOR_EXEMPLARS[axis][score]}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON extraction from LLM output.

    Returns ``{}`` on total failure; callers treat that as a parse error and
    skip the event (logged as ``parse_error`` in the run metadata).
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJECT_RE.search(text)
    if m is None:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _clip_int(value: Any, lo: int = 1, hi: int = 5) -> int | None:
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    if v < lo or v > hi:
        return None
    return v


# ----------------------------------------------------------------------
# Rubric scoring (single prompt → 1-5 per axis + contradiction)
# ----------------------------------------------------------------------


def score_rubric(
    judge_lm: LanguageModel,
    merged: str,
    p1: str,
    p2: str,
) -> dict[str, Any]:
    """Score the merged prompt on all four 1-5 rubric axes."""
    prompt = _RUBRIC_PROMPT_TEMPLATE.format(
        anchors=_format_anchors(), p1=p1, p2=p2, merged=merged
    )
    raw = judge_lm(prompt)
    parsed = _extract_json(raw)

    scores = {axis: _clip_int(parsed.get(axis)) for axis in RUBRIC_AXES}
    contradiction_present = parsed.get("contradiction_present")
    if contradiction_present not in (True, False):
        contradiction_present = None
    contradiction_span = parsed.get("contradiction_span")
    if contradiction_span in ("", "null", None):
        contradiction_span = None

    return {
        "scores": scores,
        "contradiction": {
            "present": contradiction_present,
            "span": contradiction_span,
        },
        "raw": raw,
    }


# ----------------------------------------------------------------------
# Pairwise scoring with mandatory position swap
# ----------------------------------------------------------------------


def _pairwise_once(
    judge_lm: LanguageModel,
    cand_a: str,
    cand_b: str,
    p1: str,
    p2: str,
) -> dict[str, str | None]:
    prompt = _PAIRWISE_PROMPT_TEMPLATE.format(p1=p1, p2=p2, cand_a=cand_a, cand_b=cand_b)
    raw = judge_lm(prompt)
    parsed = _extract_json(raw)
    out: dict[str, str | None] = {}
    for axis in RUBRIC_AXES:
        v = parsed.get(axis)
        if isinstance(v, str):
            v_norm = v.strip().lower()
            if v_norm in {"a", "b", "tie"}:
                out[axis] = v_norm
                continue
        out[axis] = None
    return out


def score_pairwise(
    judge_lm: LanguageModel,
    merged: str,
    best_parent: str,
    p1: str,
    p2: str,
) -> dict[str, Any]:
    """Pairwise merged-vs-best_parent with position swap.

    Result per axis ∈ {"merged", "parent", "tie", None}. ``position_swap_agreement``
    is the fraction of axes where both orderings agree on the winner. A
    disagreement is not a failure — it's logged as the judge's uncertainty.
    """
    order_ab = _pairwise_once(judge_lm, merged, best_parent, p1, p2)
    order_ba = _pairwise_once(judge_lm, best_parent, merged, p1, p2)

    translated: dict[str, str | None] = {}
    agreements = 0
    scored_axes = 0
    for axis in RUBRIC_AXES:
        a = order_ab.get(axis)
        b = order_ba.get(axis)
        # Translate raw A/B labels to "merged" / "parent" depending on order.
        chose_a = _translate(a, first_is_merged=True)
        chose_b = _translate(b, first_is_merged=False)
        if chose_a is None and chose_b is None:
            translated[axis] = None
            continue
        if chose_a is not None and chose_b is not None:
            scored_axes += 1
            if chose_a == chose_b:
                agreements += 1
                translated[axis] = chose_a
            else:
                translated[axis] = "tie"
        else:
            translated[axis] = chose_a or chose_b

    swap_agreement = (agreements / scored_axes) if scored_axes else None
    return {
        "choices": translated,
        "position_swap_agreement": swap_agreement,
        "raw": {"order_ab": order_ab, "order_ba": order_ba},
    }


def _translate(label: str | None, first_is_merged: bool) -> str | None:
    """Translate raw A/B/tie label to merged/parent/tie for the merged-vs-parent frame."""
    if label is None:
        return None
    if label == "tie":
        return "tie"
    if label == "a":
        return "merged" if first_is_merged else "parent"
    if label == "b":
        return "parent" if first_is_merged else "merged"
    return None


# ----------------------------------------------------------------------
# Event-level scoring
# ----------------------------------------------------------------------


@dataclass
class JudgeRunStats:
    total_judge_calls: int = 0
    events_scored: int = 0
    events_skipped: int = 0
    parse_errors: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)


def score_merge_event(
    judge_lm: LanguageModel,
    event: dict[str, Any],
    prompts_by_idx: dict[int, dict[str, str]],
    stats: JudgeRunStats | None = None,
    judge_lm_name: str | None = None,
) -> dict[str, Any] | None:
    """Produce a ``tier_b`` block for a single merge event.

    Requires: event has ``new_candidate_idx`` and ``parent_ids``. Skips events
    where any referenced prompt is missing from ``prompts_by_idx``.

    Currently single-predictor only: concatenates all predictor strings into
    one blob per prompt. Multi-predictor handling is a future extension.
    """
    stats = stats if stats is not None else JudgeRunStats()

    new_idx = event.get("new_candidate_idx")
    parent_ids = event.get("parent_ids") or []
    if new_idx is None or len(parent_ids) != 2:
        stats.events_skipped += 1
        stats.skipped_reasons["missing_ids"] = stats.skipped_reasons.get("missing_ids", 0) + 1
        return None

    try:
        merged_prompt = prompts_by_idx[int(new_idx)]
        p1_prompt = prompts_by_idx[int(parent_ids[0])]
        p2_prompt = prompts_by_idx[int(parent_ids[1])]
    except KeyError:
        stats.events_skipped += 1
        stats.skipped_reasons["missing_prompt"] = stats.skipped_reasons.get("missing_prompt", 0) + 1
        return None

    merged_text = _flatten_prompt(merged_prompt)
    p1_text = _flatten_prompt(p1_prompt)
    p2_text = _flatten_prompt(p2_prompt)

    # "Best parent" for pairwise: pick by full-val score if available, else p1.
    best_parent_text = _choose_best_parent(event, p1_text, p2_text)

    rubric = score_rubric(judge_lm, merged_text, p1_text, p2_text)
    stats.total_judge_calls += 1
    if all(v is None for v in rubric["scores"].values()):
        stats.parse_errors += 1

    pairwise = score_pairwise(judge_lm, merged_text, best_parent_text, p1_text, p2_text)
    stats.total_judge_calls += 2  # two orderings

    stats.events_scored += 1

    return {
        "in_preregistered_cell": True,
        "judge_lm": judge_lm_name,
        "rubric": rubric["scores"],
        "pairwise_vs_best_parent": {
            **(pairwise["choices"] or {}),
            "position_swap_agreement": pairwise["position_swap_agreement"],
        },
        "contradiction": rubric["contradiction"],
    }


def _flatten_prompt(prompt: dict[str, str]) -> str:
    """Join all predictor strings into one text blob, predictor name prefixed.

    Phase A prompts are single-predictor (``system_prompt``), so this is just the
    string itself. Multi-predictor programs get a ``# <name>`` header per block.
    """
    if len(prompt) == 1:
        return next(iter(prompt.values()))
    return "\n\n".join(f"# {k}\n{v}" for k, v in prompt.items())


def _choose_best_parent(event: dict[str, Any], p1_text: str, p2_text: str) -> str:
    """Pick the higher-scoring parent on the subsample; fall back to p1."""
    p1_sub = event.get("p1_subsample_scores") or []
    p2_sub = event.get("p2_subsample_scores") or []
    if p1_sub and p2_sub and len(p1_sub) == len(p2_sub):
        if sum(p2_sub) > sum(p1_sub):
            return p2_text
    return p1_text


# ----------------------------------------------------------------------
# JSONL driver
# ----------------------------------------------------------------------

CellFilter = Callable[[dict[str, Any]], bool]


def judge_events(
    events: Iterable[dict[str, Any]],
    prompts_by_idx: dict[int, dict[str, str]],
    judge_lm: LanguageModel,
    cell_filter: CellFilter | None = None,
    judge_lm_name: str | None = None,
    only_accepted: bool = False,
) -> tuple[list[dict[str, Any]], JudgeRunStats]:
    """Augment events with ``tier_b`` blocks for those matching ``cell_filter``.

    Non-matching events pass through unchanged. Score-aware ordering (running
    the cheaper rubric before the pairwise) keeps partial progress usable if a
    run is interrupted.
    """
    stats = JudgeRunStats()
    out: list[dict[str, Any]] = []
    for event in events:
        if cell_filter is not None and not cell_filter(event):
            out.append(event)
            continue
        if only_accepted and event.get("event") != "accepted":
            out.append(event)
            continue
        tier_b = score_merge_event(
            judge_lm=judge_lm,
            event=event,
            prompts_by_idx=prompts_by_idx,
            stats=stats,
            judge_lm_name=judge_lm_name,
        )
        if tier_b is None:
            out.append(event)
            continue
        augmented = dict(event)
        augmented["tier_b"] = tier_b
        out.append(augmented)
    return out, stats


def cell_filter_from_set(cells: set[tuple[str, str]]) -> CellFilter:
    """Build a filter that accepts events whose ``cell`` is in ``cells``."""

    def _match(event: dict[str, Any]) -> bool:
        cell = event.get("cell")
        if not cell or len(cell) != 2:
            return False
        return (str(cell[0]), str(cell[1])) in cells

    return _match


def load_candidates(path: str) -> dict[int, dict[str, str]]:
    """Load ``candidates.jsonl`` into ``{candidate_idx: prompt_dict}``."""
    prompts: dict[int, dict[str, str]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts[int(rec["candidate_idx"])] = dict(rec["prompt"])
    return prompts


def load_events(path: str) -> list[dict[str, Any]]:
    """Load ``merge_quality.jsonl`` into a list of event dicts."""
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_events(path: str, events: Iterable[dict[str, Any]]) -> None:
    """Write events back to JSONL, one record per line."""
    with open(path, "w") as f:
        for rec in events:
            f.write(json.dumps(rec) + "\n")
