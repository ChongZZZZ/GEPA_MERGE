"""Tier A merge-quality metrics — pure functions, stdlib-only.

Computed for every constructed merge (accepted and rejected). Produces a
``tier_a`` block matching the schema in MERGE_QUALITY_PLAN.md. No LLM calls,
no optimization-budget effects.

Two headline metrics, together form the 2D diagnostic plane:

- ``sentence_provenance_entropy`` — composition axis.
- ``content_coverage_fraction`` — preservation axis.

See MERGE_QUALITY_PLAN.md §Metric catalogue for definitions.
"""

from __future__ import annotations

import difflib
import math
import re
from collections.abc import Sequence
from typing import Any

PARENT_LABELS: tuple[str, ...] = ("p1", "p2")
NOVEL_LABEL = "novel"
TIE_LABEL = "tie"


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences. nltk if available, regex fallback.

    Empty/whitespace-only segments dropped; leading/trailing whitespace stripped.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        import nltk

        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in parts if s.strip()]


def _tokens(text: str) -> set[str]:
    return set((text or "").split())


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity = |A∩B|/|A∪B|. Returns 0.0 when both empty."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def shannon_entropy_normalized(labels: Sequence[Any]) -> float:
    """Normalized Shannon entropy over label frequencies.

    0 when all labels identical, 1 when labels are uniform across ``len(set(labels))``
    distinct values. Returns 0 for empty input.
    """
    if not labels:
        return 0.0
    counts: dict[Any, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    n_distinct = len(counts)
    if n_distinct <= 1:
        return 0.0
    total = len(labels)
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return h / math.log(n_distinct)


def provenance_map_predictor(
    merged: dict[str, str], p1: dict[str, str], p2: dict[str, str]
) -> dict[str, str]:
    """For each predictor, label which parent the merged value matches.

    Label ∈ {"p1", "p2", "tie" (matches both), "novel" (matches neither)}.
    """
    out: dict[str, str] = {}
    for key in merged:
        m = merged.get(key, "")
        eq_p1 = key in p1 and m == p1[key]
        eq_p2 = key in p2 and m == p2[key]
        if eq_p1 and eq_p2:
            out[key] = TIE_LABEL
        elif eq_p1:
            out[key] = "p1"
        elif eq_p2:
            out[key] = "p2"
        else:
            out[key] = NOVEL_LABEL
    return out


def predictor_provenance_entropy(prov_map: dict[str, str]) -> float:
    """Normalized entropy over per-predictor provenance labels."""
    return shannon_entropy_normalized(list(prov_map.values()))


def provenance_map_sentence(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    novel_threshold: float = 0.1,
) -> dict[str, list[str]]:
    """Per-predictor list of sentence-level parent labels.

    For each merged sentence: argmax Jaccard over the sentences of each parent,
    comparing max-Jaccard-against-p1 vs max-Jaccard-against-p2. Returns "p1" / "p2"
    / "tie" (equal, both > novel_threshold) / "novel" (both below novel_threshold).
    """
    out: dict[str, list[str]] = {}
    for key in merged:
        m_sents = split_sentences(merged.get(key, ""))
        p1_sents = [_tokens(s) for s in split_sentences(p1.get(key, ""))]
        p2_sents = [_tokens(s) for s in split_sentences(p2.get(key, ""))]
        labels: list[str] = []
        for sent in m_sents:
            t = _tokens(sent)
            max_p1 = max((jaccard(t, ps) for ps in p1_sents), default=0.0)
            max_p2 = max((jaccard(t, ps) for ps in p2_sents), default=0.0)
            if max_p1 < novel_threshold and max_p2 < novel_threshold:
                labels.append(NOVEL_LABEL)
            elif max_p1 > max_p2:
                labels.append("p1")
            elif max_p2 > max_p1:
                labels.append("p2")
            else:
                labels.append(TIE_LABEL)
        out[key] = labels
    return out


def sentence_provenance_entropy(sent_map: dict[str, list[str]]) -> float:
    """Headline composition metric. Flatten across predictors, normalized entropy."""
    flat: list[str] = []
    for labels in sent_map.values():
        flat.extend(labels)
    return shannon_entropy_normalized(flat)


def _coverage_one_direction(src: dict[str, str], tgt: dict[str, str]) -> dict[str, float]:
    """For each predictor: mean over src sentences of max-Jaccard to any tgt sentence."""
    out: dict[str, float] = {}
    for key in src:
        src_sents = [_tokens(s) for s in split_sentences(src.get(key, ""))]
        tgt_sents = [_tokens(s) for s in split_sentences(tgt.get(key, ""))]
        if not src_sents:
            out[key] = 1.0
            continue
        per_sent = [
            max((jaccard(ss, ts) for ts in tgt_sents), default=0.0) for ss in src_sents
        ]
        out[key] = sum(per_sent) / len(per_sent)
    return out


def content_coverage_fraction(
    merged: dict[str, str], p1: dict[str, str], p2: dict[str, str]
) -> dict[str, float]:
    """Headline preservation metric.

    For parent p, coverage_p = mean across predictors of (mean over p's sentences of
    max Jaccard to any merged sentence). Returns coverage_p1, coverage_p2, coverage_min.
    """
    cov_p1 = _coverage_one_direction(p1, merged)
    cov_p2 = _coverage_one_direction(p2, merged)
    cov_p1_val = sum(cov_p1.values()) / len(cov_p1) if cov_p1 else 0.0
    cov_p2_val = sum(cov_p2.values()) / len(cov_p2) if cov_p2 else 0.0
    return {
        "coverage_p1": cov_p1_val,
        "coverage_p2": cov_p2_val,
        "coverage_min": min(cov_p1_val, cov_p2_val),
    }


def jaccard_overlap(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    ancestor: dict[str, str] | None = None,
) -> dict[str, float]:
    """Word-set Jaccard: merged vs {p1, p2, ancestor}, averaged across predictors."""

    def _avg(x: dict[str, str], y: dict[str, str]) -> float:
        keys = set(x) & set(y)
        if not keys:
            return 0.0
        vals = [jaccard(_tokens(x[k]), _tokens(y[k])) for k in keys]
        return sum(vals) / len(vals)

    out = {"merged_vs_p1": _avg(merged, p1), "merged_vs_p2": _avg(merged, p2)}
    if ancestor is not None:
        out["merged_vs_anc"] = _avg(merged, ancestor)
    return out


def edit_distance_norm(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    ancestor: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Character-level normalized edit similarity (difflib.SequenceMatcher.ratio).

    Values ∈ [0, 1]; 1.0 = identical. Per predictor key.
    """
    out: dict[str, dict[str, float]] = {}
    for key in merged:
        m = merged.get(key, "")
        row: dict[str, float] = {
            "vs_p1": difflib.SequenceMatcher(None, m, p1.get(key, "")).ratio(),
            "vs_p2": difflib.SequenceMatcher(None, m, p2.get(key, "")).ratio(),
        }
        if ancestor is not None:
            row["vs_anc"] = difflib.SequenceMatcher(None, m, ancestor.get(key, "")).ratio()
        out[key] = row
    return out


def novelty_fraction(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    ancestor: dict[str, str] | None = None,
) -> float:
    """Fraction of merged words not present in union of parents (and ancestor if given)."""
    m_words: set[str] = set()
    for v in merged.values():
        m_words |= _tokens(v)
    ref_words: set[str] = set()
    for src in (p1, p2, ancestor or {}):
        for v in src.values():
            ref_words |= _tokens(v)
    if not m_words:
        return 0.0
    novel = m_words - ref_words
    return len(novel) / len(m_words)


def length_delta(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    ancestor: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    """Word-count delta per predictor: len(merged) - len(other)."""
    out: dict[str, dict[str, int]] = {}
    for key in merged:
        m = len(merged.get(key, "").split())
        row: dict[str, int] = {
            "vs_p1": m - len(p1.get(key, "").split()),
            "vs_p2": m - len(p2.get(key, "").split()),
        }
        if ancestor is not None:
            row["vs_anc"] = m - len(ancestor.get(key, "").split())
        out[key] = row
    return out


def per_instance_score_delta(
    new_scores: Sequence[float], p1_scores: Sequence[float], p2_scores: Sequence[float]
) -> list[float]:
    """new[k] - max(p1[k], p2[k]) per subsample id."""
    n = min(len(new_scores), len(p1_scores), len(p2_scores))
    return [new_scores[k] - max(p1_scores[k], p2_scores[k]) for k in range(n)]


def behavioral_delta_rate(
    new_scores: Sequence[float],
    p1_scores: Sequence[float],
    p2_scores: Sequence[float],
    epsilon: float,
) -> float:
    """Fraction of subsample ids where |new-p1|>ε AND |new-p2|>ε.

    ``epsilon`` is task-specific:
    - discrete 0/1 tasks (HoVer, PUPA) → 0.5
    - HotpotQA F1 → 0.05
    - IFBench constraint-sat → 1 / num_constraints
    """
    n = min(len(new_scores), len(p1_scores), len(p2_scores))
    if n == 0:
        return 0.0
    hits = 0
    for k in range(n):
        if abs(new_scores[k] - p1_scores[k]) > epsilon and abs(new_scores[k] - p2_scores[k]) > epsilon:
            hits += 1
    return hits / n


def subsample_rank_profile(
    new_scores: Sequence[float], p1_scores: Sequence[float], p2_scores: Sequence[float]
) -> dict[str, int]:
    """Win/tie/loss count vs max(p1, p2) per subsample id."""
    win = tie = loss = 0
    n = min(len(new_scores), len(p1_scores), len(p2_scores))
    for k in range(n):
        best_parent = max(p1_scores[k], p2_scores[k])
        if new_scores[k] > best_parent:
            win += 1
        elif new_scores[k] < best_parent:
            loss += 1
        else:
            tie += 1
    return {"win": win, "tie": tie, "loss": loss}


def compute_tier_a(
    merged: dict[str, str],
    p1: dict[str, str],
    p2: dict[str, str],
    ancestor: dict[str, str] | None = None,
    new_scores: Sequence[float] | None = None,
    p1_scores: Sequence[float] | None = None,
    p2_scores: Sequence[float] | None = None,
    epsilon: float = 0.05,
) -> dict[str, Any]:
    """Unified entry point. Returns the full tier_a JSON block."""
    prov_pred = provenance_map_predictor(merged, p1, p2)
    prov_sent = provenance_map_sentence(merged, p1, p2)

    block: dict[str, Any] = {
        "provenance_map_predictor": prov_pred,
        "predictor_provenance_entropy": predictor_provenance_entropy(prov_pred),
        "provenance_map_sentence": prov_sent,
        "sentence_provenance_entropy": sentence_provenance_entropy(prov_sent),
        "content_coverage_fraction": content_coverage_fraction(merged, p1, p2),
        "jaccard_overlap": jaccard_overlap(merged, p1, p2, ancestor),
        "edit_distance_norm": edit_distance_norm(merged, p1, p2, ancestor),
        "novelty_fraction": novelty_fraction(merged, p1, p2, ancestor),
        "length_delta": length_delta(merged, p1, p2, ancestor),
    }

    if new_scores is not None and p1_scores is not None and p2_scores is not None:
        block["per_instance_score_delta"] = per_instance_score_delta(
            new_scores, p1_scores, p2_scores
        )
        block["behavioral_delta_rate"] = behavioral_delta_rate(
            new_scores, p1_scores, p2_scores, epsilon
        )
        block["subsample_rank_profile"] = subsample_rank_profile(
            new_scores, p1_scores, p2_scores
        )

    return block


TASK_EPSILON: dict[str, float] = {
    "hotpotqa": 0.05,
    "hover": 0.5,
    "ifbench": 0.1,
}


def resolve_epsilon(task_name: str | None, default: float = 0.05) -> float:
    if task_name is None:
        return default
    return TASK_EPSILON.get(task_name, default)
