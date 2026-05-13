"""MergeQualityCallback — persists Tier A merge diagnostics to JSONL sidecars.

Produces two files in ``run_dir``:

- ``candidates.jsonl`` — one line per unique candidate index:
  ``{"candidate_idx": int, "prompt": dict[str, str]}``
- ``merge_quality.jsonl`` — one line per merge attempt (accepted or rejected)
  containing parent/ancestor ids, subsample scores, and the Tier A metric block.
  ``full_val_lift_over_best_parent`` is filled in at on_merge_accepted time.

All Tier A metrics are pure functions over prompts + cached subsample scores.
No LLM calls, no effect on ``state.total_num_evals`` or ``max_metric_calls``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

from gepa.strategies.merge_quality import compute_tier_a, resolve_epsilon

if TYPE_CHECKING:
    from gepa.core.callbacks import (
        MergeAcceptedEvent,
        MergeAttemptedEvent,
        MergePairSkippedEvent,
        MergeRejectedEvent,
        OptimizationEndEvent,
        OptimizationStartEvent,
    )


class MergeQualityCallback:
    """Writes Tier A merge diagnostics to JSONL files alongside a GEPA run.

    Args:
        run_dir: Directory to place sidecar files. If None, disables writes.
        task_name: Used to resolve per-task epsilon for ``behavioral_delta_rate``.
        cell: Optional (selection, start) tuple labeling the experiment cell.
        seed: Seed used by the parent run (for downstream filtering).
        run_id: Identifier for this run (filename/directory basename by default).
    """

    def __init__(
        self,
        run_dir: str | None,
        task_name: str | None = None,
        cell: tuple[str, str] | None = None,
        seed: int | None = None,
        run_id: str | None = None,
    ) -> None:
        self.run_dir = run_dir
        self.task_name = task_name
        self.cell = cell
        self.seed = seed
        self.run_id = run_id
        self.epsilon = resolve_epsilon(task_name)

        self.candidates_path: str | None = None
        self.events_path: str | None = None
        self._written_candidate_idxs: set[int] = set()
        self._pending_events: dict[str, dict[str, Any]] = {}

        if run_dir is not None:
            os.makedirs(run_dir, exist_ok=True)
            self.candidates_path = os.path.join(run_dir, "candidates.jsonl")
            self.events_path = os.path.join(run_dir, "merge_quality.jsonl")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_optimization_start(self, event: "OptimizationStartEvent") -> None:
        if self.candidates_path is None or self.events_path is None:
            return
        # Truncate any stale sidecars so re-runs don't concat old records.
        open(self.candidates_path, "w").close()
        open(self.events_path, "w").close()
        # Always persist the seed candidate (idx 0).
        self._write_candidate(0, event["seed_candidate"])

    def on_optimization_end(self, event: "OptimizationEndEvent") -> None:
        # Flush any still-pending merge records (no full-val lift available).
        for rec in self._pending_events.values():
            self._append_event(rec)
        self._pending_events.clear()

    # ------------------------------------------------------------------
    # Merge events
    # ------------------------------------------------------------------

    def on_merge_attempted(self, event: "MergeAttemptedEvent") -> None:
        state = event.get("state")
        if state is None or self.events_path is None:
            return

        attempt_id = event.get("attempt_id") or f"no-id-{event['iteration']}"
        parent_ids = list(event["parent_ids"])
        ancestor_idx = event.get("ancestor_idx")
        merged = event["merged_candidate"]

        p1_idx, p2_idx = int(parent_ids[0]), int(parent_ids[1])
        self._write_candidate(p1_idx, state.program_candidates[p1_idx])
        self._write_candidate(p2_idx, state.program_candidates[p2_idx])
        if ancestor_idx is not None:
            self._write_candidate(int(ancestor_idx), state.program_candidates[int(ancestor_idx)])

        trace = state.full_program_trace[-1] if state.full_program_trace else {}
        subsample_ids = list(trace.get("subsample_ids", []) or [])
        p1_scores = list(trace.get("id1_subsample_scores", []) or [])
        p2_scores = list(trace.get("id2_subsample_scores", []) or [])
        new_scores = list(trace.get("new_program_subsample_scores", []) or [])

        ancestor_prompt = (
            state.program_candidates[int(ancestor_idx)] if ancestor_idx is not None else None
        )
        tier_a = compute_tier_a(
            merged=dict(merged),
            p1=dict(state.program_candidates[p1_idx]),
            p2=dict(state.program_candidates[p2_idx]),
            ancestor=dict(ancestor_prompt) if ancestor_prompt is not None else None,
            new_scores=new_scores or None,
            p1_scores=p1_scores or None,
            p2_scores=p2_scores or None,
            epsilon=self.epsilon,
        )

        record = {
            "schedule_tick_id": attempt_id,
            "attempt_id": attempt_id,
            "event": "attempted",
            "iteration": event["iteration"],
            "parent_ids": [p1_idx, p2_idx],
            "ancestor_idx": int(ancestor_idx) if ancestor_idx is not None else None,
            "subsample_ids": subsample_ids,
            "p1_subsample_scores": p1_scores,
            "p2_subsample_scores": p2_scores,
            "new_program_subsample_scores": new_scores,
            "tier_a": tier_a,
            "full_val_lift_over_best_parent": None,
            "new_candidate_idx": None,
            "cell": list(self.cell) if self.cell else None,
            "seed": self.seed,
            "run_id": self.run_id,
        }
        self._pending_events[attempt_id] = record

    def on_merge_accepted(self, event: "MergeAcceptedEvent") -> None:
        attempt_id = event.get("attempt_id")
        if attempt_id is None:
            return
        rec = self._pending_events.pop(attempt_id, None)
        if rec is None:
            return
        rec["event"] = "accepted"
        new_idx = event["new_candidate_idx"]
        rec["new_candidate_idx"] = int(new_idx)

        state = event.get("state")
        if state is not None:
            new_full = _get_full_val_avg(state, new_idx)
            p1_full = _get_full_val_avg(state, rec["parent_ids"][0])
            p2_full = _get_full_val_avg(state, rec["parent_ids"][1])
            if new_full is not None and p1_full is not None and p2_full is not None:
                rec["full_val_lift_over_best_parent"] = float(new_full - max(p1_full, p2_full))
            if new_idx is not None:
                self._write_candidate(int(new_idx), state.program_candidates[int(new_idx)])
        self._append_event(rec)

    def on_merge_rejected(self, event: "MergeRejectedEvent") -> None:
        attempt_id = event.get("attempt_id")
        if attempt_id is None:
            return
        rec = self._pending_events.pop(attempt_id, None)
        if rec is None:
            return
        rec["event"] = "rejected"
        rec["reason"] = event.get("reason")
        self._append_event(rec)

    def on_merge_pair_skipped(self, event: "MergePairSkippedEvent") -> None:
        """Placeholder: merge.py does not currently emit these events.

        Kept for future extension. A skip event would be written as::

            {"event": "skipped", "pair": [i, j], "reason": ..., ...}
        """
        if self.events_path is None:
            return
        rec = {
            "schedule_tick_id": event.get("attempt_id"),
            "attempt_id": event.get("attempt_id"),
            "event": "skipped",
            "iteration": event["iteration"],
            "pair": list(event["pair"]),
            "pair_position": event.get("pair_position"),
            "reason": event["reason"],
            "cell": list(self.cell) if self.cell else None,
            "seed": self.seed,
            "run_id": self.run_id,
        }
        self._append_event(rec)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_candidate(self, idx: int, prompt: dict[str, str]) -> None:
        if self.candidates_path is None or idx in self._written_candidate_idxs:
            return
        with open(self.candidates_path, "a") as f:
            f.write(json.dumps({"candidate_idx": int(idx), "prompt": dict(prompt)}) + "\n")
        self._written_candidate_idxs.add(idx)

    def _append_event(self, rec: dict[str, Any]) -> None:
        if self.events_path is None:
            return
        with open(self.events_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")


def _get_full_val_avg(state: Any, idx: Any) -> float | None:
    try:
        scores = state.program_full_scores_val_set[int(idx)]
    except (IndexError, KeyError, TypeError):
        return None
    if scores is None:
        return None
    return float(scores)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)
