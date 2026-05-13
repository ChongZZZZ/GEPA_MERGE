# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
#
# Adaptive merge policy for GEPA+Merge.
# Controls *when* crossover is triggered based on candidate lineage properties.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from gepa.core.state import GEPAState


class MergePolicy(Protocol):
    """Decides whether to schedule a merge given the current optimization state."""

    def should_schedule(
        self,
        state: "GEPAState",
        merge_candidates: Sequence[int],
    ) -> bool: ...


class AlwaysMergePolicy:
    """Original GEPA behavior: schedule merge every time a new program is found."""

    def should_schedule(self, state: "GEPAState", merge_candidates: Sequence[int]) -> bool:
        return True
