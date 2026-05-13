"""One-shot warmup: download wiki.abstracts.2017 corpus + build BM25 index.

Calls the paper's own initializer so the cache lands in the same location
the HoVer / HotpotQA DSPy programs will look for it.

Run once before Phase A on HoVer/HotpotQA. Idempotent — skips if already done.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR = REPO_ROOT / "experiments" / "vendor" / "gepa-artifact"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from gepa_artifact.benchmarks.hover.hover_program import (  # noqa: E402
    init_retriever,
)

if __name__ == "__main__":
    t0 = time.time()
    print("Triggering BM25 corpus + index setup (one-time). First call downloads")
    print("wiki.abstracts.2017.tar.gz (~608MB) and builds BM25 index.")
    print("Subsequent calls are instant (diskcache + saved index).")
    print()
    init_retriever()
    print(f"\nDone in {time.time() - t0:.1f}s.")
