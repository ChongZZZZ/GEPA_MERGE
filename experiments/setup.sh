#!/usr/bin/env bash
# One-shot setup for Phase A (paper-exact path).
# Run once after cloning this repo.
#
# Usage:
#   bash experiments/setup.sh
#
# After this finishes, smoke-test with:
#   OPENAI_API_KEY='sk-...' \
#     PYTHONPATH=src .venv/bin/python -m experiments.benchmarks.run_dspy \
#     --task ifbench --use_merge --merge_selection random --merge_start immediate \
#     --merge_quality --run_dir runs/smoke_ifbench \
#     --max_metric_calls 400 --val_size 20 --train_size 30 --seed 0
#
# Then launch your slice of Phase A (see experiments/README.md).
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=============================="
echo "  GEPA+Merge Paper-Exact Setup"
echo "=============================="
echo ""

# -------------------------------------------------------------------------
# Step 1: uv and virtualenv
# -------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/5] Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed: $(uv --version)"
fi

if [ ! -d .venv ]; then
    echo "      Creating .venv..."
    uv venv
fi

# -------------------------------------------------------------------------
# Step 2: Install project + paper-extras
# -------------------------------------------------------------------------
echo ""
echo "[2/5] Syncing project + paper extras (bm25s, nltk, spacy, etc.)..."
uv sync --extra paper

echo "      Installing dspy (bypasses circular-dep self-reference)..."
uv pip install --prerelease=allow dspy

echo "      Ensuring local gepa (this repo) takes precedence over PyPI gepa..."
uv pip uninstall gepa 2>/dev/null || true
uv pip install -e .

# -------------------------------------------------------------------------
# Step 3: Vendor gepa-artifact benchmark code
# -------------------------------------------------------------------------
echo ""
echo "[3/5] Cloning gepa-ai/gepa-artifact into experiments/vendor/..."
bash experiments/setup_benchmarks.sh

# -------------------------------------------------------------------------
# Step 4: Cleanup macOS file duplicates (iCloud sync artifacts)
# -------------------------------------------------------------------------
echo ""
echo "[4/5] Scrubbing macOS duplicate files from .venv (\"* 2.py\" etc.)..."
DUPS=$(find .venv/lib/python*/site-packages -name "* 2*" -type f 2>/dev/null | wc -l | tr -d ' ')
if [ "$DUPS" != "0" ]; then
    find .venv/lib/python*/site-packages -name "* 2*" -type f -delete 2>/dev/null || true
    find .venv/lib/python*/site-packages -name "* 2" -type d -exec rm -rf {} + 2>/dev/null || true
    echo "      Removed $DUPS duplicate files."
else
    echo "      Clean — no duplicates."
fi

# -------------------------------------------------------------------------
# Step 5: Verify imports
# -------------------------------------------------------------------------
echo ""
echo "[5/5] Verifying imports..."
PYTHONPATH=src .venv/bin/python -c "
import sys
sys.path.insert(0, 'experiments/vendor/gepa-artifact')
import gepa
print(f'  gepa       OK   ({gepa.__file__})')
import dspy
print(f'  dspy       OK   ({dspy.__version__})')
import bm25s, Stemmer, diskcache, ujson, nltk, spacy, emoji
print('  all deps   OK')
import experiments.benchmarks.run_dspy  # installs compat shims
import dspy.dsp.utils
assert hasattr(dspy.dsp.utils, 'EM'), 'EM/F1 shim missing'
print('  EM/F1 shim OK')
from gepa_artifact.benchmarks.IFBench.ifbench_program import IFBenchCoT2StageProgram
from gepa_artifact.benchmarks.hotpotQA.hotpot_program import HotpotMultiHop
from gepa_artifact.benchmarks.hover.hover_program import HoverMultiHop
print('  3 DSPy programs import OK')
# IFBench's evaluator needs spaCy's en_core_web_sm (~12MB). Pre-cache so the
# first --task ifbench run does not pause to download it.
import spacy
try:
    spacy.load('en_core_web_sm')
    print('  spaCy en_core_web_sm pre-loaded OK')
except OSError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    print('  spaCy en_core_web_sm installed')
"

echo ""
echo "=============================="
echo "  Setup complete."
echo "=============================="
echo ""
echo "Next steps:"
echo "  1. (Required for HotpotQA + HoVer) Download wiki.abstracts.2017 corpus"
echo "     (~608MB, ~15 min). IFBench, MuSiQue, 2WikiMHQA don't need it."
echo ""
echo "       PYTHONPATH=src .venv/bin/python experiments/warmup_wiki_corpus.py"
echo ""
echo "  2. Smoke test (IFBench, ~5 min, ~\$0.50):"
echo ""
echo "       OPENAI_API_KEY=<key> MODEL=openai/gpt-4.1-mini TASK_LIST=ifbench \\"
echo "         SEED_ONLY=0 MERGE_ALGO_ONLY=original START_ONLY=immediate \\"
echo "         INCLUDE_NOMERGE=no BUDGET_IFBENCH=400 \\"
echo "         bash experiments/run_phase_a.sh"
echo ""
echo "  3. Full configuration sweep (§4 of the paper):"
echo "       OPENAI_API_KEY=<key> MODEL=openai/gpt-4.1-mini bash experiments/run_phase_a.sh"
echo ""
echo "  4. Behavioral probe (§6 of the paper):"
echo "       OPENAI_API_KEY=<key> MODEL=openai/gpt-4.1-mini bash experiments/run_adaptive_probe.sh"
echo ""
echo "  5. Held-out test evaluation:"
echo "       OPENAI_API_KEY=<key> RUNS_ROOT=runs/sweep bash experiments/run_test_eval.sh"
echo ""
