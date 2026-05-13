#!/usr/bin/env bash
# Vendor gepa-artifact benchmarks into experiments/vendor/.
# Run once after cloning this repo.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="$ROOT/vendor"

mkdir -p "$VENDOR"
cd "$VENDOR"

if [ ! -d "gepa-artifact" ]; then
    echo "Cloning gepa-ai/gepa-artifact (shallow)..."
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/gepa-ai/gepa-artifact.git
else
    echo "gepa-artifact already present at $VENDOR/gepa-artifact"
fi

echo "Vendored benchmarks available at:"
echo "  $VENDOR/gepa-artifact/gepa_artifact/benchmarks/"
ls "$VENDOR/gepa-artifact/gepa_artifact/benchmarks/"

# ---------------------------------------------------------------------------
# Post-clone patch: copy our team's custom benchmark loaders into the upstream
# vendor clone. These loaders are NOT in upstream gepa-ai/gepa-artifact:
#   - musique:           MuSiQue multi-hop QA (4-predictor program)
#   - twowikimultihopqa: 2WikiMultiHopQA (REPORT.md §15 voidful HF mirror)
# Source location (committed to OUR repo): gepa_merge/_vendor_patches/
# ---------------------------------------------------------------------------
PATCH_SRC="$ROOT/../_vendor_patches"
PATCH_DST="$VENDOR/gepa-artifact/gepa_artifact/benchmarks"
if [ -d "$PATCH_SRC" ]; then
    for patch in musique twowikimultihopqa; do
        if [ -d "$PATCH_SRC/$patch" ] && [ ! -d "$PATCH_DST/$patch" ]; then
            cp -r "$PATCH_SRC/$patch" "$PATCH_DST/"
            echo "Copied $patch from _vendor_patches/ → vendor."
        elif [ -d "$PATCH_DST/$patch" ]; then
            echo "$patch already present in vendor (skip copy)."
        fi
    done
else
    echo "WARNING: $PATCH_SRC not found — musique + 2wiki benchmark cells will fail to load."
fi
