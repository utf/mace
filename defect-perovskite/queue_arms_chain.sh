#!/usr/bin/env bash
# The whole arm schedule on b3, one detached process: arm A (two waves), its gates, then
# arm B (two waves) and its gates. Four GPUs at a time, 4-7 only.
#
# Arm C is NOT here. It runs only if gate 7 fires on arm A, which is a decision to be taken
# on arm A's numbers rather than queued in advance.
#
# Each stage waits on its own `queue_arms.sh`, which waits on its seeds; the gate stage
# waits on the CONDITION (the six model files) rather than on a process, per LEDGER entries
# 10 and 11.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$REPO" || exit 1

NULLS="${NULLS:-$R/aprime_nulls.json}"
REF="${REF:-$R/aprime_reference.json}"
[ -f "$REF" ] || { echo "ABORT: missing $REF"; exit 1; }
if [ ! -f "$NULLS" ]; then
    echo "=== building the null file $(date +%F' '%H:%M:%S) ==="
    python -u "$HERE/c9_null_file.py" --reference "$REF" --out "$NULLS" || exit 1
fi

stage () {   # arm, seeds
    local arm="$1" seeds="$2"
    echo "=== arm $arm, seeds $seeds  $(date +%F' '%H:%M:%S) ==="
    ARM="$arm" SEEDS="$seeds" GPU0=4 NULLS="$NULLS" bash "$HERE/queue_arms.sh"
}

gates () {   # tag
    local tag="$1"
    echo "=== gates for $tag  $(date +%F' '%H:%M:%S) ==="
    TAG="$tag" REF="$REF" bash "$HERE/queue_stage_b_gates.sh"
    # The gate script has just linked the six models into $R/${tag}_models; use that, so a
    # missing seed shows up as a shorter list rather than a brace expansion that silently
    # produces a Cartesian product of paths.
    local models=("$R/${tag}_models/${tag}_s"*.model)
    echo "  scoring ${#models[@]} models"
    echo "=== F10 on the corrected form, $tag ==="
    CUDA_VISIBLE_DEVICES=4 python -u "$HERE/c7_centred_f10.py" \
        --models "${models[@]}" \
        --device cuda --out "$R/${tag}_f10.json" 2>&1 | tail -40 || true
    if [ "$tag" = "armb" ]; then
        echo "=== gate 10: tiling drift, ideal and thermal, with the bound switch ==="
        CUDA_VISIBLE_DEVICES=6 python -u "$HERE/c3_tiling_drift.py" \
            --models "${models[@]}" --thermal 3 \
            --device cuda --out "$R/${tag}_tiling.json" 2>&1 | tail -60 || true
    fi
    echo "=== 79-atom head slope (gate 9), $tag ==="
    CUDA_VISIBLE_DEVICES=5 python -u "$HERE/b2_size_slopes.py" \
        --models "${models[@]}" \
        --device cuda --out "$R/${tag}_b2.json" 2>&1 | tail -20 || true
}

stage a "1 2 3 4"
stage a "5 6"
gates arma
stage b "1 2 3 4"
stage b "5 6"
gates armb
echo "=== chain complete $(date +%F' '%H:%M:%S) ==="
