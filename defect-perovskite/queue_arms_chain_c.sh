#!/usr/bin/env bash
# Arm C, if gate 7 fires on arm A: arm A with the log modulation widened to ln 2 (spec §4).
# Two waves of four and two, then the gates. Four GPUs at a time, 4-7 only.
#
# A SEPARATE FILE, not a branch added to queue_arms_chain.sh. That script was edited while
# a copy of it was running this cycle, which shifted every byte offset above the driver's
# resume point and would have aborted the chain after wave 1 under `set -u`. bash reads a
# script by offset; the file on disk is not the program that is running. New behaviour goes
# in a new file.
#
# Run only after `=== chain complete ===`: the gate stages use all four GPUs, so an arm C
# overlapping arm B's two-seed wave would exceed the four-GPU rule for a gate stage's
# duration.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
cd "$REPO" || exit 1

NULLS="${NULLS:-$R/aprime_nulls.json}"
REF="${REF:-$R/aprime_reference.json}"
[ -f "$REF" ]   || { echo "ABORT: missing $REF"; exit 1; }
[ -f "$NULLS" ] || { echo "ABORT: missing $NULLS"; exit 1; }

for seeds in "1 2 3 4" "5 6"; do
    echo "=== arm c, seeds $seeds  $(date +%F' '%H:%M:%S) ==="
    ARM=c SEEDS="$seeds" GPU0=4 NULLS="$NULLS" bash "$HERE/queue_arms.sh"
done

echo "=== gates for armc  $(date +%F' '%H:%M:%S) ==="
TAG=armc REF="$REF" bash "$HERE/queue_stage_b_gates.sh"
models=("$R/armc_models/armc_s"*.model)
echo "  scoring ${#models[@]} models"

echo "=== F10 on the corrected form, armc ==="
CUDA_VISIBLE_DEVICES=4 python -u "$HERE/c7_centred_f10.py" \
    --models "${models[@]}" --device cuda --out "$R/armc_f10.json" 2>&1 | tail -40 || true
echo "=== 79-atom head slope (gate 9), armc ==="
CUDA_VISIBLE_DEVICES=5 python -u "$HERE/b2_size_slopes.py" \
    --models "${models[@]}" --device cuda --out "$R/armc_b2.json" 2>&1 | tail -20 || true
echo "=== §2.2 per-site charge deviation, armc ==="
CUDA_VISIBLE_DEVICES=6 python -u "$HERE/c11_site_deviation.py" \
    --models "${models[@]}" --device cuda --out "$R/armc_site_dev.json" 2>&1 | tail -20 || true

echo "=== the ten-gate table, armc ==="
python -u "$HERE/c12_gate_table.py" --tag armc --out "$R/armc_gates.md" 2>&1 | tail -30 || true
echo "=== arm c complete $(date +%F' '%H:%M:%S) ==="
