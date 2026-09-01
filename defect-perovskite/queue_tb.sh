#!/usr/bin/env bash
# T-B: band-edge inequality on the clamp harness. V1 and V2, m in {0.2, 0.5}, 3 seeds.
#
# The wait is keyed on the step-2 controls' own COMPLETION LINE, not on a process pattern.
# Arm 1b clearing is not the gate: the step-2 controls take the same four GPUs immediately
# afterwards, and they are r1_matrix processes rather than run_train ones, so a
# `pgrep -f run_train` wait would fire T-B straight into occupied cards. Keying on the log
# line also sidesteps the self-matching pgrep that has bitten here before.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model

echo "=== T-B armed $(date +%F' '%H:%M:%S); waiting for the step-2 controls to finish ==="

# Guard: if the controls never even start, do not wait forever in silence.
deadline=$(( $(date +%s) + 6*3600 ))
while ! grep -q "step 2 controls complete" "$R/step2_b3.log" 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "=== T-B ABORT: step-2 controls did not complete within 6 h; not starting ==="
        exit 1
    fi
    sleep 120
done
echo "step-2 controls clear at $(date +%H:%M:%S)"
sleep 60
cd "$W" || exit 1

for f in "$ARCH" "$BASE"; do
    [ -f "$f" ] || { echo "=== T-B ABORT: missing $f ==="; exit 1; }
done

run () {   # gpu variant margin
    local gpu=$1 variant=$2 m=$3
    local tag="${variant}_m${m}"
    CUDA_VISIBLE_DEVICES=$gpu python defect-perovskite/tb_edge.py \
        --arch "$ARCH" --base "$BASE" --frames 48 --epochs 40 --seeds 3 \
        --variants "$variant" --margins "$m" --n-pristine 64 --pristine-batch 8 \
        --device cuda --out "$R/tb_${tag}.json" > "$R/tb_${tag}.log" 2>&1
    echo "  done $tag (exit $?)"
}

echo "=== T-B starting $(date +%F' '%H:%M:%S) ==="
run 0 v1 0.2 &
run 1 v1 0.5 &
run 2 v2 0.2 &
run 3 v2 0.5 &
wait
echo "=== T-B complete $(date +%F' '%H:%M:%S) ==="
