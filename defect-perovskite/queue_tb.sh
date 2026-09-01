#!/usr/bin/env bash
# T-B on b3, all four cards.
#
# NOTHING HERE DUPLICATES THE LOCAL RUN. The A4000 is already running V2 seeds 1-3 at both
# margins and will finish around 17:05; repeating those on b3 would be the identical
# computation -- same arch, base, frames, epochs, margins and seeds -- and a second card is
# not a confirmation of anything. So b3 takes the work that exists nowhere else:
#
#   GPU0  V1  m=0.2  seeds 1-3      the variant not running anywhere
#   GPU1  V1  m=0.5  seeds 1-3
#   GPU2  V2  m=0.2  seeds 4-6      ADDITIONAL seeds, complementing the local 1-3
#   GPU3  V2  m=0.5  seeds 4-6
#
# V2 therefore ends with six seeds against V1's three. That asymmetry is deliberate: the
# decision rule turns on whether V2 localises without the escape, so the seeds go where the
# verdict lives -- the same reasoning that gives hub2 and lig2 five seeds in the clamp matrix
# while the controls get three. Seed spread has been the weakest link in every gate reading
# on this project.
#
# The wait is keyed on the step-2 controls' own COMPLETION LINE, not a process pattern. Arm 1b
# clearing is not the gate: the controls took the same four cards straight afterwards, and
# they are r1_matrix processes rather than run_train ones, so a `pgrep -f run_train` wait
# would fire T-B into occupied GPUs. Keying on the log line also sidesteps the self-matching
# pgrep that has bitten here before.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
ARCH=$R/r2_h3_anneal_s5/r2_h3_anneal_s5.model
BASE=$R/e0_base_s1/e0_base_s1.model

echo "=== T-B armed $(date +%F' '%H:%M:%S); waiting for the step-2 controls to finish ==="

deadline=$(( $(date +%s) + 6*3600 ))
while ! grep -q "step 2 controls complete" "$R/step2_b3.log" 2>/dev/null; do
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "=== T-B ABORT: step-2 controls did not complete within 6 h; not starting ==="
        exit 1
    fi
    sleep 60
done
echo "step-2 controls clear at $(date +%H:%M:%S)"
sleep 30
cd "$W" || exit 1

for f in "$ARCH" "$BASE"; do
    [ -f "$f" ] || { echo "=== T-B ABORT: missing $f ==="; exit 1; }
done

run () {   # gpu variant margin seed_start
    local gpu=$1 variant=$2 m=$3 s0=$4
    local tag="${variant}_m${m}_s${s0}"
    CUDA_VISIBLE_DEVICES=$gpu python defect-perovskite/tb_edge.py \
        --arch "$ARCH" --base "$BASE" --frames 48 --epochs 40 \
        --seeds 3 --seed-start "$s0" \
        --variants "$variant" --margins "$m" --n-pristine 64 --pristine-batch 8 \
        --device cuda --out "$R/tb_${tag}.json" > "$R/tb_${tag}.log" 2>&1
    echo "  done $tag (exit $?)"
}

echo "=== T-B starting $(date +%F' '%H:%M:%S) ==="
run 0 v1 0.2 1 &
run 1 v1 0.5 1 &
run 2 v2 0.2 4 &
run 3 v2 0.5 4 &
wait
echo "=== T-B complete $(date +%F' '%H:%M:%S) ==="
