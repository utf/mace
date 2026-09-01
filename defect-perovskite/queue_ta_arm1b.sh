#!/usr/bin/env bash
# T-A on arm 1b's final models -- the plan's primary target, since the prediction is about
# what a from-scratch delocalised model does with Delta_bind.
#
# Runs LOCALLY on the A4000, which is free now that the R2-seed pass has finished. b3's four
# cards go to the step-2 controls the moment arm 1b clears, and adding a fifth consumer there
# would breach the four-GPU rule for no benefit: T-A trains nothing.
#
# The wait is on the .model files themselves rather than on a process or an epoch line -- the
# model is written at the very end of run_train, so its existence is the completion signal.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W

echo "=== T-A/arm1b armed $(date +%F' '%H:%M:%S); waiting for 8 final models on b3 ==="
deadline=$(( $(date +%s) + 4*3600 ))
while :; do
    n=$(ssh b3 'ls ~/runs/r2p_h3_arm1b_s*/*.model 2>/dev/null | wc -l' 2>/dev/null || echo 0)
    [ "$n" -ge 8 ] && break
    if [ "$(date +%s)" -gt "$deadline" ]; then
        echo "=== ABORT: only $n/8 models after 4 h ==="
        exit 1
    fi
    sleep 120
done
echo "all 8 models present at $(date +%H:%M:%S); pulling"
mkdir -p "$R/arm1b_final"
rsync -a b3:/home/alex/runs/r2p_h3_arm1b_s'*'/'*'.model "$R/arm1b_final/" || exit 1
ls -la "$R/arm1b_final/"

cd "$W/defect-perovskite" || exit 1
python ta_band_edge.py --models "$R"/arm1b_final/*.model \
    --n-pristine 64 --n-defect 48 --device cuda \
    --out "$R/ta_arm1b.json" 2>&1
echo "=== T-A/arm1b complete $(date +%F' '%H:%M:%S) ==="
