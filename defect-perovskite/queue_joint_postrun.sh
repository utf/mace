#!/usr/bin/env bash
# Everything the plan's sections 4 and 5 ask of the joint run's models, chained behind it.
#
# Launched at the same time as the joint run and waits for it, so the scoring happens whether
# or not anyone is watching -- and so the models are scored by code that was written BEFORE
# the numbers existed, which is the only way a registered adoption rule means anything.
#
# b3 GPUs 4-7 only.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TAG="${TAG:-joint}"
cd "$W" || exit 1

echo "=== waiting for the joint run ==="
while pgrep -f "queue_joint_run.sh" > /dev/null; do sleep 120; done
echo "=== joint run finished $(date +%F' '%H:%M:%S); scoring ==="

ARMA=()
ARMB=()
for s in 1 2 3 4 5 6; do
    f="$R/${TAG}_a${s}/${TAG}_a${s}.model"
    [ -f "$f" ] && ARMA+=("$f")
done
for s in 1 2; do
    f="$R/${TAG}_b${s}/${TAG}_b${s}.model"
    [ -f "$f" ] && ARMB+=("$f")
done
echo "  arm A (Stage-A init): ${#ARMA[@]} models"
echo "  arm B (from scratch): ${#ARMB[@]} models"
if [ "${#ARMA[@]}" -eq 0 ]; then
    echo "ABORT: no arm-A models were produced; nothing to score."
    exit 1
fi

# The adoption rule itself, per arm. Criterion 1 is the leakage detector and it is the one
# that cannot be traded against the others.
echo; echo "=== adoption rule, arm A ==="
CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b10_adoption.py \
    --models "${ARMA[@]}" --device cuda --out "$R/${TAG}_adopt_a.json" \
    > "$R/${TAG}_adopt_a.log" 2>&1 &
if [ "${#ARMB[@]}" -gt 0 ]; then
    echo "=== adoption rule, arm B (staging control) ==="
    CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/b10_adoption.py \
        --models "${ARMB[@]}" --device cuda --out "$R/${TAG}_adopt_b.json" \
        > "$R/${TAG}_adopt_b.log" 2>&1 &
fi

# Criterion 5's dilution gate, and F10 restated against the zero-initialised channel: does the
# ligand-Cl correction separate from bulk Cl now that the energy loss can see the channel and
# the uniform gauge was removed at step 0?
CUDA_VISIBLE_DEVICES=6 python -u defect-perovskite/s3_dilution.py \
    --models "${ARMA[@]}" --device cuda --out "$R/${TAG}_dilution.json" \
    > "$R/${TAG}_dilution.log" 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -u defect-perovskite/b4_per_atom_corrections.py \
    --models "${ARMA[@]}" --device cuda --out "$R/${TAG}_peratom.json" \
    > "$R/${TAG}_peratom.log" 2>&1 &
wait

# F14's modulation traces: where the hub and bulk bonds ended up under the energy loss.
echo; echo "=== modulation traces (F14) ==="
CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b9_hub_ceiling.py \
    --models "${ARMA[@]}" --n-small 24 --device cuda --out "$R/${TAG}_ceiling.json" \
    > "$R/${TAG}_ceiling.log" 2>&1 &
# Depth against both edges, joint against the pre-joint cohort.
CUDA_VISIBLE_DEVICES=5 python -u defect-perovskite/b6_depth_edges.py \
    --arms "prejoint=$R/s7_models" "joint=$R/${TAG}_a1/${TAG}_a1.model" \
    --device cuda --out "$R/${TAG}_depth.json" > "$R/${TAG}_depth.log" 2>&1 &
wait

echo; echo "=== post-run complete $(date +%F' '%H:%M:%S) ==="
for f in adopt_a adopt_b dilution peratom ceiling depth; do
    [ -f "$R/${TAG}_$f.log" ] || continue
    echo; echo "--- $f ---"
    grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|^ *$" \
        "$R/${TAG}_$f.log" | tail -18
done
