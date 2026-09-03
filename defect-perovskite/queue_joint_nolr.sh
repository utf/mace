#!/usr/bin/env bash
# The long-range control: arm A's configuration with E_LR off, and nothing else changed.
#
# WHY IT PAIRS EXACTLY. Seeds 1-4, the same seeds as joint_a1..a4, with every other setting
# identical -- 20 epochs, batch 8, float64, both large-cell upweights at 0.25, protocol on,
# on-site channel zero-initialised, gamma = 3, Gaussian 0.05, exponential envelope, linear
# modulation, Stage-A base at 0.1x the head's rate. `USE_LONG_RANGE` is the only difference,
# so a participation difference between the cohorts is attributable to E_LR and to nothing
# else. A control that changed the seeds as well would answer a different question.
#
# WHAT IT IS FOR. Carrier participation in arm A ends at 11.14, 11.19, 10.94 and 5.43 -- one
# seed roughly twice as localised as its siblings. E_LR switches on at epoch 12 of 20 in that
# cohort, so it is a candidate for the spread, and the only way to know is to run the same
# seeds without it.
#
# It waits for the joint run AND its scoring to release the GPUs, because the four-GPU cap on
# b3 is not a tunable.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$W/defect-perovskite/dataset_pbe}"
BASE=$R/e0_base_s1/e0_base_s1.model
TAG="${TAG:-nolr}"
EPOCHS="${EPOCHS:-20}"
cd "$W" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing $BASE"; exit 1; }

echo "=== waiting for the joint run to finish ==="
while ! grep -aq "joint run complete" "$R/joint_queue.log" 2>/dev/null; do sleep 120; done
echo "=== waiting for its scoring to release the GPUs ==="
while pgrep -f "[q]ueue_joint_postrun" > /dev/null; do sleep 120; done
echo "=== GPUs free $(date +%F' '%H:%M:%S); starting the E_LR-off control ==="

run () {   # gpu seed
    local gpu="$1" seed="$2" name="${TAG}_s$2"
    rm -rf "$R/$name" "$R/$name.log"
    NAME="$name" WORK_DIR="$R/$name" DATA_DIR="$DATA" MACE_REPO="$W" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
    EVAL_INTERVAL=4 USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=False \
    LR=0.005 BASE_LR_FACTOR=0.1 DEFECT_BASE_INIT="$BASE" \
    DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
    DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
    DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_COUNTING_HOP_FORM=linear \
    DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=False \
    DEFECT_PROTOCOL_ZERO_ON_SITE=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.25 \
    USE_LONG_RANGE=False \
    "$W/defect-example/train_defect_model.sh" > "$R/$name.log" 2>&1
    echo "  $name (seed $seed, E_LR OFF) exit $?"
}

echo "=== four seeds, E_LR off, $EPOCHS epochs  $(date +%F' '%H:%M:%S) ==="
gpu=4
pids=()
for seed in 1 2 3 4; do
    run "$gpu" "$seed" &
    pids+=($!)
    gpu=$((gpu + 1))
    sleep 15
done
wait "${pids[@]}"
echo "=== control complete $(date +%F' '%H:%M:%S) ==="

MODELS=()
for s in 1 2 3 4; do
    f="$R/${TAG}_s${s}/${TAG}_s${s}.model"
    [ -f "$f" ] && MODELS+=("$f")
done
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "ABORT: no control models were produced."
    exit 1
fi

echo; echo "=== participation, E_LR on against E_LR off ==="
CUDA_VISIBLE_DEVICES=4 python -u defect-perovskite/b13_participation.py \
    --on "$R"/joint_a1/joint_a1.model "$R"/joint_a2/joint_a2.model \
         "$R"/joint_a3/joint_a3.model "$R"/joint_a4/joint_a4.model \
    --off "${MODELS[@]}" \
    --baseline "$R"/s7_models/s3_on_s*.model \
    --device cuda --out "$R/${TAG}_participation.json" \
    > "$R/${TAG}_participation.log" 2>&1
grep -avE "Warning|warn|openequivariance|falling back|visitor|shape = |_Jd|np.reshape|UNDER-DET|alternative|^ *$" \
    "$R/${TAG}_participation.log" | tail -24
