#!/usr/bin/env bash
# Arms A and B of the speed cycle (spec section 4): six seeds each, head only, on the
# Stage A' production base.
#
# ARM A = sections 2.1-2.4 with the section 3 objective. Against Stage B:
#   2.1  centred on-site correction in the CORRECTED form, gamma tanh(h(x) - h(xbar))
#   2.2  per-site charges, Z_i = Z0[s] + zeta tanh(z(x) - z(xbar)), zeta = 1 e
#   2.3  no d_ref: the envelope and the Harrison init anchor at r_cov(s) + r_cov(s')
#   2.4  E_LR unchanged (density detached, branch frozen)
#   sec3 charged ENERGIES only for size classes with a neutral null (159 here); w_E gone
#   sec1 size-grouped batches, so the head solves [B, 4n, 4n] at once
# ARM B = arm A plus section 2.5, the image compensation with its bound switch.
# ARM C = arm A with the log modulation widened to ln 2. Only if gate 7 fires on arm A.
#
#   ARM=a SEEDS="1 2 3 4" GPU0=4 ./queue_arms.sh
#   ARM=b SEEDS="5 6"     GPU0=4 ./queue_arms.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$HERE/dataset_pbe}"
BASE="${BASE:-$R/aprime_prod/aprime_prod.model}"
NULLS="${NULLS:-$R/aprime_nulls.json}"
ARM="${ARM:-a}"
SEEDS="${SEEDS:-1 2 3 4}"
GPU0="${GPU0:-4}"
EPOCHS="${EPOCHS:-24}"
TAG="${TAG:-arm$ARM}"
cd "$REPO" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing base $BASE"; exit 1; }
[ -f "$NULLS" ] || { echo "ABORT: missing null file $NULLS (run c9_null_file.py)"; exit 1; }

# Arm differences, and nothing else differs between the arms.
IMAGE=False
HOP_BETA=0.4054651081081644          # ln 1.5
case "$ARM" in
    a) ;;
    b) IMAGE=True ;;
    c) HOP_BETA=0.6931471805599453 ;; # ln 2, gate-7 escape; guards unchanged
    *) echo "ABORT: unknown ARM $ARM (a, b or c)"; exit 1 ;;
esac

run () {   # gpu seed
    local gpu="$1" seed="$2" name="${TAG}_s$2"
    rm -rf "$R/$name" "$R/$name.log"
    NAME="$name" WORK_DIR="$R/$name" DATA_DIR="$DATA" MACE_REPO="$REPO" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
    EVAL_INTERVAL=4 USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=False \
    LR=0.005 BASE_LR_FACTOR=0.0 DEFECT_BASE_INIT="$BASE" \
    DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
    DEFECT_SPECTRAL_FIRST_SHELL=True \
    DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
    DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
    DEFECT_MADELUNG_SITE_ZETA=1.0 \
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_COUNTING_HOP_FORM=log DEFECT_COUNTING_HOP_BETA="$HOP_BETA" \
    DEFECT_COUNTING_DECAY_LEARNED=True DEFECT_COUNTING_DECAY_BETA=0.6931471805599453 \
    DEFECT_ON_SITE_CENTRED=True DEFECT_COUNTING_CENTRE_FORM=argument \
    DEFECT_IMAGE_COMPENSATION="$IMAGE" \
    DEFECT_PROTOCOL=True \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=True \
    DEFECT_PROTOCOL_ZERO_ON_SITE=False DEFECT_C_SHIFT_PER_CLASS=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.0 \
    DEFECT_NULL_REFERENCE="$NULLS" DEFECT_CHARGED_ENERGY_SHARE=0.25 \
    DEFECT_SIZE_GROUPED_BATCHES=True \
    USE_LONG_RANGE=True LR_START_EPOCH=0 EPS_INF=4.0 \
    DEFECT_LR_DETACH_DENSITY=True DEFECT_LR_FREEZE=True \
    DEFECT_PRECISION_POLICY=mixed DEFECT_BASE_CACHE=True \
    "$REPO/defect-example/train_defect_model.sh" > "$R/$name.log" 2>&1
    echo "  $name (seed $seed) exit $?  $(date +%F' '%H:%M:%S)"
}

echo "=== arm $ARM, seeds $SEEDS, image=$IMAGE hop_beta=$HOP_BETA  $(date +%F' '%H:%M:%S) ==="
gpu="$GPU0"
pids=()
for seed in $SEEDS; do
    run "$gpu" "$seed" &
    pids+=($!)
    gpu=$((gpu + 1))
    sleep 20
done
wait "${pids[@]}"
echo "=== arm $ARM, seeds $SEEDS complete $(date +%F' '%H:%M:%S) ==="
