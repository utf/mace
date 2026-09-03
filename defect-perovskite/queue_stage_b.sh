#!/usr/bin/env bash
# Stage B (spec section 3): six seeds, head only, on the Stage A' production base.
#
#   base       aprime_prod (frozen: head-only mask + base_lr_factor 0), outputs cached
#   head       counting head, first-shell features, gamma 3, Gaussian 0.05, exp envelope
#              L0 = 1.0 with the four learned decay lengths (beta_L = ln 2), log modulation
#              at beta = ln 1.5, centred on-site correction, Harrison init at 2.861 A,
#              warmup 5
#   E_LR       on from epoch 0, density detached, branch frozen at eps_inf 4.0
#   loss       forces on every charged frame (large-cell share 0.25); charged 159-atom
#              energies at share 0.25 of the charged energy loss, charged 79-atom energies
#              weighted by w_E; loss_gap w = 1 at 2.4 eV, composition 3,1,1; c per
#              (charge, size) calibrated with E_LR in the residual
#   precision  trunk float32, head float64 (mixed), float64 data
#   image compensation: OFF (section 2.3 probe failed F15; not adopted)
#
#   SEEDS="1 2 3 4"  GPU0=4  -> seeds on GPUs 4..7 in one wave (b3)
#   SEEDS="5 6"      GPU0=4  -> the second wave
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$HERE/dataset_pbe}"
BASE="${BASE:-$R/aprime_prod/aprime_prod.model}"
WJSON="${WJSON:-$R/c1_ood_aprime.json}"
SEEDS="${SEEDS:-1 2 3 4}"
GPU0="${GPU0:-4}"
EPOCHS="${EPOCHS:-24}"
TAG="${TAG:-stageb}"
cd "$REPO" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing base $BASE"; exit 1; }
[ -f "$WJSON" ] || { echo "ABORT: missing energy weights $WJSON"; exit 1; }

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
    DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
    DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
    DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
    DEFECT_COUNTING_HOP_FORM=log DEFECT_COUNTING_HOP_BETA=0.4054651081081644 \
    DEFECT_COUNTING_DECAY_LEARNED=True DEFECT_COUNTING_DECAY_BETA=0.6931471805599453 \
    DEFECT_ON_SITE_CENTRED=True DEFECT_IMAGE_COMPENSATION=False \
    DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
    DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=True \
    DEFECT_PROTOCOL_ZERO_ON_SITE=False DEFECT_C_SHIFT_PER_CLASS=True \
    DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
    DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.0 \
    DEFECT_ENERGY_WEIGHTS_JSON="$WJSON" DEFECT_CHARGED_ENERGY_SHARE=0.25 \
    USE_LONG_RANGE=True LR_START_EPOCH=0 EPS_INF=4.0 \
    DEFECT_LR_DETACH_DENSITY=True DEFECT_LR_FREEZE=True \
    DEFECT_PRECISION_POLICY=mixed DEFECT_BASE_CACHE=True \
    "$REPO/defect-example/train_defect_model.sh" > "$R/$name.log" 2>&1
    echo "  $name (seed $seed) exit $?  $(date +%F' '%H:%M:%S)"
}

echo "=== Stage B seeds $SEEDS  $(date +%F' '%H:%M:%S) ==="
gpu="$GPU0"
pids=()
for seed in $SEEDS; do
    run "$gpu" "$seed" &
    pids+=($!)
    gpu=$((gpu + 1))
    sleep 20
done
wait "${pids[@]}"
echo "=== Stage B seeds $SEEDS complete $(date +%F' '%H:%M:%S) ==="
