#!/bin/bash
# The Stage B training recipe as ONE function, so every queue (single-GPU waves,
# packed waves) launches identical runs. Sourced, not executed.
#   stage_b_run <gpu> <seed> <name> [madelung_range] [threads]
# Environment knobs (defaults as in queue_stage_b.sh): DATA, BASE, WJSON, EPOCHS,
# DEFECT_NULL_REFERENCE, plus any variable train_defect_model.sh reads.
stage_b_run () {
    local gpu="$1" seed="$2" name="$3" mode="${4:-full}" threads="${5:-8}"
    local here repo r
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo="$(cd "${here}/.." && pwd)"
    r=$HOME/runs
    local data="${DATA:-$here/dataset_pbe}" base="${BASE:-$r/aprime_prod/aprime_prod.model}"
    local wjson="${WJSON:-$r/c1_ood_aprime.json}" epochs="${EPOCHS:-24}"
    [ -f "$base" ] || { echo "ABORT: missing base $base"; return 1; }
    [ -f "$wjson" ] || { echo "ABORT: missing energy weights $wjson"; return 1; }
    rm -rf "$r/$name" "$r/$name.log"
    OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" \
    NAME="$name" WORK_DIR="$r/$name" DATA_DIR="$data" MACE_REPO="$repo" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MAX_NUM_EPOCHS="$epochs" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
    EVAL_INTERVAL=4 USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=False \
    LR=0.005 BASE_LR_FACTOR=0.0 DEFECT_BASE_INIT="$base" \
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
    DEFECT_ENERGY_WEIGHTS_JSON="$wjson" DEFECT_CHARGED_ENERGY_SHARE=0.25 \
    USE_LONG_RANGE=True LR_START_EPOCH=0 EPS_INF=4.0 \
    DEFECT_LR_DETACH_DENSITY=True DEFECT_LR_FREEZE=True \
    DEFECT_PRECISION_POLICY=mixed DEFECT_BASE_CACHE=True \
    DEFECT_MADELUNG_RANGE="$mode" \
    "$repo/defect-example/train_defect_model.sh" > "$r/$name.log" 2>&1
    echo "  $name (seed $seed, $mode, gpu $gpu) exit $?  $(date +%F' '%H:%M:%S)"
}
