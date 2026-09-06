#!/bin/bash
# v8.1 addendum section 8: the loss-path audit of a completed Stage-1 run, inside the run's
# own recipe (queue_stage_b.sh's environment, arm `full`), with the training loop replaced
# by `stage1_audit.py`. One GPU, one process.
#   RUN=s13ra_s1 SEED=1 GPU=4 CKPT_EPOCH=20 bash queue_stage1_audit.sh   (DEVICE=cpu when CUDA is down)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH="$HERE:$REPO" PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
RUN="${RUN:-s13ra_s1}"; SEED="${SEED:-1}"; GPU="${GPU:-4}"; CKPT_EPOCH="${CKPT_EPOCH:-20}"
DEVICE="${DEVICE:-cuda}"   # cpu when the node's CUDA is down: the audit is float64 either way
REPLAY_STEPS="${REPLAY_STEPS:-0}"
NAME="audit_${RUN}"
DATA="${DATA:-$HERE/dataset_pbe}"
BASE="${BASE:-$R/aprime_prod/aprime_prod.model}"
WJSON="${WJSON:-$R/c1_ood_aprime.json}"
export DEFECT_NULL_REFERENCE="${DEFECT_NULL_REFERENCE:-$R/aprime_nulls.json}"
CKPT="$R/$RUN/checkpoints/${RUN}_run-1_epoch-${CKPT_EPOCH}.pt"
[ -f "$CKPT" ] || { echo "ABORT: missing checkpoint $CKPT"; exit 1; }
rm -rf "$R/$NAME" "$R/$NAME.log"
mkdir -p "$R/$NAME/checkpoints"
# reuse the run's base cache (the trunk is frozen and identical); the file name carries the
# run name and the base checksum, which the audit's model reproduces for the same seed
DRAWS=4
for f in "$R/$RUN"/checkpoints/${RUN}_basecache_*.pt; do
    [ -f "$f" ] || continue
    cp "$f" "$R/$NAME/checkpoints/${NAME}_basecache_${f##*_basecache_}"
    DRAWS=3
done
cd "$REPO" || exit 1
MACE_TRAIN_MODULE=stage1_audit \
AUDIT_CHECKPOINT="$CKPT" AUDIT_OUT="$R/${NAME}.json" AUDIT_EPOCH="$CKPT_EPOCH" \
AUDIT_REPLAY_STEPS="$REPLAY_STEPS" AUDIT_DRAWS_BEFORE_EPOCH0=4 AUDIT_DRAWS_DONE="$DRAWS" \
NAME="$NAME" WORK_DIR="$R/$NAME" DATA_DIR="$DATA" MACE_REPO="$REPO" \
CUDA_VISIBLE_DEVICES="$GPU" \
MAX_NUM_EPOCHS=24 NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE="$DEVICE" DEFAULT_DTYPE=float64 \
EVAL_INTERVAL=4 USE_EMA=False PATIENCE=250 SEED="$SEED" ENABLE_CUEQ=False \
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
DEFECT_MADELUNG_RANGE="${DEFECT_MADELUNG_RANGE:-full}" \
"$REPO/defect-example/train_defect_model.sh" > "$R/$NAME.log" 2>&1
echo "audit $NAME exit $?  $(date +%F' '%H:%M:%S)"
