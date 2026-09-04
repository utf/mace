#!/usr/bin/env bash
# One epoch of arm A's exact configuration on the local box, before b3 sees it.
#
# WHAT IT IS FOR. Last cycle's smoke caught the pristine-centre guard and the c-table class
# -- the two things that would have stopped an unattended chain. This cycle's first-time
# machinery on real 79/80/159 batches: the size-grouped sampler, the neutral-reference skip,
# the per-site charge channel, the covalent-radius anchor (and the init gate under it), the
# null gate, and delta_L's collection. Every one of them is silent when it goes wrong.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
NAME="${NAME:-armsmoke_a}"
NULLS="${NULLS:-$R/aprime_nulls.json}"
BASE="${BASE:-$R/aprime_prod/aprime_prod.model}"
IMAGE="${IMAGE:-False}"
EPOCHS="${EPOCHS:-1}"
cd "$REPO" || exit 1
[ -f "$NULLS" ] || { echo "ABORT: missing $NULLS"; exit 1; }
[ -f "$BASE" ]  || { echo "ABORT: missing $BASE"; exit 1; }
rm -rf "$R/$NAME" "$R/$NAME.log"
NAME="$NAME" WORK_DIR="$R/$NAME" DATA_DIR="$HERE/dataset_pbe" MACE_REPO="$REPO" \
MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
EVAL_INTERVAL=1 USE_EMA=False PATIENCE=250 SEED=1 ENABLE_CUEQ=False \
LR=0.005 BASE_LR_FACTOR=0.0 DEFECT_BASE_INIT="$BASE" \
DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
DEFECT_SPECTRAL_FIRST_SHELL=True \
DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
DEFECT_MADELUNG_SITE_ZETA=1.0 \
DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
DEFECT_COUNTING_HOP_FORM=log DEFECT_COUNTING_HOP_BETA=0.4054651081081644 \
DEFECT_COUNTING_DECAY_LEARNED=True DEFECT_COUNTING_DECAY_BETA=0.6931471805599453 \
DEFECT_ON_SITE_CENTRED=True DEFECT_COUNTING_CENTRE_FORM=argument \
DEFECT_IMAGE_COMPENSATION="$IMAGE" \
DEFECT_PROTOCOL=True DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=True \
DEFECT_PROTOCOL_ZERO_ON_SITE=False DEFECT_C_SHIFT_PER_CLASS=True \
DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.0 \
DEFECT_NULL_REFERENCE="$NULLS" DEFECT_CHARGED_ENERGY_SHARE=0.25 \
DEFECT_SIZE_GROUPED_BATCHES=True \
USE_LONG_RANGE=True LR_START_EPOCH=0 EPS_INF=4.0 \
DEFECT_LR_DETACH_DENSITY=True DEFECT_LR_FREEZE=True \
DEFECT_PRECISION_POLICY=mixed DEFECT_BASE_CACHE=True \
"$REPO/defect-example/train_defect_model.sh" > "$R/$NAME.log" 2>&1
echo "smoke exit $?  -> $R/$NAME.log"
