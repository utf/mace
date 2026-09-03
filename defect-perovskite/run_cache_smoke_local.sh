#!/usr/bin/env bash
# F20 on the production model, local A4000: the Stage-B configuration of the Stage A' spec
# (head only, base frozen, first-shell features, E_LR detached and frozen) run twice --
# MODE=cached  : precision policy "mixed" + the base cache (section 2.5)
# MODE=uniform : the historical all-float64, uncached forward
# Same seed, same data, same everything else. The profiler table and the per-step wall time
# land in the log; the per-epoch time is read off the epoch timestamps.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$W/defect-perovskite/dataset_pbe}"
BASE=$R/e0_base_s1/e0_base_s1.model
MODE="${MODE:-cached}"
EPOCHS="${EPOCHS:-2}"
NAME="cachesmoke_${MODE}"
cd "$W" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing $BASE"; exit 1; }

if [ "$MODE" = "cached" ]; then
    POLICY=mixed; CACHE=True
else
    POLICY=uniform; CACHE=False
fi

rm -rf "$R/$NAME" "$R/$NAME.log"
NAME="$NAME" WORK_DIR="$R/$NAME" DATA_DIR="$DATA" MACE_REPO="$W" \
CUDA_VISIBLE_DEVICES=0 \
MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
EVAL_INTERVAL=1 USE_EMA=False PATIENCE=250 SEED=1 ENABLE_CUEQ=False \
LR=0.005 BASE_LR_FACTOR=0.0 DEFECT_BASE_INIT="$BASE" \
DEFECT_SPECTRAL_HEAD=True DEFECT_COUNTING_HEAD=True DEFECT_SPECTRAL_R_CUT=10.0 \
DEFECT_SPECTRAL_FIRST_SHELL=True \
DEFECT_MADELUNG_ON_SITE=True DEFECT_MADELUNG_COMPOSITION="3,1,1" \
DEFECT_MADELUNG_Z_INIT="-1,1,2" DEFECT_MADELUNG_EPS_INF=4.0 \
DEFECT_COUNTING_ON_SITE_RANGE=3.0 DEFECT_COUNTING_HOP_RANGE=0.5 \
DEFECT_COUNTING_SMEARING=gaussian DEFECT_COUNTING_T_EL=0.05 \
DEFECT_COUNTING_ENVELOPE=exp DEFECT_COUNTING_DECAY_LENGTH=1.0 \
DEFECT_COUNTING_HOP_FORM=log DEFECT_COUNTING_HOP_BETA=0.4054651081081644 \
DEFECT_COUNTING_DECAY_LEARNED=True \
DEFECT_PROTOCOL=True DEFECT_PROTOCOL_BOND_LENGTH=2.861 \
DEFECT_PROTOCOL_WARMUP=5 DEFECT_PROTOCOL_HEAD_ONLY=True \
DEFECT_PROTOCOL_ZERO_ON_SITE=True \
DEFECT_GAP_WEIGHT=1.0 DEFECT_E_GAP=2.4 DEFECT_GAP_COMPOSITION="3,1,1" \
DEFECT_TWO_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.0 \
USE_LONG_RANGE=True DEFECT_LR_DETACH_DENSITY=True DEFECT_LR_FREEZE=True \
DEFECT_PRECISION_POLICY="$POLICY" DEFECT_BASE_CACHE="$CACHE" \
"$W/defect-example/train_defect_model.sh" > "$R/$NAME.log" 2>&1
echo "  $NAME exit $?"
