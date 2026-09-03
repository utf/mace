#!/usr/bin/env bash
# One E_LR-off seed on the local A4000, in parallel with b3.
#
# TWO PURPOSES, and the second is the one that survives whatever the first says.
#
#   1. Capacity. b3's four-GPU cap is a hard rule and wave 2 holds all four until it ends, so
#      the control otherwise waits. The local box is a different machine and its GPU is idle.
#      Whether that actually helps is a question about FP64 throughput, not about scheduling:
#      this run is float64 throughout, the Quadro RTX 6000 does FP64 at 1/32 of FP32 and the
#      A4000 at 1/64, so the local card could be roughly half the speed. Measured here rather
#      than assumed -- if an epoch takes much over the 10.0 min b3 needs, this buys nothing.
#
#   2. A CROSS-HARDWARE CONTROL, which is worth having regardless. Two identical invocations
#      on the same b3 card were bit-identical (max |difference| 0.000e+00 across two epochs).
#      Nothing has tested whether that survives different silicon. Running seed 1 here and
#      also on b3 measures it directly, on a real training run rather than a smoke.
#
# Every setting matches queue_joint_nolr.sh exactly, so this seed is comparable to the b3
# ones; only the device differs.
set -uo pipefail
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
R=$HOME/runs
export PYTHONPATH=$W PATH="$HOME/micromamba/envs/py13/bin:$PATH"
DATA="${DATA:-$W/defect-perovskite/dataset_pbe}"
BASE=$R/e0_base_s1/e0_base_s1.model
SEED="${SEED:-1}"
NAME="nolrlocal_s${SEED}"
EPOCHS="${EPOCHS:-20}"
cd "$W" || exit 1
[ -f "$BASE" ] || { echo "ABORT: missing $BASE"; exit 1; }

rm -rf "$R/$NAME" "$R/$NAME.log"
NAME="$NAME" WORK_DIR="$R/$NAME" DATA_DIR="$DATA" MACE_REPO="$W" \
CUDA_VISIBLE_DEVICES=0 \
MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float64 \
EVAL_INTERVAL=4 USE_EMA=False PATIENCE=250 SEED="$SEED" ENABLE_CUEQ=False \
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
"$W/defect-example/train_defect_model.sh" > "$R/$NAME.log" 2>&1
echo "  $NAME (seed $SEED, E_LR OFF, local A4000) exit $?"
