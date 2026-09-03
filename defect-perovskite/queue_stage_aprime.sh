#!/usr/bin/env bash
# Stage A' (spec section 1): the base refit on NEUTRAL data only.
#
# Same recipe as run_e0_base.sh (140 epochs, float32, cuEq, energy 10 / total-energy 10,
# every carrier regulariser zeroed, no long-range branch), with two additions:
#   * the neutral 159-atom frames upweighted to a 0.25 realised share in the ENERGY loss and
#     in the FORCE loss, each solved on its own mass, logged every epoch;
#   * every non-parameter forward constant serialised (section 5.1, already in the model).
# No charged frame anywhere: dataset_e0 holds n = 0 frames only, and the four dataset_cf
# folds hold n = 0 frames only (each fold's null_oof.xyz is the out-of-fold neutral set).
#
#   ROLE=folds  : the four fold bases on GPUs 4-7 in one wave (b3)
#   ROLE=prod   : the production base on one GPU (local A4000 or b3)
# Waits on nothing: the caller decides when the GPUs are free.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
R=$HOME/runs
export PYTHONPATH=$REPO PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROLE="${ROLE:-folds}"
EPOCHS="${EPOCHS:-140}"
GPU0="${GPU0:-4}"
mkdir -p "$R"
cd "$REPO" || exit 1

run () {   # name data gpu seed
    local name="$1" data="$2" gpu="$3" seed="$4"
    rm -rf "$R/$name" "$R/$name.log"
    NAME="$name" WORK_DIR="$R/$name" DATA_DIR="$data" MACE_REPO="$REPO" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS=128 MAX_L=1 NUM_RADIAL_BASIS=8 R_MAX=5.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
    USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    USE_LONG_RANGE=False \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 DEFECT_GAUGE_WEIGHT=0.0 \
    DEFECT_U_L2=0.0 DEFECT_ZN_L2=0.0 DEFECT_P_L2=0.0 DEFECT_QHOST_L2=0.0 \
    DEFECT_NEUTRAL_SIZE_UPWEIGHT=0.25 DEFECT_NEUTRAL_SIZE_UPWEIGHT_ENERGY=True \
    "$REPO/defect-example/train_defect_model.sh" > "$R/$name.log" 2>&1
    echo "  $name exit $?  $(date +%F' '%H:%M:%S)"
}

echo "=== Stage A' ($ROLE) start $(date +%F' '%H:%M:%S) ==="
if [ "$ROLE" = "folds" ]; then
    pids=()
    for k in 0 1 2 3; do
        run "aprime_f$k" "$HERE/dataset_cf/fold$k" "$((GPU0 + k))" "$((100 + k))" &
        pids+=($!)
        sleep 20
    done
    wait "${pids[@]}"
elif [ "$ROLE" = "prod" ]; then
    run "aprime_prod" "$HERE/dataset_e0" "$GPU0" 1
else
    echo "unknown ROLE $ROLE"; exit 1
fi
echo "=== Stage A' ($ROLE) complete $(date +%F' '%H:%M:%S) ==="
