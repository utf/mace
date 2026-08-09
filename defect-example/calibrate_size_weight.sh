#!/usr/bin/env bash
# Calibrate lambda_size, and separate its effect on stability from the baseline's.
#
# The plan asks for lambda_size set so L_size is ~5-10% of the delta-energy term when
# active, and for the *realised* ratio to be logged. The loss is dimensionless after the
# log-ratio reformulation, so the previous weight carries no meaning at all and the value
# has to be found by measurement.
#
# Runs a CONTROL with the term off alongside the probe. Without it, any instability seen
# with the term on is unattributable -- an 8-channel model on the beta set is a small,
# noisy configuration and may well be unstable on its own.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=""

run () {
    local name="$1" weight="$2"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset_beta" \
    MAX_NUM_EPOCHS="${EPOCHS:-20}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
    BATCH_SIZE=4 VALID_BATCH_SIZE=4 DEVICE=cpu DEFAULT_DTYPE=float64 \
    USE_EMA=False PATIENCE=250 SEED=1 ENABLE_CUEQ=False \
    OMP_NUM_THREADS="${THREADS:-12}" \
    DEFECT_SIZE_WEIGHT="$weight" DEFECT_SIZE_WARMUP_EPOCHS="${WARMUP:-5}" \
    DEFECT_GAUGE_WEIGHT=0.0 \
    "${HERE}/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  ${name} (weight ${weight}) done"
}

run "cal_off" 0.0 &
run "cal_w1" "${WEIGHT:-1.0}" &
wait
echo "calibration runs complete"
