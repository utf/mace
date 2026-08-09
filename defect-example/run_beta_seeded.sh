#!/usr/bin/env bash
# Beta-set runs WITH logit seeding (gamma = 1.5), paired off against the size term.
#
# The earlier calibration pair ran at gamma = 0, which is not the production regime: with
# no seed the model sits in the pre-escape plateau that stage D exists to break, so the
# attention field the size term acts on is the uniform one. Seeding changes what it sees.
#
# Two runs so the size term's effect stays attributable, matching cal_off / cal_w2:
#   beta_seed_off   gamma 1.5, size term off  -- the control
#   beta_seed_size  gamma 1.5, size term on   -- calibrated weight
#
# The anneal is compressed to 15 epochs rather than the production 30. Over a 20-epoch run
# the default would leave gamma still live at the end, so the size term would only ever be
# observed against a seed that is still being withdrawn -- a confound. At 15 the seed is
# fully gone with five clean epochs left to read.
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
    DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
    DEFECT_SEED_ANNEAL_EPOCHS="${ANNEAL:-15}" \
    DEFECT_SIZE_WEIGHT="$weight" DEFECT_SIZE_WARMUP_EPOCHS="${WARMUP:-5}" \
    DEFECT_GAUGE_WEIGHT=0.0 \
    "${HERE}/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  ${name} (size weight ${weight}) done"
}

run "beta_seed_off" 0.0 &
run "beta_seed_size" "${WEIGHT:-2.5e-5}" &
wait
echo "seeded beta runs complete"
