#!/usr/bin/env bash
# Stage B: capacity. The accuracy lever that has never been tested on a branch that learns.
#
# The correction has only ever been trained at 8 channels with max_L = 0, which gives no
# angular resolution at all around a defect whose gap states are directional dangling
# bonds. There is already independent evidence this matters: the novelty descriptor
# separates the defect shell from bulk by 15.0 sigma with angular terms in, against
# 2.97 sigma with l = 0 only -- measured before paying for equivariant message passing.
#
# cuEquivariance is ON here and that is not optional at this width: e3nn OOMs at batch 4
# on a 16 GB card at 128ch/max_L=1, while cueq runs batch 8 in 7.3 GB and ~2.4x faster.
# It is verified numerically identical to e3nn for this model.
#
# Order per the plan: width first, then angular resolution, so the two are separable.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_stage_b.lock"
flock -n 9 || { echo "another run_stage_b.sh is running" >&2; exit 1; }

# One run at a time: a 128-channel model at batch 8 uses ~7.3 GB, so two would not fit
# alongside anything else, and contention would distort the escape-epoch comparison.
for cfg in "128ch_L0:128:0" "128ch_L1:128:1"; do
    label="${cfg%%:*}"; rest="${cfg#*:}"; ch="${rest%%:*}"; maxl="${rest##*:}"
    for seed in ${SEEDS:-1 2}; do
        name="b_${label}_s${seed}"
        rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
        NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset" \
        MAX_NUM_EPOCHS="${EPOCHS:-40}" NUM_CHANNELS=$ch MAX_L=$maxl \
        NUM_RADIAL_BASIS=8 R_MAX=4.0 \
        BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
        USE_EMA=False PATIENCE=250 SEED=$seed ENABLE_CUEQ=True \
        DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
        "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1
        echo "  ${name} done"
    done
done
echo "stage B complete"
