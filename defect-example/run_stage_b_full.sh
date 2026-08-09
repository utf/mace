#!/usr/bin/env bash
# Stage B on the FULL source: all 1341 structures, no subsampling.
#
# Everything until now used an 800-frame subsample -- a figure inherited from the very
# first exploratory run and never revisited. The full set has 504 pairs against 320, so
# ~40% more of the delta targets that carry the headline observable.
#
# Metrics are NOT comparable to the v4 (800-frame) runs: different split, different
# validation set, and a different probe floor (46.66 meV here, against 53.72 on v4).
#
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
#
# Energy weights are raised here, and this run is the first to carry them. Measured budget
# on `b_128ch_L0_s1` (loss_term_budget.py, valid split), at the old weights:
#
#   forces 71.6% | delta_forces 25.0% | delta_energy 3.4% | total_energy 0.04% | energy 0.00%
#
# The two terms that see an absolute energy are together 0.04% of the objective, so
# nothing in training meaningfully penalises a constant per-atom offset -- and forces
# cannot see one at all, since dE/dR annihilates it. That is what put every subset ~15
# meV/atom off the parity line on the 40-epoch `b_128ch_L1_s1` checkpoint.
#
# Raising *which* one matters. `base_energy` is already fit to 0.94 meV/atom, so
# ENERGY_WEIGHT has almost nothing left to correct; it goes to 10 as cheap insurance for
# the larger dataset, not because it is the lever. The lever is TOTAL_ENERGY_WEIGHT: it
# is the only term carrying the absolute energy at n != 0 frames, where the residual is
# 9.65 meV/atom, and 0.25 -> 10 takes it from 0.04% to 1.5% of the loss. DELTA_ENERGY_WEIGHT
# stays at 10: it is a paired difference, so it is blind to a constant at any weight.
#
# NOT comparable to the beta-dataset stage B runs, which used 1.0 / 0.25. That comparison
# was already broken by the dataset change; this compounds it deliberately rather than
# silently.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_stage_b_full.lock"
flock -n 9 || { echo "another run_stage_b_full.sh is running" >&2; exit 1; }

# One run at a time: a 128-channel model at batch 8 uses ~7.3 GB, so two would not fit
# alongside anything else, and contention would distort the escape-epoch comparison.
for cfg in "128ch_L0:128:0" "128ch_L1:128:1"; do
    label="${cfg%%:*}"; rest="${cfg#*:}"; ch="${rest%%:*}"; maxl="${rest##*:}"
    for seed in ${SEEDS:-1 2}; do
        name="bfull_${label}_s${seed}"
        rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
        NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset_full" \
        MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS=$ch MAX_L=$maxl \
        NUM_RADIAL_BASIS=8 R_MAX=4.0 \
        BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
        USE_EMA=False PATIENCE=250 SEED=$seed ENABLE_CUEQ=True \
        ENERGY_WEIGHT="${ENERGY_WEIGHT:-10.0}" \
        TOTAL_ENERGY_WEIGHT="${TOTAL_ENERGY_WEIGHT:-10.0}" \
        DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
        "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1
        echo "  ${name} done"
    done
done
echo "stage B (full dataset) complete"
