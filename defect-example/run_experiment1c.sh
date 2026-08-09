#!/usr/bin/env bash
# Forward plan D8 experiment 1, LOGIT route: logit_i^c += gamma_c * s_hat_i.
#
# Preferred over the u-seed. The u-seed had to be fitted through MLP_u, whose activations
# carry s_hat only along tiny singular values -- the target is in the span (residual 0.025
# at ridge 0) but needs |W| ~ 7e4, so it is a conditioning problem, not missing
# information. Seeding the logits sidesteps it: s_hat is exact and precomputed, it acts
# directly on what controls attention, and it works at step 0.
#
# This arm ANNEALS gamma to zero (per channel, on readiness), so the converged model is
# bias-free: nothing downstream carries or differentiates the descriptor, and no future
# caching of s_hat can silently break force consistency. The e1b arm is the same
# configuration with gamma left free, which is what says whether the anneal costs
# accuracy.
#
# gamma is trainable per channel, so a state for which the localisation prior is wrong can
# drive it to zero. The descriptor is rebuilt in-graph every forward so forces
# differentiate it (FD-verified to 1.7e-9); it is an ARCHITECTURE term, present at
# inference, not a training-only initialisation.
# Comparison arm is a4_v4_s{1,2,3}: same dataset, shape, seeds, settings and 40-epoch
# budget, differing only in the seed. Objective (D8.1): if escape falls from 14-16 to a
# few epochs at baseline accuracy, D2(1) is realised without the tie -- park the tie and
# keep MLP_l.
#
# The seed is applied at epoch 5, not at step 0. Measured: the target is a function of the
# trunk features, and an untrained trunk cannot express it (~4% of variance explained at
# init, 62% at epoch 5 with an adequate fit budget, 91% at epoch 40).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_exp1c.lock"
flock -n 9 || { echo "another run_experiment1c.sh is running" >&2; exit 1; }

for seed in ${SEEDS:-1 2 3}; do
    name="e1c_anneal_s${seed}"
    rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
    NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset" \
    MAX_NUM_EPOCHS="${EPOCHS:-40}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
    USE_EMA=False PATIENCE=250 SEED=$seed \
    DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
    "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1 &
done
wait
echo "experiment 1c (annealed logit seed) complete"
