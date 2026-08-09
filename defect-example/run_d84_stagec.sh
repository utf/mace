#!/usr/bin/env bash
# D8.4: re-run the Stage C protocol on the ADOPTED variant (logit seeding + anneal).
#
# This is the whole case for stage D. Unseeded, escape epoch rose with cell size --
# median 43 -> 59 across 286 -> 398 atoms, a ratio of 1.37 against a cell-size ratio of
# 1.39 -- which is what justified stage D on dynamics in the first place. If the seed
# removes that scaling, the production-scale hazard is gone. If the scaling survives, the
# seed has bought a constant factor and large cells will still plateau.
#
# Matched subsets: 135 configs, 9 base labels, 63 delta targets each, identical 288-atom
# pristine pool. Only the defect cell size differs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_d84.lock"
flock -n 9 || { echo "another run_d84_stagec.sh is running" >&2; exit 1; }

for seed in ${SEEDS:-1 2 3}; do
    for size in 286 398; do
        name="d84_n${size}_s${seed}"
        rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
        NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset_n${size}" \
        MAX_NUM_EPOCHS="${EPOCHS:-150}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
        BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
        USE_EMA=False PATIENCE=200 SEED=$seed \
        DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
        DEFECT_SEED_ANNEAL_EPOCHS="${ANNEAL_EPOCHS:-30}" \
        "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1 &
    done
    wait
    echo "seed ${seed} done"
done
echo "D8.4 complete"
