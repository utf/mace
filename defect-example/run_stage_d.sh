#!/usr/bin/env bash
# Stage D A/B: the tie (alpha = softmax(-beta u)) against the Stage A4 baseline.
#
# The comparison arm is a4_v4_s{1,2,3}, which is the same dataset, model shape, seeds and
# settings with DEFECT_ALPHA_MODE=logits. Only the attention mode differs.
#
# Gate (plan D5): held-out RMSE_dE/RMSE_dF no worse than the baseline within seed noise,
# AND at least one of D2(1) plateau elimination or D2(2) Delta u as a binding energy.
#
# EPOCHS must match the baseline arm exactly -- 40, the budget a4_v4_s{1,2,3} was stopped
# at. Comparing a 40-epoch arm against a 200-epoch one would be meaningless, and at this
# budget some seeds may not escape the plateau at all (v3 saw escape at epochs 22-45 on
# the full set), so read the result as "behaviour at 40 epochs" and not as converged
# accuracy. That is the sharper test of D2(1), which is a claim about plateau length,
# and the weaker one for D5's accuracy clause.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
exec 9>"${TMPDIR:-/tmp}/mace_stage_d.lock"
flock -n 9 || { echo "another run_stage_d.sh is running; refusing" >&2; exit 1; }

for seed in ${SEEDS:-1 2 3}; do
    name="d_tied_s${seed}"
    rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
    NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset" \
    MAX_NUM_EPOCHS="${EPOCHS:-40}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
    USE_EMA=False PATIENCE=250 SEED=$seed DEFECT_ALPHA_MODE=tied \
    "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1 &
done
wait
echo "stage D tied runs complete"
