#!/usr/bin/env bash
# Stage E: enable the long-range branch, with the screening amplitude FROZEN.
#
# The branch is not inert at q = 0. q_i^carrier = a (alpha_i^e - alpha_i^h) is a
# compensated but pointwise non-zero charge whose self-term is the electron-hole
# interaction, and q_i^pol carries a factor sum_c n_c. Part of the observable belongs
# there and Delta E_SR is currently absorbing it -- a correctness issue, not a tuning one.
#
# `a` is frozen at 1/sqrt(eps_inf): it is not identifiable from a dipole term alone, so a
# free `a` would drift to whatever absorbs the e-h energy and then read as a fitted
# screening constant. eps_inf = 6.5 is a LITERATURE value for 4H-SiC, not the DFPT
# calculation the plan asks for, so 1/a^2 cannot be checked against it as an independent
# result.
#
# Run on the seeded variant: the unseeded model puts only 2-11% of alpha on the defect
# shell and that share declines with cell size, so exporting its alpha as a latent charge
# would give a wrong electron-hole term while Delta E still looked fine.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_stage_e.lock"
flock -n 9 || { echo "another run_stage_e.sh is running" >&2; exit 1; }

python -c "import les" 2>/dev/null || { echo "LES not importable; stage E needs it" >&2; exit 1; }

for seed in ${SEEDS:-1 2 3}; do
    name="e_lr_s${seed}"
    rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
    NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset" \
    MAX_NUM_EPOCHS="${EPOCHS:-40}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
    BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
    USE_EMA=False PATIENCE=250 SEED=$seed \
    DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
    USE_LONG_RANGE=True FREEZE_AMPLITUDE=True EPS_INF=6.5 \
    "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1 &
done
wait
echo "stage E complete"
