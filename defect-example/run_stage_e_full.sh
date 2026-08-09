#!/usr/bin/env bash
# Stage E at production scale: the bfull L1 configuration with the long-range branch on.
#
# Third leg of the size-transferability comparison (HANDOFF section 29). The question is
# whether more data and more capacity, plus the long-range term, hold `alpha` on the
# defect as the cell grows -- or whether softmax dilution is structural and survives all
# three.
#
# Differences from run_stage_e.sh, which was an 8-channel probe on the small dataset:
#   * 128 channels / max_L = 1 on dataset_full, matching the bfull runs it is compared to
#   * the raised energy weights (10 / 10), so it stays comparable to bfull
#
# cuEquivariance is ON, and at this width it is not optional: e3nn was measured to OOM at
# batch 4 at 128ch/max_L=1 on this 16 GB A4000 *without* the long-range branch, while cueq
# runs batch 8 in 7.3 GB. It used to be blocked here -- conversion with use_long_range was
# untested, and rightly so, because the conversion rebuilds the model from an extracted
# config that was silently dropping `freeze_amplitude` and handing back a *trainable*
# screening amplitude. That is fixed and verified by verify_cueq_long_range.py (amplitude
# bit-identical and still frozen, E_LR still contributing, forces agreeing to 6e-6 eV/A).
#
# The batch size is still probed rather than assumed, since the long-range branch adds
# memory on top of the 7.3 GB measured without it.
#
# `a` stays FROZEN at 1/sqrt(eps_inf) with eps_inf = 6.5 (literature 4H-SiC, not DFPT), as
# in the original stage E: it is not identifiable from a dipole term alone, so a free `a`
# drifts to whatever absorbs the electron-hole energy and then reads as a fitted screening
# constant.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_stage_e_full.lock"
flock -n 9 || { echo "another run_stage_e_full.sh is running" >&2; exit 1; }

python -c "import les" 2>/dev/null || { echo "LES not importable; stage E needs it" >&2; exit 1; }

SEED="${SEED:-1}"
NAME="${NAME:-efull_128ch_L1_s${SEED}}"
LOG="$HOME/runs/${NAME}.log"

for batch in ${BATCH_CANDIDATES:-8 4 2 1}; do
    echo "=== attempting ${NAME} at batch size ${batch} ==="
    rm -rf "$HOME/runs/$NAME" "$LOG"
    NAME=$NAME WORK_DIR="$HOME/runs/$NAME" DATA_DIR="${HERE}/dataset_full" \
    MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS=128 MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX=4.0 \
    BATCH_SIZE=$batch VALID_BATCH_SIZE=$batch DEVICE=cuda DEFAULT_DTYPE=float32 \
    USE_EMA=False PATIENCE=250 SEED=$SEED ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DEFECT_SIZE_WEIGHT="${DEFECT_SIZE_WEIGHT:-0.0}" \
    DEFECT_SIZE_WARMUP_EPOCHS="${DEFECT_SIZE_WARMUP_EPOCHS:-20}" \
    DEFECT_LOGIT_SEED_GAMMA="${GAMMA:-1.5}" DEFECT_SEED_ANNEAL=True \
    USE_LONG_RANGE=True FREEZE_AMPLITUDE=True EPS_INF=6.5 \
    "${HERE}/train_defect_model.sh" > "$LOG" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "${NAME} finished at batch ${batch}"
        exit 0
    fi
    if grep -qiE "out of memory|CUDA error" "$LOG"; then
        echo "  batch ${batch} ran out of memory, stepping down"
        continue
    fi
    echo "  ${NAME} failed at batch ${batch} for a reason other than memory (exit ${status}):" >&2
    tail -20 "$LOG" >&2
    exit $status
done
echo "${NAME} could not be fitted on this card at any candidate batch size" >&2
exit 1
