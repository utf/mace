#!/usr/bin/env bash
# Stage B continuation: resume the existing 128-channel runs to a total of 140 epochs.
#
# Uniform TOTAL rather than "+100 each": MACE keeps the best checkpoint, not necessarily
# the last, so the runs resume from different epochs (39 and 37 observed). Equal total
# budget is what makes the arms comparable; equal added epochs would not.
#
# Every setting must match the original run or --restart_latest will load a checkpoint
# into a differently-shaped model. The seed gain is already annealed to zero in all four
# checkpoints, so this continues a bias-free model: the hook re-reads gamma at startup,
# finds zero, and holds it there.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"${TMPDIR:-/tmp}/mace_stage_b_cont.lock"
flock -n 9 || { echo "another stage B continuation is running" >&2; exit 1; }

TOTAL="${TOTAL_EPOCHS:-140}"
for cfg in "128ch_L0:128:0" "128ch_L1:128:1"; do
    label="${cfg%%:*}"; rest="${cfg#*:}"; ch="${rest%%:*}"; maxl="${rest##*:}"
    for seed in ${SEEDS:-1 2}; do
        name="b_${label}_s${seed}"
        if [ ! -d "$HOME/runs/$name/checkpoints" ]; then
            echo "  ${name}: no checkpoints, skipping" >&2
            continue
        fi
        NAME=$name WORK_DIR="$HOME/runs/$name" DATA_DIR="${HERE}/dataset" \
        MAX_NUM_EPOCHS="$TOTAL" NUM_CHANNELS=$ch MAX_L=$maxl \
        NUM_RADIAL_BASIS=8 R_MAX=4.0 \
        BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
        USE_EMA=False PATIENCE=300 SEED=$seed ENABLE_CUEQ=True \
        DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
        "${HERE}/train_defect_model.sh" --restart_latest \
            >> "$HOME/runs/$name.log" 2>&1
        echo "  ${name} continued to ${TOTAL}"
    done
done
echo "stage B continuation complete"
