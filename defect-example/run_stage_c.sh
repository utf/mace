#!/usr/bin/env bash
# Forward plan stage C (direct version): does plateau length scale with cell size?
#
# The refuted claim was that the *logit* gradient is 1/N-suppressed. The untested one is
# that the defect-specific fraction of the *u* gradient goes as n_d/N, which would make
# plateau length grow with N -- and every production cell is larger than these.
#
# Two matched datasets, 63 pairs each, same seed, same steps per epoch, same LR, differing
# only in cell size (286 vs 398 atoms, a 1.39x lever arm). Neither has pristine frames, so
# loss composition matches too. Escape epoch is the metric; RMSE is not comparable across
# them because the two subsets have different target spreads.
#
# Decision: escape epoch rising with N => the plateau is a production-scale hazard and
# stage D is justified on dynamics. Flat => a fixed ~20-epoch cost, and D must justify
# itself on the Delta-u diagnostic alone.
set -euo pipefail

# Refuse to run twice at once. Both instances would rm -rf and rewrite the same
# ~/runs/c_n*_s* directories, so the second silently destroys the first's results and the
# two interleave on the GPU -- which also invalidates any wall-clock comparison. This
# happened once; the lock is cheaper than noticing it again.
exec 9>"${TMPDIR:-/tmp}/mace_stage_c.lock"
if ! flock -n 9; then
    echo "another run_stage_c.sh is already running (lock held); refusing to start" >&2
    exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS="${EPOCHS:-150}"
SEEDS="${SEEDS:-1 2 3}"

for seed in ${SEEDS}; do
    for size in 286 398; do
        name="c_n${size}_s${seed}"
        rm -rf "$HOME/runs/$name" "$HOME/runs/$name.log"
        NAME=$name WORK_DIR="$HOME/runs/$name" \
        DATA_DIR="${HERE}/dataset_n${size}" \
        MAX_NUM_EPOCHS="${EPOCHS}" NUM_CHANNELS=8 MAX_L=0 NUM_RADIAL_BASIS=4 R_MAX=4.0 \
        BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda DEFAULT_DTYPE=float32 \
        USE_EMA=False PATIENCE=200 SEED=$seed \
        "${HERE}/train_defect_model.sh" > "$HOME/runs/$name.log" 2>&1 &
    done
    # One seed (both cell sizes) at a time: two concurrent runs fit the A4000 comfortably,
    # six do not, and contention would distort the wall-clock comparison anyway.
    wait
    echo "seed ${seed} done"
done

echo "=== stage C: escape epoch by cell size ==="
printf "%-14s %8s %10s\n" run escape_ep final_dE
for seed in ${SEEDS}; do
    for size in 286 398; do
        log="$HOME/runs/c_n${size}_s${seed}.log"
        [ -f "$log" ] || continue
        rows=$(grep -E "Epoch [0-9]+: head" "$log" \
            | sed -E 's/.*Epoch ([0-9]+).*RMSE_dE= *([0-9.]+).*RMSE_dF= *([0-9.]+).*/\1 \2 \3/')
        # Escape = first epoch whose dF falls 10% below the plateau value (the dF metric
        # is flat to four significant figures until the attention localises).
        plateau=$(echo "$rows" | head -1 | awk '{print $3}')
        esc=$(echo "$rows" | awk -v p="$plateau" '$3 < 0.9*p {print $1; exit}')
        printf "%-14s %8s %10s\n" "c_n${size}_s${seed}" "${esc:-none}" \
            "$(echo "$rows" | tail -1 | awk '{print $2}')"
    done
done
