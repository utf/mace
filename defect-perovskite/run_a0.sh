#!/usr/bin/env bash
# A0 -- the contemporaneous reference rate.
#
# Production configuration exactly as it stands: two-field head (alpha = softmax(l), u = MLP),
# seed anneal on the gap gate, size hinge, joint base, long-range branch off. Nothing here is
# new; the point is to measure today's localisation rate under today's code, so the spectral
# arms are compared against a number produced by the same trunk, data and optimiser rather
# than against a rate quoted from earlier runs.
#
# Eight seeds, because the quantity of interest is a RATE. Historical arms sat at 2 of 4 and
# four seeds cannot separate 50% from 25% or 75%.
#
# Runs four at a time: each run holds ~3 GB of the A4000's 16 GB, and a single run leaves the
# device far from saturated.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
CONCURRENCY="${CONCURRENCY:-4}"

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"

run () {
    local seed="$1" name="a0_s$1"
    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA" \
    MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="$seed" \
    ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
    USE_LONG_RANGE=False \
    DEFECT_SIZE_WEIGHT="${SIZE_WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  a0_s${seed} done (exit $?)"
}

pids=()
for seed in $(seq 1 8); do
    run "$seed" &
    pids+=($!)
    sleep 15                      # stagger so runs do not race on E0s/statistics
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
done
wait "${pids[@]}"
echo "A0 complete"
