#!/usr/bin/env bash
# E2 -- component H alone: the spectral head on a JOINTLY trained base.
#
# Does threshold localisation survive signal laundering? A0 measured the laundering baseline
# at 2 of 8 with the softmax head on a joint base. E2 changes only the head. If it lifts the
# rate materially, the bound-state threshold is doing work on its own; if it does not, the
# frozen base in E3 is carrying the result, and E2 is what separates those.
#
# The size hinge and the seed anneal are OFF, as component H requires: a bound state's weight
# is N-independent by construction, so the size ladder verifies that property rather than a
# hinge enforcing it, and there are no logits to seed.
#
# GPU BUDGET: four devices at a time on this machine, by instruction. CUDA_VISIBLE_DEVICES is
# set per run to one explicit device, so nothing enumerates a card this arm was not given.
#
# DEVICE COUNT AND PROCESS COUNT ARE SEPARATE. Measured with four runs going: 25-34% GPU
# utilisation, 2.7 GB of 24 GB per run, CPU load 4.3 of 32 cores. These runs are nowhere near
# GPU-bound, so packing two per card fits all eight seeds into a single pass -- still four
# devices, roughly half the wall-clock.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
GPUS=(${GPUS_LIST:-0 1 2 3})          # four devices, never more
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
CONCURRENCY=$(( ${#GPUS[@]} * RUNS_PER_GPU ))

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"
echo "using GPUs: ${GPUS[*]} -- ${RUNS_PER_GPU} run(s) each, ${CONCURRENCY} concurrent"

run () {
    local seed="$1" gpu="$2" name="e2_s$1"
    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    CUDA_VISIBLE_DEVICES="$gpu" \
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA" \
    MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="$seed" \
    ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    USE_LONG_RANGE=False \
    DEFECT_SPECTRAL_HEAD=True DEFECT_SPECTRAL_STATES="${STATES:-6}" \
    DEFECT_SPECTRAL_SMEARING="${SMEARING:-0.020}" \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  e2_s${seed} done (exit $?) gpu=${gpu}"
}

pids=()
for seed in $(seq 1 8); do
    # Round-robin over DEVICES, not over the concurrency limit, so with 8 seeds and 4 cards
    # each card gets exactly two.
    gpu=${GPUS[$(( (seed - 1) % ${#GPUS[@]} ))]}
    run "$seed" "$gpu" &
    pids+=($!)
    sleep 15
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
done
wait "${pids[@]}"
echo "E2 complete"
