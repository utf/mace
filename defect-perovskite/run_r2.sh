#!/usr/bin/env bash
# R2 -- free-attention screen on H2 and H3 (plan T4).
#
# Two-timescale base, replacing the hard freeze. E1 showed hard-freezing costs 20-33 meV/A
# because the base cannot adapt to charged geometries at all; a joint base launders the site
# signal. So: Stage-A init, base LR = 0 through epoch 30 (passing seeds settle by ~30), then
# base LR = 0.01x head LR. The drift guard (in run_train) reverts the base to its epoch-30
# weights if validation hub mass falls by more than 0.05 after release.
#
# Gauge: T5 penalty, not the mean subtraction -- the subtraction made lambda cell-size
# dependent at ~2.6 meV across the ladder, against a <= 1 meV gate.
#
# Bandwidth anneal arm: all hoppings scaled by s(e) = 4^(1 - e/20) for e <= 20, then 1.
# Annealing only enters HERE: before R1 inverts the hub/cage preference it would select the
# cage, which is the failure mode the whole cycle is about.
#
# 50-epoch screen (passing seeds settle by ~30; a0_s6 is the known late mover, so anything
# promising is re-run at full length before a rate is recorded).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
GPUS=(${GPUS_LIST:-0 1 2 3})
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
CONCURRENCY=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
HEADS="${HEADS:-h2 h3}"
ARMS="${ARMS:-noanneal anneal}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"

BASE1="$HOME/runs/e0_base_s1/e0_base_s1.model"
BASE2="$HOME/runs/e0_base_s2/e0_base_s2.model"
for b in "$BASE1" "$BASE2"; do
    [ -f "$b" ] || { echo "missing Stage-A base: $b" >&2; exit 1; }
done

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"
echo "heads: $HEADS | arms: $ARMS | seeds: $SEEDS | GPUs ${GPUS[*]} x${RUNS_PER_GPU}"

run () {
    local head="$1" arm="$2" seed="$3" gpu="$4"
    local name="r2_${head}_${arm}_s${seed}"
    local base="$BASE1"
    [ $((seed % 2)) -eq 0 ] && base="$BASE2"

    local sigma=False first=True
    [ "$head" = "h3" ] && sigma=True
    local anneal=0.0
    [ "$arm" = "anneal" ] && anneal=4.0

    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    (
    export CUDA_VISIBLE_DEVICES="$gpu"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA" \
    MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="${EPOCHS:-50}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="$seed" \
    ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    USE_LONG_RANGE=False \
    DEFECT_SPECTRAL_HEAD=True DEFECT_SPECTRAL_DECAY=True \
    DEFECT_SPECTRAL_FIRST_SHELL="$first" DEFECT_SPECTRAL_SIGMA="$sigma" \
    DEFECT_SPECTRAL_GAUGE_PENALTY=True DEFECT_EPS_GAUGE_WEIGHT="${GAUGE_W:-1.0}" \
    DEFECT_SPECTRAL_ANNEAL_S0="$anneal" DEFECT_SPECTRAL_ANNEAL_EPOCHS=20 \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 \
    DEFECT_BASE_INIT="$base" BASE_LR_FACTOR=0.0 \
    DEFECT_BASE_RELEASE_EPOCH=30 DEFECT_BASE_RELEASE_FACTOR=0.01 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    )
    echo "  ${name} done (exit $?) gpu=${gpu} base=$(basename "$base")"
}

pids=(); i=0
for head in $HEADS; do
  for arm in $ARMS; do
    for seed in $SEEDS; do
      gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
      i=$(( i + 1 ))
      run "$head" "$arm" "$seed" "$gpu" &
      pids+=($!)
      sleep 10
      while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
    done
  done
done
wait "${pids[@]}"
echo "R2 complete"
