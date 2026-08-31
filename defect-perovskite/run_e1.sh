#!/usr/bin/env bash
# E1 -- component S in isolation: the tied head on a frozen, pre-trained base.
#
# The head is alpha = softmax(-beta eps) with beta = 10, which removes the alpha/u gauge
# freedom. On a JOINTLY trained base that variant gave 0 of 8 correct, with failures coherent
# but sublattice-level: ~0.20 attention mass on the two-atom shell against 1.000 for correct
# models, and a within-species eps spread on Pb of 0.036-0.043 eV against 0.184-0.186 eV.
# Gauge deletion alone is therefore insufficient.
#
# E1 changes only the base. Stage A is frozen, so the base cannot absorb the site signal out
# from under the correction, and the residual at a charged geometry stands in for the neutral
# pair partner the dataset does not contain. E0 measured that this residual carries the
# signal: two atoms of 79 hold 32% of the squared force residual against 14-16% for a
# carrier-free null.
#
# The prediction to test is specific and not just "more seeds pass": the within-species eps
# spread on Pb should rise from ~0.04 toward ~0.18 eV EVEN IF the shell mass stays partial.
# That is what would show the base was laundering the signal.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
# GPUS_LIST empty means "one device, whatever CUDA picks" -- the single-GPU workstation.
# Setting it pins one explicit device per run so several cards can share the arm.
GPUS=(${GPUS_LIST:-})
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
if [ ${#GPUS[@]} -gt 0 ]; then
    CONCURRENCY=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
else
    CONCURRENCY="${CONCURRENCY:-4}"
fi

# Alternate the two Stage-A bases across seeds so the arm's rate is not a property of one
# base. They agreed closely in E0 (shell/bulk 3.93 vs 4.01) so this should not matter -- which
# is exactly why it is worth being able to check rather than assume.
BASE1="$HOME/runs/e0_base_s1/e0_base_s1.model"
BASE2="$HOME/runs/e0_base_s2/e0_base_s2.model"
for b in "$BASE1" "$BASE2"; do
    [ -f "$b" ] || { echo "missing Stage-A base: $b" >&2; exit 1; }
done

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"

run () {
    local seed="$1" gpu="${2:-}" name="e1_s$1"
    local base="$BASE1"
    [ $((seed % 2)) -eq 0 ] && base="$BASE2"
    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    (
    [ -n "$gpu" ] && export CUDA_VISIBLE_DEVICES="$gpu"
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
    DEFECT_ALPHA_MODE=tied DEFECT_BETA=10.0 \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_BASE_INIT="$base" BASE_LR_FACTOR=0.0 \
    DEFECT_SIZE_WEIGHT="${SIZE_WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    )
    echo "  e1_s${seed} done (exit $?) base=$(basename "$base")"
}

pids=()
i=0
for seed in $SEEDS; do
    gpu=""
    [ ${#GPUS[@]} -gt 0 ] && gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    i=$(( i + 1 ))
    run "$seed" "$gpu" &
    pids+=($!)
    sleep 15
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
done
wait "${pids[@]}"
echo "E1 complete"
