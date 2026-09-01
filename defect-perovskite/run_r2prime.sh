#!/usr/bin/env bash
# R2', the revised arms. Staging is demoted from default to control.
#
# Why the default changed. Staging has never localised anything (E1 0/8; R1/R2 void). The
# only localisations on record -- A0, 2/8 -- were joint training FROM SCRATCH with the size
# hinge, and from scratch WITHOUT the hinge was 0/8. The discriminating ingredient is the
# hinge, not the staging. The hinge is a dilution constraint, and Test 2 has now established
# from DFT that the physics it encodes holds here: the hub force does not dilute when the
# cell doubles (R_DFT = 0.95, CI [0.66, 1.34]).
#
# Staging also has a cost that works against localisation: a base that never saw charged
# geometries carries extrapolation error everywhere (E1: 20-33 meV/A against ~12 for A0), and
# the head must absorb it during the very epochs in which the state forms. Diffuse, ligand-
# shaped carriers absorb distributed error better than two atoms do, so a frozen base rewards
# the wrong states.
#
#   arm 1   from scratch, joint base + head, DCL from the start, two-size upweight, anneal
#   arm 1b  as arm 1 without DCL -- isolates the constraint's own contribution
#   arm 2   the staged configuration, now a control, with upweight and anneal
#
# No hinge-era exemption or span guard anywhere: there is no exemption clause.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
# GPU 0-3 are off-limits on b3 by instruction (2026-09-01): GPU3 has a recurring
# hardware fault -- it has now died three times -- and 0-2 are reserved. Four cards,
# so this still satisfies the standing at-most-four-devices rule. Override with
# GPUS_LIST if you are on a different machine.
GPUS=(${GPUS_LIST:-4 5 6 7})
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
CONCURRENCY=$(( ${#GPUS[@]} * RUNS_PER_GPU ))
ARMS="${ARMS:-arm1 arm1b}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
HEAD="${HEAD:-h3}"
EPOCHS="${EPOCHS:-50}"
# DCL is deferred to step 3: forcing compactness on a head whose compact states cannot
# reproduce the force footprint (R1: every clamp ~0, free ~0.8) would buy worse forces and
# an arbitrary compact site -- which R1 already told us. Its flags are NOT declared yet;
# declaring inert flags is how the bandwidth anneal sat dead for a whole cycle.

BASE1="$HOME/runs/e0_base_s1/e0_base_s1.model"
BASE2="$HOME/runs/e0_base_s2/e0_base_s2.model"

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"
echo "arms: $ARMS | head: $HEAD | seeds: $SEEDS | GPUs ${GPUS[*]} x${RUNS_PER_GPU}"

run () {
    local arm="$1" seed="$2" gpu="$3"
    local name="r2p_${HEAD}_${arm}_s${seed}"
    local sigma=False
    [ "$HEAD" = "h3" ] && sigma=True

    # Staging applies to arm 2 only. Arms 1 and 1b train base and head jointly from step 0,
    # so there is no release stage and therefore no rollback guard -- the guard exists to
    # protect a release, and with nothing released it would only ever freeze a base that was
    # never frozen.
    local base_init="" base_lr=1.0 release_epoch=0
    if [ "$arm" = "arm2" ]; then
        base_init="$BASE1"
        [ $((seed % 2)) -eq 0 ] && base_init="$BASE2"
        base_lr=0.0
        release_epoch=30
    fi

    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    (
    export CUDA_VISIBLE_DEVICES="$gpu"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA" MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="$EPOCHS" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="$seed" ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 USE_LONG_RANGE=False \
    DEFECT_SPECTRAL_HEAD=True DEFECT_SPECTRAL_DECAY=True \
    DEFECT_SPECTRAL_FIRST_SHELL=True DEFECT_SPECTRAL_SIGMA="$sigma" \
    DEFECT_SPECTRAL_GAUGE_PENALTY=True DEFECT_EPS_GAUGE_WEIGHT="${GAUGE_W:-1.0}" \
    DEFECT_SPECTRAL_ANNEAL_S0=4.0 DEFECT_SPECTRAL_ANNEAL_EPOCHS=20 \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 \
    DEFECT_TWO_SIZE_UPWEIGHT="${UPWEIGHT:-0.25}" \
    DEFECT_BASE_INIT="$base_init" BASE_LR_FACTOR="$base_lr" \
    DEFECT_BASE_RELEASE_EPOCH="$release_epoch" DEFECT_BASE_RELEASE_FACTOR=0.01 \
        "${REPO}/defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    )
    echo "  ${name} done (exit $?) gpu=${gpu}"
}

pids=(); i=0
for arm in $ARMS; do
  for seed in $SEEDS; do
    gpu=${GPUS[$(( i % ${#GPUS[@]} ))]}
    i=$(( i + 1 ))
    run "$arm" "$seed" "$gpu" &
    pids+=($!)
    sleep 10
    while [ "$(jobs -rp | wc -l)" -ge "$CONCURRENCY" ]; do sleep 20; done
  done
done
wait "${pids[@]}"
echo "R2' complete"
