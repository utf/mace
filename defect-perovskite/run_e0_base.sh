#!/usr/bin/env bash
# E0 / Stage A: train the geometry-only base on n = 0 frames (pristine + V_Cl0).
#
# The correction branch is disabled two ways, because the data alone is not enough:
#
#   * The energy path vanishes structurally at n = 0 (verified by e0_inertness_probe.py:
#     delta_sr is exactly 0, energy == base_energy, forces == base_forces, and none of the 38
#     correction-head tensors receives gradient).
#   * The four output regularisers -- u_l2 on carrier_readouts, p_l2 on polarisation,
#     qhost_l2 on latent_charges_host, zn_l2 on the counter embedding -- act on model outputs,
#     not parameters, so at their DEFAULTS they push gradient into the shared trunk even on a
#     carrier-free frame. Same for the size hinge and the seed-anneal schedule via the logit
#     network. All are zeroed here. Leaving them at defaults would produce something that is
#     not a geometry-only base and would quietly falsify E0's premise.
#
# These checkpoints are also Stage A for the later arms (E1, E3): train once, freeze, reuse.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_e0}"

mkdir -p "$HOME/runs"
echo "code: $(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain | wc -l) files"

run () {
    local seed="$1" name="e0_base_s$1"
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
    USE_LONG_RANGE=False \
    DEFECT_SEED_ANNEAL=False DEFECT_LOGIT_SEED_GAMMA=0.0 \
    DEFECT_SIZE_WEIGHT=0.0 DEFECT_GAUGE_WEIGHT=0.0 \
    DEFECT_U_L2=0.0 DEFECT_ZN_L2=0.0 DEFECT_P_L2=0.0 DEFECT_QHOST_L2=0.0 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  ${name} done (exit $?)"
}

# Both seeds share the one GPU: the model is small (128 channels, max_L=1) and a single run
# leaves the device far from saturated, so concurrency costs much less than a second pass.
run 1 &
P1=$!
sleep 20   # stagger, so the two runs do not race on E0s/statistics computation
run 2 &
P2=$!
wait $P1 $P2
echo "E0 base training complete"
