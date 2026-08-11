#!/usr/bin/env bash
# CsPbCl3 V_Cl, PBE: one model with the long-range branch and one without.
#
# This is the first dataset with q != 0, so it is the first that can identify `a` at all.
# Under the neutral-reference labelling the counter charge equals the absolute cell charge,
# so V_Cl+ carries a real monopole and the long-range branch has something to fit --
# unlike 4H-SiC, where every frame was net neutral and `a` was unidentifiable.
#
# Supervision, per the dataset plan: there are NO paired frames (the two charge states share
# only pristine copies; the V_Cl geometries are disjoint because the source paper trains
# separate models per charge state). So the delta terms have no support and are zeroed, and
# everything rests on L_base + L_tot with E_base receiving gradient.
#
# `a` stays frozen at 1/sqrt(eps_inf) with an interim eps_inf, flagged as such: an unfrozen
# `a` during the transition would absorb any residual labelling error and then read as a
# fitted screening constant. Replace with DFPT before quoting 1/a^2.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"

run () {
    local name="$1" long_range="$2"
    echo "=== ${name} (use_long_range=${long_range}) ==="
    rm -rf "$HOME/runs/$name" "$HOME/runs/${name}.log"
    NAME="$name" WORK_DIR="$HOME/runs/$name" DATA_DIR="$DATA" \
    MACE_REPO="$REPO" \
    MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
    NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
    BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE=cuda \
    DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="${SEED:-1}" \
    ENABLE_CUEQ=True \
    ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
    DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
    DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
    USE_LONG_RANGE="$long_range" FREEZE_AMPLITUDE=True EPS_INF="${EPS_INF:-4.0}" \
    DEFECT_SIZE_WEIGHT="${SIZE_WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
    "${HERE}/../defect-example/train_defect_model.sh" > "$HOME/runs/${name}.log" 2>&1
    echo "  ${name} done (exit $?)"
}

run "perov_nolr_s1" False
run "perov_lr_s1" True
echo "perovskite runs complete"
