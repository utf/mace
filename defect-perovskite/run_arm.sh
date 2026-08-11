#!/usr/bin/env bash
# One experiment arm from the feature-collision plan.
#
#   ARM=A0  2a only               -- seed anneal regated on site structure
#   ARM=A1  2a + 2b(i)            -- and q^host computed from detached features
#   ARM=Z   2a + a = 0            -- the CONTROL
#
# Z is the measurement that discriminates. A0 and nolr differ in more than one energy
# term: enabling the long-range branch constructs additional modules, and `_mlp` draws its
# first layer from the GLOBAL rng, so the two have different trunk and readout
# initialisations at the same seed. That is not a small confound in a project where escape
# epochs have differed 22 vs 45 on identical data. Z holds a = 0, so q^carrier = a alpha is
# zero and carrier^2 = 0 -- energetically identical to nolr, structurally identical to A0,
# same rng stream. Collapse under Z exonerates carrier^2 and indicts the initialisation or
# code path; localisation under Z indicts carrier^2.
#
# Both carry Step A (q^pol off) and the section 4 repartition, which are settled. The
# question each asks is in the plan: A0, does holding the seed longer let the energy fit
# win despite the collision; A1, does removing the pathway fix it structurally.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM="${ARM:-A0}"
SEED="${SEED:-1}"
NAME="perov_${ARM}_s${SEED}"
EPS="${EPS_INF:-4.0}"
case "$ARM" in
    A0) DETACH=False ;;
    A1) DETACH=True ;;
    # a = 1/sqrt(eps_inf) = 1e-6, so carrier^2 ~ a^2 = 1e-12 eV: zero for every practical
    # purpose, with no code change and therefore no new code path of its own.
    Z)  DETACH=False; EPS=1e12 ;;
    *) echo "unknown ARM '$ARM' (want A0, A1 or Z)" >&2; exit 2 ;;
esac

rm -rf "$HOME/runs/$NAME" "$HOME/runs/${NAME}.log"
NAME="$NAME" WORK_DIR="$HOME/runs/$NAME" DATA_DIR="${DATA:-$HERE/dataset_pbe}" \
MACE_REPO="$REPO" \
MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS=128 MAX_L=1 \
NUM_RADIAL_BASIS=8 R_MAX=5.0 \
BATCH_SIZE=8 VALID_BATCH_SIZE=8 DEVICE=cuda \
DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="$SEED" \
ENABLE_CUEQ=True \
ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
USE_LONG_RANGE=True FREEZE_AMPLITUDE=True EPS_INF="$EPS" \
USE_POLARISATION=False HOST_CARRIER_COUPLING=False \
HOST_CHARGE_DETACHED="$DETACH" \
DEFECT_SIZE_WEIGHT=1e-4 DEFECT_SIZE_WARMUP_EPOCHS=20 \
"${REPO}/defect-example/train_defect_model.sh" > "$HOME/runs/${NAME}.log" 2>&1
echo "${NAME} finished (exit $?)"
