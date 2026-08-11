#!/usr/bin/env bash
# Step A: retrain the perovskite long-range model with q^pol ABLATED.
#
# delta_lr then reduces to carrier^2 + host.carrier, the latter physical and constant
# (-1.906 eV, constant to five decimals). Measured at inference on the existing weights,
# ablation already gives delta_lr flat to -0.003 in the exponent and 6.6 meV of drift over
# a 3.4x size range, against +0.971 and 52 eV before.
#
# The justification for trying the ablation FIRST rather than as a fallback: after gating,
# the polarisation response is contained inside the receptive field (<=0.001 e beyond 6 A
# against a ~10 A receptive field), and a neutral cloud contained inside the receptive field
# has a local electrostatic energy -- its self-energy and its interaction with the periodic
# host field both sample only where the cloud is -- so delta_SR can already represent it.
# The long-range branch is only structurally required for the monopole and the
# slowly-converging multipoles of the carrier, which q^carrier carries.
#
# Also active in this run, independent of q^pol:
#   * the k = 0 neutralising background in LatentEwald.energy (-24 meV at these cells)
#   * the delocalised-carrier exemption guard in the size hinge
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
NAME="${NAME:-perov_lr_stepA_s1}"

rm -rf "$HOME/runs/$NAME" "$HOME/runs/${NAME}.log"
NAME="$NAME" WORK_DIR="$HOME/runs/$NAME" DATA_DIR="$DATA" \
MACE_REPO="$REPO" \
MAX_NUM_EPOCHS="${EPOCHS:-140}" NUM_CHANNELS="${CHANNELS:-128}" MAX_L=1 \
NUM_RADIAL_BASIS=8 R_MAX="${R_MAX:-5.0}" \
BATCH_SIZE="${BATCH:-8}" VALID_BATCH_SIZE="${BATCH:-8}" DEVICE="${DEVICE:-cuda}" \
DEFAULT_DTYPE=float32 USE_EMA=False PATIENCE=250 SEED="${SEED:-1}" \
ENABLE_CUEQ=True \
ENERGY_WEIGHT=10.0 TOTAL_ENERGY_WEIGHT=10.0 \
DELTA_ENERGY_WEIGHT=0.0 DELTA_FORCES_WEIGHT=0.0 \
DEFECT_LOGIT_SEED_GAMMA=1.5 DEFECT_SEED_ANNEAL=True \
USE_LONG_RANGE=True FREEZE_AMPLITUDE=True EPS_INF="${EPS_INF:-4.0}" \
USE_POLARISATION=False \
DEFECT_SIZE_WEIGHT="${SIZE_WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
"${REPO}/defect-example/train_defect_model.sh" > "$HOME/runs/${NAME}.log" 2>&1
echo "${NAME} done (exit $?) -- log at $HOME/runs/${NAME}.log"
