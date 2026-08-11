#!/usr/bin/env bash
# Step A + section 4: q^pol ABLATED and the q^host-carrier cross term REMOVED.
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
# The cross term is removed because it is degenerate with Delta E_SR -- same pooling form,
# same alpha, one field learned and one fixed-shape -- and, measured under hand-set
# attention, it favours the Cs sublattice over the vacancy shell by 1.25 eV against 0.18 eV
# of short-range difference, size-independent to 18 meV. The decisive part is not the
# magnitude: a classical point-charge potential is deepest at CATION sites, so it drags a
# hole onto Cs on principle. In a halide perovskite the VBM is Cl 3p with Pb 6s antibonding
# character, so a hole belongs on Cl/Pb and never on Cs, which contributes no frontier
# states -- the collapsed answer is chemically impossible, not merely different. Rescaling
# q^host or a changes the magnitude of that pull but not its sign, so no reparameterisation
# repairs it.
#
# NO `a` ramp in this run, deliberately. The ramp exists to stop the electrostatic term
# capturing attention early; removing the term removes that reason, and running both would
# confound the test of whether the repartition alone suffices.
#
# Also active, independent of q^pol:
#   * the k = 0 neutralising background in LatentEwald.energy
#   * the delocalised-carrier exemption guard in the size hinge
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/.." && pwd)"
export PATH="$HOME/micromamba/envs/py13/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA="${DATA:-$HERE/dataset_pbe}"
NAME="${NAME:-perov_lr_a4_s1}"

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
USE_POLARISATION=False HOST_CARRIER_COUPLING=False \
DEFECT_SIZE_WEIGHT="${SIZE_WEIGHT:-1e-4}" DEFECT_SIZE_WARMUP_EPOCHS=20 \
"${REPO}/defect-example/train_defect_model.sh" > "$HOME/runs/${NAME}.log" 2>&1
echo "${NAME} done (exit $?) -- log at $HOME/runs/${NAME}.log"
