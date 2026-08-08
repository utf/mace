#!/usr/bin/env bash
#
# Train MACEDefect on the 4H-SiC divacancy dataset produced by extract_defect_dataset.py.
#
#   python extract_defect_dataset.py --n-structures 200 --seed 42
#   ./train_defect_model.sh
#
# Every setting below can be overridden from the environment, and any extra flags are
# passed straight through to mace_run_train:
#
#   NUM_CHANNELS=128 MAX_NUM_EPOCHS=600 ./train_defect_model.sh --restart_latest
#
# ---------------------------------------------------------------------------------------
# What this run does and does not test
# ---------------------------------------------------------------------------------------
# The divacancy excitation is an intra-defect promotion within the minority spin channel:
# one electron and one hole, so the net charge is zero on every frame. Two consequences.
#
#   * The long-range branch is nearly unconstrained here. With q = 0 the carrier latent
#     charge sums to zero per cell, so there is no monopole and the screening amplitude
#     `a` barely enters the energy. The section 12.4 gate "1/a^2 agrees with DFPT
#     eps_inf" therefore CANNOT be closed with this data, however clean the run looks.
#     USE_LONG_RANGE defaults to False for that reason: it keeps the first run to the
#     short-range correction, where E_total(R, 0) == E_base(R) holds exactly and there is
#     no unidentifiable parameter in the way. Turning it on is experiment two.
#
#     Corollary trap: with q = 0 the screening amplitude never moves off its
#     softplus^-1(1/sqrt(EPS_INF)) initialisation, because nothing in the loss pulls on
#     it. A post-run `a` will therefore look plausible precisely because it was never
#     fitted. Do not read it as a result.
#
#   * The other open gate, "the logit gap stays stable during training", is NOT
#     observable from a training run as the code stands. `logit_gap` and
#     `screening_amplitude` are produced by MACEDefect.forward and `screening_amplitude`
#     is consumed by the eps_inf prior, but neither is written to the DefectRMSE table
#     or the log, despite section 7 of the blueprint saying they are. Both open gates of
#     section 12.4 therefore survive this experiment.
#
# Stage Two / SWA is deliberately not used. `get_swa` has no branch for `--loss defect`
# and silently falls back to a plain energy/forces loss, which trains the total energy
# against the referenced labels and destroys the base/delta decomposition.
#
# Sample size: at the extraction script's default --n-structures 200 --valid-fraction
# 0.1, the validation set holds 7 pairs, so the "RMSE dE" column will bounce hard from
# epoch to epoch on pure sampling noise. Re-extract with --n-structures 400 or
# --valid-fraction 0.2 before reading anything into it.
#
# The band edges in band_edges.json are a fitted gauge, not PBEsol band edges, so the
# absolute energies this model returns are on a referenced scale. Only their differences
# at fixed carrier counts mean anything; see section 8 of the implementation blueprint.
# ---------------------------------------------------------------------------------------

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run against this working tree rather than whatever `mace_run_train` on PATH resolves
# to: MACEDefect is not in any released mace-torch, so a non-editable install elsewhere
# in the environment would fail with "argument --model: invalid choice: 'MACEDefect'".
# Set MACE_REPO to point somewhere else, or install this tree with `pip install -e .`
# and the PYTHONPATH below becomes a no-op.
MACE_REPO="${MACE_REPO:-$(cd "${HERE}/.." && pwd)}"
export PYTHONPATH="${MACE_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

if ! python -c "import mace.modules as m; assert hasattr(m, 'MACEDefect')" 2>/dev/null; then
    echo "the mace on PYTHONPATH has no MACEDefect; expected the defect branch at ${MACE_REPO}" >&2
    exit 1
fi

# ---- data -----------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-${HERE}/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.xyz}"
VALID_FILE="${VALID_FILE:-${DATA_DIR}/valid.xyz}"
BAND_EDGES_FILE="${BAND_EDGES_FILE:-${DATA_DIR}/band_edges.json}"

# ---- run ------------------------------------------------------------------------------
NAME="${NAME:-MACEDefect_SiC_divacancy}"
WORK_DIR="${WORK_DIR:-${HERE}/runs/${NAME}}"
SEED="${SEED:-1}"
DEVICE="${DEVICE:-$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"
# float64: MACE's training default, and the base-branch labels are ~2600 eV totals whose
# meaningful structure is at the meV level, which is at the edge of float32's resolution.
# It is the expensive choice on GPU (FP64 is ~1:2 of FP32 on A100/H100 and ~1:64 on
# consumer cards, and it forfeits TF32 entirely). At 800 structures that is affordable;
# if you scale the dataset up, benchmark DEFAULT_DTYPE=float32 against a float64 run
# rather than assuming either way.
DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"

# ---- model shape ----------------------------------------------------------------------
R_MAX="${R_MAX:-5.0}"
NUM_CHANNELS="${NUM_CHANNELS:-64}"
MAX_L="${MAX_L:-1}"
NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
NUM_RADIAL_BASIS="${NUM_RADIAL_BASIS:-8}"
CORRELATION="${CORRELATION:-3}"

# ---- defect heads ---------------------------------------------------------------------
CARRIER_FEATURE_DIM="${CARRIER_FEATURE_DIM:-32}"
COUNTER_EMBEDDING_DIM="${COUNTER_EMBEDDING_DIM:-32}"
CARRIER_MLP_HIDDEN="${CARRIER_MLP_HIDDEN:-64}"
USE_LONG_RANGE="${USE_LONG_RANGE:-False}"
# 4H-SiC high-frequency dielectric constant, ~6.5 (DFPT). Sets the initial gauge of the
# screening amplitude; only meaningful when USE_LONG_RANGE=True.
EPS_INF="${EPS_INF:-6.5}"
EPS_INF_PRIOR_WEIGHT="${EPS_INF_PRIOR_WEIGHT:-0.0}"

# ---- loss weights ---------------------------------------------------------------------
# L_base (energy/forces on the n = 0 surface), L_delta (paired charge-state differences,
# weighted highest because they are algebraically free of base-model error) and L_tot
# (unpaired excited frames, base branch detached -- note its energy term is downweighted
# to 0.1 but its FORCE term runs at the full FORCES_WEIGHT).
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-100.0}"
DELTA_ENERGY_WEIGHT="${DELTA_ENERGY_WEIGHT:-10.0}"
DELTA_FORCES_WEIGHT="${DELTA_FORCES_WEIGHT:-100.0}"
TOTAL_ENERGY_WEIGHT="${TOTAL_ENERGY_WEIGHT:-0.1}"
# The L2 on u is load-bearing, not housekeeping: it is what makes the optimiser buy
# localisation rather than large cancelling readouts, and hence what keeps the logit gap
# open. Record the value used alongside any reported result.
DEFECT_U_L2="${DEFECT_U_L2:-1e-4}"
DEFECT_P_L2="${DEFECT_P_L2:-1e-4}"
DEFECT_QHOST_L2="${DEFECT_QHOST_L2:-1e-4}"

# ---- optimisation ---------------------------------------------------------------------
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-4}"
LR="${LR:-0.005}"
EMA_DECAY="${EMA_DECAY:-0.99}"
PATIENCE="${PATIENCE:-50}"

# ---- preflight ------------------------------------------------------------------------
# All of this fails in seconds on a login node rather than after a queue wait.
for path in "${TRAIN_FILE}" "${VALID_FILE}" "${BAND_EDGES_FILE}"; do
    if [[ ! -f "${path}" ]]; then
        echo "missing ${path}; run 'python ${HERE}/extract_defect_dataset.py' first" >&2
        exit 1
    fi
done

if [[ "${USE_LONG_RANGE}" == "True" ]]; then
    # les is not on PyPI; it is pinned by commit in requirements/les.txt. Without it
    # MACEDefect raises only once the model is built, i.e. after the data has loaded.
    if ! python -c "import les" 2>/dev/null; then
        echo "USE_LONG_RANGE=True but the 'les' package is not importable." >&2
        echo "Install it, or run against a checkout with PYTHONPATH=/path/to/les/src" >&2
        exit 1
    fi
fi

if [[ "${DEVICE}" == "cuda" ]] && ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "DEVICE=cuda but torch reports no CUDA device" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}"

echo "Training ${NAME} on ${DEVICE} (${DEFAULT_DTYPE}), long-range branch: ${USE_LONG_RANGE}"
echo "Model: ${NUM_CHANNELS} channels, max_L=${MAX_L}, ${NUM_INTERACTIONS} interactions, r_max=${R_MAX}"
echo "Data:  $(grep -c 'Lattice=' "${TRAIN_FILE}") train / $(grep -c 'Lattice=' "${VALID_FILE}") valid frames from ${DATA_DIR}"
echo "Outputs in ${WORK_DIR}"
if [[ "${DEVICE}" == "cuda" ]]; then
    python - <<'PY'
import torch
name = torch.cuda.get_device_name(0)
total = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU:   {name}, {total:.0f} GiB")
PY
fi

python -m mace.cli.run_train \
    --name="${NAME}" \
    --model="MACEDefect" \
    --loss="defect" \
    --error_table="DefectRMSE" \
    --train_file="${TRAIN_FILE}" \
    --valid_file="${VALID_FILE}" \
    --band_edges_file="${BAND_EDGES_FILE}" \
    --E0s="average" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --stress_key="REF_stress" \
    --carrier_counts_key="carrier_counts" \
    --host_key="host" \
    --pair_id_key="pair_id" \
    --r_max="${R_MAX}" \
    --num_channels="${NUM_CHANNELS}" \
    --max_L="${MAX_L}" \
    --num_interactions="${NUM_INTERACTIONS}" \
    --num_radial_basis="${NUM_RADIAL_BASIS}" \
    --correlation="${CORRELATION}" \
    --carrier_feature_dim="${CARRIER_FEATURE_DIM}" \
    --counter_embedding_dim="${COUNTER_EMBEDDING_DIM}" \
    --carrier_mlp_hidden="${CARRIER_MLP_HIDDEN}" \
    --use_long_range="${USE_LONG_RANGE}" \
    --eps_inf="${EPS_INF}" \
    --eps_inf_prior_weight="${EPS_INF_PRIOR_WEIGHT}" \
    --energy_weight="${ENERGY_WEIGHT}" \
    --forces_weight="${FORCES_WEIGHT}" \
    --delta_energy_weight="${DELTA_ENERGY_WEIGHT}" \
    --delta_forces_weight="${DELTA_FORCES_WEIGHT}" \
    --total_energy_weight="${TOTAL_ENERGY_WEIGHT}" \
    --defect_u_l2="${DEFECT_U_L2}" \
    --defect_p_l2="${DEFECT_P_L2}" \
    --defect_qhost_l2="${DEFECT_QHOST_L2}" \
    --max_num_epochs="${MAX_NUM_EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --valid_batch_size="${VALID_BATCH_SIZE}" \
    --lr="${LR}" \
    --ema \
    --ema_decay="${EMA_DECAY}" \
    --patience="${PATIENCE}" \
    --eval_interval=1 \
    --default_dtype="${DEFAULT_DTYPE}" \
    --device="${DEVICE}" \
    --seed="${SEED}" \
    --save_cpu \
    --work_dir="${WORK_DIR}" \
    --model_dir="${WORK_DIR}" \
    --log_dir="${WORK_DIR}/logs" \
    --checkpoints_dir="${WORK_DIR}/checkpoints" \
    --results_dir="${WORK_DIR}/results" \
    "$@"
