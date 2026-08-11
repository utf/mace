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
# Halved from 32/64. The correction branch is dominated by the first layer of the eight
# carrier MLPs (in_dim = carrier_feature_dim*n_layers + counter_embedding_dim -> hidden),
# which was 68.8% of all model parameters at 8 channels. The counter embedding maps six
# numbers, and the readouts produce one scalar each, so neither needed the width.
COUNTER_EMBEDDING_DIM="${COUNTER_EMBEDDING_DIM:-16}"
CARRIER_MLP_HIDDEN="${CARRIER_MLP_HIDDEN:-32}"
USE_LONG_RANGE="${USE_LONG_RANGE:-False}"
# Attention mode (plan stage D). 'logits' is the baseline: a separate network per channel.
# 'tied' sets alpha = softmax(-DEFECT_BETA * u), deriving the attention from the carrier
# site energy itself -- which removes the additive gauge freedom in the logits, forbids
# bound-but-delocalised states, drops MLP_l entirely, and makes the reported gap a
# physical binding energy. DEFECT_BETA is a recorded gauge constant, not a tuning knob.
# Additive novelty logit bias: logit_i^c += gamma_c * s_hat_i, gamma trainable per channel.
# Preferred over the u-seed: it acts directly on what controls attention, works at step 0
# with no fit through the readout, and the descriptor is recomputed in-graph so forces
# differentiate it (FD-verified to 1.7e-9). It IS an architecture term, present at
# inference -- not a training-only initialisation.
DEFECT_LOGIT_SEED_GAMMA="${DEFECT_LOGIT_SEED_GAMMA:-0.0}"
# Anneal that gain to zero over training, per channel, on readiness. Strongly recommended
# whenever the seed is on: it leaves the converged model bias-free, so nothing downstream
# has to carry or differentiate the descriptor, and no future caching of s_hat can
# silently break force consistency.
DEFECT_SEED_ANNEAL="${DEFECT_SEED_ANNEAL:-True}"
# Absolute epoch by which gamma reaches zero. Absolute rather than a fraction of
# max_num_epochs because early stopping makes the run length unknown in advance, and
# a fractional schedule can let a model converge and stop with the seed still active.
DEFECT_SEED_ANNEAL_EPOCHS="${DEFECT_SEED_ANNEAL_EPOCHS:-30}"
DEFECT_ALPHA_MODE="${DEFECT_ALPHA_MODE:-logits}"
DEFECT_BETA="${DEFECT_BETA:-10.0}"
# 4H-SiC high-frequency dielectric constant, ~6.5 (DFPT). Sets the initial gauge of the
# screening amplitude; only meaningful when USE_LONG_RANGE=True.
# Freeze `a` at 1/sqrt(EPS_INF) rather than fitting it (plan stage E). With q = 0 on every
# frame there is no monopole for `a` to scale, so a free `a` is unidentifiable and drifts
# to whatever absorbs the electron-hole energy.
FREEZE_AMPLITUDE="${FREEZE_AMPLITUDE:-False}"
# q^pol channel. A whole-cell mean gives neutrality, not locality, so an ungated q^pol puts
# a fixed per-species charge on every atom in the crystal and its Madelung energy grows with
# N. USE_POLARISATION=False ablates the channel; POL_GATE=True gates it on the smeared
# carrier density instead. Only meaningful when USE_LONG_RANGE=True.
USE_POLARISATION="${USE_POLARISATION:-True}"
POL_GATE="${POL_GATE:-False}"
POL_GATE_LAMBDA="${POL_GATE_LAMBDA:-6.0}"
POL_GATE_HOPS="${POL_GATE_HOPS:-2}"
# Keep the q^host-carrier cross term in delta_lr. False repartitions the branch so it
# carries only what Delta E_SR structurally cannot: the monopole self-interaction and
# interactions between separated carriers.
HOST_CARRIER_COUPLING="${HOST_CARRIER_COUPLING:-True}"
EPS_INF="${EPS_INF:-6.5}"
EPS_INF_PRIOR_WEIGHT="${EPS_INF_PRIOR_WEIGHT:-0.0}"

# ---- loss weights ---------------------------------------------------------------------
# L_base (energy/forces on the n = 0 surface, i.e. pristine frames only), L_delta (paired
# charge-state differences, weighted highest because they are algebraically free of
# base-model error) and L_tot (see below -- now every n != 0 frame, base branch NOT
# detached; its energy term is moderate but its FORCE term runs at the full
# FORCES_WEIGHT, which is what supervises forces at defect geometries).
ENERGY_WEIGHT="${ENERGY_WEIGHT:-1.0}"
FORCES_WEIGHT="${FORCES_WEIGHT:-100.0}"
DELTA_ENERGY_WEIGHT="${DELTA_ENERGY_WEIGHT:-10.0}"
DELTA_FORCES_WEIGHT="${DELTA_FORCES_WEIGHT:-100.0}"
# L_tot is back ON, and now applies at every n != 0 frame with E_base receiving gradient
# (plan A5.3). It was off because corrected labels leave no n = 0 reference at defect
# geometries, which left total forces there supervised by nothing at all. The protection
# against the base branch absorbing correction physics is structural rather than the old
# stopgrad: the correction is intensive (sum_i alpha_i = 1) and E_base is extensive, so a
# bulk-wide base error cannot be absorbed at any localisation of alpha.
# Moderate on the energy, full on the forces (the force term uses FORCES_WEIGHT).
TOTAL_ENERGY_WEIGHT="${TOTAL_ENERGY_WEIGHT:-0.25}"
# Plan A5.3 asks for the base branch on a low learning rate, so it is not a moving target
# for the correction. Measured on dataset_beta, 25 epochs, one knob at a time:
#
#   BASE_LR_FACTOR=0.25 (with L_tot on)   RMSE_E =  789 meV/atom, RMSE_F = 177
#   BASE_LR_FACTOR=0.25 (with L_tot off)  RMSE_E = 1545 meV/atom, RMSE_F = 179
#   BASE_LR_FACTOR=1.0  (with L_tot on)   RMSE_E =   18 meV/atom, RMSE_F =  89
#
# So it stays at 1.0 here. The "moving target" argument assumes a base branch that is
# already trained; this one starts from random init and has to learn the whole SiC
# potential, and quartering its learning rate simply leaves it undertrained -- which
# L_tot then exposes directly, because it now supervises total energy at defect
# geometries. Lower it when warm-starting from a trained base (e.g. the Stage E retrain).
BASE_LR_FACTOR="${BASE_LR_FACTOR:-1.0}"
# Pristine frames are the only real n = 0 labels, and there are just 53 of them against
# ~666 defect frames, so they are replayed at high weight.
CONFIG_TYPE_WEIGHTS="${CONFIG_TYPE_WEIGHTS:-{\"ideal\":5.0}}"
# Keeps u close to linear in n, the condition under which counter dependency kills the
# E_base gauge (plan A5.4). Not a gauge-fixing device in itself.
DEFECT_ZN_L2="${DEFECT_ZN_L2:-1e-3}"
# Size-extensivity hinge (size plan section 4) and the level-mode gauge penalty
# (forward plan stage D-opt). Both default OFF, so every existing run is byte-identical;
# their diagnostics are logged either way, which is the point of computing them
# unconditionally.
DEFECT_SIZE_WEIGHT="${DEFECT_SIZE_WEIGHT:-0.0}"
DEFECT_SIZE_RATIO="${DEFECT_SIZE_RATIO:-1e4}"
DEFECT_SIZE_TOL="${DEFECT_SIZE_TOL:-1e-3}"
DEFECT_SIZE_WARMUP_EPOCHS="${DEFECT_SIZE_WARMUP_EPOCHS:-20}"
DEFECT_GAUGE_WEIGHT="${DEFECT_GAUGE_WEIGHT:-0.0}"
# The L2 on u was declared load-bearing (plan 3.2): the mechanism that makes the optimiser
# buy localisation rather than large cancelling readouts. It is off here (fix plan 1.5).
# Two reasons. It never did that job -- at 1e-4 * mean(u^2) it was seven orders below the
# data term, and being a *mean* it is intensive, so it cannot penalise a bulk-wide level of
# u at all. And the synthetic anchors now pin u_bulk = 0 from data, which is the honest
# version of the same constraint. If the Stage 2 tie is ever removed, a regulariser must
# return on the residual head v, not on u.
DEFECT_U_L2="${DEFECT_U_L2:-0.0}"
# Zero-init of MLP_u's last layer. ON: it is an optimisation-conditioning choice, not a
# requirement of the n = 0 identity (that is structural, via the counter prefactor). A
# random u is noise the optimiser must first destroy, and it ends up back at u ~ 0 with
# the attention still uniform; from zero, u grows along the gradient instead. Measured
# over three seeds: escape 3/3 with it, 1/3 without. Set False to ablate.
DEFECT_ZERO_U_INIT="${DEFECT_ZERO_U_INIT:-True}"
# cuEquivariance acceleration of the trunk. Verified numerically identical to e3nn for
# the short-range model (tests/unit/test_defect_cueq.py: 9e-14 on energies, 2e-16 on
# forces in float64). It converts the TRUNK only -- the correction heads are dense MLPs
# and are already as fast as they get.
#
# Measured on the A4000, 128 channels / max_L=1, 286-atom cells, float32:
#
#   batch  backend   ms/step   peak MiB   ms/frame
#       2  e3nn        343.3       8257      171.7
#       2  cueq        159.8       1835       79.9
#       2  oeq         126.2       4604       63.1
#       4  e3nn          OOM          -          -
#       4  cueq        139.5       3648       34.9
#       4  oeq         233.4       9144       58.3
#       8  cueq        141.1       7274       17.6
#       8  oeq           OOM          -          -
#
# The memory column is the headline: cueq needs 4.5x less than e3nn, which is what lets
# the batch grow, and throughput follows -- 17.6 ms/frame at batch 8 against e3nn's 171.7
# at batch 2, a ~10x improvement. OpenEquivariance is faster per step at batch 2 but uses
# 2.5x the memory of cueq and OOMs at batch 8, so it loses on throughput here.
#
# It does NOT pay at small width: 8 channels / max_L=0 measured 0.85x, i.e. a slowdown,
# where kernel launch overhead dominates. Leave it off for debug runs.
#
# Not supported with USE_LONG_RANGE=True (untested). The cueq+oeq hybrid does not work in
# this tree: the cueq conversion does not persist cueq_config on the model, so chaining
# the two loses the cueq half and fails on a state_dict mismatch.
ENABLE_CUEQ="${ENABLE_CUEQ:-False}"
DEFECT_P_L2="${DEFECT_P_L2:-1e-4}"
DEFECT_QHOST_L2="${DEFECT_QHOST_L2:-1e-4}"

# ---- optimisation ---------------------------------------------------------------------
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4}"
VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-4}"
LR="${LR:-0.005}"
EMA_DECAY="${EMA_DECAY:-0.99}"
PATIENCE="${PATIENCE:-50}"
# Validation runs inside `ema.average_parameters()` (train.py:299), so with EMA on the
# reported metrics are those of the averaged weights, not the ones being optimised. Set
# USE_EMA=False to read the raw weights -- the control that separates "not learning" from
# "learning, but the average lags" (fix plan stage 0.3).
USE_EMA="${USE_EMA:-True}"

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

case "${USE_EMA}" in
    True | true | TRUE | 1) EMA_ARGS=(--ema "--ema_decay=${EMA_DECAY}") ;;
    *)
        EMA_ARGS=()
        echo "EMA disabled: validation metrics reflect the raw (optimised) weights."
        ;;
esac

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
    --defect_zero_u_init="${DEFECT_ZERO_U_INIT}" \
    --enable_cueq="${ENABLE_CUEQ}" \
    --defect_logit_seed_gamma="${DEFECT_LOGIT_SEED_GAMMA}" \
    --defect_seed_anneal="${DEFECT_SEED_ANNEAL}" \
    --defect_seed_anneal_epochs="${DEFECT_SEED_ANNEAL_EPOCHS}" \
    --defect_alpha_mode="${DEFECT_ALPHA_MODE}" \
    --defect_beta="${DEFECT_BETA}" \
    --use_long_range="${USE_LONG_RANGE}" \
    --freeze_amplitude="${FREEZE_AMPLITUDE}" \
    --use_polarisation="${USE_POLARISATION}" \
    --pol_gate="${POL_GATE}" \
    --pol_gate_lambda="${POL_GATE_LAMBDA}" \
    --pol_gate_hops="${POL_GATE_HOPS}" \
    --host_carrier_coupling="${HOST_CARRIER_COUPLING}" \
    --eps_inf="${EPS_INF}" \
    --eps_inf_prior_weight="${EPS_INF_PRIOR_WEIGHT}" \
    --energy_weight="${ENERGY_WEIGHT}" \
    --forces_weight="${FORCES_WEIGHT}" \
    --delta_energy_weight="${DELTA_ENERGY_WEIGHT}" \
    --delta_forces_weight="${DELTA_FORCES_WEIGHT}" \
    --total_energy_weight="${TOTAL_ENERGY_WEIGHT}" \
    --defect_u_l2="${DEFECT_U_L2}" \
    --defect_zn_l2="${DEFECT_ZN_L2}" \
    --defect_size_weight="${DEFECT_SIZE_WEIGHT}" \
    --defect_size_ratio="${DEFECT_SIZE_RATIO}" \
    --defect_size_tol="${DEFECT_SIZE_TOL}" \
    --defect_size_warmup_epochs="${DEFECT_SIZE_WARMUP_EPOCHS}" \
    --defect_gauge_weight="${DEFECT_GAUGE_WEIGHT}" \
    --base_lr_factor="${BASE_LR_FACTOR}" \
    --config_type_weights="${CONFIG_TYPE_WEIGHTS}" \
    --defect_p_l2="${DEFECT_P_L2}" \
    --defect_qhost_l2="${DEFECT_QHOST_L2}" \
    --max_num_epochs="${MAX_NUM_EPOCHS}" \
    --batch_size="${BATCH_SIZE}" \
    --valid_batch_size="${VALID_BATCH_SIZE}" \
    --lr="${LR}" \
    ${EMA_ARGS[@]+"${EMA_ARGS[@]}"} \
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
