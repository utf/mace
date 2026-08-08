#!/usr/bin/env bash
#
# SLURM submission template for the 4H-SiC divacancy MACEDefect run.
#
#   sbatch submit_slurm.sh
#
# >>> EVERY LINE MARKED [EDIT] IS A PLACEHOLDER <<<  They are guesses about your cluster,
# not working values. Check them before the first submission.
#
# The job is deliberately single-GPU. At 800 structures the run is small, and MACE's
# --distributed path adds a DDP loss reduction that DefectLoss's masked terms have not
# been exercised against. Scale the model before scaling the node count.
#
#SBATCH --job-name=macedefect-sic
#SBATCH --partition=gpu                  # [EDIT] your GPU partition
#SBATCH --account=                       # [EDIT] your allocation, or delete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                     # [EDIT] some sites use --gpus=1 or -C gpu
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#
# Requeue on preemption is OPT-IN: uncomment BOTH lines below to enable it. Left off,
# nothing triggers a requeue and the --restart_latest below is simply a no-op.
##SBATCH --signal=B:USR1@120
##SBATCH --requeue

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- environment [EDIT] ---------------------------------------------------------------
# Replace with whatever your site uses: module load, conda activate, venv, container.
# module load cuda/12.4
# source /path/to/venv/bin/activate
# conda activate mace

# Point at the defect branch. `pip install -e .` in the repo makes this redundant but
# harmless; without either, mace_run_train resolves to a released mace-torch that has no
# MACEDefect and the job dies immediately.
export MACE_REPO="${MACE_REPO:-$(cd "${HERE}/.." && pwd)}"

# Threads: MACE is GPU-bound here, and letting OpenMP grab every core on a shared node
# slows the dataloader down rather than speeding anything up.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

# ---- run settings ---------------------------------------------------------------------
export NAME="${NAME:-MACEDefect_SiC_divacancy}"
export WORK_DIR="${WORK_DIR:-${HERE}/runs/${NAME}}"
export DEVICE=cuda

# A production-sized model. 64 channels is the local default; 128 with max_L=1 is a
# reasonable first cluster run for ~300-400 atom cells. If you hit OOM, drop BATCH_SIZE
# to 2 before dropping NUM_CHANNELS -- the batch is 4 x ~350 atoms and dominates
# activation memory, and float64 doubles all of it.
export NUM_CHANNELS="${NUM_CHANNELS:-128}"
export MAX_L="${MAX_L:-1}"
export NUM_INTERACTIONS="${NUM_INTERACTIONS:-2}"
export R_MAX="${R_MAX:-5.0}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export VALID_BATCH_SIZE="${VALID_BATCH_SIZE:-4}"
export MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-500}"
export LR="${LR:-0.005}"
export DEFAULT_DTYPE="${DEFAULT_DTYPE:-float64}"

# Short-range first: with q = 0 the screening amplitude is unidentifiable, so the
# long-range branch costs time and adds a parameter nothing fits. See notes.md section 2.
export USE_LONG_RANGE="${USE_LONG_RANGE:-False}"

echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Is)"
nvidia-smi || true

# Deliberately NOT `srun`. For a single-task single-GPU job srun buys nothing, and whether
# it forwards the exported settings above depends on the site's SLURM_EXPORT_ENV -- some
# clusters set it to NONE, in which case train_defect_model.sh would silently fall back to
# its own defaults (64 channels, not 128; a different WORK_DIR) rather than failing. That
# is a wrong-model run, not a crash. If your site needs srun, use `srun --export=ALL ...`.
#
# --restart_latest resumes from the last checkpoint if the job was requeued (see the
# opt-in block above). It is a no-op on a fresh run.
bash "${HERE}/train_defect_model.sh" --restart_latest

echo "finished at $(date -Is)"
