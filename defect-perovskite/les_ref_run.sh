#!/bin/bash
# W7.2 -- MACE-LES from scratch following the REFERENCE recipe (ChengUCB/les_fit,
# MLIPs/MACE-LES-new/fit.sh), with the latent charges LIVE at the library's own initialisation.
#
# What differs from the run that diverged (W7, base_v3_les), and why each is suspected:
#   scheduler_patience  50 -> 15   the reference's value. MY RUN'S LR WAS STILL 0.01 AT EPOCH 90:
#                                  with patience 50 the plateau scheduler essentially never
#                                  fired, and the forces were degrading for ~17 epochs before
#                                  the NaN (10.19 -> 12.00 -> 15.56 -> 21.40 -> nan).
#   loss                stress -> weighted (E + F only), stress weight 0 as the reference sets
#   weights             1 : 100 -> 10 : 1000 (same ratio, the reference's absolute values)
#   start_stage_two     80 % -> 50 % of epochs, as the reference does
#   patience            2048 -> 200, as the reference sets
#   latent charges      ZEROED -> library default. The zeroing PINNED them (q = 0 is a
#                                 stationary point of a quadratic Ewald energy), so the two
#                                 runs that used it were not LES models at all.
# lr is NOT set by the reference either: both use MACE's default 1e-2. The suspicion is the
# SCHEDULE, not the initial value.
#   usage: les_ref_run.sh NAME TRAIN VALID [EXTRA...]
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; TRAIN=$2; VALID=$3; shift 3
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS=${LES_EPOCHS:-320}
START2=${LES_START_STAGE_TWO:-160}
PATIENCE=${LES_SCHED_PATIENCE:-15}
SCALE=${LES_SCALE:-1.0}
python - "$NAME" "$TRAIN" "$VALID" "$EPOCHS" "$START2" "$PATIENCE" "$SCALE" "$@" <<'PY'
import sys
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
name, train, valid, epochs, start2, patience, scale = sys.argv[1:8]
extra = sys.argv[8:]
sys.argv = ["run_train",
  "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1",
  "--device", "cuda", "--default_dtype", "float32",
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--atomic_numbers", "[17, 55, 82]",
  # architecture: MH-1's, as every base in this campaign
  "--model", "MACELES",
  "--r_max", "6.0", "--num_radial_basis", "10", "--num_cutoff_basis", "5", "--max_ell", "3",
  "--interaction", "RealAgnosticResidualNonLinearInteractionBlock",
  "--interaction_first", "RealAgnosticResidualNonLinearInteractionBlock",
  "--num_interactions", "2",
  "--hidden_irreps", "512x0e+512x1o", "--edge_irreps", "128x0e+128x1o",
  "--MLP_irreps", "16x0e", "--gate", "silu", "--correlation", "3",
  "--radial_type", "bessel", "--radial_MLP", "[64, 64, 64]",
  "--pair_repulsion", "--distance_transform", "Agnesi",
  "--use_edge_irreps_first", "True", "--use_agnostic_product", "True",
  "--apply_cutoff", "False",
  "--les_readout_scale", scale,
  # the reference recipe
  "--loss", "weighted", "--energy_weight", "10", "--forces_weight", "1000",
  "--lr", "0.01", "--amsgrad", "--scheduler", "ReduceLROnPlateau",
  "--scheduler_patience", patience, "--patience", "200",
  "--ema", "--ema_decay", "0.99", "--clip_grad", "10.0",
  "--stage_two", "--start_stage_two", start2, "--max_num_epochs", epochs,
  "--batch_size", "8", "--valid_batch_size", "8",
  "--error_table", "PerAtomRMSE", "--eval_interval", "1",
  "--restart_latest", "--save_cpu", "--enable_cueq", "True"] + extra
main()
PY
echo "SCRATCH_EXIT $?"
