#!/bin/bash
# W7 -- base v3: MACE trained FROM SCRATCH with MACE-MH-1's architecture, on the same neutral
# data, the same E0s and the same validation split as base v2, so the three sit in one table.
#
# ARCHITECTURE = MH-1's, read off the checkpoint with `extract_config_mace_model` and passed
# explicitly: 512x0e+512x1o hidden, two RealAgnosticResidualNonLinear interaction blocks,
# edge irreps 128x0e+128x1o in the first block, element-agnostic product, correlation 3,
# max_ell 3, 10 Bessel + p = 5 polynomial cutoff at r_max 6.0, Agnesi distance transform, ZBL
# pair repulsion, radial MLP [64, 64, 64], MLP_irreps 16x0e. The only deliberate differences
# are the ones a single-material three-element model must have: 3 elements instead of 89, one
# head instead of six, and the scale/shift refitted from this data (`rms_forces_scaling`).
#
# TRAINING = the README's from-scratch recipe, NOT base v2's fine-tune settings: lr 1e-2,
# amsgrad, ReduceLROnPlateau, EMA 0.99, Stage Two for the last 20 % of epochs, and the epoch
# count from the documented heuristic (~200k gradient updates: 1619 frames / batch 8 x 1000
# epochs = 202k). `--loss stress` because the labels carry REF_stress and the base v2 table has
# a stress column -- the same 1 : 100 : 1 energy/forces/stress the docs use, with the code's
# Stage Two weights 1000 : 100 : 10.
#
# E0s: the SAME estimated isolated-atom energies base v2 used, carried as IsolatedAtom frames
# in train_iso.xyz at config_weight 1000 and kept as training points. They came from the
# foundation model's residuals, so they are a constant reference rather than DFT -- used here
# because sharing base v2's reference is what makes the energy columns comparable.
#
#   usage: scratch_run.sh NAME MODEL TRAIN_XYZ VALID_XYZ [EXTRA ARGS...]
#          MODEL is ScaleShiftMACE or MACELES
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; MODEL=$2; TRAIN=$3; VALID=$4; shift 4
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS=${SCRATCH_EPOCHS:-1000}
START2=${SCRATCH_START_STAGE_TWO:-800}
python - "$NAME" "$MODEL" "$TRAIN" "$VALID" "$EPOCHS" "$START2" "$@" <<'PY'
import sys
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
name, model, train, valid, epochs, start2 = sys.argv[1:7]
extra = sys.argv[7:]
sys.argv = ["run_train",
  "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1",
  "--device", "cuda", "--default_dtype", "float32",
  # ---- data: identical to base v2
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--atomic_numbers", "[17, 55, 82]",
  # ---- architecture: MACE-MH-1's
  "--model", model,
  "--r_max", "6.0", "--num_radial_basis", "10", "--num_cutoff_basis", "5", "--max_ell", "3",
  "--interaction", "RealAgnosticResidualNonLinearInteractionBlock",
  "--interaction_first", "RealAgnosticResidualNonLinearInteractionBlock",
  "--num_interactions", "2",
  "--hidden_irreps", "512x0e+512x1o", "--edge_irreps", "128x0e+128x1o",
  "--MLP_irreps", "16x0e", "--gate", "silu", "--correlation", "3",
  "--radial_type", "bessel", "--radial_MLP", "[64, 64, 64]",
  "--pair_repulsion", "--distance_transform", "Agnesi",
  "--use_edge_irreps_first", "True", "--use_agnostic_product", "True",
  "--apply_cutoff", "False",          # MH-1's setting; the code default is True
  "--scaling", "rms_forces_scaling",
  # ---- from-scratch training strategy (README)
  "--loss", "stress", "--energy_weight", "1", "--forces_weight", "100", "--stress_weight", "1",
  "--compute_stress", "True",
  "--lr", "0.01", "--amsgrad", "--scheduler", "ReduceLROnPlateau", "--clip_grad", "10.0",
  "--ema", "--ema_decay", "0.99",
  "--stage_two", "--start_stage_two", start2, "--max_num_epochs", epochs,
  "--batch_size", "8", "--valid_batch_size", "8",
  "--error_table", "PerAtomRMSEstressvirials", "--eval_interval", "1",
  "--restart_latest", "--save_cpu", "--enable_cueq", "True"] + extra
main()
PY
echo "SCRATCH_EXIT $?"
