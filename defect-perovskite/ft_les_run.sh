#!/bin/bash
# W7.1 -- base v3 LES-FT: MACE-LES initialised from the MACE-MH-1 weights and DIRECT
# (single-head, no replay) fine-tuned, per the MACE finetuning-guidance page's NAIVE
# fine-tuning column -- which is a different recipe from the multihead one base v2 used:
#
#            naive / direct (this run)      multihead replay (base v2)
#   lr       1e-3                           1e-4
#   EMA      0.999                          0.9999
#   clip     1.0                            1.0
#   wd       0.0                            0.0
#   loss     constant 10 : 10, no stage two  constant 10 : 10, no stage two
#   replay   none                           10k fps/combinations pseudolabels
#
# The guide is explicit that naive fine-tuning has the worst MD stability and the greatest
# drift from the pretrained distribution of the methods it compares -- that is the experiment,
# and the epoch trajectory is reported so the drift is visible rather than assumed. The guide
# gives 10-30 epochs for multihead and is silent for naive; this runs 60 with best-validation
# selection so the minimum and any drift past it are both in the record.
#
# `--model MACELES` with `--foundation_model` becomes `FoundationMACELES`: MH-1's architecture
# and weights, plus LES's latent-charge readouts, which `load_foundations_elements` does NOT
# touch -- they are new modules and are learned here.
#
# E0s: the same estimated isolated-atom energies base v2 and the from-scratch runs use, carried
# as IsolatedAtom frames at config_weight 1000 and kept as training points -- which is also the
# guide's own advice ("include true isolated-atom configurations as explicit training points").
#
#   usage: ft_les_run.sh NAME TRAIN_XYZ VALID_XYZ [EXTRA...]
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; TRAIN=$2; VALID=$3; shift 3
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS=${FT_EPOCHS:-60}
python - "$NAME" "$TRAIN" "$VALID" "$EPOCHS" "$@" <<'PY'
import sys
sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
name, train, valid, epochs = sys.argv[1:5]
extra = sys.argv[5:]
sys.argv = ["run_train",
  "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1",
  "--device", "cuda", "--default_dtype", "float32",
  # ---- foundation: MH-1, DIRECT fine-tuning (no replay head)
  "--model", "MACELES",
  "--foundation_model", "/home/alex/.cache/mace/macemh1model",
  "--foundation_head", "omat_pbe",
  "--multiheads_finetuning", "False",
  # ---- data: identical to base v2 and to the from-scratch runs
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--atomic_numbers", "[17, 55, 82]",
  # ---- the guide's NAIVE fine-tuning settings
  "--lr", "0.001", "--ema", "--ema_decay", "0.999",
  "--weight_decay", "0.0", "--clip_grad", "1.0",
  "--energy_weight", "10", "--forces_weight", "10", "--stress_weight", "1",
  "--loss", "stress", "--compute_stress", "True",
  # no stage two: the guide says constant weights, no force-then-energy transition
  "--max_num_epochs", epochs, "--batch_size", "8", "--valid_batch_size", "8",
  "--error_table", "PerAtomRMSEstressvirials", "--eval_interval", "1",
  "--restart_latest", "--save_cpu", "--enable_cueq", "True"] + extra
main()
PY
echo "SCRATCH_EXIT $?"
