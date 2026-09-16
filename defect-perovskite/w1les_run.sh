#!/bin/bash
# L0 (LES head line, 2026-09-15): base v2's registered multi-head fine-tune of MACE-MH-1 -- the
# omat_pbe head, the sel10k fps/combinations replay, energy 10 / forces 10, lr 1e-4 forced, EMA
# 0.9999, clip 1.0, wd 0, 30 epochs, batch 8, float32, cueq, E0s carried as IsolatedAtom frames --
# with ONE addition: `--model MACELES --les_readout_scale 0.01`, the latent-charge readouts of
# LES starting small and LIVE (LES-FT2's value). Nothing else is tuned for the LES term.
#   usage: w1les_run.sh NAME TRAIN_XYZ VALID_XYZ         (W1LES_EPOCHS=1 for a smoke)
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; NAME=$1; TRAIN=$2; VALID=$3
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
EPOCHS=${W1LES_EPOCHS:-30}
python - "$NAME" "$TRAIN" "$VALID" "$EPOCHS" <<'PY'
import sys; sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
name, train, valid, epochs = sys.argv[1:5]
sys.argv = ["run_train",
  "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1", "--device", "cuda", "--default_dtype", "float32",
  "--foundation_model", "/home/alex/.cache/mace/macemh1model", "--foundation_head", "omat_pbe", "--multiheads_finetuning", "True",
  "--pt_train_file", "/home/alex/.cache/mace/replay_mh1_omat_pbe_sel10k.xyz", "--atomic_numbers", "[17, 55, 82]",
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--energy_weight", "10", "--forces_weight", "10",
  "--force_mh_ft_lr", "True", "--lr", "0.0001", "--ema", "--ema_decay", "0.9999", "--weight_decay", "0.0", "--clip_grad", "1.0",
  "--max_num_epochs", epochs, "--batch_size", "8", "--valid_batch_size", "8", "--error_table", "PerAtomRMSE", "--eval_interval", "1", "--save_cpu", "--enable_cueq", "True",
  # ---- the one addition
  "--model", "MACELES", "--les_readout_scale", "0.01"]
main()
PY
echo "W1LES_EXIT $?"
