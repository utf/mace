#!/bin/bash
# D2: multi-head DFT fine-tune of a D1 student with the distillation set as replay.
#   usage: d2_run.sh STUDENT_MODEL NAME        (D2_EPOCHS=1 for a smoke; D2_TRAIN/D2_VALID override the DFT files)
# Heads: 'Default' = the plan's `dft` head (dataset_e0, DFT labels, estimated-E0 isolated atoms at
# config_weight 1000); 'pt_head' = the plan's `teacher` head (replay_1616.xyz: 1616 D0 frames
# stratified by family + the teacher-labelled isolated atoms). Shared backbone, SHARED latent-charge
# readout transferred from the student (unscaled), per-head energy readouts; base v2's optimiser
# settings (energy 10 / forces 10, lr 1e-4 forced, EMA 0.9999, clip 1, wd 0, 30 epochs, batch 8,
# float32, cueq).
STUDENT=$1; NAME=$2
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TRAIN=${D2_TRAIN:-$D/dataset_e0/train_iso.xyz}; VALID=${D2_VALID:-$D/dataset_e0/valid.xyz}; REPLAY=${D2_REPLAY:-$D/dataset_d0/replay_1616.xyz}
EPOCHS=${D2_EPOCHS:-30}; BATCH=${D2_BATCH:-8}
python - "$STUDENT" "$NAME" "$TRAIN" "$VALID" "$REPLAY" "$EPOCHS" "$BATCH" <<'PY'
import sys; sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
student, name, train, valid, replay, epochs, batch = sys.argv[1:8]
sys.argv = ["run_train", "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1", "--device", "cuda", "--default_dtype", "float32",
  "--model", "MACELES", "--foundation_model", student, "--foundation_head", "Default", "--multiheads_finetuning", "True",
  "--pt_train_file", replay, "--atomic_numbers", "[17, 55, 82]", "--les_shared_readout", "True",
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--energy_weight", "10", "--forces_weight", "10",
  "--force_mh_ft_lr", "True", "--lr", "0.0001", "--ema", "--ema_decay", "0.9999", "--weight_decay", "0.0", "--clip_grad", "1.0",
  "--max_num_epochs", epochs, "--batch_size", batch, "--valid_batch_size", batch, "--error_table", "PerAtomRMSE", "--eval_interval", "1",
  "--restart_latest", "--save_cpu", "--enable_cueq", "True"]
main()
PY
echo "D2_EXIT $?"
