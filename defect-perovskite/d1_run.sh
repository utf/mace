#!/bin/bash
# D1 student arms on the D0 distillation set (registered flags: tracker "D1 flags and budgets").
#   usage: d1_run.sh ARM NAME        ARM in light | full-scratch | full-init     (D1_EPOCHS=1 for a smoke)
# light        : 1 interaction, r_max 6.0, 96x0e+96x1o, max_L 1, corr 3, 8 radial; les_fit recipe; 80 epochs, Stage Two at 40
# full-scratch : MH-1 shape from scratch (les_ref_run.sh's architecture); les_fit recipe; 40 epochs, Stage Two at 20
# full-init    : MH-1 shape, MH-1 weights (FoundationMACELES, no replay head); lr 1e-4, EMA 0.9999, 10/10, clip 1; 30 epochs
ARM=$1; NAME=$2
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=$W/defect-perovskite
R=$HOME/runs/$NAME; mkdir -p $R; cd $W
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH="$HOME/micromamba/envs/py13/bin:$PATH"
TRAIN=${D1_TRAIN:-$D/dataset_d0/train.xyz}; VALID=${D1_VALID:-$D/dataset_d0/valid.xyz}
LR=${D1_LR:-0.01}      # divergence guard relaunches at 0.005 (registered 14:30)
BATCH=${D1_BATCH:-8}   # light: 32 (user, 13:55 -- 110 ms/step at batch 8 was launch-bound); full arms: 8
case $ARM in
  light)        EPOCHS=${D1_EPOCHS:-80}; START2=$((EPOCHS / 2)) ;;
  full-scratch) EPOCHS=${D1_EPOCHS:-40}; START2=$((EPOCHS / 2)) ;;
  full-init)    EPOCHS=${D1_EPOCHS:-30}; START2=0 ;;
  light2)       EPOCHS=${D1_EPOCHS:-80}; START2=$((EPOCHS / 2)) ;;   # light width, 2 interactions (12 A reach)
  full1)        EPOCHS=${D1_EPOCHS:-40}; START2=$((EPOCHS / 2)) ;;   # MH-1 width, 1 interaction (6 A reach)
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
python - "$ARM" "$NAME" "$TRAIN" "$VALID" "$EPOCHS" "$START2" "$BATCH" "$LR" <<'PY'
import sys; sys.path.insert(0, "/home/alex/src/mace/.claude/worktrees/size-extensivity")
from mace.cli.run_train import main
arm, name, train, valid, epochs, start2, batch, lr = sys.argv[1:9]
common = ["run_train", "--name", name, "--work_dir", f"/home/alex/runs/{name}", "--seed", "1",
  "--device", "cuda", "--default_dtype", "float32",
  "--train_file", train, "--valid_file", valid, "--keep_isolated_atoms", "True",
  "--energy_key", "REF_energy", "--forces_key", "REF_forces", "--stress_key", "REF_stress",
  "--atomic_numbers", "[17, 55, 82]", "--model", "MACELES",
  "--batch_size", batch, "--valid_batch_size", batch, "--error_table", "PerAtomRMSEstressvirials", "--eval_interval", "1",
  "--max_num_epochs", epochs, "--restart_latest", "--save_cpu", "--enable_cueq", "True"]
les_fit = ["--loss", "stress", "--energy_weight", "10", "--forces_weight", "1000", "--stress_weight", "10", "--compute_stress", "True",
  "--lr", lr, "--amsgrad", "--scheduler", "ReduceLROnPlateau", "--scheduler_patience", "15", "--patience", "200",
  "--ema", "--ema_decay", "0.99", "--clip_grad", "10.0", "--stage_two", "--start_stage_two", start2,
  "--les_readout_scale", "1.0"]
full_shape = ["--r_max", "6.0", "--num_radial_basis", "10", "--num_cutoff_basis", "5", "--max_ell", "3",
  "--interaction", "RealAgnosticResidualNonLinearInteractionBlock", "--interaction_first", "RealAgnosticResidualNonLinearInteractionBlock",
  "--num_interactions", "2", "--hidden_irreps", "512x0e+512x1o", "--edge_irreps", "128x0e+128x1o",
  "--MLP_irreps", "16x0e", "--gate", "silu", "--correlation", "3", "--radial_type", "bessel", "--radial_MLP", "[64, 64, 64]",
  "--pair_repulsion", "--distance_transform", "Agnesi", "--use_edge_irreps_first", "True", "--use_agnostic_product", "True", "--apply_cutoff", "False"]
light_shape = ["--r_max", "6.0", "--num_radial_basis", "8", "--num_cutoff_basis", "5", "--max_ell", "1",
  "--interaction", "RealAgnosticResidualNonLinearInteractionBlock", "--interaction_first", "RealAgnosticResidualNonLinearInteractionBlock",
  "--num_interactions", "1", "--hidden_irreps", "96x0e+96x1o",
  "--MLP_irreps", "16x0e", "--gate", "silu", "--correlation", "3", "--radial_type", "bessel", "--radial_MLP", "[64, 64, 64]",
  "--pair_repulsion", "--distance_transform", "Agnesi", "--apply_cutoff", "False"]
def with_layers(shape, n):
    i = shape.index("--num_interactions"); return shape[:i + 1] + [str(n)] + shape[i + 2:]
if arm == "light":
    sys.argv = common + light_shape + les_fit
elif arm == "full-scratch":
    sys.argv = common + full_shape + les_fit
elif arm == "light2":
    sys.argv = common + with_layers(light_shape, 2) + les_fit
elif arm == "full1":
    sys.argv = common + with_layers(full_shape, 1) + les_fit
else:   # full-init: MH-1 weights, direct fine-tune, base v2's optimiser settings
    sys.argv = common + ["--foundation_model", "/home/alex/.cache/mace/macemh1model", "--foundation_head", "omat_pbe",
      "--multiheads_finetuning", "False", "--les_readout_scale", "0.01",
      "--loss", "stress", "--energy_weight", "10", "--forces_weight", "10", "--stress_weight", "1", "--compute_stress", "True",
      "--lr", "0.0001", "--ema", "--ema_decay", "0.9999", "--weight_decay", "0.0", "--clip_grad", "1.0"]
main()
PY
echo "D1_EXIT $?"
