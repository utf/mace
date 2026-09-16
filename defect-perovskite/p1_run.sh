#!/bin/bash
# Plan P Phase 1: base v5 = MACE-Field checkpoint fine-tuned on the neutral defect data with
# pseudolabelled replay, E_PP (Phase 0 r_split) in the forward. Usage: p1_run.sh NAME EPOCHS OUTDIR [extra args]
NAME=$1; EPOCHS=$2; OUT=$3; shift 3
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
CKPT=/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model
mkdir -p $OUT; cd $W
export PYTHONPATH=$W
python -c "from mace.cli.run_train import main; main()" \
  --name $NAME \
  --heads "{'Default': {'train_file': '$W/defect-perovskite/dataset_e0/train.xyz', 'valid_file': '$W/defect-perovskite/dataset_e0/valid.xyz'}}" \
  --foundation_model $CKPT --foundation_head pt_head \
  --pt_train_file /home/alex/runs/dscc/p1/replay_fps1000.xyz \
  --multiheads_finetuning True --pseudolabel_replay True \
  --E0s foundation --model MACEField --loss universal_field --error_table PerAtomFieldRMSE \
  --energy_weight 10 --forces_weight 10 --stress_weight 1 \
  --polarization_weight 1000 --becs_weight 100 --polarizability_weight 100 \
  --compute_forces True --compute_stress True --compute_polarization True --compute_becs True --compute_polarizability True \
  --force_mh_ft_lr True --lr 0.0001 --ema --ema_decay 0.9999 --weight_decay 0.0 --clip_grad 1.0 \
  --max_num_epochs $EPOCHS --batch_size 8 --valid_batch_size 8 --default_dtype float32 --enable_cueq True \
  --pp_r_split 9.645 --pp_eps_inf 4.0 --seed 1 --eval_interval 1 \
  --results_dir $OUT --checkpoints_dir $OUT/checkpoints --log_dir $OUT --model_dir $OUT --downloads_dir $OUT \
  --device cuda "$@"
