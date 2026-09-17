#!/bin/bash
# Plan P Phase 1, amendment A5: base v7 = MACE-Field (mp-dielectric head) fine-tuned on the
# neutral defect data with MACE-Field response targets at P 10 / Z* 1 / alpha 1 (anchor test, arm b),
# the relabelled base v2 replay without response labels (replay_v4), head-grouped batches so the
# replay skips the field derivatives, estimated E0s via isolated atoms, no E_PP in the forward.
# Usage: p1v3_run.sh NAME LR OUTDIR [extra args]
NAME=$1; LR=$2; OUT=$3; shift 3
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; D=/home/alex/runs/dscc/p1v2
CKPT=/home/alex/src/macefield/MACEField-ASE-MD/MACEField-omat-dielectric-converged.model
mkdir -p $OUT; cd $W; export PYTHONPATH=$W
python -c "from mace.cli.run_train import main; main()" \
  --name $NAME \
  --heads "{'Default': {'train_file': '$D/train_v2_iso.xyz', 'valid_file': '$D/valid_v2.xyz'}}" \
  --foundation_model $CKPT --foundation_head mp-dielectric \
  --pt_train_file $D/replay_v4.xyz \
  --multiheads_finetuning True --pseudolabel_replay False --keep_isolated_atoms True \
  --model MACEField --loss universal_field --error_table PerAtomFieldRMSE \
  --energy_weight 10 --forces_weight 10 --stress_weight 1 \
  --polarization_weight 10 --becs_weight 1 --polarizability_weight 1 \
  --compute_forces True --compute_stress True --compute_polarization True --compute_becs True --compute_polarizability True \
  --force_mh_ft_lr True --lr $LR --ema --ema_decay 0.9999 --weight_decay 0.0 --clip_grad 1.0 \
  --max_num_epochs 30 --batch_size 4 --valid_batch_size 4 --default_dtype float32 --enable_cueq True \
  --pp_r_split 0 --pp_eps_inf 4.0 --seed 1 --eval_interval 1 --head_grouped_batches True \
  --results_dir $OUT --checkpoints_dir $OUT/checkpoints --log_dir $OUT --model_dir $OUT --downloads_dir $OUT \
  --device cuda "$@"
