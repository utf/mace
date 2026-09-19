#!/bin/bash
# Phase 2b smoke: one epoch of each new arm (and the control with the BZ regulariser) on the local card.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; cd $W; export PYTHONPATH=$W PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CUDA_DEV:-0}
BASE=/home/alex/runs/dscc/p1v2/base_v7_lr3e4/base.pt
run() { name=$1; shift; R=/home/alex/runs/dscc/p2b/smoke_$name; rm -rf $R; mkdir -p $R
  python defect-perovskite/dscc_train.py --name smoke_$name --seed 0 --fold 0 --epochs 1 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 1 "$@" --run_dir $R --device cuda > $R/nohup.log 2>&1
  echo "$name exit $? | $(grep -h 'epoch 0\|BZ regulariser\|Phase 2b\|Traceback\|Error' $R/nohup.log | tail -3 | cut -c1-220)"; }
run a --host_coupling mulliken --cache_base 1
run bs --host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --near_field mulliken --r_c 8.0 --delta 2.0
run b0 --host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --near_field none --r_c 8.0 --delta 2.0
echo SMOKE_DONE
