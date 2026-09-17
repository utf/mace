#!/bin/bash
# Plan P Phase 2 arm A (W6 as shipped on v5), seeds 0 1 2 on the local A4000 in series.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; BASE=$1
declare -A FOLD=([0]=0 [1]=1 [2]=2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W
cd $W
for s in 0 1 2; do
  R=/home/alex/runs/dscc/dscc_p2a_s$s; mkdir -p $R
  echo "START $(date)" > $R/STATUS
  python defect-perovskite/dscc_train.py --name dscc_p2a_s$s --seed $s --fold ${FOLD[$s]} --epochs 60 --directional 1 --regime B \
    --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 \
    --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 \
    --host_coupling mulliken --cache_base 1 --run_dir $R > $R/nohup.log 2>&1 && echo "DONE $(date)" >> $R/STATUS || echo "FAILED $(date)" >> $R/STATUS
done
