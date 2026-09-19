#!/bin/bash
# Plan P Phase 2b: the control's remaining seeds (A s3/s4/s6) with the BZ regulariser, local card, in series.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; BASE=$1
declare -A FOLD=([3]=3 [4]=0 [6]=2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W CUDA_VISIBLE_DEVICES=0
cd $W
for s in 3 4 6; do
  R=/home/alex/runs/dscc/dscc_p2a_s$s; rm -rf $R; mkdir -p $R
  echo "START $(date)" > $R/STATUS
  python defect-perovskite/dscc_train.py --name dscc_p2a_s$s --seed $s --fold ${FOLD[$s]} --epochs 60 --directional 1 --regime B \
    --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 \
    --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 1 \
    --host_coupling mulliken --cache_base 1 --run_dir $R --device cuda > $R/nohup.log 2>&1 && echo "DONE $(date)" >> $R/STATUS || echo "FAILED $(date)" >> $R/STATUS
done
