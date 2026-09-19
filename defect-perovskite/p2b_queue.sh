#!/bin/bash
# Plan P Phase 2b §4: queue B_s and B_0 (seeds 0 1 2 3 4 6) on b3 (GPUs 5/7 via w3_queue.sh); the
# control's remaining seeds (A s3/s4/s6, with the BZ regulariser) run on the local card (p2b_local.sh).
#   usage: p2b_queue.sh BASE
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
BASE=$1
COMMON="--epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 1"
POL="--host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --r_c 8.0 --delta 2.0"
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
Q=/home/alex/runs/p2b_queue.txt; : > $Q
for s in 0 1 2 3 4 6; do
  echo "dscc_p2bs_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL --near_field mulliken" >> $Q
  echo "dscc_p2b0_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL --near_field none" >> $Q
done
echo "queue: $(wc -l < $Q) lines -> $Q (b3: B_s and B_0, seeds 0 1 2 3 4 6, interleaved)"
