#!/bin/bash
# Plan P Phase 2b §5 localisation remedy: the adopted arm (B_s) retrained once with w_gap x 3
# (3.0), no other change; six seeds; names dscc_p2bsg_s*.   usage: p2b_queue_gap3.sh BASE
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
BASE=$1
COMMON="--epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 0 --bz_readout 1 --gap_weight 3.0"
POL="--host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --r_c 8.0 --delta 2.0 --near_field mulliken"
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
Q=/home/alex/runs/p2b_queue.txt; : > $Q
for s in 0 1 2 3 4 6; do echo "dscc_p2bsg_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL" >> $Q; done
echo "queue: $(wc -l < $Q) lines -> $Q (B_s with w_gap 3.0, seeds 0 1 2 3 4 6)"
