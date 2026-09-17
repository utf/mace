#!/bin/bash
# Plan P Phase 2: queue the arms on b3 (GPUs 5/7 via w3_queue.sh) and arm A on the local card.
#   usage: p2_queue.sh BASE EPS_M     (EPS_M = v5's polycrystalline eps_inf for arm B(M))
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
BASE=$1; EPS_M=$2
COMMON="--epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0"
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
Q=/home/alex/runs/p2_queue.txt; : > $Q
for s in 0 1 2 3 4 6; do
  echo "dscc_p2b_s$s|--seed $s --fold ${FOLD[$s]} $COMMON --host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --run_dir /home/alex/runs/dscc/p2/dscc_p2b_s$s" >> $Q
done
for s in 0 1 2; do
  echo "dscc_p2bm_s$s|--seed $s --fold ${FOLD[$s]} $COMMON --host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf $EPS_M --run_dir /home/alex/runs/dscc/p2/dscc_p2bm_s$s" >> $Q
done
for s in 3 4 6; do
  echo "dscc_p2a_s$s|--seed $s --fold ${FOLD[$s]} $COMMON --host_coupling mulliken --cache_base 1 --run_dir /home/alex/runs/dscc/p2/dscc_p2a_s$s" >> $Q
done
echo "queue: $(wc -l < $Q) lines -> $Q (b3: arms B, B(M) and A seeds 3/4/6); local: A seeds 0/1/2 via p2_local.sh"
