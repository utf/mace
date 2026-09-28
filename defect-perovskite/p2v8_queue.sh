#!/bin/bash
# Plan P Phase 2b on base v8 (registered 2026-09-28): queue B_s (all six seeds first) then B_0 on b3 (GPUs 5/7 via w3_queue.sh).
# Names carry the v8 tag (dscc_p2v8bs_s*, dscc_p2v8b0_s*) so the v7 records are untouched.   usage: p2v8_queue.sh BASE
BASE=$1
COMMON="--epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 0 --bz_readout 1"
POL="--host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --r_c 8.0 --delta 2.0"
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
Q=/home/alex/runs/p2v8_queue.txt; : > $Q
for s in 0 1 2 3 4 6; do echo "dscc_p2v8bs_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL --near_field mulliken" >> $Q; done
for s in 0 1 2 3 4 6; do echo "dscc_p2v8b0_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL --near_field none" >> $Q; done
echo "queue: $(wc -l < $Q) lines -> $Q (b3: B_s x6 then B_0 x6, seeds 0 1 2 3 4 6)"
