#!/bin/bash
# Plan P Phase 2c Step 1 rerun (registered 2026-10-11 18:00): arm B_w with the s-p on-site half-width (0.1) and the gap term on H0 - W; otherwise third run (bw3: the spike fix), seeds 0 2 6 (folds 0 2 2), s_init = B_s w01's epoch-59 s per seed, W6 schedule on base_v8_w01,
# per-epoch level readouts (held fold 79-atom +1 frames; the 639-atom label-tiled structure (b)). Queue for b3 GPUs 5/7 via w3_queue.sh.   usage: p2cw_queue.sh BASE
BASE=$1
COMMON="--epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 0 --bz_readout 1"
POL="--host_coupling polarisation --field_provider base --cache_base 0 --r_split 4.5 --eps_inf 4.0 --r_c 8.0 --delta 2.0"
BW="--near_field potential --delta_rule sp_splitting --delta_frac 0.1 --level_readout 1 --embed_readout /home/alex/runs/dscc/p2c/embed_b_label_w01.json ${BW3_FLAGS:?set BW3_FLAGS to the spike fix flags}"
declare -A FOLD=([0]=0 [2]=2 [6]=2)
declare -A SINIT=([0]=0.376 [2]=0.411 [6]=0.555)
Q=/home/alex/runs/p2cw3_queue.txt; : > $Q
for s in 0 2 6; do echo "dscc_p2cwbw3_s$s|--seed $s --fold ${FOLD[$s]} $COMMON $POL $BW --s_init ${SINIT[$s]}" >> $Q; done
echo "queue: $(wc -l < $Q) lines -> $Q (b3: B_w seeds 0 2 6)"
