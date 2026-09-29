#!/bin/bash
# Phase 2b on base_v8_w01: control arm A on the local card. Seed 0 = the floor check dscc_p2v8x_aw01_s0 (identical command), copied under the arm name when it finishes;
# then seeds 1 2 3 4 6 in series, all with --bz_gap 0 --bz_readout 1.
W=/home/alex/src/mace/.claude/worktrees/size-extensivity; BASE=/home/alex/runs/dscc/p1v8/base_v8_w01/base.pt
declare -A FOLD=([1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$W CUDA_VISIBLE_DEVICES=0 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd $W
until grep -qE "DONE|FAILED" /home/alex/runs/dscc/dscc_p2v8x_aw01_s0/STATUS 2>/dev/null; do sleep 120; done
if grep -q DONE /home/alex/runs/dscc/dscc_p2v8x_aw01_s0/STATUS; then rm -rf /home/alex/runs/dscc/dscc_p2v8wa_s0; cp -r /home/alex/runs/dscc/dscc_p2v8x_aw01_s0 /home/alex/runs/dscc/dscc_p2v8wa_s0; fi
for s in 1 2 3 4 6; do
  R=/home/alex/runs/dscc/dscc_p2v8wa_s$s; rm -rf $R; mkdir -p $R
  echo "START $(date)" > $R/STATUS
  python defect-perovskite/dscc_train.py --name dscc_p2v8wa_s$s --seed $s --fold ${FOLD[$s]} --epochs 60 --directional 1 --regime B \
    --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 \
    --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 --bz_gap 0 --bz_readout 1 \
    --host_coupling mulliken --cache_base 1 --run_dir $R --device cuda > $R/nohup.log 2>&1 && echo "DONE $(date)" >> $R/STATUS || echo "FAILED $(date)" >> $R/STATUS
done
