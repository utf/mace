#!/bin/bash
# L1 (plain W6 on base_v4_lesmh_prod) on the LOCAL A4000: the seeds in $SEEDS one after another
# (a W6 step peaks at 11.6 GB of 16). Flags = the registered W6 command (launch_w6_s0.sh) with
# the L0 base; nothing else changes. Folds by the W0.3 map.
#   usage: L1_SEEDS="0 1 2" l1_local_queue.sh
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
BASE=${L1_BASE:-/home/alex/runs/base_v4_lesmh_prod/base_v4_lesmh_prod_base.pt}
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W
for s in ${L1_SEEDS:-0 1 2}; do
  R=/home/alex/runs/dscc/dscc_l1_s$s
  [ -d $R ] && { echo "$(date '+%F %T') $R exists, skipping"; continue; }
  mkdir -p $R
  echo "$(date '+%F %T') starting dscc_l1_s$s fold ${FOLD[$s]}"
  /home/alex/micromamba/envs/py13/bin/python defect-perovskite/dscc_train.py \
    --name dscc_l1_s$s --seed $s --fold ${FOLD[$s]} --epochs 60 --directional 1 --regime B \
    --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 \
    --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 \
    --run_dir $R > $R/nohup.log 2>&1
  echo "$(date '+%F %T') dscc_l1_s$s exit $?"
done
