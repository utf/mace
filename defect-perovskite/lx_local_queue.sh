#!/bin/bash
# Generic local sequential runner for one arm of the LES head line (the A4000 takes one W6-type
# run at a time). Env: LX_NAME (run prefix, e.g. dscc_l2), LX_SEEDS ("0 1 2"), LX_BASE, LX_EXTRA
# (extra dscc_train.py flags), LX_INIT_PAT (optional: printf pattern with %d for --init_state).
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
declare -A FOLD=([0]=0 [1]=1 [2]=2 [3]=3 [4]=0 [6]=2)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W
for s in ${LX_SEEDS:-0 1 2}; do
  R=/home/alex/runs/dscc/${LX_NAME}_s$s
  [ -d $R ] && { echo "$(date '+%F %T') $R exists, skipping"; continue; }
  mkdir -p $R
  init=""; [ -n "${LX_INIT_PAT:-}" ] && init="--init_state $(printf "$LX_INIT_PAT" $s)"
  echo "$(date '+%F %T') starting ${LX_NAME}_s$s fold ${FOLD[$s]} $init"
  /home/alex/micromamba/envs/py13/bin/python defect-perovskite/dscc_train.py \
    --name ${LX_NAME}_s$s --seed $s --fold ${FOLD[$s]} --epochs 60 --directional 1 --regime B \
    --base $LX_BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 \
    --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 \
    $LX_EXTRA $init --run_dir $R > $R/nohup.log 2>&1
  echo "$(date '+%F %T') ${LX_NAME}_s$s exit $?"
done
