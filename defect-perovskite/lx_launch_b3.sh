#!/bin/bash
# Queue seeds 3 4 6 of one arm on b3 GPUs 5/7 through w3_queue.sh. Args: NAME BASE "EXTRA" [INIT_PAT]
# (INIT_PAT: printf pattern with %d, a b3 path to the same-seed source model.pt)
W=/home/alex/src/mace/.claude/worktrees/size-extensivity
NAME=$1; BASE=$2; EXTRA=$3; INIT_PAT=${4:-}
Q=/home/alex/runs/${NAME}_queue.txt; : > $Q
for s in 3 4 6; do
  f=$([ $s = 3 ] && echo 3 || ([ $s = 4 ] && echo 0 || echo 2))
  init=""; [ -n "$INIT_PAT" ] && init="--init_state $(printf "$INIT_PAT" $s)"
  echo "${NAME}_s$s|--seed $s --fold $f --epochs 60 --directional 1 --regime B --base $BASE --coupling 0 --scf_free 1 --route_b 1 --beta_b 0.0 --beta_a 0.0 --energy_weight 0.05 --base_float32 1 --setup_batch_size 4 --base_cache_batch_size 4 --gpu_memory_fraction 0 $EXTRA $init" >> $Q
done
cd $W
W3_GPUS="5 7" W3_PER_GPU=1 W3_HEADROOM=13000 W3_QUEUE=$Q W3_LOG=/home/alex/runs/${NAME}_queue.log nohup bash defect-perovskite/w3_queue.sh > /dev/null 2>&1 &
echo "$(date '+%F %T') $NAME runner started on b3 ($(wc -l < $Q) lines)"
